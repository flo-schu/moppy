import os
import re
from functools import lru_cache
import tempfile
import inspect
import warnings

import numpy as np
import xarray as xr
import arviz as az
from matplotlib import pyplot as plt

from pymob.simulation import SimulationBase
from pymob.utils.store_file import is_number
from pymob.utils.errors import import_optional_dependency
from pymob.inference.base import InferenceBackend

import pyabc
from pathos import multiprocessing as mp

class PyabcBackend(InferenceBackend):
    def __init__(
        self, 
        simulation: SimulationBase
    ):
        super().__init__(simulation=simulation)
        self.parameter_map = {}
        
        self.simulation = simulation
        self.config = simulation.config
        self.evaluator = self.model_parser()
        self.prior = self.prior_parser(simulation.free_model_parameters)
        self.distance_function = self.distance_function_parser()
        self.observations = simulation.observations

        self.abc = None
        self.history = None
        self.posterior = None
        self.config = simulation.config
   
    @property
    def sampler(self):
        # TODO: Remove when all methods have been changed
        warnings.warn("Deprecation warning. Use `Simulation.config.OPTION` API")
        return self.config.inference_pyabc.sampler
    
    @property
    def population_size(self):
        # TODO: Remove when all methods have been changed
        warnings.warn("Deprecation warning. Use `Simulation.config.OPTION` API")
        return self.config.inference_pyabc.population_size
    
    @property
    def minimum_epsilon(self):
        # TODO: Remove when all methods have been changed
        warnings.warn("Deprecation warning. Use `Simulation.config.OPTION` API")
        return self.config.inference_pyabc.minimum_epsilon
    
    @property
    def min_eps_diff(self):
        # TODO: Remove when all methods have been changed
        warnings.warn("Deprecation warning. Use `Simulation.config.OPTION` API")
        return self.config.inference_pyabc.min_eps_diff
    
    @property
    def max_nr_populations(self):
        # TODO: Remove when all methods have been changed
        warnings.warn("Deprecation warning. Use `Simulation.config.OPTION` API")
        return self.config.inference_pyabc.max_nr_populations
    
    @property
    def database(self):
        dbp = self.config.inference_pyabc.database_path
        if os.path.isabs(dbp):
            return dbp
        else:
            return os.path.join(self.config.case_study.output_path, dbp)
    
    
    @staticmethod
    def param_to_prior(par):
        parname = par.name
        distribution, cluttered_arguments = par.prior.split("(", 1)
        param_strings = cluttered_arguments.split(")", 1)[0].split(",")
        params = {}
        arraypattern = r"\[(\d+(\.\d+)?(\s+\d+(\.\d+)?)*|\s*)\]"
        
        for parstr in param_strings:

            key, expression = parstr.split("=")
            
            # check if val is a number
            if is_number(expression):
                value = float(expression)
            
            # check if val is an array
            elif re.fullmatch(arraypattern, expression):
                expression = expression.removeprefix("[").removesuffix("]")
                value = np.array([float(v) for v in expression.split(" ")])

            else:
                raise NotImplementedError(
                    f"Prior format {expression} is not implemented. "
                    "Use e.g."
                    "\n- normal(loc=1.0,scale=0.5) for 1-dimensional priors, or"
                    "\n- normal(loc=[1.0 2.4 1],scale=0.5) for n-dimensional priors." 
                )

            params.update({key:value})

        return parname, distribution, params

    def prior_parser(self, free_model_parameters: list):
        prior_dict = {}
        for mp in free_model_parameters:
            name, distribution, params = self.param_to_prior(par=mp)
            pmap, dist_map = self.array_param_to_1d(name, distribution, params)

            self.parameter_map.update(pmap)
            for subpar_name, subpar_dist in dist_map.items():
                prior = pyabc.RV(subpar_dist["dist"], **subpar_dist["params"])
                prior_dict.update({subpar_name: prior})

        return pyabc.Distribution(**prior_dict)

    @staticmethod
    def map_parameters(theta, parameter_map):
        theta_mapped = {}
        for par_name, subpar_name in parameter_map.items():
            if isinstance(subpar_name, list):
                subparams = np.array([theta[p] for p in subpar_name])
            else:
                subparams = theta[subpar_name]
            
            theta_mapped.update({par_name: subparams})
        return theta_mapped

    @staticmethod
    def array_param_to_1d(name, distribution, dist_param_dict):
        # return a distribution if all parameters of the distribution are floats
        # and map with name

        if np.all([isinstance(v, float) for _, v in dist_param_dict.items()]):
            return {name:name}, {name: {"dist": distribution, "params": dist_param_dict}}
        else:
            args_as_arrays = {k:np.array(v, ndmin=1) for k, v in dist_param_dict.items()}
            broadcasted_args = np.broadcast_arrays(*tuple(args_as_arrays.values()))
            # args = {k: v for k, v in zip(dist_param_dict.keys(), broadcasted_args)}

            param_map = {name: []}
            dist_map = {}
            for i, args in enumerate(zip(*broadcasted_args)):
                p_subname = f"{name}_{i}"
                param_map[name].append(p_subname)
                dist_kwargs = {
                    "dist": distribution,
                    "params": {k:v for k, v in zip(dist_param_dict.keys(), args)}
                }
                dist_map.update({p_subname: dist_kwargs})

            return param_map, dist_map

    
    def plot(self):
        plot = self.config.inference.plot
        if plot is None:
            return
        elif isinstance(plot, str):
            try:
                plot_func = getattr(self.simulation, plot)
                plot_func(self.simulation)
            except AttributeError:
                warnings.warn(
                    f"Plot function {plot} was not found in the plot.py module "
                    "Make sure the name has been spelled correctly or try to "
                    "set the function directly to 'sim.config.inference.plot'.",
                    category=UserWarning
                )
        else:
            plot(self.simulation)
        

    def model_parser(self):
        extra_kwargs = {}
        for extra in self.extra_vars:
            extra_kwargs.update({extra: self.simulation.observations[extra].values})

        obj_func_signature = inspect.signature(self.simulation.objective_function)
        obj_func_params = list(obj_func_signature.parameters.keys())
    
        def model(theta):
            theta_mapped = self.map_parameters(theta, self.parameter_map)
            evaluator = self.simulation.dispatch(theta=theta_mapped, **extra_kwargs)
            evaluator(seed=self.simulation.RNG.integers(1000))
            res = {k: np.array(v) for k, v in evaluator.Y.items()}
            res.update({p: theta_mapped[p] for p in obj_func_params if p in theta_mapped})
            return res
        return model
    
    def distance_function_parser(self):
        obj_func_signature = inspect.signature(self.simulation.objective_function)
        obj_func_params = list(obj_func_signature.parameters.keys())
            
        def distance_function(x, x0):
            Y = {k: v for k, v in x.items() if k in self.config.data_structure.data_variables}
            
            theta_obj_func = {p: x[p] for p in obj_func_params if p in x}
            obj_name, obj_value = self.simulation.objective_function(
                results=Y, **theta_obj_func 
            )
            return obj_value
        
        return distance_function

    def run(self):
        n_cores = self.config.multiprocessing.n_cores
        print(f"Using {n_cores} CPU cores", flush=True)

        sampler = self.config.inference_pyabc.sampler.lower()
        # before launch server in bash with `redis-server --port 1803`
        if sampler == "RedisEvalParallelSampler".lower():
            abc_sampler = pyabc.sampler.RedisEvalParallelSampler(
                host="localhost", 
                password=self.config.inference_pyabc_redis.password, 
                port=self.config.inference_pyabc_redis.port
            )

        elif sampler == "SingleCoreSampler".lower():
            abc_sampler = pyabc.sampler.SingleCoreSampler()

        elif sampler == "MulticoreParticleParallelSampler".lower():
            abc_sampler = pyabc.sampler.MulticoreParticleParallelSampler(
                n_procs=n_cores
            )

        elif sampler == "MulticoreEvalParallelSampler".lower():
            abc_sampler = pyabc.sampler.MulticoreEvalParallelSampler(
                n_procs=n_cores
            )

        else:
            raise NotImplementedError(
                "Sampler is not implemented. Choose one of: 'RedisEvalParallelSampler', " +
                "'SingleCoreSampler'"
            )


        self.abc = pyabc.ABCSMC(
            models=self.evaluator, 
            parameter_priors=self.prior, 
            distance_function=self.distance_function, 
            sampler=abc_sampler,
            population_size=self.config.inference_pyabc.population_size
        )

        self.history = self.abc.new("sqlite:///" + self.database)

        self.abc.run(
            minimum_epsilon=self.config.inference_pyabc.minimum_epsilon,
            min_eps_diff=self.config.inference_pyabc.min_eps_diff,
            max_nr_populations=self.config.inference_pyabc.max_nr_populations
        )


    def load_results(self):
        if self.history is None:
            self.history = pyabc.History(f"sqlite:///" + self.database)
        
        # set history id
        db_id = self.config.inference_pyabc_redis.history_id
        self.history.id = self.history._find_latest_id() if db_id == -1 else db_id
        
        mod_id = self.config.inference_pyabc_redis.model_id
        samples, w = self.history.get_distribution(m=mod_id, t=self.history.max_t)
        
        # re-sort parameters based on prior order
        samples = samples[self.prior.keys()]

        posterior = xr.DataArray(
            samples.values.reshape((1,*samples.values.shape)),
            coords=dict(
                chain=[1],
                draw=range(len(samples)),
                parameter=list(samples.columns)
            )
        )
        
        idata = az.from_dict(
            posterior={key: col.values for key, col in samples.items()},
            dims=self.posterior_data_structure,
            coords=self.posterior_coordinates
        )
        self.idata = idata
        # posterior
        self.posterior = Posterior(posterior)

    @property
    def posterior_data_structure(self):
        data_structure = self.simulation.data_structure.copy()
        data_structure_loglik = {f"{dv}_obs": dims for dv, dims in data_structure.items()}
        data_structure.update(data_structure_loglik)
        return data_structure

    @property
    def posterior_coordinates(self):
        posterior_coords = self.simulation.coordinates.copy()
        posterior_coords.update({
            "draw": list(range(self.population_size)), 
            "chain": [0]
        })
        return posterior_coords

    def plot_chains(self):
        assert self.history is not None, AssertionError(
            "results must be loaded before they can be accessed. "
            "Call load_results() before executing plot_chains. "
            "E.g. sim.inferer.load_results()"
        )

        distributions = []
        for t in range(self.history.max_t + 1):
        # for t in range(2):
            print(f"iteration: {t}", end="\r")
            par_values, _ = self.history.get_distribution(m=0, t=t)
            post = par_values.to_xarray().to_array()
            post["id"] = range(len(post.id))
            post = post.assign_coords(iteration=t)
            distributions.append(post)

        trajectory = xr.concat(distributions, dim="iteration")
        
        parameters = trajectory.coords["variable"].values
        nit = len(trajectory.iteration)
        color="tab:blue"
        Np = len(parameters)
        fig, axes = plt.subplots(nrows=Np, ncols=2, figsize=(10, 2*Np),
            gridspec_kw=dict(width_ratios=[1, 1]))
        
        if len(axes.shape) == 1:
            axes = axes.reshape((Np, 2))
        
        for i, par in enumerate(parameters):
            ax = axes[i, 0]

            ax.plot(trajectory.iteration, trajectory.sel(variable=par), 
                color=color,
                # s=5,
                alpha=.01)
                
            # takes the last 
            ax2 = axes[i, 1]
            ax2.hist(trajectory.sel(variable=par, iteration=nit-1), 
                    alpha=.85, color=color)
            ax2.set_ylabel(f"{par.split('.')[-1].replace('_', ' ')}")
            
            ax.set_xlabel("iteration")
            ax.set_yscale("log")
            ax.set_ylabel(f"{par.split('.')[-1].replace('_', ' ')}")

        fig.subplots_adjust(wspace=.3, hspace=.4)

        return fig

    def store_results(self):
        """results are stored by default in database"""
        pass
   
    @lru_cache
    def posterior_predictions(self, n=50, seed=1):
        assert self.posterior is not None, AssertionError(
            "results must be loaded before Posterior can be accessed. "
            "Call load_results() before executing plot_chains. "
            "E.g. sim.inferer.load_results()"
        )
        rng = np.random.default_rng(seed)
        post = self.posterior
        total_samples = post.samples.shape[1]

        # draw samples from posterior
        posterior_samples = rng.choice(post.samples.draw, size=n, replace=False)

        def predict(posterior_sample_id):
            params = post.draw(i=posterior_sample_id)
            evaluator = self.simulation.dispatch(params.to_dict())
            evaluator()
            res = evaluator.results
            res = res.assign_coords({"draw": posterior_sample_id, "chain": 1})
            res["params"] = params.samples
            res = res.expand_dims(("chain", "draw"))
            return res
        
        print(f"Using {self.config.multiprocessing.n_cores} CPUs")
        if self.config.multiprocessing.n_cores == 1:
            results = list(map(predict, posterior_samples))
        else:
            with mp.ProcessingPool(self.config.multiprocessing.n_cores) as pool:        
                results = pool.map(predict, posterior_samples)
            
        return xr.combine_by_coords(results)

    def plot_predictions(
            self, data_variable: str, x_dim: str, ax=None, subset={}
        ):
        obs = self.simulation.observations.sel(subset)
        
        post_pred = self.posterior_predictions(
            n=self.config.inference_pyabc_redis.n_predictions, 
            # seed only controls the parameters samples drawn from posterior
            seed=self.config.simulation.seed
        ).sel(subset)

        hdi = az.hdi(post_pred, .95)

        if ax is None:
            _, ax = plt.subplots(1,1)
        
        y_mean = post_pred[data_variable].mean(dim=("chain", "draw"))
        ax.plot(
            post_pred[x_dim].values, y_mean.values, 
            color="black", lw=.8
        )

        ax.fill_between(
            post_pred[x_dim].values, *hdi[data_variable].values.T,  # type: ignore
            alpha=.5, color="grey"
        )

        ax.plot(
            obs[x_dim].values, obs[data_variable].values, 
            marker="o", ls="", ms=3
        )
        
        ax.set_ylabel(data_variable)
        ax.set_xlabel(x_dim)

        return ax

class Posterior:
    def __init__(self, samples):
        self.samples = samples

    def __repr__(self):
        return str(self.samples)

    def to_dict(self):
        theta = self.samples
        return {par:float(theta.sel(parameter=par)) 
            for par in theta.parameter.values}

    def draw(self, i):
        return Posterior(self.samples.sel(draw=i, chain=1))

    def mean(self):
        return Posterior(self.samples.mean(dim=("chain", "draw")))


