import numpy as np
import ast
from typing import (
    Dict,
    Tuple,
    Mapping,
    Callable,
    Iterable,
    Optional,
    Any,
    List,
    Protocol,
    Literal
)
from abc import ABC, abstractmethod
import warnings

from matplotlib import pyplot as plt
import matplotlib as mpl
import arviz as az
import itertools as it
import tqdm

from pymob.simulation import SimulationBase
from pymob.sim.parameters import Param, RandomVariable, Expression
from pymob.sim.config import Datastructure
from pymob.utils.config import lookup_from

import logging

class TqdmLogger:
    """File-like class redirecting tqdm progress bar to given logging logger."""
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def write(self, msg: str) -> None:
        self.logger.info(msg.lstrip("\r"))

    def flush(self) -> None:
        pass


class Errorfunction(Protocol):
    def __call__(
        self, 
        theta: Dict[str, List[float]], 
    ) -> Any:
        ...

class Distribution:
    """The distribution is a pre-initialized distibution with human friendly 
    interface to construct a distribution in an arbitrary backend.

    The necessary adjustments to make the distribution backend specific
    are done by passing more context to Distribution class to the
    _context variable
    """
    distribution_map: Dict[str,Tuple[Callable, Dict[str,str]]] = {}
    parameter_converter: Callable = staticmethod(lambda x: x)
    _context = {}
    _import_map = {"np": "numpy", "numpy":"numpy", "jnp":"jax.numpy", "jax.numpy": "jax.numpy"}

    def __init__(
        self, 
        name: str, 
        random_variable: RandomVariable, 
        dims: Tuple[str, ...],
        shape: Tuple[int, ...],
    ) -> None:
        self.name = name
        self._dist_str = random_variable.distribution
        self._parameter_expression = random_variable.parameters
        self._obs_transform: Optional[Expression] = random_variable.obs
        self._obs_transform_inv: Optional[Expression] = random_variable.obs_inv
        self.dims = dims
        self.shape = shape if len(shape) > 0 else ()

        dist, params, uargs = self.parse_distribution(random_variable)
        self.distribution: Callable = dist
        self.parameters: Dict[str, Expression] = params
        self.undefined_args: set = uargs

        self.obs_transform_func = self.create_transform_func(self._obs_transform, "obs_transform")
        self.obs_transform_func_inv = self.create_transform_func(self._obs_transform_inv, "obs_transform_inv")

    def __str__(self) -> str:
        dist = self.dist_name
        params = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        dimshape = tuple([f'{d}={s}' for d, s in zip(self.dims, self.shape)])
        return f"{dist}({params}, dims={dimshape}, obs={self._obs_transform})"
    
    def __repr__(self) -> str:
        return str(self)

    @property
    def dist_name(self) -> str:
        return self._dist_str
    
    def create_transform_func(self, transform, func_name):
        if transform is None:
            return None
        
        else:
            expr = transform
            
            imports = [a for a in expr.undefined_args if a in self._import_map]
            args = [a for a in expr.undefined_args if a not in self._import_map]

            # Create the function arguments
            args = [ast.arg(arg=arg_name, annotation=None) for arg_name in args]
            
            # Build a function definition from the expression
            func_def = ast.FunctionDef(
                name=f"{func_name}_{self.name}",
                args=ast.arguments(
                    posonlyargs=[], args=args, vararg=None, kwonlyargs=[], 
                    kw_defaults=[], defaults=[]
                ),
                body=[ast.Return(value=expr.expression.body)],
                decorator_list=[]
            )
        
            import_statements = [
                ast.Import(names=[ast.alias(name=self._import_map[i], asname=i)])
                for i in imports
            ]
            module = ast.Module(body=[*import_statements, func_def], type_ignores=[])
            module = ast.fix_missing_locations(module)

            # compile the function and retrieve object
            code = compile(module, filename="<ast>", mode="exec")
            func_env = {}
            exec(code, func_env)    
            func = func_env[f"{func_name}_{self.name}"]
            
            return func
    
    def construct(self, context: Iterable[Mapping], extra_kwargs: Dict = {}):
        _context = {arg: lookup_from(arg, context) for arg in self.undefined_args}
        _context.update(self._context)
        # evaluate the parameters given a context
        params = {
            key: self.parameter_converter(value.evaluate(context=_context)) 
            for key, value in self.parameters.items()
        }
        return self.distribution(**params, **extra_kwargs)

    def _get_distribution(self, distribution: str) -> Tuple[Callable, Dict[str, str]]:
        return self.distribution_map[distribution]

    def parse_distribution(self, random_variable: RandomVariable) -> Tuple[Any,Dict[str,Expression],set]:

        distribution_mapping = self._get_distribution(random_variable.distribution)

        if not isinstance(distribution_mapping, tuple):
            distribution = distribution_mapping
            distribution_mapping = (distribution, {})
        
        assert len(distribution_mapping) == 2, (
            "distribution and parameter mapping must be "
            "a tuple of length 2."
        )

        distribution, parameter_mapping = distribution_mapping
        mapped_params = {}
        underfined_args = set()
        for key, val in random_variable.parameters.items():
            mapped_key = parameter_mapping.get(key, key)
            underfined_args = underfined_args.union(val.undefined_args)
            mapped_params.update({mapped_key:val})

        return distribution, mapped_params, underfined_args


class InferenceBackend(ABC):
    _distribution = Distribution
    idata: az.InferenceData
    prior: Dict[str,Distribution]
    log_likelihood: Errorfunction
    gradient_log_likelihood: Errorfunction
    chains = 1
    draws = 1

    def __init__(
        self, 
        simulation: SimulationBase,
    ) -> None:
        
        self.simulation = simulation
        self.config = simulation.config

        self.indices = {v.name: np.array(v.values) for _, v in self.simulation.indices.items()}
        # parse model components
        self.prior = self.parse_model_priors(
            parameters=self.config.model_parameters.free,
            dim_shapes=self.simulation.parameter_shapes,
            data_structure=self.config.data_structure,
            indices=self.indices
        )

        self.evaluator = self.parse_deterministic_model()

        self.error_model = self.parse_error_model(
            error_models=self.config.error_model.all
        )
        

    @abstractmethod
    def parse_deterministic_model(self):
        pass

    @abstractmethod
    def parse_probabilistic_model(self):
        pass

    @property
    def extra_vars(self):
        return self.config.inference.extra_vars
    
    @property
    def n_predictions(self):
        return self.config.inference.n_predictions
    
    @property
    def EPS(self):
        return self.config.inference.eps

    @property
    def posterior_data_structure(self) -> Dict[str, List[str]]:
        data_structure = self.simulation.data_structure.copy()
        data_structure_loglik = {f"{dv}_obs": dims for dv, dims in data_structure.items()}
        data_structure_residuals = {f"{dv}_res": dims for dv, dims in data_structure.items()}
        parameter_dims = {k: list(v) for k, v in self.simulation.parameter_dims.items() if len(v) > 0}
        parameter_dims_unconstrained = {
            f"{k}_normal_base": list(v) for k, v 
            in self.simulation.parameter_dims.items() if len(v) > 0
        }
        data_structure.update(data_structure_loglik)
        data_structure.update(data_structure_residuals)
        data_structure.update(parameter_dims)
        data_structure.update(parameter_dims_unconstrained)
        return data_structure
    
    @property
    def posterior_coordinates(self) -> Dict[str, List[str|int]]:
        chains = self.chains
        draws = self.draws

        posterior_coords = {k: list(v) for k, v in self.simulation.dimension_coords.items()}
        posterior_coords.update({
            "draw": list(range(draws)), 
            "chain": list(range(chains))
        })
        return posterior_coords

    @classmethod
    def parse_model_priors(
        cls, 
        parameters: Dict[str,Param], 
        dim_shapes: Dict[str,Tuple[int, ...]],
        data_structure: Datastructure,
        indices: Dict[str, Any] = {}
    ):
        priors = {}
        hyper_ = []
        for key, par in parameters.items():
            if par.prior is None:
                raise AttributeError(
                    f"No prior was defined for parameter '{key}'. E.g.: "+
                    f"`sim.config.model_parameters.{key}.prior = 'lognorm(loc=1, scale=2)'`"
                )
            
            for k, v in par.prior.parameters.items():
                for ua in v.undefined_args:
                    if ua in indices:
                        continue

                    elif ua in cls._distribution._import_map:
                        continue

                    elif ua in priors:
                        continue

                    elif ua in data_structure.data_variables:
                        # allows observations
                        continue

                    elif ua.strip("_y0") in data_structure.data_variables:
                        # allows initial values
                        continue

                    else:
                        raise KeyError(
                            f"Parameter '{key}' defines a prior '{par.prior.model_ser()}' that will try "+
                            f"to access the variable '{ua}' before it is defined "+
                            "in 'sim.indices', 'Distribution._import_map' or previously "+
                            "defined priors. Please double check the prior definition for errors, "+
                            f"specify the needed parameter, or check the parameter order. "+
                            "If needed, use "+
                            f"config.model_parameters.reorder([..., '{ua}', '{key}', ...]) "+
                            "to arrange the parameters in the correct order."
                        )

            dist = cls._distribution(
                name=key, 
                random_variable=par.prior,
                dims=par.dims,
                shape=dim_shapes[key]
            )
            priors.update({key: dist})
        return priors
    
    @classmethod
    def parse_error_model(
        cls,
        error_models: Dict[str,RandomVariable], 
    ):
        error_model = {}
        for data_var, error_distribution in error_models.items():
            error_dist = cls._distribution(
                name=data_var, 
                random_variable=error_distribution,
                dims=(),
                shape=(),
            )
               
            error_model.update({data_var: error_dist})
        return error_model

    @abstractmethod
    def create_log_likelihood(self) -> Tuple[Errorfunction,Errorfunction]:
        """This method creates a log likelihood function and potentially
        function to compute the gradients.
        """
        pass

    def plot_likelihood_landscape(
        self, 
        parameters: Tuple[str, str],
        log_likelihood_func: Callable,
        gradient_func: Optional[Callable] = None,
        bounds: Tuple[List[float],List[float]] = ([-10, 10], [-10,10]),
        n_grid_points: int = 100,
        n_vector_points: int = 50,
        normal_base = False,
        ax: Optional[plt.Axes] = None,
    ):
        """Plots the likelihood for each coordinate pair of two model parameters
        Parameters are taken from the standardized scale and transformed 

        For some reason the likelihood and gradients need to be transposed after
        calculating to work with meshgrid. The coordinates can be 

        Parameters
        ----------

        parameters : Tuple[str,str]
            The parameters to be plotted against each other

        log_likelihood_func : Callable
            Must be a vectorized function that can dictionary with two keys that
            contain array values as input parameters and returns a 1-D array of 
            log-likelihood values.

        gradient_func : Callable
            If, in addition to the likelihood values, also gradients should
            be computed, a function to compute the gradients must be provided 

        bounds : Tuple[Tuple[float,float],Tuple[float,float]]
            Fallback bounds for the parameter grid to calculate the likelihood
            function. This is only used if bounds are not provided by the parameters

        n_grid_points : int
            The number of grid points per side, to calculate the likelihood
            function for. Scales n^2
        """
        par_x, par_y = parameters
        bounds_x, bounds_y = bounds
        px = self.config.model_parameters.all[par_x]
        py = self.config.model_parameters.all[par_y]

        bounds_x[0] = px.min if px.min is not None else bounds_x[0]
        bounds_x[1] = px.max if px.max is not None else bounds_x[1]
        bounds_y[0] = py.min if py.min is not None else bounds_y[0]
        bounds_y[1] = py.max if py.max is not None else bounds_y[1]

        x = np.linspace(*bounds_x, n_grid_points)
        y = np.linspace(*bounds_y, n_grid_points)

        normal_base_str = "_normal_base" if normal_base else ""

        grid = {
            f"{p}{normal_base_str}": np.expand_dims(v, 1) for p, v 
            in zip(parameters, np.array(list(it.product(x, y))).T)
        }

        loglik = []
        for i in tqdm.tqdm(
            range(len(x) * len(y)), 
            desc="Function evaluations", 
            mininterval=5,
            file=TqdmLogger(self.simulation.logger)
        ):
            loglik_i = log_likelihood_func({k: v[i] for k, v in grid.items()})
            loglik.append(loglik_i)

        loglik = np.array(loglik)

        # loglikelihood must be transposed to work with meshgrid
        Z = loglik.reshape((n_grid_points, n_grid_points)).T
        X, Y = np.meshgrid(x, y)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8,6))

        cmap = mpl.colormaps["viridis"]  # type: ignore
        contours = ax.contourf(X, Y, Z, cmap=cmap, levels=50)

        if gradient_func is not None:
            xv = np.linspace(*bounds_x, n_vector_points)
            yv = np.linspace(*bounds_y, n_vector_points)
            Xv, Yv = np.meshgrid(xv, yv)

            gridv = {
                f"{p}{normal_base_str}": np.expand_dims(v, 1) for p, v 
                in zip(parameters, np.array(list(it.product(xv, yv))).T)
            }

            u = []
            v = []
            for i in tqdm.tqdm(
                range(len(xv) * len(yv)), 
                desc="Gradient evaluations", 
                mininterval=5,
                file=TqdmLogger(self.simulation.logger)
            ):
                grads = gradient_func({k: v[i] for k, v in gridv.items()})
                u.append(grads[f"{par_x}{normal_base_str}"])
                v.append(grads[f"{par_y}{normal_base_str}"])

            u = np.array(u)
            v = np.array(v)

            # gradients must be transposed to work with meshgrid
            U = u.reshape((n_vector_points, n_vector_points)).T
            V = v.reshape((n_vector_points, n_vector_points)).T

            vector_field = ax.quiver(Xv, Yv, U, V, angles="xy", width=0.001)

        ax.figure.colorbar(
            contours,
            ax=ax,
            label="log-likelihood"
        )
        ax.set_xlabel(f"{par_x} (normalized)")
        ax.set_ylabel(f"{par_y} (normalized)")

        return ax
    
    @abstractmethod
    def prior_predictions(self):
        pass

    @abstractmethod
    def posterior_predictions(self):
        pass

    def plot_prior_predictions(
            self, data_variable: str, x_dim: str, ax=None, subset={}, 
            n=None, seed=None, plot_preds_without_obs=False,
            prediction_data_variable: Optional[str] = None,
            **plot_kwargs
        ):
        if n is None:
            n = self.n_predictions

        if seed is None:
            seed = self.config.simulation.seed
        
        idata = self.prior_predictions(
            n=n, 
            # seed only controls the parameters samples drawn from posterior
            seed=seed
        )

        ax = self.plot_predictions(
            observations=self.simulation.observations,
            predictions=idata.prior_predictive, # type: ignore
            data_variable=data_variable,
            plot_preds_without_obs=plot_preds_without_obs,
            x_dim=x_dim,
            ax=ax,
            subset=subset,
            prediction_data_variable=prediction_data_variable,
            **plot_kwargs,
        )

        return ax

    def plot_posterior_predictions(
            self, data_variable: str, x_dim: str, ax=None, subset={},
            n=None, seed=None, plot_preds_without_obs=False,
            prediction_data_variable: Optional[str] = None,
            **plot_kwargs
        ):
        # TODO: This method should be trashed. It is not really useful
        if n is None:
            n = self.n_predictions
        
        if seed is None:
            seed = self.config.simulation.seed
        
        predictions = self.posterior_predictions(
            n=n, 
            # seed only controls the parameters samples drawn from posterior
            seed=seed
        )
        
        ax = self.plot_predictions(
            observations=self.simulation.observations,
            predictions=predictions,
            data_variable=data_variable,
            plot_preds_without_obs=plot_preds_without_obs,
            x_dim=x_dim,
            ax=ax,
            subset=subset,
            prediction_data_variable=prediction_data_variable,
            **plot_kwargs
        )

        return ax

    def plot(self):
        self.plot_diagnostics()

        plot = self.config.inference.plot
        if plot is None:
            return
        elif isinstance(plot, str):
            try:
                plot_func = getattr(self.simulation._plot, plot)
                plot_func(self.simulation)
            except AttributeError:
                warnings.warn(
                    f"Plot function {plot} was not found in the {self.simulation._plot} module "+
                    "Make sure the name has been spelled correctly or try to "+
                    "set the function directly to 'sim.config.inference.plot'.",
                    category=UserWarning
                )
        else:
            plot(self.simulation)

    @staticmethod
    def plot_predictions(
            observations,
            predictions,
            data_variable: str,
            x_dim: str, 
            ax=None, 
            plot_preds_without_obs=False,
            subset={},
            mode: Literal["mean+hdi", "draws"]="mean+hdi",
            plot_options: Dict={"obs": {}, "pred_mean": {}, "pred_draws": {}, "pred_hdi": {}},
            prediction_data_variable: Optional[str] = None,
        ):
        warnings.warn(
            "Use of 'sim.inferer.plot_predictions' is deprecated. Use the "+
            "Plotting backend: pymob.sim.plot.SimulationPlot",
            category=DeprecationWarning
        )
        # filter subset coordinates present in data_variable
        subset = {k: v for k, v in subset.items() if k in observations.coords}
        
        if prediction_data_variable is None:
            prediction_data_variable = data_variable

        # select subset
        if prediction_data_variable in predictions:
            preds = predictions.sel(subset)[prediction_data_variable]
        else:
            raise KeyError(
                f"{prediction_data_variable} was not found in the predictions "+
                "consider specifying the data variable for the predictions "+
                "explicitly with the option `prediction_data_variable`."
            )
        try:
            obs = observations.sel(subset)[data_variable]
        except KeyError:
            obs = preds.copy().mean(dim=("chain", "draw"))
            obs.values = np.full_like(obs.values, np.nan)
        
        # stack all dims that are not in the time dimension
        if len(obs.dims) == 1:
            # add a dummy batch dimension
            obs = obs.expand_dims("batch")
            obs = obs.assign_coords(batch=[0])

            preds = preds.expand_dims("batch")
            preds = preds.assign_coords(batch=[0])


        stack_dims = [d for d in obs.dims if d not in [x_dim, "chain", "draw"]]
        obs = obs.stack(i=stack_dims)
        preds = preds.stack(i=stack_dims)
        N = len(obs.coords["i"])
            
        hdi = az.hdi(preds, .95)[f"{prediction_data_variable}"]

        if ax is None:
            ax = plt.subplot(111)
        
        y_mean = preds.mean(dim=("chain", "draw"))

        for i in obs.i:
            if obs.sel(i=i).isnull().all() and not plot_preds_without_obs:
                # skip plotting combinations, where all values are NaN
                continue
            
            if mode == "mean+hdi":
                kwargs_hdi = dict(color="black", alpha=0.1)
                kwargs_hdi.update(plot_options.get("pred_hdi", {}))
                ax.fill_between(
                    preds[x_dim].values, *hdi.sel(i=i).values.T, # type: ignore
                    **kwargs_hdi
                )

                kwargs_mean = dict(color="black", lw=1, alpha=max(1/N, 0.05))
                kwargs_mean.update(plot_options.get("pred_mean", {}))
                ax.plot(
                    preds[x_dim].values, y_mean.sel(i=i).values, 
                    **kwargs_mean
                )
            elif mode == "draws":
                kwargs_draws = dict(color="black", lw=0.5, alpha=max(1/N, 0.05))
                kwargs_draws.update(plot_options.get("pred_draws", {}))
                ys = preds.sel(i=i).stack(sample=("chain", "draw"))
                ax.plot(
                    preds[x_dim].values, ys.values, 
                    **kwargs_draws
                )
            else:
                raise NotImplementedError(
                    f"Mode '{mode}' not implemented. "+
                    "Choose 'mean+hdi' or 'draws'."
                )

            kwargs_obs = dict(marker="o", ls="", ms=3, color="tab:blue")
            kwargs_obs.update(plot_options.get("obs", {}))
            ax.plot(
                obs[x_dim].values, obs.sel(i=i).values, 
                **kwargs_obs
            )
        
        ax.set_ylabel(data_variable)
        ax.set_xlabel(x_dim)

        return ax

    def plot_diagnostics(self):
        """
        TODO: Should be outsourced to pymob.sim.plot
        """
        var_names = [
            k for k, v in self.config.model_parameters.free.items()
            if v.prior.distribution != "deterministic"
        ]

        if hasattr(self.idata, "sample_stats"):
            if "diverging" in self.idata["sample_stats"]:
                self.idata["sample_stats"]["diverging"] = self.idata["sample_stats"].diverging.astype(int)

        if hasattr(self.idata, "posterior"):
            axes = az.plot_trace(
                self.idata,
                var_names=var_names
            )
            fig = plt.gcf()
            fig.tight_layout()
            fig.savefig(f"{self.simulation.output_path}/trace.png")
            
            axes = az.plot_pair(
                self.idata, 
                divergences=True, 
                var_names=var_names
            )
            fig = plt.gcf()
            fig.tight_layout()
            fig.savefig(f"{self.simulation.output_path}/pairs_posterior.png")

    def check_prior_for_nans(self, idata):
        nans = idata.prior.isnull().sum().to_array()
        if np.any(nans > 0):
            prior_names = nans.where(nans > 0, drop=True)["variable"].values
            warnings.warn(
                f"NaNs occurred in the prior draws of {list(prior_names)}. "+
                "Make sure your priors are correctly specified. "+
                "I.e. prior parameters are inside the support of the distribution "+
                "(negative values, out of bounds values), this is especially "+
                "relevant if hyper-priors are used. "+
                f"Prior: {self.prior}",
                category=UserWarning
            )
            return False
        else:
            return True
      