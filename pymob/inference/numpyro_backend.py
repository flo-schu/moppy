from functools import partial, lru_cache
import glob
import re
import warnings
from typing import (
    Tuple, Dict, Union, Optional, Callable, Literal, List, Any,
    Protocol
)

from tqdm import tqdm
import numpy as np
import xarray as xr
import arviz as az
from matplotlib import pyplot as plt
import sympy
from arviz.data.inference_data import SUPPORTED_GROUPS_ALL

import pymob
from pymob.simulation import SimulationBase
from pymob.sim.parameters import Expression, NumericArray
from pymob.inference.base import Errorfunction, InferenceBackend, Distribution
from pymob.inference.analysis import (
    cluster_chains, rename_extra_dims, plot_posterior_samples,
    add_cluster_coordinates
)
from pymob.inference.numpyro_dist_map import (
    scipy_to_numpyro, transformed_dist_map, transform, inv_transform
)

import numpyro
from numpyro.infer import Predictive
from numpyro.distributions import Normal
from numpyro.distributions.distribution import DistributionMeta
from numpyro import handlers
from numpyro import infer

import jax
import jax.numpy as jnp
import sympy2jax

SUPPORTED_GROUPS_ALL.extend([
    "prior_model_fits", 
    "prior_residuals",
    "posterior_model_fits",
    "posterior_residuals",
    "unconstrained_prior",
    "unconstrained_posterior"
])


sympy2jax_extra_funcs = {
    sympy.Array: jnp.array,
    sympy.Tuple: tuple,
}


def catch_patterns(expression_str):
    # tries to match array notation [0 1 2]
    pattern = r"\[(\d+(\.\d+)?(\s+\d+(\.\d+)?)*|\s*)\]"
    if re.fullmatch(pattern, expression_str) is not None:
        expression_str = expression_str.replace(" ", ",") \
            .removeprefix("[").removesuffix("]")
        return f"stack({expression_str})"

    return expression_str

# distribution_map =  {
#     "lognorm": (LogNormalTrans, {"scale": "loc", "s": "scale"}),
#     "binom": (dist.Binomial, {"n":"total_count", "p":"probs"}),
#     "normal": dist.Normal,
#     "halfnorm": HalfNormalTrans,
#     "poisson": (dist.Poisson, {"mu": "rate"}),
# }

class ErrorModelFunction(Protocol):
    def __call__(
        self, 
        theta: Dict, 
        simulation_results: Dict, 
        indices: Dict,
        observations: Dict, 
        masks: Dict,
        make_predictions: bool
    ) -> Any:
        ...

class NumpyroDistribution(Distribution):
    distribution_map: Dict[str,Tuple[DistributionMeta, Dict[str,str]]] = scipy_to_numpyro
    parameter_converter = staticmethod(lambda x: jnp.array(x))

    def _get_distribution(self, distribution: str) -> Tuple[DistributionMeta, Dict[str, str]]:
        # TODO: This is not satisfying. I think the transformed distributions
        # should only be used when this is explicitly specified.
        # I really wonder, why this makes such a large change in numpyro
        numpyro_dist, parameter_mapping = self.distribution_map[distribution]
        if numpyro_dist in transformed_dist_map:
            transformed_dist = transformed_dist_map[numpyro_dist]
            return transformed_dist, parameter_mapping
        else:
            # could not find the transformed distribution using normal
            return numpyro_dist, parameter_mapping

    @property
    def dist_name(self):
        return self.distribution.__name__

class NumpyroBackend(InferenceBackend):
    _distribution = NumpyroDistribution
    prior: Dict[str, DistributionMeta]

    def __init__(
        self, 
        simulation: SimulationBase
    ):
        """Initializes the NumpyroBackend with a Simulation object.

        Parameters
        ----------
        simulation : SimulationBase
            An initialized simulation.
        """
        super().__init__(simulation=simulation)
        # parse preprocessing
        if self.user_defined_preprocessing is not None:
            self.preprocessing = getattr(
                self.simulation._prob,
                self.user_defined_preprocessing
            )

        # parse the probability model
        if self.user_defined_probability_model is not None:
            inference_model = getattr(
                self.simulation._prob, 
                self.user_defined_probability_model
            )
            self.inference_model = partial(
                inference_model,
                indices = self.indices
            )

        # combine the model
        else:
            self.inference_model = self.parse_probabilistic_model()

        if self.user_defined_error_model is not None:
            user_error_model = getattr(
                self.simulation._prob,
                self.user_defined_error_model
            )

            self.inference_model = partial(
                self.inference_model,
                user_error_model=user_error_model
            )

        self.check_tolerance_and_jax_mode()
    

    @property
    def user_defined_probability_model(self):
        return self.config.inference_numpyro.user_defined_probability_model
    
    @property
    def user_defined_error_model(self):
        return self.config.inference_numpyro.user_defined_error_model
    
    @property
    def user_defined_preprocessing(self):
        return self.config.inference_numpyro.user_defined_preprocessing

    @property
    def gaussian_base_distribution(self):
        return self.config.inference_numpyro.gaussian_base_distribution
    
    @property
    def chains(self):
        if self.config.inference_numpyro.kernel == "svi":
            return 1
        else:
            return self.config.inference_numpyro.chains
    
    @property
    def draws(self):
        return self.config.inference_numpyro.draws
    
    @property
    def warmup(self):
        return self.config.inference_numpyro.warmup
    
    @property
    def thinning(self):
        return self.config.inference_numpyro.thinning

    @property
    def svi_iterations(self):
        return self.config.inference_numpyro.svi_iterations
    
    @property
    def svi_learning_rate(self):
        return self.config.inference_numpyro.svi_learning_rate
    
    @property
    def kernel(self):
        return self.config.inference_numpyro.kernel
    
    @property
    def adapt_state_size(self):
        return self.config.inference_numpyro.sa_adapt_state_size

    @property
    def init_strategy(self):
        strategy = self.config.inference_numpyro.init_strategy
        return getattr(infer, strategy)

    @staticmethod
    def generate_transform(expression: Expression):
        # check for parentheses in expression
        
        # Parse the expression without knowing the symbol names in advance
        parsed_expression = sympy.sympify(str(expression), evaluate=False)
        free_symbols = tuple(parsed_expression.free_symbols)

        # Transform expression to jax expression
        func = sympy2jax.SymbolicModule(
            parsed_expression, 
            extra_funcs=sympy2jax_extra_funcs, 
            make_array=True
        )

        return {"transform": func, "args": [str(s) for s in free_symbols]}

    def check_tolerance_and_jax_mode(self):
        x64 = jax.config.read("jax_enable_x64")
        atol = self.config.jaxsolver.atol
        if atol < 1e-8 and not x64:
            warnings.warn(
                "Jax is not running in 64 bit mode but the precision is smaller "+
                f"than 1e-8 (config.jaxsolver.atol={atol} < 1e-8). "
                "Increase the absolute tolerance value or run jax in 64 bit mode. "
                "Script: `jax.config.update('jax_enable_x64', True)`. "
                "Commandline: Set environmental variable `JAX_ENABLE_X64=True`.",
                category=UserWarning
            )
        else:
            print("Jax 64 bit mode:", x64)
            print("Absolute tolerance:", atol)


    def parse_deterministic_model(self) -> Callable:
        """Parses an evaluation function from the Simulation object, which 
        takes a single argument theta and defaults to passing no seed to the
        deterministic evaluator.

        Returns
        -------
        callable
            The evaluation function
        """
        def evaluator(theta, y0={}, x_in={}, seed=None):
            evaluator = self.simulation.dispatch(theta=theta, y0=y0, x_in=x_in)
            evaluator(seed)
            return evaluator.Y
        
        return evaluator

    def model(self):
        pass


    def observation_parser(self) -> Tuple[Dict,Dict]:
        """Transform a xarray.Dataset into a dictionary of jnp.Arrays. Creates
        boolean arrays of masks for nan values (missing values are tagged False)

        Returns
        -------
        Tuple[Dict,Dict]
            Dictionaries of observations (data) and masks (missing values)
        """
        obs = self.simulation.observations #\
            # .transpose(*self.simulation.dimensions)
        y0 = self.simulation.model_parameters.get("y0", {})
        observed_data_vars = self.config.data_structure.observed_data_variables + self.extra_vars

        if len(observed_data_vars) == 0:
            warnings.warn(
                "No observed data_variables were found. Is this correct? "+
                "Make sure you have marked "+
                "All relevant data variables as observed before running inference"+
                "`sim.config.data_variables.MY_DATA_VAR.observed = True`",
                category=UserWarning
            )

        # this was implemented to deal with import export and the required precision
        # for jax 
        if jax.config.jax_enable_x64:
            bit = "64"
        else:
            bit = "32"

        masks = {}
        observations = {}
        for d in observed_data_vars:
            o = jnp.array(self.cast_to_precision(obs[d].values, precision=bit))
            m = jnp.logical_not(jnp.isnan(o))
            observations.update({d:o})
            masks.update({d:m})
        
        # add y0 values to the observation dict
        for d in self.config.data_structure.data_variables:
            if d in y0:
                observations.update({
                    f"{d}_y0": jnp.array(self.cast_to_precision(y0[d], precision=bit))
                })

        return observations, masks
    
    @staticmethod
    def cast_to_precision(value, precision="64"):
        if precision in str(value.dtype):
            return value
        
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(f"float{precision}")
        elif np.issubdtype(value.dtype, np.integer):
            return value.astype(f"int{precision}")
        else:
            return value

    def parse_probabilistic_model(self):
        EPS = self.EPS
        prior = self.prior.copy()
        # TODO: This should be passed to the model, becuase if the batch
        #       dimension is subsetted, the index size is not anymore the same
        #       and correspondingly, be parsed during parse_obs or so
        indices = self.indices
        error_model = self.error_model.copy()
        extra = {"EPS": EPS}
        gaussian_base = self.gaussian_base_distribution
        data_variables_y0 = [
            f"{dv}_y0" for dv in self.config.data_structure.data_variables
        ]

        def sample_prior(prior: Dict, obs: Dict, indices: Dict):
            theta = {}
            context = [theta, indices, obs, extra]
            for prior_name, prior_dist in prior.items():
                if prior_dist._dist_str == "deterministic":
                    theta_i = prior_dist.construct(
                        context=context, 
                        extra_kwargs={"name": prior_name}
                    )
                else:  
                    dist = prior_dist.construct(context=context)

                    # TODO: distribution expansion is not so trivial unfortunately
                    dist = dist.expand(batch_shape=prior_dist.shape)

                    theta_i = numpyro.sample(
                        name=prior_name,
                        fn=dist,
                    )
                theta.update({prior_name: theta_i})

            return {}, theta
        
        def sample_prior_gaussian_base(prior: Dict, obs: Dict, indices=indices):
            theta = {}
            theta_base = {}
            context = [theta, indices, obs, extra]
            for prior_name, prior_dist in prior.items():
                if prior_dist._dist_str == "deterministic":
                    theta_i = prior_dist.construct(
                        context=context, 
                        extra_kwargs={"name": prior_name}
                    )
                else:    
                    dist = prior_dist.construct(context=context)

                    try:
                        transforms = getattr(dist, "transforms")
                    except:
                        raise RuntimeError(
                            f"The specified distribution {prior_dist} had no transforms. If setting "+
                            "the option 'inference.numpyro.gaussian_base_distribution = 1', "+
                            "you are only allowed to use parameter distribution, which can "+
                            "be specified as transformed normal distributions. "+
                            f"Currently {transformed_dist_map.keys()} are specified"+
                            "You can use the numypro.distributions.TransformedDistribution "+
                            "API to specify additional distributions with transforms."+
                            "And pass them to the inferer by updating the distribution map: "+
                            "sim.inferer.distribution_map.update({'newdist': your_new_distribution})"
                        )

                    # sample from a random normal distribution
                    # CHECK: Expanding before 
                    # TODO: potentially just use the base dist here, because this is then
                    # already truncated
                    theta_base_i = numpyro.sample(
                        name=f"{prior_name}_normal_base",
                        fn=dist.base_dist.expand(batch_shape=prior_dist.shape),
                    )


                    # apply the transforms 
                    theta_i = numpyro.deterministic(
                        name=prior_name,
                        value=transform(transforms=transforms, x=theta_base_i)
                    )

                    theta_base.update({prior_name: theta_base_i})
                theta.update({prior_name: theta_i})

            return theta_base, theta

        def likelihood(theta, simulation_results, indices, observations, masks, make_predictions):
            """Uses lookup and error model from the local function context"""
            context = [simulation_results, theta, indices, observations, extra]
            for error_model_name, error_model_dist in error_model.items():
                dist = error_model_dist.construct(context=context)

                # this assumes that the transform function is only ever 
                # characterized by algebraic transforms of the observations 
                # and the determinsitic model of that variable 
                # This should cover the vast majority of cases.
                if error_model_dist.obs_transform_func is None:
                    # TODO: consider if this should go into the 2nd ifelse condition
                    #       because then, I could actually rename _obs and _res 
                    #       and downstream use this information for creating inverse
                    #       function
                    if make_predictions:
                        obs = None
                    else:
                        obs = observations[error_model_name]
                    
                    _ = numpyro.sample(
                        name=error_model_name + "_obs",
                        fn=dist.mask(masks[error_model_name]),
                        obs=obs
                    )
                else:
                    residuals = numpyro.deterministic(
                        name=error_model_name + "_res",
                        value=error_model_dist.obs_transform_func(
                            obs=observations[error_model_name], 
                            **{error_model_name: simulation_results[error_model_name]}, 
                        )
                    )

                    if make_predictions:
                        residuals = None

                    _ = numpyro.sample(
                        name=error_model_name + "_obs",
                        fn=dist.mask(masks[error_model_name]),
                        obs=residuals
                    )


        def model(
            solver, obs, masks, 
            only_prior: bool = False, 
            user_error_model: Optional[ErrorModelFunction] = None,
            make_predictions: bool = False,
        ):
            # construct priors with numpyro.sample and sample during inference
            if gaussian_base:
                theta_gaussian, theta_ = sample_prior_gaussian_base(
                    prior=prior, 
                    obs=obs, 
                    indices=indices,
                )
            else:
                _, theta_ = sample_prior(
                    prior=prior, 
                    obs=obs, 
                    indices=indices,
                )
            
            if only_prior:
                return
            
            # calculate deterministic simulation with parameter samples
            y0 = {k.replace("_y0",""): v for k, v in theta_.items() if k in data_variables_y0}
            theta = {k: v for k, v in theta_.items() if k not in data_variables_y0}
            sim_results = solver(theta=theta, y0=y0)

            # store data_variables as deterministic model output
            for deterministic_name, deterministic_value in sim_results.items():
                _ = numpyro.deterministic(
                    name=deterministic_name, 
                    value=deterministic_value
                )

            if user_error_model is None:
                _ = likelihood(
                    theta=theta_,
                    simulation_results=sim_results,
                    indices=indices,
                    observations=obs,
                    masks=masks,
                    make_predictions=make_predictions,
                )
            else:
                _ = user_error_model(
                    theta=theta,
                    simulation_results=sim_results,
                    indices=indices,
                    observations=obs,
                    masks=masks,
                    make_predictions=make_predictions,
                )

        return model

    @staticmethod
    def preprocessing(**kwargs):
        return kwargs

    def run(self, print_debug=True, render_model=True):
        # set parameters of JAX and numpyro
        # jax.config.update("jax_enable_x64", True)

        # generate random keys
        key = jax.random.PRNGKey(self.simulation.config.simulation.seed)
        key, *subkeys = jax.random.split(key, 20)
        keys = iter(subkeys)

        # parse observations and masks for missing data
        obs, masks = self.observation_parser()

        model_kwargs = self.preprocessing(
            obs=obs, 
            masks=masks,
        )

        # prepare model and print information about shapes
        model = partial(
            self.inference_model, 
            solver=self.evaluator, 
            **model_kwargs
        )    

        if render_model:
            try:
                import graphviz
                graph = numpyro.render_model(model, render_distributions=True)
                graph.render(
                    filename=f"{self.simulation.output_path}/probability_model",
                    view=False, cleanup=True, format="png",
                ) 
            except graphviz.backend.ExecutableNotFound:
                warnings.warn(
                    "Model is not rendered, because the graphviz executable is "+
                    "not found. Try search for 'graphviz executables not found' "+
                    "and the used OS. This should be an easy fix :-)"
                )


        if print_debug:
            with numpyro.handlers.seed(rng_seed=1):
                trace = numpyro.handlers.trace(model).get_trace()
            print(numpyro.util.format_shapes(trace)) # type: ignore
            
        # run inference
        if self.kernel.lower() == "sa" or self.kernel.lower() == "nuts":
            sampler, mcmc = self.run_mcmc(
                model=model,
                keys=keys,
                kernel=self.kernel.lower()
            )

            # create arviz idata
            self.idata = self.nuts_posterior(
                mcmc=mcmc, model=model, key=next(keys), obs=obs
            )

        elif self.kernel.lower() == "svi" or self.kernel.lower() == "map":
            if not self.gaussian_base_distribution:
                raise RuntimeError(
                    "SVI is only supported if parameter distributions can be "+
                    "re-parameterized as gaussians. Please set "+
                    "inference.numpyro.gaussian_base_distribution = 1 "+
                    "and if needed use distributions from the loc-scale family "+
                    "to specify the model parameters."
                )
            
            svi, guide, svi_result = self.run_svi(
                model=model,
                keys=keys,
                kernel=self.kernel.lower(),
                learning_rate=self.svi_learning_rate,
                iterations=self.svi_iterations,
            )

            # plot loss curve
            fig, ax = plt.subplots(1, 1)
            ax.plot(svi_result.losses)
            ax.set_yscale("log")
            ax.set_ylabel("Loss")
            ax.set_xlabel("Iteration")
            fig.savefig(f"{self.simulation.output_path}/svi_loss_curve.png")

            # save idata and print summary
            draws = 1 if self.kernel.lower() == "map" else self.draws
            self.idata = self.svi_posterior(
                svi_result, model, guide, next(keys), draws
            )
            print(az.summary(self.idata))

        else:
            raise NotImplementedError(
                f"Kernel {self.kernel} is not implemented. "+
                "Use one of nuts, sa, svi, map"
            )
        
    def run_mcmc(self, model, keys, kernel):
        if kernel == "sa":
            sampler = infer.SA(
                model=model,
                dense_mass=True,
                adapt_state_size=self.config.inference_numpyro.sa_adapt_state_size,
                init_strategy=self.init_strategy,
            )

        elif kernel == "nuts":
            sampler = infer.NUTS(
                model, 
                dense_mass=self.config.inference_numpyro.nuts_dense_mass, 
                step_size=self.config.inference_numpyro.nuts_step_size,
                adapt_mass_matrix=True,
                adapt_step_size=True,
                max_tree_depth=self.config.inference_numpyro.nuts_max_tree_depth,
                target_accept_prob=self.config.inference_numpyro.nuts_target_accept_prob,
                init_strategy=self.init_strategy
            )
        else:
            raise NotImplementedError(
                f"MCMC kernel {kernel} not implemented. Use one of 'sa', 'nuts'"
            )


        mcmc = infer.MCMC(
            sampler=sampler,
            num_warmup=self.warmup,
            num_samples=self.draws * self.thinning,
            num_chains=self.chains,
            thinning=self.thinning,
            progress_bar=True,
        )
    
        # run inference
        mcmc.run(next(keys))
        mcmc.print_summary()

        return sampler, mcmc

    @staticmethod
    def run_svi(model, keys, learning_rate, iterations, kernel):

        init_fn = partial(infer.init_to_uniform, radius=1)
        if kernel == "svi":
            guide = infer.autoguide.AutoMultivariateNormal(model, init_loc_fn=init_fn)
        elif kernel == "map":
            guide = numpyro.infer.autoguide.AutoDelta(model, init_loc_fn=init_fn)
        else:
            raise NotImplementedError(
                f"SVI kernel {kernel} is not implemented. "+
                "Use one of 'map', 'svi'"
            )

        optimizer = numpyro.optim.ClippedAdam(step_size=learning_rate, clip_norm=10)
        svi = infer.SVI(model=model, guide=guide, optim=optimizer, loss=infer.Trace_ELBO())
        svi_result = svi.run(next(keys), iterations, stable_update=True)

        if kernel == "svi":
            cov = svi_result.params['auto_scale_tril'].dot(
                svi_result.params['auto_scale_tril'].T
            )
            median = guide.median(svi_result.params)

        return svi, guide, svi_result

    @property
    def posterior(self):
        warnings.warn(
            "Discouraged use of inferer.posterior API"+
            "use inferer.idata.posterior instead."
        )
        return self.idata.posterior  # type: ignore

    def create_log_likelihood(
        self, 
        seed=1,
        return_type:Literal["joint-log-likelihood", "full", "summed-by-site", "summed-by-prior-data", "custom"]="joint-log-likelihood",
        check=True,
        custom_return_fn: Optional[Callable] = None,
        scaled=True,
        vectorize=False,
        gradients=False,
    ) -> Tuple[Errorfunction,ErrorModelFunction]:
        """Log density relies heavily on the substitute utility
        
        The log density is the scaled log-likelihood. In case the the scale handler
        is  used, log_density reflects this. Usually, the scaled log-density
        should be returned, because it is loss used for the optimizer/sampler

        The general method is actually quite simple. Values of all SAMPLE
        sites are replaced according to the key: value pairs in `theta`.

        Then the model is calculated and the trace is obtained. Everything
        else is then just post-processing of the sites. Here the log_prob
        function of the sites in the trace are used and the values of the
        sites are inserted. 

        Note that the log-density can randomly fluctuate, if not all
        sites are replaced.

        Note that the data-loglik can be used to calculate a maximum-likelihood
        estimate. Because it is independent of the prior

        The method is equivalent using the log_likelihood method, but returns
        only the likelihood of the data given the model parameters.
        
        .. :code-block: python
        
           def log_likelihood(theta: dict):
               theta = {f"{key}_normal_base": val for key, val in theta.items()}
               loglik = numpyro.infer.util.log_likelihood(
                   model=seeded_model,
                   posterior_samples=theta, 
                   batch_ndims=0,
                   solver=sim.inferer.evaluator,
                   obs=data,
                   masks=masks,
               )
              return loglik
           
           jax.vmap(log_likelihood)(theta)

        Parameters
        ----------


        return_type : str
            The information which should be returned. With increasing level
            of computation:
            
            joint-log-likelihood: returns a single value, the entire log
                likelihood of the model, given the values in theta
            full: joint-log, loglik-prior of each site and value, 
                loglik-data of each site and value 
            summed-by-site: joint-loglik, loglik-prior of sites, 
                loglik-data of sites
            summed-by-prior-data:
                joint-loglik, prior-loglik, data-loglik
            custom:
                uses the full log

        """
        key = jax.random.PRNGKey(seed)
        obs, masks = self.observation_parser()

        model_kwargs = self.preprocessing(
            obs=obs, 
            masks=masks,
        )
        
        # prepare model
        model = partial(
            self.inference_model, 
            solver=self.evaluator, 
            **model_kwargs
        )    

        seeded_model = numpyro.handlers.seed(model, key)
   
        def log_density(
            theta: Dict[str, float|List[NumericArray]], 
        ):
            """
            Calculate the log-probability of different sites of the probabilistic
            model

            Parameters
            ----------

            theta : Dict
                Dictionary of priors (sites) which should be deterministically 
                fixed (substituted).

            """
            
            if self.gaussian_base_distribution:
                theta_ = theta.copy()
                theta = {}
                for key, val in theta_.items():
                    if "_normal_base" in key:
                        theta_i = {key: val}
                    else:
                        theta_i = {f"{key}_normal_base": val}

                    theta.update(theta_i)
            else:
                pass
        
            joint_log_density, trace = numpyro.infer.util.log_density( # type: ignore
                model=seeded_model,
                model_args=(),
                model_kwargs={},
                params=theta
            )

            joint_log_density = joint_log_density

            if check:
                sites_in_theta = {
                    name: True 
                    if name in theta
                    else False
                    for name, site in {
                        name: site for name, site in trace.items() 
                        if site["type"] == "sample" 
                        and not site["is_observed"]
                    }.items()
                }

                if not all(list(sites_in_theta.values())):
                    missing_sites = [name for name, in_theta in sites_in_theta.items() if not in_theta]
                    warnings.warn(
                        f"Sites: {missing_sites} were not specified in theta. "+
                        "Log-likelihood will not be fully defined by theta. "+
                        "Results should be independent of the given seed"
                    )


            if return_type == "joint-log-likelihood":
                return joint_log_density
            
            def get_scale(site):
                # get the scale factor for the log probability
                scale = site["scale"]
                if scale is None or not scaled:
                    return jnp.array(1.0, dtype=float)
                else:
                    return scale
            
            prior_loglik = {
                name: site["fn"].log_prob(site["value"]) * get_scale(site)
                for name, site in trace.items()
                if site["type"] == "sample" and not site["is_observed"]
            }

            data_loglik = {
                name: site["fn"].log_prob(site["value"]) * get_scale(site)
                for name, site in trace.items()
                if site["type"] == "sample" and site["is_observed"]
            }

            if return_type == "full":
                return joint_log_density, prior_loglik, data_loglik


            if return_type == "custom":
                return custom_return_fn(joint_log_density, prior_loglik, data_loglik)

            prior_loglik_sum = {
                key: jnp.sum(value) for key, value in prior_loglik.items()
            }

            data_loglik_sum = {
                key: jnp.sum(value) for key, value in data_loglik.items()
            }

            if return_type == "summed-by-site":
                return joint_log_density, prior_loglik_sum, data_loglik_sum

            prior_loglik_total = jnp.sum(jnp.array(list(prior_loglik_sum.values())))
            data_loglik_total = jnp.sum(jnp.array(list(data_loglik_sum.values())))
            
            if return_type == "summed-by-prior-data":
                return joint_log_density, prior_loglik_total, data_loglik_total
            
            raise NotImplementedError(f"return_type flag: {return_type} is not implemented")

        if gradients:
            if not (return_type == "joint-log-likelihood" or return_type == "custom"):
                raise ValueError(
                    "Gradients need a single return value to be computed. "+
                    f"Either choose `return_type={'joint-log-likelihood'}` or "+
                    f"`return_type={'custom'}` and specify an approxpriate "+
                    "`custom_return_fn`."
                )
            else:
                grad_log_density = jax.grad(log_density)

        else:
            grad_log_density = lambda x: None

        if vectorize:
            return jax.vmap(log_density), jax.vmap(grad_log_density)

        else:
            return log_density, grad_log_density

    def check_log_likelihood(
        self, 
        theta: Optional[Dict[str, float|NumericArray]]=None,
        vectorize=False
    ):
        log_density, _ = self.create_log_likelihood(
            seed=self.config.simulation.seed,
            return_type="full",
            check=True,
            vectorize=vectorize
        )
        
        if theta is not None:
            pass
        elif self.config.inference_numpyro.gaussian_base_distribution:
            theta = {k: 0.0 for k, _ in self.config.model_parameters.free.items()}
        else:
            # TODO: replace by prior sample, or prior mean, ...
            theta = self.config.model_parameters.value_dict
            
        
        llsum, llpri, lldat = log_density(theta=theta)
        if not np.isnan(llsum) and not np.isnan(llsum):
            return llsum, llpri, lldat
        
        nanlogliks_pri = [k for k, g in llpri.items() if np.any(np.isnan(g)) or np.any(np.isinf(g))]
        nanlogliks_dat = [k for k, g in lldat.items() if np.any(np.isnan(g)) or np.any(np.isinf(g))]
        
        if len(nanlogliks_dat + nanlogliks_pri) > 0:
            warnings.warn(
                f"Log-likelihoods {nanlogliks_dat + nanlogliks_pri} contained "+
                "NaN or inf values. The gradient based "+
                "samplers will not be able to sample from this model. Make sure "+
                "that all functions are numerically well behaved. "+
                "Inspect the model with `jax.debug.print('{}',x)` "+
                "https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html#exploring-debug-callback "+
                "Or look at the functions step by step to find the position where "+
                "jnp.grad(func)(x) evaluates to NaN"
            )
        return llsum, llpri, lldat

    def check_gradients(
        self, 
        theta: Optional[Dict[str, float|NumericArray]]=None,
        vectorize=False
    ):

        _, grad_log_density = self.create_log_likelihood(
            seed=self.config.simulation.seed,
            return_type="joint-log-likelihood",
            check=False,
            gradients=True,
            vectorize=vectorize
        )

        if theta is not None:
            pass
        elif self.config.inference_numpyro.gaussian_base_distribution:
            # this works, because in the grad function the parameters are
            # renamed with f'{k}_normal_base' if the gaussian base distribution
            # is used.
            theta = {k: 0.0 for k, _ in self.config.model_parameters.free.items()}
        else:
            # TODO: replace by prior sample, or prior mean, ...
            theta = self.config.model_parameters.value_dict
            
        grads = grad_log_density(theta)
        nangrads = [k for k, g in grads.items() if np.any(np.isnan(g)) or np.any(np.isinf(g))]
        if len(nangrads) > 0:
            warnings.warn(
                f"Gradients {nangrads} contained NaN or infinity values. The gradient based "+
                "samplers will not be able to sample from this model. Make sure "+
                "that all functions are numerically well behaved. "+
                "Inspect the model with `jax.debug.print('{}',x)` "+
                "https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html#exploring-debug-callback "+
                "Or look at the functions step by step to find the position where "+
                "jnp.grad(func)(x) evaluates to NaN"
            )

        return grads


    def predict_observations(self, model, posterior_samples, key, n=100):
        """
        there is a very small remark in the numpyro API that explains that
        if data input for observed variables is None, the data are sampled
        from the distributions instead of returning the input data
        https://num.pyro.ai/en/stable/getting_started.html#a-simple-example-8-schools
        """
        data_vars = self.config.data_structure.observed_data_variables
        predictive = Predictive(
            model=partial(model, make_predictions=True),
            # use all samples that were not generated in the likelihood function
            posterior_samples={
                k: v for k, v in posterior_samples.items() if k not in 
                [f"{k}_obs" for k in data_vars] + [f"{k}_res" for k in data_vars]
            },
            return_sites=[f"{k}_obs" for k in data_vars],
            num_samples=n, 
            batch_ndims=2
        )
        # using the same key, to generate predictions the same way
        # as for the the posterior samples
        predictions = predictive(key)

        if self.user_defined_probability_model is not None:
            return predictions

        observations = {}
        for data_var in self.config.data_structure.observed_data_variables:
            if data_var not in self.error_model:
                # catches the case where a user defined error model was provided
                # and error models were not defined.
                # TODO: Use trace handler to obtain the different components
                #       prior, likelihood and deterministic transforms to populate
                #       missing components
                if f"{data_var}_obs" not in predictions:
                    raise KeyError(
                        f"`{data_var}` was not found in the predictions, "+
                        "but is marked as an observed data variable. Either "+
                        f"set `sim.config.data_variables.{data_var}.observed = False` " +
                        f"or generate predictions for '{data_var}' in the error-model."
                    )
                obs = predictions[f"{data_var}_obs"]
            else:
                if self.error_model[data_var].obs_transform_func is not None:
                    transform_inv = self.error_model[data_var].obs_transform_func_inv
                    if transform_inv is None:
                        warnings.warn(
                            "Cannot make predictions of observations from normalized "+
                            "observations (residuals). Please provide an inverse observation "+
                            f"transform: e.g. `sim.config.error_model['{data_var}'].obs_inv = ...`."+
                            "residuals are denoted as 'res'. See Lotka-volterra case "+
                            "study for an example. "
                        )
                        # this will fetch the residuals (they are called _obs)
                        obs = predictions[f"{data_var}_obs"]
                    else:
                        obs = transform_inv(
                            res=predictions[f"{data_var}_obs"],
                            **{data_var: posterior_samples[data_var]}
                        )
                    
                    # get the deterministic residuals res=y_obs-y_det regardless
                    res = posterior_samples[f"{data_var}_res"] 
                    observations.update({f"{data_var}_res": res})
                else:
                    obs = predictions[f"{data_var}_obs"]

            observations.update({f"{data_var}_obs": obs})

        return observations

    def calculate_log_likelihood(self, model, posterior_samples):
        return numpyro.infer.log_likelihood(
            model=model, 
            posterior_samples=posterior_samples, 
            batch_ndims=2, 
        )



    @lru_cache
    def prior_predictions(self, n=None, seed=1):
        if n is None:
            n = self.n_predictions
            
        key = jax.random.PRNGKey(seed)
        obs, masks = self.observation_parser()

        model_kwargs = self.preprocessing(obs=obs, masks=masks)

        # prepare model
        model = partial(
            self.inference_model, 
            solver=self.evaluator, 
            **model_kwargs
        )    
   
        prior_predictive = Predictive(
            model, num_samples=n, batch_ndims=2
        )
        prior_samples = prior_predictive(key)

        log_likelihood = self.calculate_log_likelihood(
            model=model, posterior_samples=prior_samples
        )

        obs_predictions = self.predict_observations(
            model=model, posterior_samples=prior_samples, key=key, n=n
        )

        return self.to_arviz_idata(
            prior=prior_samples,
            prior_predictive=obs_predictions,
            observed_data=obs,
            log_likelihood=log_likelihood,
            n_draws=n,
            n_chains=1
        )
    

    def to_arviz_idata(
        self,
        prior: Dict[str, NumericArray] = {},
        posterior: Dict[str, NumericArray] = {},
        log_likelihood: Dict[str, NumericArray] = {},
        prior_predictive: Dict[str, NumericArray] = {},
        posterior_predictive: Dict[str, NumericArray] = {},
        observed_data: Dict[str, NumericArray] = {},
        n_draws: Optional[int] = None,
        n_chains: Optional[int] = None,
        **kwargs,
    ):
        """Create an Arviz idata object from samples.
        TODO: Outsource to base.InferenceBackend
        """
        posterior_coords = self.posterior_coordinates
        # posterior_coords["chain"] = [0]
        if n_draws is not None:
            posterior_coords["draw"] = list(range(n_draws))

        if n_chains is not None:
            posterior_coords["chain"] = list(range(n_chains))
        data_structure = self.posterior_data_structure

        data_variables = self.config.data_structure.data_variables
        obs_data_variables = self.config.data_structure.observed_data_variables
        prior_keys = self.simulation.model_parameter_dict.keys()
        
        prior_ = {k: v for k, v in prior.items() if k in prior_keys}
        unconstrained_prior_ = {
            k: v for k, v in prior.items() 
            if k in [f"{k}_normal_base" for k in prior_keys]
        }
        prior_model_fits_ = {
            k: v for k, v in prior.items() if k in data_variables
        }
        prior_residuals_ = {
            k.replace("_res", ""): v 
            for k, v in prior_predictive.items() 
            if k in [f"{d}_res" for d in data_variables]
        }
        prior_predictive_ = {
            k.replace("_obs", ""): v 
            for k, v in prior_predictive.items() 
            if k in [f"{d}_obs" for d in data_variables]
        }

        posterior_ = {k: v for k, v in posterior.items() if k in prior_keys}
        unconstrained_posterior_ = {
            k: v for k, v in posterior.items() 
            if k in [f"{k}_normal_base" for k in prior_keys]
        }
        posterior_model_fits_ = {
            k: v for k, v in posterior.items() if k in data_variables
        }
        posterior_residuals_ = {
            k.replace("_res", ""): v 
            for k, v in posterior_predictive.items() 
            if k in [f"{d}_res" for d in data_variables]
        }
        posterior_predictive_ = {
            k.replace("_obs", ""): v 
            for k, v in posterior_predictive.items() 
            if k in [f"{d}_obs" for d in data_variables]
        }
        
        likelihood_ = {
            k.replace("_obs", ""): v 
            for k, v in log_likelihood.items() 
            if k in [f"{d}_obs" for d in obs_data_variables]
        }
        
        observed_data_ = {
            k: v for k, v in observed_data.items() 
            if k in obs_data_variables
        }
        
        if len(prior_) > 0 and len(prior_keys) != len(prior_):
            miss_keys = [k for k in prior_keys if k not in prior]
        else:
            miss_keys = []

        if len(posterior_) > 0 and len(prior_keys) != len(posterior_):
            miss_keys = [k for k in prior_keys if k not in posterior]
        else:
            miss_keys = []

        if len(miss_keys) > 0:
            warnings.warn(
                f"Parameters {miss_keys} "+
                "were not found in the predictions. Make sure that the prior "+
                "names match the names in user_defined_probability_model = "+
                f"{self.config.inference_numpyro.user_defined_probability_model}.",
                category=UserWarning
            )

        idata = az.from_dict(
            prior=prior_,
            prior_predictive=prior_predictive_,
            posterior=posterior_,
            posterior_predictive=posterior_predictive_,
            observed_data=observed_data_,
            log_likelihood=likelihood_,
            **kwargs,
            dims=data_structure,
            coords=posterior_coords,
        )

        idata.add_groups(group_dict={
                "prior_model_fits": prior_model_fits_,
                "posterior_model_fits": posterior_model_fits_,
                "prior_residuals": prior_residuals_,
                "posterior_residuals": posterior_residuals_,
                "unconstrained_prior": unconstrained_prior_,
                "unconstrained_posterior": unconstrained_posterior_,
            }, 
            dims=data_structure,
            coords=posterior_coords
        )

        for group_key in idata.groups():
            if self.config.simulation.batch_dimension in idata[group_key].coords:
                idata[group_key] = idata[group_key].assign_coords({  # type: ignore
                  k: (self.config.simulation.batch_dimension, v) 
                    for k, v in self.indices.items()
                }) 
                idata[group_key].attrs.update({"pymob_version": pymob.__version__})

        return idata
    

    def posterior_draws_from_svi(self, guide, svi_result, n, key):
        # prepare model without obs, so obs are sampled from the posterior
        params = svi_result.params

        # this gets the parameters of the normal_base_distributions
        predictive = Predictive(
            guide, params=params, 
            num_samples=n, batch_ndims=2
        )
        posterior_samples = predictive(key)
        return posterior_samples


    def svi_posterior(self, svi_result, model, guide, key, n=1000):
        # TODO: Harmonize SVI posterior and nuts posterior. The base function
        # should only accept only samples from the posterior. Then the tool
        # could be used to generate observations with higher time resolution
        obs, masks = self.observation_parser()

        posterior_samples = self.posterior_draws_from_svi(
            guide=guide, svi_result=svi_result, n=n, key=key
        )

        # this gets all the parameters and predictions
        predictive = Predictive(
            model, posterior_samples, #params=params, 
            num_samples=n, batch_ndims=2
        )
        posterior = predictive(key)

        # passing complete posterior is identical to only passing only posterior_samples
        log_likelihood = self.calculate_log_likelihood(
            model=model, posterior_samples=posterior_samples, 
        )

        # here only the transformed data are passed. I suppose this is enough,
        # because. The transformed data are the only thing required for the
        obs_predictions = self.predict_observations(
            model=model, posterior_samples=posterior, key=key, n=n
        )

        # TODO: When passing the complete posterior to the calculate_log_likelihood
        # function the results are the same as when only using the posterior,
        # but when passing it to predict_observations, the results are different
        # I suppose this is correct, but make sure to udnerstand what's going on
        # and then document it.
        complete_posterior = {}
        complete_posterior.update(posterior_samples)
        complete_posterior.update(posterior)

        return self.to_arviz_idata(
            posterior=complete_posterior,
            posterior_predictive=obs_predictions,
            log_likelihood=log_likelihood,
            observed_data=obs,
            n_draws=n
        )



    def nuts_posterior(self, mcmc, model, key, obs):
        samples = jax.device_get(mcmc.get_samples(group_by_chain=True))

        priors = list(self.prior.keys())
        data_variables = self.config.data_structure.data_variables
        
        log_likelihood = self.calculate_log_likelihood(
            model=model, posterior_samples=samples
        )
        
        posterior_predictive = self.predict_observations(
            model, posterior_samples=samples, key=key, n=None
        )

        unconstrained_prior = [f"{p}_normal_base" for p in priors]
        return self.to_arviz_idata(
            posterior={
                k: v for k, v in samples.items() 
                if k in priors + unconstrained_prior + data_variables
            },
            posterior_predictive=posterior_predictive,
            log_likelihood=log_likelihood,
            observed_data=obs,
            sample_stats=mcmc.get_extra_fields(group_by_chain=True)
        )

    @staticmethod
    def get_dict(group: xr.Dataset):
        data_dict = group.to_dict()["data_vars"]
        return {k: np.array(val["data"]) for k, val in data_dict.items()}


    @lru_cache
    def posterior_predictions(self, n: Optional[int]=None, seed=1):
        # TODO: It may be necessary that the coordinates should be passed as 
        # constant data. Because if the model is compiled with them once, 
        # according to the design philosophy of JAX, the model will not 
        # be evaluated again. But considering that the jitted functions do take
        # coordinates as an input argument, maybe I'm okay. This should be
        # tested.
        posterior = self.idata.posterior  # type: ignore
        n_samples = posterior.sizes["chain"] * posterior.sizes["draw"]
        data_variables_y0 = [
            f"{dv}_y0" for dv in self.config.data_structure.data_variables
        ]

        if n is not None:
            key = jax.random.PRNGKey(seed)

            n_draws = int(n / posterior.sizes["chain"])
            # the same selection of draws will be applied to all chains. This
            # any other form will result in an array, where a lot of nans 
            # are present of the size n * chains, while we want size n
            selection = jax.random.choice(
                key=key, 
                a=posterior.draw.values, 
                replace=False, 
                shape=(n_draws, ) # type: ignore
            )
            posterior = posterior.isel(draw=selection)


        preds = []
        with tqdm(
            total=posterior.sizes["chain"] * posterior.sizes["draw"],
            desc="Posterior predictions"
        ) as pbar:
            for chain in posterior.chain:
                for draw in posterior.draw:
                    theta_arr = posterior.sel(draw=draw, chain=chain)
                    theta_dict = self.get_dict(theta_arr)
                    
                    # calculate deterministic simulation with parameter samples
                    y0 = {k.replace("_y0",""): v for k, v in theta_dict.items() if k in data_variables_y0}
                    theta = {k: v for k, v in theta_dict.items() if k not in data_variables_y0}
                    evaluator = self.simulation.dispatch(theta=theta, y0=y0)
                    evaluator()
                    ds = evaluator.results

                    ds = ds.assign_coords({"chain": chain, "draw": draw})
                    ds = ds.expand_dims(("chain", "draw")) # type:ignore
                    preds.append(ds)
                    pbar.update(1)

        # key = jax.random.PRNGKey(seed)
        # model = partial(self.model, solver=self.evaluator)    
        # predict = numpyro.infer.Predictive(model, posterior_samples=posterior, batch_ndims=2)
        # predict(key, obs=obs, masks=masks)
        

        return xr.combine_by_coords(preds)

    def store_results(self, output=None):
        if output is not None:
            self.idata.to_netcdf(output)
        else:
            self.idata.to_netcdf(f"{self.simulation.output_path}/numpyro_posterior.nc")

    def load_results(self, file="numpyro_posterior.nc"):
        self.idata = az.from_netcdf(f"{self.simulation.output_path}/{file}")




    # This is a separate script!    
    def combine_chains(self, chain_location="chains", drop_extra_vars=[], cluster_deviation="std"):
        """Combine chains if chains were computed in a fully parallelized manner
        (on different machines, jobs, etc.). 

        In addition, the method drops all data variables and '..._norm' priors 
        (i.e. helper priors with a normal base). This is done, in order to
        create slim data objects for storage.

        Parameters
        ----------
        chain_location : str, optional
            location of the chains, relative to the simulation.output_path, this
            parameter is simulteneously the string appended to the saved 
            posterior. By default "chains"
        drop_extra_vars : List, optional
            any additional variables to drop from the posterior
        """
        sim = self.simulation
        pseudo_chains = glob.glob(
            f"{sim.output_path}/{chain_location}/*/numpyro_posterior.nc"
        )

        # just be aware that in the case of MAP this is not an acutal posterior.
        # But it can behave like one with multiple chains (1 for each start)
        idata = az.from_netcdf(pseudo_chains[0])
        posterior = self.drop_vars_from_posterior(idata.posterior, drop_extra_vars) # type: ignore
        log_likelihood = idata.log_likelihood # type: ignore

        # iterate over the posterior files with a progress bar (depending on the
        # size and number of posteriors this op needs time and memory)
        tqdm_iterator = tqdm(
            enumerate(pseudo_chains[1:], start=1), 
            total=len(pseudo_chains)-1,
            desc="Concatenating posteriors"
        )
        for i, f in tqdm_iterator:
            idata = az.from_netcdf(f)
            ccord = {"chain": np.array([i])}
            
            # add chain coordinate to posterior and likelihood
            idata.posterior = self.drop_vars_from_posterior(idata.posterior, drop_extra_vars) # type: ignore
            idata.posterior = idata.posterior.assign_coords(ccord) # type: ignore
            idata.log_likelihood = idata.log_likelihood.assign_coords(ccord) # type: ignore

            # concatenate chains
            posterior = xr.concat([posterior, idata.posterior], dim="chain") # type: ignore
            log_likelihood = xr.concat(
                [log_likelihood, idata.log_likelihood], # type: ignore
                dim="chain"
            )

        posterior = rename_extra_dims(
            posterior, 
            new_dim="substance", 
            new_coords=sim.observations.attrs["substance"]
        )

        # store mutlichain inferencedata to the main output directory
        # this is also a slim posterior that only contains the necessary information
        # posterior and likelihood and has therefore a small file size
        idata_multichain = az.InferenceData(
            posterior=posterior, 
            log_likelihood=log_likelihood,
            observed_data=idata.observed_data, # type: ignore
        )

        idata_multichain = add_cluster_coordinates(idata_multichain, cluster_deviation)
        print("Clusters:", idata_multichain.posterior.cluster)

        return idata_multichain            

    def drop_vars_from_posterior(self, posterior, drop_extra_vars):
        """drops extra variables if they are included in the posterior
        """
        drop_vars = [k for k in list(posterior.data_vars.keys()) if "_norm" in k]
        drop_vars = drop_vars + self.config.data_structure.data_variables + drop_extra_vars
        drop_vars = [v for v in drop_vars if v in posterior]
        drop_coords = [c for c in list(posterior.coords.keys()) if c.split("_dim_")[0] in drop_vars]

        posterior = posterior.drop(drop_vars)
        posterior = posterior.drop(drop_coords)

        return posterior