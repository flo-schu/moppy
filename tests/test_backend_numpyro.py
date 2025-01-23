import pytest
import numpy as np
from click.testing import CliRunner
from matplotlib import pyplot as plt
from pymob.solvers.diffrax import JaxSolver
from pymob.inference.numpyro_backend import NumpyroBackend
from pymob.sim.parameters import Param

from tests.fixtures import (
    init_simulation_casestudy_api, 
    init_lotka_volterra_case_study_hierarchical_from_settings,
)

def test_diffrax_exception():
    # with proper scripting API define JAX model here or import from fixtures
    sim = init_simulation_casestudy_api("test_scenario")

    # diffrax returns infinity for all computed values after which the solver 
    # breaks due to raching maximum number of steps. 
    # This function calculates the number of inf values
    n_inf = lambda x: (x.results.to_array() == np.inf).values.sum() / len(x.data_variables)

    ub_alpha = 5.0  # alpha values above do not yield reasonable fits for beta = 0.02
    alpha = np.logspace(-2, 3, 20)  # we sample alpha from 0.01-100

    badness = []
    for a in alpha:
        eva = sim.dispatch({"alpha": a, "beta": 0.02})
        eva()

        badness.append(n_inf(eva))

    # the tests make sure that parameters within feasible bounds result in simulation
    # results without inf values and do contain inf values when parameters above
    # feasible bounds are sampled.
    badness_for_feasible_alpha = np.array(badness)[np.where(alpha < ub_alpha)[0]]
    assert sum(badness_for_feasible_alpha) == 0

    badness_for_infeasible_alpha = np.array(badness)[np.where(alpha >= ub_alpha)[0]]
    assert sum(badness_for_infeasible_alpha) > 0


def test_convergence_user_defined_probability_model():
    sim = init_simulation_casestudy_api("test_scenario")

    sim.config.data_structure.wolves.observed = False
    sim.config.data_structure.rabbits.observed = False

    sim.config.inference_numpyro.kernel = "nuts"
    sim.config.inference_numpyro.user_defined_probability_model = "parameter_only_model"
    sim.config.inference_numpyro.user_defined_preprocessing = "dummy_preprocessing"

    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    
    posterior_median = sim.inferer.idata.posterior.median( # type: ignore
        ("chain", "draw"))[["beta", "alpha"]] 
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_median.to_dataarray().values,
        np.array([0.02, 0.5]),
        rtol=1e-1, atol=1e-3
    )


def test_convergence_nuts_kernel():
    sim = init_simulation_casestudy_api("test_scenario")

    sim.config.inference_numpyro.kernel = "nuts"
    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    
    posterior_mean = sim.inferer.idata.posterior.mean( # type: ignore
        ("chain", "draw"))[sim.model_parameter_names]
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=1e-2, atol=1e-3
    )

def test_convergence_nuts_kernel_jaxsolver():
    sim = init_simulation_casestudy_api("test_scenario")
    sim.solver = JaxSolver
    # TODO can the lotka volterra case study be run with the Jax solver
    # without throw_exception = False
    sim.dispatch_constructor(throw_exception=False)
    
    sim.config.inference_numpyro.kernel = "nuts"
    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    
    posterior_mean = sim.inferer.idata.posterior.mean( # type: ignore
        ("chain", "draw"))[sim.model_parameter_names]
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=1e-2, atol=1e-3
    )


def test_convergence_svi_kernel_jaxsolver_truncated_alpha():
    sim = init_simulation_casestudy_api("test_scenario_truncated_alpha")
    sim.solver = JaxSolver
    # TODO can the lotka volterra case study be run with the Jax solver
    # without throw_exception = False
    sim.dispatch_constructor(throw_exception=False)
    
    sim.config.inference_numpyro.gaussian_base_distribution = True

    sim.config.inference_numpyro.kernel = "svi"
    sim.config.inference_numpyro.svi_iterations = 10_000
    sim.config.inference_numpyro.svi_learning_rate = 0.01
    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    
    posterior_mean = sim.inferer.idata.posterior.mean( # type: ignore
        ("chain", "draw"))[sim.model_parameter_names]
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=1e-2, atol=1e-3
    )

    sim.inferer.plot_diagnostics()

    # TODO: Test traceplot

def test_convergence_svi_kernel():
    sim = init_simulation_casestudy_api("test_scenario")

    sim.config.inference_numpyro.kernel = "svi"
    sim.config.inference_numpyro.svi_iterations = 10_000
    sim.config.inference_numpyro.svi_learning_rate = 0.01
    # this samples the model with standard normal distributions
    # and rescales them according to the transformations of the specified 
    # parameter distributions to the normal
    sim.config.inference_numpyro.gaussian_base_distribution = True

    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    sim.inferer.idata.posterior_predictive # type: ignore

    posterior_mean = sim.inferer.idata.posterior.mean( # type: ignore
        ("chain", "draw"))[sim.model_parameter_names]
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=1e-2, atol=1e-3
    )


    # posterior predictions
    fig, axes = plt.subplots(2,1, sharex=True)
    for data_var, ax in zip(sim.config.data_structure.data_variables, axes):
        ax = sim.inferer.plot_posterior_predictions( # type: ignore
            data_variable=data_var, 
            prediction_data_variable=data_var,
            x_dim="time",
            ax=ax
        )


def test_convergence_map_kernel():
    sim = init_simulation_casestudy_api("test_scenario")

    sim.config.inference_numpyro.kernel = "map"
    sim.config.inference_numpyro.svi_iterations = 2000
    sim.config.inference_numpyro.svi_learning_rate = 0.01
    # this samples the model with standard normal distributions
    # and rescales them according to the transformations of the specified 
    # parameter distributions to the normal
    sim.config.inference_numpyro.gaussian_base_distribution = True

    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    sim.inferer.idata.posterior_predictive # type: ignore

    posterior_mean = sim.inferer.idata.posterior.mean( # type: ignore
        ("chain", "draw"))[sim.model_parameter_names]
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=1e-2, atol=1e-3
    )


def test_convergence_sa_kernel():
    sim = init_simulation_casestudy_api("test_scenario")

    sim.config.inference_numpyro.kernel = "sa"
    sim.config.inference_numpyro.init_strategy = "init_to_sample"
    sim.config.inference_numpyro.warmup = 2000
    sim.config.inference_numpyro.draws = 1000
    sim.config.inference_numpyro.sa_adapt_state_size = 10

    sim.set_inferer(backend="numpyro")
    sim.inferer.run()
    
    posterior_mean = sim.inferer.idata.posterior.mean( # type: ignore
        ("chain", "draw"))[sim.model_parameter_names]
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=1e-2, atol=1e-3
    )



    # posterior predictions
    for data_var in sim.config.data_structure.data_variables:
        ax = sim.inferer.plot_posterior_predictions( # type: ignore
            data_variable=data_var, 
            prediction_data_variable=data_var,
            x_dim="time"
        )


def test_convergence_hierarchical_lotka_volterra():
    sim = init_lotka_volterra_case_study_hierarchical_from_settings()

    sim.config.model_parameters.alpha_species_hyper = Param(
        prior="halfnorm(scale=5)", dims=('rabbit_species',) , # type: ignore
        hyper=True, free=True
    )
    # sim.config.model_parameters.alpha_species_sigma_hyper = Param(
    #     prior="halfnorm(scale=5)", dims=('rabbit_species',) , # type: ignore
    #     hyper=True, free=True
    # )
    sim.config.model_parameters.alpha_sigma = Param(
        prior="halfnorm(scale=1)", hyper=True, free=True # type:ignore
    )
    sim.config.model_parameters.alpha_species.prior="norm(loc=[[alpha_species_hyper[0]],[alpha_species_hyper[1]]],scale=0.1)" # type: ignore
    sim.config.model_parameters.alpha.prior="lognorm(s=alpha_sigma,scale=alpha_species[rabbit_species_index, experiment_index])" # type: ignore
    sim.config.model_parameters.beta.prior="lognorm(s=1,scale=1)" # type: ignore
    # sim.config.model_parameters.sigma = Param(value=1.0, prior="halfnorm(scale=5)", free=True) # type: ignore
    
    # this reorders the parameters, beginning with alpha_species_hyper
    sim.config.model_parameters.reorder(["alpha_species_hyper", "alpha_sigma"])

    sim.config.error_model.rabbits = "norm(loc=0, scale=1, obs=(obs-rabbits)/jnp.sqrt(rabbits+1e-6),obs_inv=res*jnp.sqrt(rabbits+1e-6)+rabbits)"
    sim.config.error_model.wolves = "norm(loc=0, scale=1, obs=(obs-wolves)/jnp.sqrt(wolves+1e-6),obs_inv=res*jnp.sqrt(wolves+1e-6)+wolves)"


    # using SVI, because it is much faster than NUTS. 
    sim.config.inference_numpyro.kernel = "svi"
    sim.config.inference_numpyro.svi_iterations = 5_000
    sim.config.inference_numpyro.svi_learning_rate = 0.005
    sim.config.inference_numpyro.gaussian_base_distribution = True
    sim.config.jaxsolver.max_steps = 1e5
    sim.config.jaxsolver.throw_exception = False
    sim.config.inference_numpyro.init_strategy = "init_to_median"
    sim.solver = JaxSolver
    sim.dispatch_constructor()
    sim.set_inferer("numpyro")

    sim.config.case_study.scenario = "lotka_volterra_hierarchical_presimulated_v1"
    sim.config.create_directory("scenario", force=True)
    sim.config.save(force=True)
    sim.inferer.run()

    # this works
    np.testing.assert_allclose(
        sim.inferer.idata.posterior.beta.mean(("chain", "draw")), 
        sim.config.model_parameters.beta.value,
        atol=0.0003,
        rtol=0.0001
    )

    # TODO: CUrrently this is not very accurate. But it is a sufficient test
    # that hierarchical models can be fitted
    # correct
    np.testing.assert_allclose(
        sim.inferer.idata.posterior.alpha_species_hyper.mean(("chain", "draw")), 
        (1, 3),
        atol=0.1,
        rtol=0.01
    )



def test_hierarchical_lotka_volterra_user_defined_prob_model():
    pytest.skip()    
    sim = init_lotka_volterra_case_study_hierarchical_from_settings()
    sim.config.inference_numpyro.user_defined_probability_model = "hierarchical_lotka_volterra"

    sim.solver = JaxSolver
    sim.config.inference_numpyro.kernel = "svi"
    sim.config.inference_numpyro.svi_iterations = 2_000
    sim.config.inference_numpyro.svi_learning_rate = 0.005
    sim.config.jaxsolver.max_steps = 1e5
    sim.config.jaxsolver.throw_exception = False
    sim.config.inference_numpyro.init_strategy = "init_to_uniform"
    sim.dispatch_constructor()
    sim.set_inferer("numpyro")

    # TODO: Test this model with SVI
    sim.inferer.run()

    np.testing.assert_almost_equal(
        sim.inferer.idata.posterior.beta, 
        sim.config.model_parameters.beta.value
    )
    np.testing.assert_almost_equal(
        sim.inferer.idata.posterior.alpha_species, 
        sim.config.model_parameters.alpha_species.value
    )


def test_commandline_api_infer():
    from pymob.infer import main
    runner = CliRunner()
    
    args = "--case_study=lotka_volterra_case_study "+\
        "--scenario=test_scenario "+\
        "--inference_backend=numpyro"
    result = runner.invoke(main, args.split(" "))

    if result.exception is not None:
        raise result.exception


if __name__ == "__main__":
    pass