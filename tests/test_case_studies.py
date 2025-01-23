"""
TODO: Case study tests should be outsourced to the different tests. Nevertheless
the tests should be run when pymob is tested, but not in here but in the 
case studies.
"""


import numpy as np
import pytest

from tests.fixtures import (
    init_simulation_casestudy_api,
    init_case_study_and_scenario,
    init_guts_casestudy_constant_exposure,
    init_guts_casestudy_variable_exposure,
    init_bufferguts_leo_casestudy,
    init_bufferguts_casestudy,
)


def test_multiple_case_study_import():
    sim = init_simulation_casestudy_api()
    sim = init_guts_casestudy_constant_exposure()
    sim = init_bufferguts_casestudy()
    sim = init_bufferguts_casestudy()
    sim = init_simulation_casestudy_api()
    sim = init_case_study_and_scenario(
        case_study="tktd_rna_pulse", 
        scenario="rna_pulse_3_6c_substance_specific"
    )





def test_bufferguts_variable_exposure():
    pytest.skip()
    sim = init_bufferguts_casestudy()

    # sim.use_symbolic_solver(do_compile=True)
    # evaluator = sim.dispatch(theta={})
    # evaluator()
    # sol_symbolic = evaluator.results

    sim.use_discrete_solver()
    evaluator = sim.dispatch(theta={})
    evaluator()
    sol_discrete = evaluator.results

    sim.use_jax_solver()
    sim.dispatch_constructor(rtol=1e-7, atol=1e-8, max_steps=1e6)
    evaluator = sim.dispatch(theta={})
    evaluator()
    sol_numerical = evaluator.results

    # make sure errors are small between exact solution and numerical solution
    # the errors come from:
    # a) integrating not exactly to t_eq
    # b) using a numerical switch in the ODE solution.
    diff = (
        sol_numerical.sel(time=np.arange(0,sim.t_max))
        - sol_discrete.sel(time=np.arange(0,sim.t_max))
    )[["B", "D", "H", "S"]]
    max_delta = np.abs(diff).max().to_array()
    np.testing.assert_array_less(max_delta, [1e-2, 1e-8, 1e-8])


    axes = sim._plot.plot_multiexposure(sol_numerical, vars=["exposure", "B", "D", "H", "S"], color="tab:blue", label_prefix="ODE")
    axes = sim._plot.plot_multiexposure(sol_discrete, vars=["exposure", "B", "D", "H", "S"], axes=axes, color="tab:red", linestyle="--", label_prefix="exact")
    fig = axes[0].figure
    fig.savefig(f"{sim.output_path}/solution_comparison.png")


def test_bufferguts_conditional_binomial():
    pytest.skip()
    sim = init_case_study_and_scenario(case_study="bufferguts", scenario="simulate_lab_experiment")
    
    sim.config.inference_numpyro.user_defined_error_model = "conditional_survival_error_model"
    sim.dispatch_constructor()
    sim.set_inferer("numpyro")

    sim.config.inference_numpyro.kernel = "nuts"
    sim.config.inference_numpyro.svi_iterations = 1000
    sim.config.inference_numpyro.chains = 1
    sim.inferer.run(print_debug=False, render_model=False)

    sim.inferer.idata.posterior_predictive

    sim.inferer.idata.posterior
    # TODO: Test output somehow



def test_bufferguts_multinomial():
    pytest.skip()
    sim = init_case_study_and_scenario(case_study="bufferguts", scenario="simulate_lab_experiment_multinomial")
    
    sim.config.inference_numpyro.user_defined_error_model = "conditional_survival_error_model"
    sim.dispatch_constructor()
    sim.set_inferer("numpyro")
    
    sim.config.inference_numpyro.kernel = "map"
    sim.config.inference_numpyro.svi_iterations = 10
    sim.config.inference_numpyro.chains = 1
    sim.inferer.run()

    sim.inferer.idata


def test_bufferguts_hybrid_solution():
    sim = init_bufferguts_casestudy()

    sim.use_hybrid_solver()
    evaluator = sim.dispatch(theta={})
    evaluator()
    sol_symbolic = evaluator.results

    sim.use_jax_solver()
    sim.dispatch_constructor(rtol=1e-7, atol=1e-8, max_steps=1e6)
    evaluator = sim.dispatch(theta={})
    evaluator()
    sol_numerical = evaluator.results

    # make sure errors are small between exact solution and numerical solution
    # the errors come from:
    # a) integrating not exactly to t_eq
    # b) using a numerical switch in the ODE solution.
    diff = (
        sol_numerical.sel(time=np.arange(0,sim.t_max))
        - sol_symbolic.sel(time=np.arange(0,sim.t_max))
    )[["B", "D", "H", "S"]]
    max_delta = np.abs(diff).max().to_array()
    # large errors are partially due to the solver running at 64 bit precision 
    # (exact to 8 digits)
    np.testing.assert_array_less(max_delta, [0.25, 0.25, 0.25, 0.25])


    axes = sim._plot.plot_multiexposure(sol_numerical, vars=["exposure", "B", "D", "H", "S"], color="tab:blue", label_prefix="ODE")
    axes = sim._plot.plot_multiexposure(sol_symbolic, vars=["exposure", "B", "D", "H", "S"], axes=axes, color="tab:red", linestyle="--", label_prefix="exact")
    fig = axes[0].figure
    fig.savefig(f"{sim.output_path}/solution_comparison.png")

def test_bufferguts_leo():
    sim = init_bufferguts_leo_casestudy()
    sim.initialize(["Beta-Cyfluthrin_M-051896_acute_oral.xlsx", "Exposure", "BufferGUTS_SD"])
    sim.setup_numpyro_inferer()

    e = sim.dispatch({})
    e()
    e.results

    sim.inferer.run()
    sim.inferer.store_results()
    fig = sim.plot_posterior_predictions()
    fig.savefig(f"{sim.output_path}/posterior_model_fits.png")


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    pass
    # test_multiple_case_study_import()
    # test_constant_exposure()
    # test_variable_exposure()
    # test_bufferguts_variable_exposure()
    # test_bufferguts_hybrid_solution()
