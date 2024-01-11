import numpy as np

from pymob.utils.store_file import prepare_casestudy

def create_simulation():
    config = prepare_casestudy(
        case_study=("test_case_study", "test_scenario"),
        config_file="settings.cfg"
    )
    from case_studies.test_case_study.sim import Simulation
    return Simulation(config=config)

def test_diffrax_exception():
    # with proper scripting API define JAX model here or import from fixtures
    sim = create_simulation()

    # diffrax returns infinity for all computed values after which the solver 
    # breaks due to raching maximum number of steps. 
    # This function calculates the number of inf values
    n_inf = lambda x: (x.results.to_array() == np.inf).values.sum() / len(x.data_variables)

    ub_alpha = 5.0  # alpha values above do not yield reasonable fits for beta = 0.02
    alpha = np.logspace(-2, 3, 20)  # we sample alpha from 0.01-100

    badness = []
    for a in alpha:
        eva = sim.dispatch({"alpha": 5, "beta": 0.02})
        eva()

        badness.append(n_inf(eva))

    # the tests make sure that parameters within feasible bounds result in simulation
    # results without inf values and do contain inf values when parameters above
    # feasible bounds are sampled.
    badness_for_feasible_alpha = np.array(badness)[np.where(alpha < ub_alpha)[0]]
    assert sum(badness_for_feasible_alpha) == 0

    badness_for_infeasible_alpha = np.array(badness)[np.where(alpha >= ub_alpha)[0]]
    assert sum(badness_for_infeasible_alpha) > 0

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    test_diffrax_exception()