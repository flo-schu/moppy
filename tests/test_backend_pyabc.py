import pytest
import numpy as np
from click.testing import CliRunner

from tests.fixtures import init_simulation_casestudy_api


def test_convergence():
    pytest.skip()
    sim = init_simulation_casestudy_api()
    sim.set_inferer(backend="pyabc")
    sim.inferer.run()
    sim.inferer.load_results()
    
    posterior_mean = sim.inferer.idata.posterior.mean(("chain", "draw"))
    true_parameters = sim.model_parameter_dict
    
    # tests if true parameters are close to recovered parameters from simulated
    # data
    np.testing.assert_allclose(
        posterior_mean.to_dataarray().values,
        np.array(list(true_parameters.values())),
        rtol=5e-2, atol=1e-5
    )


def test_commandline_api_infer():
    # TODO: This will run, once methods are available for 
    # - prior_predictive_checks, 
    # - store_results, 
    # - posterior_predictive_checks 
    pytest.skip()
    from pymob.infer import main
    runner = CliRunner()
    
    args = "--case_study=lotka_volterra_case_study "\
        "--scenario=test_scenario "\
        "--inference_backend=pyabc"
    result = runner.invoke(main, args.split(" "))

    if result.exception is not None:
        raise result.exception

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_commandline_API_infer()