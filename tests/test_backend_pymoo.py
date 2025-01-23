import pytest
import json
import numpy as np
from click.testing import CliRunner

from tests.fixtures import init_simulation_casestudy_api


def test_convergence():
    sim = init_simulation_casestudy_api()
    sim.set_inferer(backend="pymoo")
    sim.inferer.run()

    with open(f"{sim.config.case_study.output_path}/pymoo_params.json", "r") as f:
        pymoo_results = json.load(f)

    estimated_parameters = pymoo_results["X"]
    true_parameters = sim.model_parameter_dict
    
    np.testing.assert_allclose(
        np.array(list(estimated_parameters.values())),
        np.array(list(true_parameters.values())),
        rtol=1e-1, atol=5e-3
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
        "--inference_backend=pymoo"
    result = runner.invoke(main, args.split(" "))

    if result.exception is not None:
        raise result.exception


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_scripting_api_pyabc()