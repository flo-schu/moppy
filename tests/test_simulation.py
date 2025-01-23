import pytest
import xarray as xr
import numpy as np
from click.testing import CliRunner

from pymob.simulation import SimulationBase
from pymob.sim.config import Param, DataVariable

from tests.fixtures import init_simulation_casestudy_api, linear_model

def test_simulation():
    sim = init_simulation_casestudy_api()

    evalu = sim.dispatch(theta=sim.model_parameter_dict)
    evalu()

    ds = evalu.results
    ds_ref = xr.load_dataset(f"{sim.data_path}/simulated_data.nc")

    np.testing.assert_allclose(
        (ds - ds_ref).to_array().values,
        0
    )

def test_minimal_simulation():
    sim = SimulationBase()
    linreg, x, y, y_noise, parameters = linear_model()

    obs = xr.DataArray(y_noise, coords={"x": x}).to_dataset(name="y")
    sim.observations = obs
    
    from pymob.sim.solvetools import solve_analytic_1d
    sim.model = linreg
    sim.solver = solve_analytic_1d

    sim.config.model_parameters.a = Param(value=10, free=False)
    sim.config.model_parameters.b = Param(value=3, free=True , prior="normal(loc=0,scale=10)") # type:ignore
    sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict

    sim.dispatch_constructor()
    evaluator = sim.dispatch(theta={"b":3})
    evaluator()
    evaluator.results

    np.testing.assert_allclose(evaluator.results.y.values, y * 3 + 10)

    # this tests automatic updating of the parameterize method with partial
    sim.config.model_parameters.a = Param(value=0, free=False)
    sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict
    evaluator = sim.dispatch(theta={"b":3})
    evaluator()
    evaluator.results

    np.testing.assert_allclose(evaluator.results.y.values, y * 3)

    sim.config.model_parameters.sigma_y = Param(free=True , prior="lognorm(scale=1,s=1)") # type:ignore
    sim.config.error_model.y = "normal(loc=y,scale=sigma_y)"

    sim.set_inferer("numpyro")
    sim.inferer.config.inference_numpyro.kernel = "nuts"
    # sim.inferer.config.inference_pyabc.min_eps_diff = 0.001
    sim.inferer.run()
    b = float(sim.inferer.idata.posterior["b"].mean()) # type: ignore
    sigma_y = float(sim.inferer.idata.posterior["sigma_y"].mean()) # type: ignore

    # test that the _model parameters of the Simulation remain unchanged. This is
    # achieved throgh deepcopying the dictionary on setting partial
    assert sim._model_parameters["parameters"]["b"] == 3
    np.testing.assert_allclose(b, parameters["b"], atol=0.05, rtol=0.05)
    np.testing.assert_allclose(sigma_y, parameters["sigma_y"], atol=0.05, rtol=0.05)

def test_input_parsing():
    sim = SimulationBase()
    sim.config.data_structure.A = DataVariable(dimensions=["x", "y"], observed=False)

    sim.config.simulation.y0 = ["X=0"]
    
    try:
        sim.parse_input(input="y0")
        threw_error = False
    except KeyError:
        threw_error = True

    assert threw_error        


    sim.config.data_structure.X = DataVariable(dimensions=["x"])
    test_coordinates = {"x": np.linspace(1, 10, 5), "y": np.array([1, 2])}
    sim.coordinates = test_coordinates
    
    y0 = sim.parse_input(input="y0", drop_dims=[])
    np.testing.assert_equal(y0.X.values, np.zeros(sim.dimension_sizes["x"]))

    y0 = sim.parse_input(input="y0", drop_dims=["x"])
    assert float(y0.X.values) == 0.0


    sim.config.data_structure.X = DataVariable(dimensions=["y", "x"])
    y0 = sim.parse_input(input="y0", drop_dims=[])
    np.testing.assert_equal(y0.X.values, np.zeros((2, 5)))


    sim.config.data_structure.X = DataVariable(dimensions=["x", "y"])
    y0 = sim.parse_input(input="y0", drop_dims=[])
    np.testing.assert_equal(y0.X.values, np.zeros((5, 2)))

    # test if broadcasting is done correctly
    sim.config.data_structure.X = DataVariable(dimensions=["x", "y"])
    sim.config.simulation.y0 = ["X=Array([0, 1])"]
    y0 = sim.parse_input(input="y0", drop_dims=[])
    np.testing.assert_equal(y0.X.isel(y=0), np.zeros((5)))
    np.testing.assert_equal(y0.X.isel(y=1), np.ones((5)))

    # test if broadcasting is done correctly
    sim.config.data_structure.X = DataVariable(dimensions=["y", "x"])
    sim.config.simulation.y0 = ["X=Array([0, 1, 2, 3,4])"]
    y0 = sim.parse_input(input="y0", drop_dims=[])
    np.testing.assert_equal(y0.X.isel(y=0), np.arange(5))
    np.testing.assert_equal(y0.X.isel(y=1), np.arange(5))

    # test if broadcasting is done correctly
    sim.config.data_structure.X = DataVariable(dimensions=["y", "x"])
    sim.config.simulation.y0 = ["X=Array([[0,1,2,3,4],[1,2,3,4,5]])"]
    y0 = sim.parse_input(input="y0", drop_dims=[])
    np.testing.assert_equal(y0.X.isel(y=0), np.arange(5))
    np.testing.assert_equal(y0.X.isel(y=1), np.arange(1, 6))

    # test if broadcasting is done correctly
    sim.config.data_structure.X = DataVariable(dimensions=["y", "x"])
    sim.config.simulation.y0 = ["X=Array([[0,1,2,3,4],[1,2,3,4,5]])"]
    y0 = sim.parse_input(input="y0", drop_dims=[])
    np.testing.assert_equal(y0.X.isel(y=0), np.arange(5))
    np.testing.assert_equal(y0.X.isel(y=1), np.arange(1, 6))

    # test if broadcasting is done correctly
    sim.config.data_structure.X = DataVariable(dimensions=["y", "x"])
    sim.config.simulation.y0 = ["X=Array([0, 1])"]
    y0 = sim.parse_input(input="y0", drop_dims=["x"])
    np.testing.assert_equal(y0.X.values, np.array([0,1]))

    sim.config.data_structure.X = DataVariable(dimensions=["y", "x"])
    sim.config.simulation.y0 = ["X=0"]
    y0 = sim.parse_input(input="y0", drop_dims=["y"])
    np.testing.assert_equal(y0.X.values, np.array([0,0,0,0,0]))

    sim.config.data_structure.X = DataVariable(dimensions=["y", "x"])
    sim.config.simulation.y0 = ["X=0"]
    y0 = sim.parse_input(input="y0", drop_dims=["x","y"])
    assert float(y0.X.values) == 0.0

    X = xr.DataArray(
        data=np.random.normal(size=(5,2)), 
        coords=sim.coordinates
    ).to_dataset(name="X")

    # test addition of observations when data variables had been specified
    # before
    sim.config.data_structure.X = DataVariable(dimensions=["y", "x"])
    sim.observations = X
    assert all([np.all(sc == tc) for (sk,sc), (tk,tc) in 
                zip(sim.coordinates.items(), test_coordinates.items())])

    sim.config.data_structure.remove("X")
    sim.observations = X

    sim.config.simulation.y0 = ["X=X"]
    y0 = sim.parse_input(input="y0", reference_data=X.isel(y=0), drop_dims=["y"])
    np.testing.assert_equal(y0.X.values, X.isel(y=0).X.values)
    
    sim.config.simulation.y0 = ["X=2*X"]
    y0 = sim.parse_input(input="y0", reference_data=X.isel(y=0), drop_dims=["y"])
    np.testing.assert_equal(y0.X.values, X.isel(y=0).X.values * 2)

    sim.config.simulation.y0 = ["X=exp(X)"]
    y0 = sim.parse_input(input="y0", reference_data=X.isel(y=0, x=0), drop_dims=["y", "x"])
    np.testing.assert_equal(y0.X.values, np.exp(X.isel(y=0, x=0).X.values))

def test_indexing_simulation():
    pytest.skip()

def test_no_error_from_repeated_setup():
    sim = init_simulation_casestudy_api()  # already executes setup
    sim.setup()


def test_commandline_api_simulate():
    from pymob.simulate import main
    runner = CliRunner()
    
    args = "--case_study=lotka_volterra_case_study "\
        "--scenario=test_scenario"
    result = runner.invoke(main, args.split(" "))

    if result.exception is not None:
        raise result.exception


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
