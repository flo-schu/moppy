import os
import time
import numpy as np
import xarray as xr

from pymob.sim.config import Param, DataVariable
from pymob.solvers import JaxSolver, SolverBase
from pymob.solvers.base import rect_interpolation
from tests.fixtures import (
    init_simulation_casestudy_api, 
    init_lotkavolterra_simulation_replicated,
    init_bufferguts_casestudy, 
    setup_solver
)

from pymob import SimulationBase

def test_solver_preprocessing():
    sim = init_simulation_casestudy_api()
    sim.config.model_parameters.gamma = Param(value=0.3)
    sim.config.model_parameters.delta = Param(value=0.01)
    
    # test SolverBase
    solver = setup_solver(sim, solver=SolverBase)
    y0 = sim.parse_input("y0",drop_dims=["time"])
    y0_solver = solver.preprocess_y_0(sim.validate_model_input(y0))
    np.testing.assert_equal([y.shape for y in y0_solver], [(1,1), (1,1)])

    # test jax solver
    solver = setup_solver(sim, solver=JaxSolver)
    y0 = sim.parse_input("y0",drop_dims=["time"])
    y0_solver = solver.preprocess_y_0(sim.validate_model_input(y0))
    np.testing.assert_equal([y.shape for y in y0_solver], [(1,1), (1,1)])

def test_solver_preprocessing_replicated():
    sim = init_lotkavolterra_simulation_replicated()

    # test SolverBase
    solver = setup_solver(sim, solver=SolverBase)
    y0 = sim.parse_input("y0",drop_dims=["time"])
    y0_solver = solver.preprocess_y_0(sim.validate_model_input(y0))
    np.testing.assert_equal([y.shape for y in y0_solver], [(2,1), (2,1)])
    np.testing.assert_equal(y0_solver, [np.array([[9],[5]]), np.array([[40], [50]])])

    sim.config.simulation.batch_dimension = "idx" # this is not existing
    solver = setup_solver(sim, solver=JaxSolver)
    y0 = sim.parse_input("y0",drop_dims=["time"])
    y0_solver = solver.preprocess_y_0(sim.validate_model_input(y0))
    np.testing.assert_equal([y.shape for y in y0_solver], [(1,2), (1,2)])
    np.testing.assert_equal(y0_solver, [np.array([[9, 5]]), np.array([[40, 50]])])
    
    sim.config.simulation.batch_dimension = "id" # This exists!
    solver = setup_solver(sim, solver=JaxSolver)
    y0 = sim.parse_input("y0",drop_dims=["time"])
    y0_solver = solver.preprocess_y_0(sim.validate_model_input(y0))
    np.testing.assert_equal([y.shape for y in y0_solver], [(2,1), (2,1)])
    np.testing.assert_equal(y0_solver, [np.array([[9],[5]]), np.array([[40], [50]])])
    
    # test parameter processing
    theta = sim.model_parameter_dict
    theta_solver_ode, theta_solver_pp = solver.preprocess_parameters(theta)
    np.testing.assert_equal(
        [t.shape for t in theta_solver_ode], 
        [(2,1), (2,1), (2,1), (2,1)]
    )


def test_solver_preprocessing_complex_parameters():
    sim = init_lotkavolterra_simulation_replicated()

    # step 1 of mischief ;) add a parameter with more than 1 dimension
    # without declaring these coordinates. By using the same number
    # of parameters as the batch_dimension no error is thrown.
    # This is an expected behavior, but will issue a warning if declared
    # without dimensions
    sim.config.model_parameters.gamma.value = [0,1]
    sim.config.model_parameters.gamma.dims = ("id",)

    # test SolverBase
    solver = setup_solver(sim, solver=SolverBase)

    # test parameter processing
    theta = sim.model_parameter_dict
    theta_solver_ode, theta_solver_pp = solver.preprocess_parameters(theta)

    # step 2 of mischief ;) this is not allowed and should raise an exception
    sim.config.model_parameters.gamma.value = [0,1,2]
    sim.config.model_parameters.gamma.dims = ()

    # test parameter processing
    solver = setup_solver(sim, solver=SolverBase)
    theta = sim.model_parameter_dict

    try:
        theta_solver_ode, theta_solver_pp = solver.preprocess_parameters(theta)
        raise AssertionError(
            "This behavior should throw an exception. Multidimensional"+
            "Array values were specified without explicitly defining dimensions"
        )
    except ValueError:
        pass
    
    # step 3: fixing mischief conducted
    # there should be a quick fix for this
    # TODO: Add a dummy coordinate for the parameter dimension.
    sim.config.model_parameters.gamma.dims = ("test_dim",)
    
    try:
        sim.dimension_coords
        raise AssertionError(
            "This behavior should throw an exception. Parameter dimension"+
            "was specified without explicitly defining coordinates for that "+
            "dimension."
        )
    except KeyError:
        pass
    

    # step 4: try to fix but should not work because dimension size does not match 
    # value shape
    sim.coordinates["test_dim"] = ["a", "b"]

    sim.dimension_coords
    sim.dimension_sizes

    solver = setup_solver(sim, solver=SolverBase)
    theta = sim.model_parameter_dict

    try:
        theta_solver_ode, theta_solver_pp = solver.preprocess_parameters(theta)
        raise AssertionError(
            "This behavior should throw an exception. Multidimensional"+
            "Array values were specified without explicitly defining dimensions"
        )
    except ValueError:
        pass
    
    
    # step 5: Correct the coordinate shape
    sim.coordinates["test_dim"] = ["a", "b", "c"]

    solver = setup_solver(sim, solver=SolverBase)
    theta = sim.model_parameter_dict
    theta_solver_ode, theta_solver_pp = solver.preprocess_parameters(theta)

    np.testing.assert_equal(
        [t.shape for t in theta_solver_ode], 
        [(2,1), (2,1), (2,3), (2,1)]
    )

def test_solver_dimensional_order():
    sim = init_lotkavolterra_simulation_replicated()
    theta = sim.model_parameter_dict
    sim.solver = JaxSolver

    sim.data_structure_and_dimensionality
    sim.dispatch_constructor()
    e = sim.dispatch(theta)
    e()
    res_id_time = e.results

    # reorder dimensions of the data variables to see if they can be processed
    sim.config.data_structure.wolves.dimensions = ["time", "id"] # type: ignore
    sim.config.data_structure.rabbits.dimensions = ["time", "id"] # type: ignore

    sim.dispatch_constructor()
    e = sim.dispatch(theta)
    e()
    res_time_id = e.results

    np.testing.assert_equal(
        (res_id_time.to_array() - res_time_id.to_array()).values, 0
    )




def test_benchmark_time():
    sim = init_simulation_casestudy_api()

    cpu_time_start = time.process_time()
    sim.benchmark(n=100)
    cpu_time_stop = time.process_time()

    t = cpu_time_stop - cpu_time_start

    if t > 2:
        raise AssertionError(f"Benchmarking took too long: {t}s. Expected t < 2s")


def test_benchmark_jaxsolver():
    sim = init_simulation_casestudy_api()

    # dispatch is constructed in `init_simulation_case_study`
    e = sim.dispatch({})
    e()
    a = e.results

    # construct the dispatch again with a different solver
    sim.solver = JaxSolver
    from diffrax import Dopri5
    sim.dispatch_constructor(diffrax_solver=Dopri5, rtol=1e-6)
    e = sim.dispatch({})
    e()
    b = e.results

    np.testing.assert_allclose(a.to_array(), b.to_array(), atol=1e-3)

    cpu_time_start = time.process_time()
    sim.benchmark(n=100)
    cpu_time_stop = time.process_time()

    t = cpu_time_stop - cpu_time_start

    if t > 1:
        raise AssertionError(f"Benchmarking took too long: {t}s. Expected t < 1s")


def test_rect_interpolation():
    sim: SimulationBase
    sim = init_bufferguts_casestudy(scenario="testing")
    sim.use_jax_solver() # type: ignore

    # x input is defined on the interval [0,179]
    x_in = sim.parse_input(input="x_in", reference_data=sim.observations, drop_dims=[])
    # rect_interpolation adds duplicates the last y_179 for x_180
    x_in = rect_interpolation(x_in=x_in, x_dim="time")
    sim.model_parameters["x_in"] = x_in

    # Interpolations in diffrax now jump each discontinuity until the last time
    # that is evaluated by the solver. This is not jumped, so that it can be 
    # retrieved by SaveAt
    # https://github.com/patrick-kidger/diffrax/issues/58
    # It seems like this is the intended behavior of diffrax, 

    # run the simulation until exactly the last time, which is a discontinuity
    # this works and will not return a discontinuity, because jump_ts in the
    # PIDController of diffrax, is told that it shoud jump all ts that are 
    # smaller than x_stop
    sim.coordinates["time"] = np.linspace(0, sim.t_max - 1, 1000) #type: ignore
    sim.dispatch_constructor(max_steps=1e5, throw_exception=True, pcoeff=0.0, icoeff=0.25)
    e = sim.dispatch(theta={})
    e()

    # assert that the interpolation produces no infinity values 
    np.testing.assert_array_equal(
        (e.results == np.inf).sum().to_array().values, 
        np.array([0, 0, 0, 0, 0])
    )

    # test if the simulation also works with the normal time vector
    sim.reset_coordinate(dim="time")
    sim.dispatch_constructor(max_steps=1e5, throw_exception=True, pcoeff=0.0, icoeff=0.25)
    e = sim.dispatch(theta={})
    e()

    # assert that the interpolation produces no infinity values 
    np.testing.assert_array_equal(
        (e.results == np.inf).sum().to_array().values, 
        np.array([0, 0, 0, 0, 0])
    )

    # run the simulaton until the added interpolation provided by rect_interpolation
    # until t=180
    sim.coordinates["time"] = np.linspace(0, sim.t_max, 1000) # type: ignore
    sim.dispatch_constructor(max_steps=1e5, throw_exception=True, pcoeff=0.0, icoeff=0.25)
    e = sim.dispatch(theta={})
    e()

    # assert that the interpolation works without problems if the last specified
    # value is not a discontinuity
    np.testing.assert_array_equal(
        (e.results == np.inf).sum().to_array().values, 
        np.array([0, 0, 0, 0, 0])
    )

    # run the simulaton until the added interpolation provided by rect_interpolation
    # until t=500
    # This behavior is correctly caught, by pymob, before it can ocurr, because
    # an interpolation over the provided ts, and ys is not possible. And would
    # result in a difficult to diagnose max_steps error
    try:
        sim.coordinates["time"] = np.linspace(0, sim.t_max + 0.01, 1000) # type: ignore
        sim.dispatch_constructor(max_steps=1e5, throw_exception=True, pcoeff=0.0, icoeff=0.25)
        threw_error = False
    except AssertionError:
        threw_error = True

    if not threw_error: 
        AssertionError(
            "The solver should fail if interpolation is attempted "
            "to be done further than the intended range."        
        )

def test_no_interpolation():
    sim: SimulationBase
    sim = init_bufferguts_casestudy(scenario="testing")
    sim.use_jax_solver() # type: ignore
    
    # x input is defined on the interval [0,179]
    x_in = sim.parse_input(input="x_in", reference_data=sim.observations, drop_dims=[])
    sim.model_parameters["x_in"] = x_in

    # run the simulaton until the provided x_input until t=179
    sim.coordinates["time"] = np.linspace(0, sim.t_max - 1, 1000) # type: ignore

    # without interpolation, the time needs to be 
    sim.dispatch_constructor(max_steps=1e6, throw_exception=True, pcoeff=0.0, icoeff=0.25)
    e = sim.dispatch(theta={})
    e()

    # assert that the interpolation works without problems if the last specified
    # value is not a discontinuity
    np.testing.assert_array_equal(
        (e.results == np.inf).sum().to_array().values, 
        np.array([0, 0, 0, 0, 0])
    )


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    # test_solver_dimensional_order()
    # test_solver_preprocessing_replicated()
    # test_benchmark_jaxsolver()