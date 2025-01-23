from typing import Literal
import numpy as np
import xarray as xr
import pytest

from pymob.solvers.diffrax import JaxSolver
from pymob.sim.config import Config, DataVariable, Modelparameters
from pymob.sim.parameters import Param, RandomVariable, Expression, OptionRV
from pymob.simulation import SimulationBase
from pymob.utils.store_file import prepare_casestudy
from pymob.examples import linear_model

rng = np.random.default_rng(1)

def init_simulation_casestudy_api(scenario="test_scenario"):
    config = prepare_casestudy(
        case_study=("lotka_volterra_case_study", scenario),
        config_file="settings.cfg",
        pkg_dir="case_studies"
    )
    
    from case_studies.lotka_volterra_case_study.sim import Simulation
    sim = Simulation(config=config)
    sim.config.import_casestudy_modules(reset_path=True)
    sim.setup()
    return sim

def init_case_study_and_scenario(case_study, scenario, package="case_studies") -> SimulationBase:
    """Generic utility to import and setup a case study and scenario"""
    config = Config(
        f"{package}/{case_study}/scenarios/{scenario}/settings.cfg"
    )
    config.case_study.package = package
    config.import_casestudy_modules(reset_path=True)
    Simulation = config.import_simulation_from_case_study()
    
    sim = Simulation(config)
    sim.setup()

    return sim

def init_lotkavolterra_simulation_replicated():
    sim = init_simulation_casestudy_api()
    sim.config.case_study.scenario = "test_replicated"
    sim._observations = xr.Dataset()

    sim.config.simulation.batch_dimension = "id"

    sim.config.model_parameters.gamma = Param(value=0.3)
    sim.config.model_parameters.delta = Param(value=0.01)

    sim.config.data_structure.wolves = DataVariable(
        dimensions=["id", "time"], observed=False, dimensions_evaluator=["id", "time"])
    sim.config.data_structure.rabbits = DataVariable(
        dimensions=["id", "time"], observed=False, dimensions_evaluator=["id", "time"]
    )
    
    sim.coordinates["id"] = [0,1]
    sim.coordinates["time"] = [0,1,2,3,4,5]
    
    sim.config.simulation.y0 = ["wolves=Array([9, 5])", "rabbits=Array([40, 50])"]
    y0 = sim.parse_input("y0",drop_dims=["time"])
    sim.model_parameters["y0"] = y0
    
    sim.config.create_directory("results", force=True)
    sim.config.create_directory("scenario", force=True)
    sim.config.save(force=True)
    
    return sim


def init_bufferguts_casestudy(scenario="testing"):
    """This is an external case study used for local testing. The test study
    will eventually added also to the remote as an example, but until this happens
    the test is skipped on the remote.
    """
    config = Config()
    config.case_study.name = "bufferguts"
    config.case_study.scenario = scenario
    config.case_study.package = "../pollinator-tktd/case_studies"
    config.case_study.simulation = "BuffergutsSimulation"
    config.import_casestudy_modules(reset_path=True)

    if "sim" in config._modules:       
        Simulation = config.import_simulation_from_case_study()
        sim = Simulation(config)
        return sim
    else:
        pytest.skip()

def init_bufferguts_leo_casestudy(scenario="testing"):
    """This is an external case study used for local testing. The test study
    will eventually added also to the remote as an example, but until this happens
    the test is skipped on the remote.
    """
    config = Config()
    config.case_study.name = "bufferguts"
    config.case_study.scenario = scenario
    config.case_study.package = "../pollinERA/case_studies"
    config.case_study.simulation = "BuffergutsSimulation"
    config.import_casestudy_modules(reset_path=True)

    if "sim" in config._modules:       
        Simulation = config.import_simulation_from_case_study()
        sim = Simulation(config)
        return sim
    else:
        pytest.skip()

def init_guts_casestudy_constant_exposure(scenario="testing_guts_constant_exposure"):
    """This is an external case study used for local testing. The test study
    will eventually added also to the remote as an example, but until this happens
    the test is skipped on the remote.
    """
    config = Config()
    config.case_study.name = "bufferguts"
    config.case_study.scenario = scenario
    config.case_study.package = "../pollinator-tktd/case_studies"
    config.case_study.simulation = "GutsSimulationConstantExposure"
    config.import_casestudy_modules(reset_path=True)

    if "sim" in config._modules:       
        Simulation = config.import_simulation_from_case_study()
        sim = Simulation(config)
        return sim
    else:
        pytest.skip()


def init_guts_casestudy_variable_exposure(scenario="testing_guts_variable_exposure"):
    """This is an external case study used for local testing. The test study
    will eventually added also to the remote as an example, but until this happens
    the test is skipped on the remote.
    """
    config = Config()
    config.case_study.name = "bufferguts"
    config.case_study.scenario = scenario
    config.case_study.package = "../pollinator-tktd/case_studies"
    config.case_study.simulation = "GutsSimulationVariableExposure"
    config.import_casestudy_modules(reset_path=True)

    if "sim" in config._modules:       
        Simulation = config.import_simulation_from_case_study()
        sim = Simulation(config)
        return sim
    else:
        pytest.skip()


def setup_solver(sim: SimulationBase, solver: type):
    sim.solver = solver
    sim.dispatch_constructor()
    return sim.evaluator._solver


def create_composite_priors():
    prior_mu = RandomVariable(distribution="normal", parameters={"loc": Expression("5"), "scale": Expression("2.0")})
    prior_k = RandomVariable(distribution="normal", parameters={"loc": Expression("mu"), "scale": Expression("1.0")})
    prior_k = RandomVariable(distribution="normal", parameters={"loc": Expression("mu + [5,5]"), "scale": Expression("1.0")})

    mu = Param(prior=prior_mu, hyper=True, dims=("experiment",))
    k = Param(value=np.array([5,23.1]), prior=prior_k, dims=("experiment",))

    theta = Modelparameters() # type: ignore
    theta.mu = mu
    theta.k = k

    return theta


def create_composite_priors_wrong_order():
    prior_mu = RandomVariable(distribution="normal", parameters={"loc": Expression("5"), "scale": Expression("2.0")})
    prior_k = RandomVariable(distribution="normal", parameters={"loc": Expression("mu"), "scale": Expression("1.0")})

    k = Param(prior=prior_k)
    mu = Param(prior=prior_mu, hyper=True, dims=("experiment",))

    theta = Modelparameters() # type: ignore
    theta.k = k
    theta.mu = mu

    return theta

def init_lotka_volterra_case_study_hierarchical_from_script() -> SimulationBase:
    config = Config()
    config.case_study.name = "lotka_volterra_case_study"
    config.case_study.scenario = "test_hierarchical"
    config.case_study.simulation = "HierarchicalSimulation"
    config.import_casestudy_modules(reset_path=True)
    Simulation = config.import_simulation_from_case_study()
    
    sim = Simulation(config)
    sim.initialize_from_script()
    sim.config.save(force=True)

    return sim

def init_lotka_volterra_case_study_hierarchical_from_settings(
    scenario: Literal[
        "test_hierarchical",
        "test_hierarchical_presimulated",
        "lotka_volterra_hierarchical_presimulated_v1",
    ] = "test_hierarchical_presimulated"
) -> SimulationBase:
    config = Config(f"case_studies/lotka_volterra_case_study/scenarios/{scenario}/settings.cfg")
    config.case_study.package = "case_studies"
    config.import_casestudy_modules(reset_path=True)
    Simulation = config.import_simulation_from_case_study()
    
    sim = Simulation(config)
    sim.setup()

    return sim

def create_simulation_for_test_numpyro_behavior():
    config = Config("case_studies/lotka_volterra_case_study/scenarios/test_scenario/settings.cfg")
    config.case_study.name = "lotka_volterra_case_study"
    config.case_study.scenario = "test_scenario"
    config.import_casestudy_modules(reset_path=True)
    sim = config.import_simulation_from_case_study()
    sim = sim(config)
    sim.setup()

    sim.solver = JaxSolver
    sim.config.case_study.simulation = "Simulation_v2"
    sim.config.jaxsolver.throw_exception = False
    sim.config.inference_numpyro.kernel = "nuts"
    sim.config.inference_numpyro.init_strategy = "init_to_median"
    sim.config.inference_numpyro.gaussian_base_distribution = False
    sim.config.inference_numpyro.user_defined_probability_model = "lotka_volterra"
    sim.config.inference_numpyro.user_defined_preprocessing = "dummy_preprocessing"

    sim.config.error_model.wolves = "norm(loc=0,scale=1,obs=(obs-wolves)/jnp.sqrt(wolves+1e-6),obs_inv=res*jnp.sqrt(wolves+1e-06)+wolves)"
    sim.config.error_model.rabbits = "norm(loc=0,scale=1,obs=(obs-rabbits)/jnp.sqrt(rabbits+1e-6),obs_inv=res*jnp.sqrt(rabbits+1e-06)+rabbits)"

    sim.config.model_parameters.gamma = Param(value=0.3, free=False)
    sim.config.model_parameters.delta = Param(value=0.01, free=False)

    sim.config.simulation.input_files = []
    sim.config.case_study.scenario = "test_numpyro_behavior"
    sim.config.create_directory("scenario", force=True)
    sim.config.save(force=True)
    
