import pytest
from click.testing import CliRunner
import tempfile
from pymob.simulation import SimulationBase, Config
from pymob.sim.config import DataVariable, Datastructure, configure
from pymob.sim.parameters import Param, RandomVariable, Expression
from pymob.utils.store_file import import_package
from pymob.solvers.scipy import solve_ivp_1d
from sympy import Function
import xarray as xr
import numpy as np
import os

scenario = "case_studies/lotka_volterra_case_study/scenarios/test_scenario_scripting_api"

def test_simulation():
    sim = SimulationBase()
    sim.config.case_study.name = "lotka_volterra_case_study"
    sim.config.case_study.scenario = "test_scenario_scripting_api"
    sim.config.case_study.observations = "simulated_data.nc"
    sim.config.case_study.data_path = None

    # load data before specifying dims
    sim.config.case_study.data = os.path.abspath("case_studies/lotka_volterra_case_study/data")
    sim.observations = xr.load_dataset(sim.config.input_file_paths[0])    

    # try wrong specification
    threw_error = None
    try:
        sim.config.data_structure.rabbits = DataVariable(dimensions=["hash"])
        sim.config.data_structure.wolves = DataVariable(dimensions=["time"])
        sim.observations = xr.load_dataset(sim.config.input_file_paths[0])
        threw_error = False
    except KeyError:
        threw_error = True

    assert threw_error

    sim.validate()

    sim.config.data_structure.rabbits = DataVariable(dimensions=["time"])
    sim.config.data_structure.wolves = DataVariable(dimensions=["time"])
    
    # load data by providing an absolute path
    sim.config.case_study.data = os.path.abspath("case_studies/lotka_volterra_case_study/data")
    sim.observations = xr.load_dataset(sim.config.input_file_paths[0])    

    # load data by providing a relative path
    sim.config.case_study.data = "case_studies/lotka_volterra_case_study/data"
    sim.observations = xr.load_dataset(sim.config.input_file_paths[0])    
    
    # load data by providing no path (the default 'data' directory in the case study)
    sim.config.case_study.data = None
    sim.observations = xr.load_dataset(sim.config.input_file_paths[0])    

    sim.config.case_study.output = None

    # resetting the path is important in case other case studies were imported
    # in the same testing session
    sim.config.import_casestudy_modules(reset_path=True)
    sim.config.simulation.model = "lotka_volterra"

    sim.solver = solve_ivp_1d
    sim.setup()
    sim.config.save(
        fp=f"{scenario}/test_settings.cfg",
        force=True, 
    )
    sim.save_observations(filename=sim.config.case_study.observations, force=True)

def test_load_generated_settings():
    sim = SimulationBase(f"{scenario}/test_settings.cfg")
    assert sim.config.case_study.name == "lotka_volterra_case_study"
    assert sim.config.case_study.scenario == "test_scenario_scripting_api"
    assert sim.config.case_study.package == "case_studies"
    assert sim.config.case_study.data == None
    assert sim.config.case_study.data_path == "case_studies/lotka_volterra_case_study/data"
    assert sim.config.case_study.output == None
    assert sim.config.case_study.output_path == \
        "case_studies/lotka_volterra_case_study/results/test_scenario_scripting_api"

def test_load_interpolated_settings():
    sim = SimulationBase(f"{scenario}/interp_settings.cfg")
    expected_output = \
        "./case_studies/lotka_volterra_case_study/results/test_scenario_scripting_api"
    assert sim.config.case_study.output == expected_output



def test_standalone_casestudy():
    wd = os.getcwd()
    case_study_name = "lotka_volterra_case_study_standalone"
    root = os.path.join(str(tempfile.tempdir), case_study_name)
    os.mkdir(root)
    os.chdir(root)
    
    # this is the syntax for setting up a standalone case study
    # currently root cannot be set with the config backend, but needs
    # to be specified with `chdir`
    sim = SimulationBase()
    sim.config.case_study.name = "."
    sim.config.case_study.scenario = "test_scenario_standalone"
    sim.config.case_study.package = "."

    os.makedirs(sim.config.case_study.output_path)
    os.makedirs(sim.config.case_study.data_path)
    os.makedirs(sim.config.case_study.scenario_path)
    sim.config.save(force=True)

    # test if all files exist and remove test directory
    os.chdir(wd)
    file_structure = [
        f"{tempfile.tempdir}/lotka_volterra_case_study_standalone",
        f"{tempfile.tempdir}/lotka_volterra_case_study_standalone/data",
        f"{tempfile.tempdir}/lotka_volterra_case_study_standalone/results",
        f"{tempfile.tempdir}/lotka_volterra_case_study_standalone/results/test_scenario_standalone",
        f"{tempfile.tempdir}/lotka_volterra_case_study_standalone/scenarios",
        f"{tempfile.tempdir}/lotka_volterra_case_study_standalone/scenarios/test_scenario_standalone",
        f"{tempfile.tempdir}/lotka_volterra_case_study_standalone/scenarios/test_scenario_standalone/settings.cfg",
    ]
    
    for p in reversed(file_structure):
        assert os.path.exists(p)
        if os.path.isdir(p):
            os.rmdir(p)
        else:
            os.remove(p)

def test_parameter_parsing():
    config = Config()

    io = "value=1.0 dims=[] min=0.0 max=3.0 hyper=False free=True"

    # test scripting input
    test = Param(value=1.0, min=0.0, max=3.0)
    config.model_parameters.test = test

    # test dict input
    config.model_parameters.test = test.model_dump(exclude_none=True)
    assert config.model_parameters.test == test # type: ignore

    # test validation
    config.model_parameters.test = io
    assert config.model_parameters.test == test # type: ignore

    # test serialization
    serialized = config.model_parameters.model_dump(mode="json")
    assert serialized == {"test": io}


def test_parameter_array():
    config = Config()

    io = "value=[1.0,2.0,3.0] dims=['test_dim'] hyper=False free=True"

    # test scripting input
    test = Param(value=np.array([1.0,2.0,3.0]))
    config.model_parameters.test = test

    # test dict input
    config.model_parameters.test = test.model_dump(exclude_none=True)
    assert config.model_parameters.test == test # type: ignore

    # test config file input
    config.model_parameters.test = io
    assert config.model_parameters.test == test  # type: ignore
    
    # test serialization
    serialized = config.model_parameters.model_dump(mode="json")
    assert serialized == {"test": io}

def test_prior():
    config = Config()

    io = "lognorm(scale=[1.0,1.0,a],s=1.5)"

    test_prior = RandomVariable(
        distribution="lognorm", 
        parameters={"scale": Expression("[1.0,1.0,a]"), "s": Expression("1.5")},
    )

    # test scripting input
    test_param = Param(value=np.array([1.0,2.0,3.0]))
    test_param.prior = test_prior

    # test dict input
    test_param.prior = test_prior.model_dump(exclude_none=True)
    assert test_param.prior == test_prior # type: ignore

    # test config file input
    test_param.prior = io
    assert test_param.prior == test_prior  # type: ignore
    
    # test serialization
    config.model_parameters.test = test_param
    serialized = test_prior.model_dump(mode="json")
    config.model_parameters.model_dump(mode="json")
    assert serialized == io

def test_parameter_array_with_prior():
    config = Config()

    io = "value=[1.0,2.0,3.0] dims=[] prior=lognorm(scale=[1.0,1.0,1.0],s=1.0) hyper=False free=True"
    test = Param(value=[1.0,2.0,3.0], prior="lognorm(scale=[1.0,1.0,1.0],s=1.0)") # type:ignore

    # test config file input
    config.model_parameters.test = io
    assert config.model_parameters.test == test  # type: ignore
    
    # test scripting input
    config.model_parameters.test = test

    # test dict input
    config.model_parameters.test = test.model_dump(exclude_none=True)
    assert config.model_parameters.test == test # type: ignore

    # test serialization
    serialized = config.model_parameters.model_dump(mode="json")
    assert serialized == {"test": io}



def test_model_parameters():
    config = Config()

    a = Param(value=1)
    b = Param(value=5, free=False)

    config.model_parameters.a = a
    config.model_parameters.b = b

    frmp = config.model_parameters.free
    fimp = config.model_parameters.fixed
    almp = config.model_parameters.all

    assert frmp == {"a": a}
    assert fimp == {"b": b}
    assert almp == {"a": a, "b":b}

def test_error_model():
    config = Config()

    io = "norm(loc=1,scale=2)"
    
    # test config file input
    test = RandomVariable(
        distribution="norm",
        parameters=dict(loc=Expression("1"),scale=Expression("2")),  # type:ignore
    )

    # test config file input
    config.error_model.test = io
    assert config.error_model.test == test  # type: ignore
    
    # test scripting input
    config.error_model.test = test

    # test dict input
    config.error_model.test = test.model_dump(exclude_none=True)
    assert config.error_model.test == test # type: ignore

    # test serialization
    serialized = config.error_model.model_dump(mode="json", exclude_none=True)
    assert serialized == {"test": io}

def test_error_model_with_obs():
    config = Config()

    io = "lognorm(scale=[1.0,1.0,1.0],s=1.0,obs=b/jnp.sqrt(2))"
    test = RandomVariable(
        distribution="lognorm",
        parameters=dict(scale=Expression("[1.0,1.0,1.0]"),s=Expression("1.0")),  # type:ignore
        obs=Expression("b/jnp.sqrt(2)")
    )

    # test config file input
    config.error_model.test = io
    assert config.error_model.test == test  # type: ignore
    
    # test scripting input
    config.error_model.test = test

    # test dict input
    config.error_model.test = test.model_dump(exclude_none=True)
    assert config.error_model.test == test # type: ignore

    # test serialization
    serialized = config.error_model.model_dump(mode="json")
    assert serialized == {"test": io}



def test_data_variables():
    config = Config()
    config.case_study.name = "lotka_volterra_case_study"
    config.case_study.scenario = "test_scenario_scripting_api"
    config.data_structure.wolves = DataVariable(dimensions=["time"], min=0)
    assert config.data_structure.data_variables == ["wolves"]
    config.save(force=True)
    
    config.data_structure = {"wolves": dict(dimensions = ["time"])} # type: ignore
    assert config.data_structure.data_variables == ["wolves"]


    config.data_structure.B = DataVariable(dimensions=["a", "b"], dimensions_evaluator=["b","a"])
    assert config.data_structure.dimdict == {"wolves": ["time"], "B": ["a", "b"]}
    assert config.data_structure.var_dim_mapper == {"wolves": [0], "B": [1,0]}



def test_commandline_api_configure():
    runner = CliRunner()
    
    args = [
        "--file=case_studies/lotka_volterra_case_study/scenarios/test_scenario_scripting_api/test_settings.cfg",
        "-o inference.numpyro.kernel=nuts",
        "-o simulation.y0 = wolves=wolves rabbits=rabbits",
        "-o inference.n_predictions=10",
        "-o model-parameters.test_parameter=",
    ]
    result = runner.invoke(configure, args)

    if result.exception is not None:
        raise result.exception


if __name__ == "__main__":
    pass