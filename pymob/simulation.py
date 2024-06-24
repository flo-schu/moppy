import os
import inspect
import warnings
from typing import Optional, List, Union, Literal
from types import ModuleType
import configparser
from functools import partial
import multiprocessing as mp
from typing import Callable, Dict
from multiprocessing.pool import ThreadPool, Pool
import re

import numpy as np
import xarray as xr
import dpath as dp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from toopy import param, benchmark

from pymob.utils.config import lambdify_expression, lookup_args
from pymob.utils.errors import errormsg, import_optional_dependency
from pymob.utils.store_file import scenario_file, parse_config_section
from pymob.sim.evaluator import Evaluator, create_dataset_from_dict, create_dataset_from_numpy
from pymob.sim.base import stack_variables
from pymob.sim.config import Config

config_deprecation = "Direct access of config options will be deprecated. Use `Simulation.config.OPTION` API instead"

def is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


def flatten_parameter_dict(model_parameter_dict, exclude_params=[]):
    """Takes a dictionary of key value pairs where the values may be
    floats or arrays. It flattens the arrays and adds indexes to the keys.
    In addition a function is returned that back-transforms the flattened
    parameters."""
    parameters = model_parameter_dict

    flat_params = {}
    empty_map = {}
    for par, value in parameters.items():
        if par in exclude_params:
            continue
        if is_iterable(value):
            empty_map.update({par: np.full_like(value, fill_value=np.nan)})
            for i, subvalue in enumerate(value):
                subpar = f"{par}___{i}"
                flat_params.update({subpar: subvalue})

        else:
            flat_params.update({par: value})

        if par not in empty_map:
            empty_map.update({par: np.nan})


    def reverse_mapper(parameters):
        param_dict = empty_map.copy()

        for subpar, value in parameters.items():
            subpar_list = subpar.split("___")

            if len(subpar_list) > 1:
                par, par_index = subpar_list
                param_dict[par][int(par_index)] = value
            elif len(subpar_list) == 1:
                par, = subpar_list
                param_dict[par] = value

        return param_dict
    
    return flat_params, reverse_mapper


def update_parameters_dict(config, x, parnames):
    for par, val, in zip(parnames, x):
        key_exist = dp.set(config, glob=par, value=val, separator=".")
        if key_exist != 1:
            raise KeyError(
                f"prior parameter name: {par} was not found in config. " + 
                f"make sure parameter name was spelled correctly"
            )
    return config

def get_return_arguments(func):
    ode_model_source = inspect.getsource(func)
    
    # extracts last return statement of source
    return_statement = ode_model_source.split("\n")[-2]

    # extract arguments returned by ode_func
    return_args = return_statement.split("return")[1]

    # strip whitespace and separate by comma
    return_args = return_args.replace(" ", "").split(",")

    return return_args

class SimulationBase:
    model: Callable
    solver: Callable
    mod: ModuleType
    prob: ModuleType
    mdata: ModuleType
    mplot: ModuleType

    def __init__(
        self, 
        config: Optional[Union[str,configparser.ConfigParser]] = None, 
    ) -> None:
        
        self.config = Config(config=config)
        self._observations: xr.Dataset = xr.Dataset()
        self._coordinates: Dict = {}
        self.var_dim_mapper: Dict[str, List[str]] = {}
        self.free_model_parameters: List = []

        self.model_parameters: Dict = {}
        # self.observations = None
        self._objective_names: List = []
        self.indices: Dict = {}

        # seed gloabal RNG
        self._seed_buffer_size: int = self.config.multiprocessing.n_cores * 2
        self.RNG = np.random.default_rng(self.config.simulation.seed)
        self._random_integers = self.create_random_integers(n=self._seed_buffer_size)
     

        # simulation
        # self.setup()
        
    def setup(self):
        """Simulation setup routine, when the following methods have been 
        defined:
        
        coords = self.set_coordinates(input=self.input_file_paths)
        self.coordinates = self.create_coordinates(coordinate_data=coords)
        self.var_dim_mapper = self.create_dim_index()
        init-methods
        ------------

        self.initialize --> may be replaced by self.set_observations

        """

        self.initialize(input=self.config.input_file_paths)
        self.var_dim_mapper = self.create_dim_index()
        
        # coords = self.set_coordinates(input=self.config.input_file_paths)
        # self.coordinates = self.create_coordinates(coordinate_data=coords)
        self.free_model_parameters  = self.set_free_model_parameters()

        output_dir = self.config.case_study.output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        scenario_dir = self.config.case_study.scenario_path
        if not os.path.exists(scenario_dir):
            os.makedirs(scenario_dir)
            print(f"Created directory: {scenario_dir}")

        # TODO: set up logger
        self.parameterize = partial(self.parameterize, model_parameters=self.model_parameters)
        self.config.simulation.n_ode_states = self.infer_ode_states()

    @property
    def observations(self):
        assert isinstance(self._observations, xr.Dataset), "Observations must be an xr.Dataset"
        return self._observations

    @observations.setter
    def observations(self, value):
        self._observations = value
        self.create_data_scaler()
        self.coordinates = self.set_coordinates(input=self.config.input_file_paths)
        

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        self._coordinates = self.create_coordinates(coordinate_data=value)


    def __repr__(self) -> str:
        return (
            f"Simulation(case_study={self.config.case_study.name}, "
            f"scenario={self.config.case_study.scenario})"
        )

    def load_functions(self):
        _model = self.config.simulation.model
        if _model is not None:
            self.model = getattr(self.mod, _model)

        _solver = self.config.simulation.solver
        if _solver is not None:
            self.solver = getattr(self.mod, _solver)

    def set_coordinates(self, input):
        dimensions = self.config.simulation.dimensions
        return [self.observations[dim].values for dim in dimensions]

    def benchmark(self, n=100, **kwargs):
        evaluator = self.dispatch(theta=self.model_parameter_dict, **kwargs)
        evaluator(seed=1) 

        @benchmark
        def run_bench():
            for i in range(n):
                evaluator(seed=self.RNG.integers(100))
        
        print(f"\nBenchmarking with {n} evaluations")
        print(f"=================================")
        run_bench()
        print(f"=================================\n")
        
    def infer_ode_states(self) -> int:
        if self.config.simulation.n_ode_states == -1:
            try: 
                return_args = get_return_arguments(self.model)
                n_ode_states = len(return_args)
                warnings.warn(
                    "The number of ODE states was not specified in "
                    "the config file [simulation] > 'n_ode_states = <n>'. "
                    f"Extracted the return arguments {return_args} from the "
                    "source code. "
                    f"Setting 'n_ode_states={n_ode_states}."
                )
            except:
                warnings.warn(
                    "The number of ODE states was not specified in "
                    "the config file [simulation] > 'n_ode_states = <n>' "
                    "and could not be extracted from the return arguments."
                )
                n_ode_states = -1
        else:
            n_ode_states = self.config.simulation.n_ode_states

        return n_ode_states
        
    def dispatch(self, theta, **evaluator_kwargs):
        """Dispatch an evaluator, which will compute the model at parameters
        (theta). Evaluators are advantageous, because they are easier serialized
        than the whole simulation object. Comparison can then happen back in 
        the simulation.

        Theoretically, this could also be used to constrain coordinates etc, 
        before evaluating.  
        """
        model_parameters = self.parameterize(theta)
        
        # TODO: make sure the evaluator has all arguments required for solving
        # model
        # TODO: Check if model is bound. If yes extract
        if hasattr(self.solver, "__func__"):
            solver = self.solver.__func__
        else:
            solver = self.solver

        if hasattr(self.model, "__func__"):
            model = self.model.__func__
        else:
            model = self.model
        
        if self.solver_post_processing is not None:
            # TODO: Handle similar to solver and model
            post_processing = getattr(self.mod, self.solver_post_processing)
        else:
            post_processing = None

        stochastic = self.config.simulation.modeltype
            
        evaluator = Evaluator(
            model=model,
            solver=solver,
            parameters=model_parameters,
            dimensions=self.dimensions,
            n_ode_states=self.config.simulation.n_ode_states,
            var_dim_mapper=self.var_dim_mapper,
            data_structure=self.data_structure,
            data_variables=self.data_variables,
            coordinates=self.coordinates,
            # TODO: pass the whole simulation settings section
            stochastic=True if stochastic == "stochastic" else False,
            indices=self.indices,
            post_processing=post_processing,
            **evaluator_kwargs
        )

        return evaluator

    def parse_input(self, data=None, input=Literal["y0", "x_in"], drop_dims=["time"]):
        """Parses a config string e.g. y=Array([0]) or a=b to a numpy array 
        and looks up symbols in the elements of data, where data items are
        key:value pairs of a dictionary, xarray items or anything of this form

        The values are broadcasted along the remaining dimensions in the obser-
        vations that have not been dropped. Input refers to the argument in
        the config file. 

        This method is useful to prepare y0s from observations or to broadcast
        starting values along batch dimensions.
        """
        # parse dims and coords
        input_dims = {k:v for k, v in self.observations.dims.items() if k not in drop_dims}
        input_coords = {k:v for k, v in self.observations.coords.items() if k in input_dims}
        
        if input == "y0":
            input_list = self.config.simulation.y0
        elif input == "x_in":
            input_list = self.config.simulation.x_in
        else:
            raise NotImplementedError(f"Input type {input}: is not implemented")

        input_dataset = xr.Dataset()
        for input_expression in input_list:
            key, expr = input_expression.split("=")
            
            func, args = lambdify_expression(expr)

            kwargs = lookup_args(args, data)

            value = func(**kwargs)

            if not isinstance(value, xr.DataArray):
                value.shape != tuple(input_dims.values())
                value = np.broadcast_to(value, tuple(input_dims.values()))
                value = xr.DataArray(value, coords=input_coords)

            else:
                value = xr.DataArray(value.values, coords=input_coords)

            input_dataset[key] = value


        return input_dataset


    def reshape_observations(self, observations, reduce_dim):
        """This method reduces the dimensionality of the observations. 
        Compiling xarray datasets from multiple experiments with different 
        IDs and different endpoints, lead to blown up datasets where 
        all combinations (even though they were not tested) are filled with
        NaNs. Reducing such artificial dimensions by flattening the arrays
        is the aim of this method. 

        TODO: There should be tests, whether the method is applicable (this
        may be already caught with the assertion)

        TODO: The method should be generally applicable
        """

        raise NotImplementedError(
            "reshape_observations is an experimental method. "
            "Using this method may have unexpected results."
        )
        
        # currently the method is still based on the damage-proxy project
        substances = observations.attrs[reduce_dim]

        stacked_obs = stack_variables(
            ds=observations.copy(),
            variables=["cext_nom", "cext", "cint"],
            new_coordinates=substances,
            new_dim="substance",
            pattern=lambda var, coord: f"{var}_{coord}"
        ).transpose(*self.dimensions, reduce_dim)

        # reduce cext_obs
        cext_nom = stacked_obs["cext_nom"]
        assert np.all(
            (cext_nom == 0).sum(dim=reduce_dim) == len(substances) - 1
        ), "There are mixture treatments in the SingleSubstanceSim."


        # VECTORIZED INDEXING IS THE KEY
        # https://docs.xarray.dev/en/stable/user-guide/indexing.html#vectorized-indexing

        # Defining a function to reduce the "reduce_dim" dimension to length 1
        def index(array, axis=None, **kwargs):
            # Check that there is exactly one non-zero value in 'substance'
            non_zero_count = (array != 0).sum(axis=axis)
            
            if not (non_zero_count == 1).all():
                raise ValueError(f"Invalid '{reduce_dim}' dimension. It should have exactly one non-zero value.")
            
            return np.where(array != 0)[axis]


        # Applying the reduction function using groupby and reduce
        red_data = stacked_obs.cext_nom
        new_dims = [d for d in red_data.dims if d != reduce_dim]

        reduce_dim_idx = red_data\
            .groupby(*new_dims)\
            .reduce(index, dim=reduce_dim)\
            .rename(f"{reduce_dim}_index")
        
        if stacked_obs.dims[reduce_dim] == 1:
            reduce_dim_idx = reduce_dim_idx.squeeze()
        
        reduce_dim_id_mapping = stacked_obs[reduce_dim]\
            .isel({reduce_dim: reduce_dim_idx})\
            .drop(reduce_dim)\
            .rename(f"{reduce_dim}_id_mapping")
        
        reduce_dim_idx = reduce_dim_idx.assign_coords({
            f"{reduce_dim}": reduce_dim_id_mapping
        })

        # this works because XARRAY is amazing :)
        stacked_obs["cext_nom"] = stacked_obs["cext_nom"].sel({reduce_dim: reduce_dim_id_mapping})
        stacked_obs["cext"] = stacked_obs["cext"].sel({reduce_dim: reduce_dim_id_mapping})
        stacked_obs["cint"] = stacked_obs["cint"].sel({reduce_dim: reduce_dim_id_mapping})
        
        # drop old dimension and add dimension as inexed dimension
        # this is necessary, as the reduced dimension needs to disappear from
        # the coordinates.
        stacked_obs = stacked_obs.drop_dims(reduce_dim)
        stacked_obs = stacked_obs.assign_coords({
            f"{reduce_dim}": reduce_dim_id_mapping,
            f"{reduce_dim}_index": reduce_dim_idx,
        })
        
        indices = {
            reduce_dim: reduce_dim_idx
        }

        return stacked_obs, indices


    def evaluate(self, theta):
        """Wrapper around run to modify paramters of the model.
        """
        self.model_parameters = self.parameterize(theta)
        return self.run()
    
    def compute(self):
        """
        A wrapper around run, which catches errors, logs, does post processing
        """
        warnings.warn("Discouraged to use self.Y constructs. Instability suspected.", DeprecationWarning, 2)
        self.Y = self.evaluate(theta=self.model_parameter_dict)

    def interactive(self):
        # optional imports
        extra = "'interactive' dependencies can be installed with pip install pymob[interactive]"
        widgets = import_optional_dependency("ipywidgets", errors="raise", extra=extra)
        if widgets is not None:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        else:
            raise ImportError(f"ipywidgets is not available and needs to be installed")

        def interactive_output(func, controls):
            out = widgets.Output(layout={'border': '1px solid black'})
            def observer(change):
                theta={key:s.value for key, s in sliders.items()}
                widgets.interaction.show_inline_matplotlib_plots()
                with out:
                    clear_output(wait=True)
                    func(theta)
                    widgets.interaction.show_inline_matplotlib_plots()
            for k, slider in controls.items():
                slider.observe(observer, "value")
            widgets.interaction.show_inline_matplotlib_plots()
            observer(None)
            return out

        sliders = {}
        for par in self.free_model_parameters:
            s = widgets.FloatSlider(
                par.value, description=par.name, min=par.min, max=par.max,
                step=par.step
            )
            sliders.update({par.name: s})

        def func(theta):
            extra = self.config.inference.extra_vars
            extra = [extra] if isinstance(extra, str) else extra
            extra_vars = {v: self.observations[v] for v in extra}
            evaluator = self.dispatch(theta=theta, **extra_vars)
            evaluator()
            self.plot(results=evaluator.results)

        out = interactive_output(func=func, controls=sliders)

        display(widgets.HBox([widgets.VBox([s for _, s in sliders.items()]), out]))
    
    def set_inferer(self, backend):
        extra = (
            "set_inferer(backend='{0}') was not executed successfully, because "
            "'{0}' dependencies were not found. They can be installed with "
            "pip install pymob[{0}]. Alternatively:"
        )

        if backend == "pyabc":
            pyabc = import_optional_dependency(
                "pyabc", errors="raise", extra=extra.format("pyabc")
            )
            if pyabc is not None:
                from pymob.inference.pyabc_backend import PyabcBackend
            
            self.inferer = PyabcBackend(simulation=self)

        elif backend == "pymoo":
            pymoo = import_optional_dependency(
                "pymoo", errors="raise", extra=extra.format("pymoo2")
            )
            if pymoo is not None:
                from pymob.inference.pymoo_backend import PymooBackend

            self.inferer = PymooBackend(simulation=self)

        elif backend == "numpyro":
            numpyro = import_optional_dependency(
                "numpyro", errors="raise", extra=extra.format("numpyro")
            )
            if numpyro is not None:
                from pymob.inference.numpyro_backend import NumpyroBackend

            self.inferer = NumpyroBackend(simulation=self)
    
        else:
            raise NotImplementedError(f"Backend: {backend} is not implemented.")

    def check_dimensions(self, dataset):
        """Check if dataset dimensions match the specified dimensions.
        TODO: Name datasets for referencing them in errormessages
        """
        ds_dims = list(dataset.dims.keys())
        in_dims = [k in self.dimensions for k in ds_dims]
        assert all(in_dims), IndexError(
            "Not all dataset dimensions, were not found in specified dimensions. "
            f"Settings(dims={self.dimensions}) != dataset(dims={ds_dims})"
        )
        
    def dataset_to_2Darray(self, dataset: xr.Dataset) -> xr.DataArray: 
        self.check_dimensions(dataset=dataset)
        array_2D = dataset.stack(multiindex=self.config.simulation.dimensions)
        return array_2D.to_array().transpose("multiindex", "variable")

    def array2D_to_dataset(self, dataarray: xr.DataArray) -> xr.Dataset: 
        dataset_2D = dataarray.to_dataset(dim="variable")      
        return dataset_2D.unstack().transpose(*self.config.simulation.dimensions)

    def create_data_scaler(self):
        """Creates a scaler for the data variables of the dataset over all
        remaining dimensions.
        In addition produces a scaled copy of the observations
        """
        # make sure the dataset follows the order of variables specified in
        # the config file. This is important so also in the simulation results
        # the scalers are matched.
        ordered_dataset = self.observations[self.config.simulation.data_variables]
        obs_2D_array = self.dataset_to_2Darray(dataset=ordered_dataset)
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        
        # add bounds to array of observations and fit scaler
        lower_bounds = np.array(self.config.simulation.data_variables_min)
        upper_bounds = np.array(self.config.simulation.data_variables_max)
        stacked_array = np.row_stack([lower_bounds, upper_bounds, obs_2D_array])
        scaler.fit(stacked_array)

        self.scaler = scaler
        self.print_scaling_info()

        scaled_obs = self.scale_(self.observations)
        self.observations_scaled = scaled_obs

    def print_scaling_info(self):
        scaler = type(self.scaler).__name__
        for i, var in enumerate(self.config.simulation.data_variables):
            print(
                f"{scaler}(variable={var}, "
                f"min={self.scaler.data_min_[i]}, max={self.scaler.data_max_[i]})"
            )

    def scale_(self, dataset: xr.Dataset):
        ordered_dataset = dataset[self.config.simulation.data_variables]
        data_2D_array = self.dataset_to_2Darray(dataset=ordered_dataset)
        obs_2D_array_scaled = data_2D_array.copy() 
        obs_2D_array_scaled.values = self.scaler.transform(data_2D_array) # type: ignore
        return self.array2D_to_dataset(obs_2D_array_scaled)

    @property
    def results(self):
        warnings.warn("Discouraged to use results property.", DeprecationWarning, 2)
        return self.create_dataset_from_numpy(
            Y=self.Y, 
            Y_names=self.config.simulation.data_variables, 
            coordinates=self.coordinates
        )

    def results_to_df(self, results):
        if isinstance(results, xr.Dataset):
            return results
        elif isinstance(results, dict):
            return create_dataset_from_dict(
                Y=results, 
                coordinates=self.coordinates,
                data_structure=self.data_structure,
            )
        elif isinstance(results, np.ndarray):
            return create_dataset_from_numpy(
                Y=results,
                Y_names=self.config.simulation.data_variables,
                coordinates=self.coordinates,
            )
        else:
            raise NotImplementedError(
                "Results returned by the solver must be of type Dict or np.ndarray."
            )
    

    @property
    def results_scaled(self):
        scaled_results = self.scale_(self.results)
        # self.check_scaled_results_feasibility(scaled_results)
        return scaled_results

    def scale_results(self, Y):
        ds = self.create_dataset_from_numpy(
            Y=Y, 
            Y_names=self.config.simulation.data_variables, 
            coordinates=self.coordinates
        )
        return self.scale_(ds)

    def check_scaled_results_feasibility(self, scaled_results):
        """Parameter inference or optimization over many variables can only succeed
        in reasonable time if the results that should be compared are on approximately
        equal scales. The Simulation class, automatically estimates the scales
        of result variables, when observations are provided. 

        Problems can occurr when observations are on very narrow ranges, but the 
        simulation results can take much larger or lower values for that variable.
        As a result the inference procedure will almost exlusively focus on the
        optimization of this variable, because it provides the maximal return.

        The function warns the user, if simulation results largely deviate from 
        the scaled minima or maxima of the observations. In this case manual 
        minima and maxima should be given
        """
        max_scaled = scaled_results.max()
        min_scaled = scaled_results.min()
        if isinstance(self.scaler, MinMaxScaler):
            for varkey, varval in max_scaled.variables.items():
                if varval > 2:
                    warnings.warn(
                        f"Scaled results for '{varkey}' are {float(varval.values)} "
                        "above the ideal maximum of 1. "
                        "You should specify explicit bounds for the results variable."
                    )

            for varkey, varval in min_scaled.variables.items():
                if varval < -1:
                    warnings.warn(
                        f"Scaled results for '{varkey}' are {float(varval.values)} "
                        "below the ideal minimum of 0. "
                        "You should specify explicit bounds for the results variable."
                    )

    def validate(self):
        # TODO: run checks if the simulation was set up correctly
        #       - do observation dimensions match the model output (run a mini
        #         simulation with reduced coordinates to verify)
        #       -
        if len(self.config.simulation.data_variables) == 0:
            raise RuntimeError(
                "No data_variables were specified. "
                "Specify like sim.config.simulation.data_variables = ['a', 'b'] "
                "Or in the simulation section of the config file. "
                "Data variables track the state variables of the simulation. "
                "If you want to do inference, they must match the variables of "
                "the observations."
            )

                    
        if len(self.config.simulation.dimensions) == 0:
            raise RuntimeError(
                "No dimensions of the simulation were specified. "
                "Which observations are you expecting? "
                "'time' or 'id' are reasonable choices. But it all depends on "
                "your data. Dimensions must match your data if you want to do "
                "Parameter inference."
            )

    @staticmethod
    def parameterize(free_parameters: list[param.Param], model_parameters) -> dict:
        """
        Optional. Set parameters and initial values of the model. 
        Must return a dictionary with the keys 'y0' and 'parameters'
        
        Can be used to define parameters directly in the script or from a 
        parameter file.

        Arguments
        ---------

        input: List[str] file paths of parameter/input files
        theta: List[Param] a list of Parameters. By default the parameters
            specified in the settings.cfg are used in this list. 

        returns
        -------

        tulpe: tuple of parameters, can have any length.
        """
        parameters = model_parameters["parameters"]
        y0 = model_parameters["y0"]

        parameters.update({p.name: p.value for p in free_parameters})
        return {"y0": y0, "parameters": parameters} 

    def run(self):
        """
        Implementation of the forward simulation of the model. Needs to return
        X and Y

        returns
        -------

        X: np.ndarray | xr.DataArray
        Y: np.ndarray | xr.DataArray
        """
        raise NotImplementedError
    
    def objective_function(self, results, **kwargs):
        func = getattr(self, self.config.inference.objective_function)
        obj = func(results, **kwargs)

        if obj.ndim == 0:
            obj_value = float(obj)
            obj_name = "objective"
        elif obj.ndim == 1:
            obj_value = obj.values
            obj_name = list(obj.coords["variable"].values)
        else:
            raise ValueError("Objectives should be at most 1-dimensional.")

        if len(self._objective_names) == 0:
            self._objective_names = obj_name

        return obj_name, obj_value

    def total_average(self, results):
        """objective function returning the total MSE of the entire dataset"""
        
        diff = (self.scale_(self.results_to_df(results)) - self.observations_scaled).to_array()
        return (diff ** 2).mean()

    def prior(self):
        raise NotImplementedError

    def initialize(self, input):
        """
        initializes the simulation. Performs any extra work, not done in 
        parameterize or set_coordinates. 
        """
        pass
    
    def dump(self):
        pass
        
    
    def plot(self, results):
        pass

    def create_coordinates(self, coordinate_data):
        if not isinstance(coordinate_data, (list, tuple)):
            coordinate_data = (coordinate_data, )

        assert len(self.config.simulation.dimensions) == len(coordinate_data), errormsg(
            f"""number of dimensions, specified in the configuration file
            must match the coordinate data (X) returned by the `run` method.
            """
        )

        coord_zipper = zip(self.config.simulation.dimensions, coordinate_data)
        coords = {dim: x_i for dim, x_i in coord_zipper}
        return coords

    @staticmethod
    def create_dataset_from_numpy(Y, Y_names, coordinates):
        warnings.warn(
            "Use `create_dataset_from_numpy` defined in sim.evaluator",
            category=DeprecationWarning
        )
        n_vars = Y.shape[-1]
        n_dims = len(Y.shape)
        assert n_vars == len(Y_names), errormsg(
            """The number of datasets must be the same as the specified number
            of data variables declared in the `settings.cfg` file.
            """
        )

        # transpose Y to put the variable dimension first, then add the
        # remaining dimensions in order
        Y_transposed = Y.transpose((n_dims - 1, *range(n_dims - 1)))

        data_arrays = []
        for y, y_name in zip(Y_transposed, Y_names):
            da = xr.DataArray(y, coords=coordinates, name=y_name)
            data_arrays.append(da)

        dataset = xr.merge(data_arrays)

        return dataset

    @staticmethod
    def option_as_list(opt):
        # TODO: Remove when all methods have been updated to the new config API
        if not isinstance(opt, (list, tuple)):
            opt_list = [opt]
        else:
            opt_list = opt

        return opt_list

    @property
    def input_file_paths(self):
        # TODO: Remove when all method has been updated to the new config API
        return self.config.input_file_paths

    # config as properties
    @property
    def dimensions(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.simulation.dimensions

    @property
    def data_variables(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.simulation.data_variables

    @property
    def n_ode_states(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.simulation.n_ode_states
    
    @n_ode_states.setter
    def n_ode_states(self, n_ode_state):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        self.config.simulation.n_ode_states = n_ode_state

    @property
    def solver_post_processing(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.simulation.solver_post_processing

    @property
    def input_files(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.simulation.input_files
  
    @property
    def case_study_path(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.case_study.package

    @property
    def root_path(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.case_study.root

    @property
    def case_study(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.case_study.name

    @property
    def scenario(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.case_study.scenario

    @property
    def scenario_path(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.case_study.scenario_path

    # TODO Outsource model parameters also to config (if it makes sense)
    @property
    def model_parameter_values(self):
        return [p.value for p in self.free_model_parameters]
    
    @property
    def model_parameter_names(self):
        return [p.name for p in self.free_model_parameters]
    
    @property
    def n_free_parameters(self):
        return len(self.free_model_parameters)

    @property
    def model_parameter_dict(self):
        return {p.name:p.value for p in self.free_model_parameters}


    @property
    def output_path(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.case_study.output_path

    @property
    def data_path(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.case_study.data_path
       

    @property
    def data_variable_bounds(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        lower_bounds = self.config.simulation.data_variables_min
        upper_bounds = self.config.simulation.data_variables_max
        return lower_bounds, upper_bounds

    @property
    def objective(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.inference.objective_function

    @property
    def n_objectives(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.inference.n_objectives

    @property
    def objective_names(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.inference.objective_names

    @property
    def n_cores(self):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        return self.config.multiprocessing.n_cores
    
    @n_cores.setter
    def n_cores(self, value):
        # TODO: Remove when all method has been updated to the new config API
        warnings.warn(config_deprecation, DeprecationWarning)
        self.config.multiprocessing.cores = value

    def create_random_integers(self, n: int):
        return self.RNG.integers(low=0, high=int(1e18), size=n).tolist()
        
    def refill_consumed_seeds(self):
        n_seeds_left = len(self._random_integers)
        if n_seeds_left == self.config.multiprocessing.n_cores:
            n_new_seeds = self._seed_buffer_size - n_seeds_left
            new_seeds = self.create_random_integers(n=n_new_seeds)
            self._random_integers.extend(new_seeds)
            print(f"Appended {n_new_seeds} new seeds to sim.")
        
    def draw_seed(self):
        # return None       
        # the collowing has no multiprocessing stability when the simulation is
        # serialized directly
        self.refill_consumed_seeds()
        seed = self._random_integers.pop(0)
        return seed

    def set_free_model_parameters(self):
        if self.config._config.has_section("model-parameters"):
            warnings.warn(
                "config section 'model-parameters' is deprecated, "
                "use 'free-model-parameters' and 'fixed-model-parameters'", 
                DeprecationWarning
            )
            params = self.config.model_parameters.model_dump()
        elif self.config._config.has_section("free-model-parameters"):
            params = self.config.model_parameters.model_dump()
        else:
            warnings.warn("No parameters were specified.")
            params = {}
        
        # create a nested dictionary from model parameters
        parameter_dict = {}
        for par_key, par_value in params.items():
            dp.new(parameter_dict, par_key, par_value, separator=".")

        parse = lambda x: None if x is None else float(x)

        # create Param instances
        parameters = []
        for param_name, param_dict in parameter_dict.items():
            if isinstance(value, (int, float)):
                p = param.FloatParam(
                    value=parse(param_dict.get("value")),
                    name=param_name,
                    min=parse(param_dict.get("min")),
                    max=parse(param_dict.get("max")),
                    step=parse(param_dict.get("step")),
                    prior=param_dict.get("prior", None)
                )
            else:
                # check for array notation
                pattern = r"(\d+(\.\d+)?(\s+\d+(\.\d+)?)*|\s*)"
                if re.fullmatch(pattern, value):
                    value = np.array([float(v) for v in value.split(" ")])
                    p = param.ArrayParam(
                        value=value,
                        name=param_name,
                        min=param_dict.get("min", None),
                        max=param_dict.get("max", None),
                        step=param_dict.get("step", None),
                        prior=param_dict.get("prior", None)
                    )
                else:
                    raise NotImplementedError(
                        f"Parameter specification '{value}' cannot be parsed."
                    )
            parameters.append(p)

        return parameters

    @property
    def fixed_model_parameters(self):
        fixed_parameters = {}
        params = parse_config_section(self.config._config["fixed-model-parameters"])
        for k, v in params.items():
            vlist = v.split(" ")
            floatlist = [float(v) for v in vlist]
            if len(vlist) == 1:
                v_ = floatlist[0]

            else: 
                v_ = np.array(floatlist)

            fixed_parameters.update({k: v_})

        return fixed_parameters

    @property
    def error_model(self):
        em = parse_config_section(self.config._config["error-model"], method="strfloat")
        return em

    @property
    def evaluator_dim_order(self):
        return self.config.simulation.evaluator_dim_order

    def create_dim_index(self) -> Dict[str, List[str]]:
        # TODO: If a dimensionality config seciton is implemented this function
        # may become superflous
        sim_dims = self.dimensions
        evaluator_dims = self.evaluator_dim_order
        obs_ordered = self.observations.transpose(*sim_dims)

        var_dim_mapper = {}
        for var in self.data_variables:
            obs_var_dims = obs_ordered[var].dims
            var_dim_mapper.update({
                var: [obs_var_dims.index(e_i) for e_i in evaluator_dims if e_i in obs_var_dims]
            })

        return var_dim_mapper
    
    @property
    def data_structure(self):
        # TODO: If a dimensionality config seciton is implemented this function
        # may become superflous
        obs_ordered = self.observations.transpose(*self.dimensions)

        data_structure = {}
        for var in self.data_variables:
            obs_var_dims = obs_ordered[var].dims
            data_structure.update({
                var: list(obs_var_dims)
            })

        return data_structure

    def reorder_dims(self, Y):
        results = {}
        for var, mapper in self.var_dim_mapper.items():
            results.update({
                var: Y[var][np.array(mapper)]
            })
    
        return results
