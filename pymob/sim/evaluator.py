from typing import Callable, Dict, List, Optional, Sequence, Tuple
import inspect
from frozendict import frozendict
from copy import deepcopy
import xarray as xr
import numpy as np
from numpy.typing import NDArray
from pymob.solvers.base import mappar, SolverBase

def create_dataset_from_numpy(Y, Y_names, coordinates):
    DeprecationWarning(
        "This method will be discontinued in future relases. "
        "Use 'create_dataset_from_dict' instead and return a dictionary from "
        "the solver or post processing respectively. The needed variable names "
        "to create the dictionary can be obtained from the data_variables "
        "argument in the solver signature. "
    )
    n_vars = Y.shape[-1]
    n_dims = len(Y.shape)
    assert n_vars == len(Y_names), (
        "The number of datasets must be the same as the specified number"
        "of data variables declared in the `settings.cfg` file."
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

def create_dataset_from_dict(Y: dict, data_structure, coordinates, var_dim_mapper):
    arrays = {}
    for k, v in Y.items():
        dims = data_structure.get(k, tuple(coordinates.keys()))
        coords = {d: coordinates[d] for d in dims}
        dim_order = var_dim_mapper.get(k) # use get which returns None if there is no mapping

        # TODO: it would be an idea to leave the batch dimension in the return 
        #       from the solver and process it here. 1st advantage is that the
        #       name of the batch_dimension could reported in the error message
        #       second advantage is that the output from JAXSolver (which may)
        #       be passed to numpyro, has a batch dimension, which makes it 
        #       useful for numpyro plate notation (and more consistent)
        v_permuted_dims = v.transpose(dim_order)
        try:
            da = xr.DataArray(v_permuted_dims, coords=coords, dims=dims)
        except ValueError as err:
            raise ValueError(
                f"{err} for the data variable '{k}'. This can have multiple causes: "
                f"1) Have you specified a batch dimensions? If you have a "
                f"multi-replicate design, this is typically 'ID', or 'replicate_id' "
                f"or similar. You can set the batch dimension, by setting the "
                f"option config.simulation.batch_dimension = '...' "
                f"2) The dimensional order of the data variables differs from the "
                f"Solver dimensional order. You can try specifying "
                f"evaluator dimensions for '{k}' in order to redorder the dimensions: "
                f"sim.config.data_structure.{k} = DataVariable(..., dimensions_evaluator=[...]). " 
                f"Hint: The solvers usually put the batch dimension (e.g. 'id') first."
            )
        arrays.update({k: da})

    return xr.Dataset(arrays)

class Evaluator:
    """The Evaluator is an instance to evaluate a model. It's purpose is primarily
    to create objects that can be spawned and evaluated in parallel and can 
    individually track the results of a simulation or a parameter inference
    process. If needed the evaluations can be tracked and results can later
    be collected.
    
    Seed may not be set as a property, because this should be something passed
    through
    """
    result: xr.Dataset

    def __init__(
            self,
            model: Callable,
            solver: type|Callable,
            dimensions: Sequence[str],
            dimension_sizes: Dict[str, int],
            parameter_dims: Dict[str, Tuple[str, ...]],
            n_ode_states: int,
            var_dim_mapper: Dict,
            data_structure: Dict,
            data_structure_and_dimensionality: Dict,
            coordinates: Dict[str, NDArray],
            coordinates_input_vars: Dict[str, Dict[str, Dict[str, NDArray]]],
            dims_input_vars: Dict[str, Dict[str, Tuple[str, ...]]],
            coordinates_indices: Dict,
            data_variables: Sequence[str],
            stochastic: bool,
            batch_dimension: str,
            indices: Dict = {},
            post_processing: Optional[Callable] = None,
            solver_options: Dict = {},
            **kwargs
        ) -> None:
        """_summary_

        Parameters
        ----------
        model : Callable
            the ODE model to be solved by the evaluator
        solver : Callable
            a function to solve the ODE model with
        parameters : Dict
            A dictionary of model and post_processing parameters. Do not have
            to be in any particular order
        dimensions : List
            A list of the dimensions of the simulations
        n_ode_states : int
            The number of ODE states tracked
        var_dim_mapper : List
            A list of variables and their associated dimensions. This is relevant
            for simulations, where not all data variables have the same dimensional
            structure
        data_structure : Dict
            Similar to the var_dim_mapper, but additionally contains the coordinates
        coordinates : Dict
            The coordinates of each dimension in a dict
        data_variables : List
            The data variables of the simulation
        stochastic : bool
            Whether the model is a stochastic or a deterministic model
        indices : Optional[Dict], optional
            Indices, which should be used to map potentially nested parameters
            to a flat array for batch processing the simulations, by default {}
        post_processing : Optional[Callable], optional
            A function that takes a dictionary of simulation results and 
            parameters as an input and adds new variables to the results, 
            by default None, meaning that no post processing of the ODE solution
            is performed
        """
        
        self._parameters = frozendict()
        self.dimensions = dimensions
        self.dimension_sizes = dimension_sizes
        self.n_ode_states = n_ode_states
        self.var_dim_mapper = var_dim_mapper
        self.data_structure = data_structure
        self.data_structure_and_dimensionality = data_structure_and_dimensionality
        self.data_variables = data_variables
        self.coordinates = coordinates
        self.coordinates_input_vars = coordinates_input_vars
        self.coordinates_indices = coordinates_indices
        self.is_stochastic = stochastic
        self.indices = indices
        self.batch_dimension = batch_dimension
        self.solver_options = solver_options
        

        

        self.parameter_dims = self._regularize_batch_dimensions(
            arg_names=list(mappar(model, {}, to="names")) + list(mappar(post_processing, {}, to="names")), # type: ignore
            arg_dims=parameter_dims
        )

        self.dims_input_vars = {}
        self.dims_input_vars["y0"] = self._regularize_batch_dimensions(
            arg_names=list(self.coordinates_input_vars["y0"].keys()),
            arg_dims=dims_input_vars["y0"]
        )
        self.dims_input_vars["x_in"] = self._regularize_batch_dimensions(
            arg_names=list(self.coordinates_input_vars["x_in"].keys()),
            arg_dims=dims_input_vars["x_in"]
        )

        # can be initialized
        if post_processing is None:
            self.post_processing = lambda results, time, interpolation: results
        else: 
            self.post_processing = post_processing
                
        # can be initialized
        # set additional arguments of evaluator
        _ = [setattr(self, key, val) for key, val in kwargs.items()]

        self._signature = {}

        if callable(model):
            if hasattr(model, "__func__"):
                self.model = model.__func__
            else:
                self.model = model
        else:
            raise NotImplementedError(
                f"The model {model} must be provided as a callable."
            )

        # can be initialized
        if isinstance(solver, type):
            if issubclass(solver, SolverBase):
                frozen_coordinates_input_vars = frozendict({
                    k_input_var: frozendict({
                        k_datavar: frozendict({
                            k_coord: tuple(v_coord)
                            for k_coord, v_coord in v_datavar.items()
                        }) 
                        for k_datavar, v_datavar in v_input_var.items()
                    }) 
                    for k_input_var, v_input_var in coordinates_input_vars.items()
                })

                frozen_dims_input_vars = frozendict({
                    k_input_var: frozendict({
                        k_datavar: tuple(dims_datavar)
                        for k_datavar, dims_datavar in v_input_var.items()
                    }) 
                    for k_input_var, v_input_var in self.dims_input_vars.items()
                })

                frozen_coordinates_indices = frozendict({
                    k: tuple(v) for k, v in coordinates_indices.items()
                })

                data_structure_dims = frozendict({
                    dv: frozendict({d: lendim for d, lendim in dimdict.items()}) 
                    for dv, dimdict 
                    in self.data_structure_and_dimensionality.items()
                })

                frozen_coordinates = frozendict({
                    k: tuple(v) for k, v in self.coordinates.items()
                })

                solver_extra_options = frozendict({
                    k:v for k, v in kwargs.items() 
                    if k in solver.__match_args__
                })

                solver_options.update(solver_extra_options)
                

                self._solver = solver(
                    model=self.model,
                    post_processing=self.post_processing,
                    
                    coordinates=frozen_coordinates,
                    coordinates_input_vars=frozen_coordinates_input_vars,
                    dims_input_vars=frozen_dims_input_vars,
                    coordinates_indices=frozen_coordinates_indices,
                    dimensions=tuple(self.dimensions),
                    dimension_sizes=frozendict(self.dimension_sizes),
                    parameter_dims=frozendict(self.parameter_dims),
                    data_variables=tuple(self.data_variables),
                    data_structure_and_dimensionality=data_structure_dims,

                    indices=frozendict({k: tuple(v.values) for k, v in self.indices.items()}),
                    n_ode_states=self.n_ode_states,
                    is_stochastic=self.is_stochastic,
                    batch_dimension=self.batch_dimension,
                    **solver_options

                )
            else:
                raise NotImplementedError(
                    f"If solver is passed as a class of type {type(solver)}. "
                    "Must be a subclass of `pymob.solvers.base.SolverBase`. "
                    "Alternatively pass a callable."
                )
        elif callable(solver):
            if hasattr(solver, "__func__"):
                self._solver = solver.__func__
            else:
                self._solver = solver
            self.get_call_signature()

        else:
            raise NotImplementedError(
                f"Solver {solver} is neither a subclass of "
                "`pymob.solvers.base.SolverBase` nor a callable."
            )

    def _regularize_batch_dimensions(
        self, 
        arg_names: List[str], 
        arg_dims: Dict[str, Tuple[str, ...]]
    ) -> Dict[str, Tuple[str, ...]]:
        _param_dims = {}
        for par_name, par_dims in arg_dims.items():
            if par_name in arg_names:
                if self.batch_dimension in par_dims:
                    if par_dims[0] != self.batch_dimension:
                        raise ValueError(
                            f"If the batch dimension '{self.batch_dimension}' is "+
                            f"specified in a model parameter it must always be "+
                            f"the 0th dimension ('{self.batch_dimension}', ...). "+
                            f"For parameter '{par_name}' you have provided {par_dims}"
                        )
                    else:
                        # everything okay in this case the dimensional specification
                        # is good
                        pass
                    
                else:
                    # add the batch dimension at index 0. If no or other
                    # dimensions have been specified but not the batch dimension
                    par_dims = (self.batch_dimension, *par_dims)
            
            else:
                # if the parameter is not part of the model args, there
                # is no need to do anything
                pass
            
            _param_dims.update({par_name: par_dims})
        return _param_dims

    # can be initialized
    def get_call_signature(self):
        if isinstance(self._solver, SolverBase):
            signature = inspect.signature(self._solver.solve)
        elif inspect.isfunction(self._solver) or inspect.ismethod(self._solver):
            signature = inspect.signature(self._solver)
        else:
            raise TypeError(f"{self._solver} must be SolverBase class or a function")
        
        model_args = [a for a in signature.parameters.keys() if a != "parameters"]

        for a in model_args:
            if a not in self.allowed_model_signature_arguments:
                raise ValueError(
                    f"'{a}' in model signature is not an attribute of the Evaluator. "
                    f"Use one of {self.allowed_model_signature_arguments}, "
                    f"or set as evaluator_kwargs in the call to "
                    "'SimulationBase.dispatch'" 
                )
            
            # add argument to signature for call to model
            if a != "seed":
                self._signature.update({a: getattr(self, a)})
        
    
    @property
    def allowed_model_signature_arguments(self):
        return [a for a in self.__dict__.keys() if a[0] != "_"] + ["seed"]

    def __call__(self, seed=None):
        if seed is not None:
            self._signature.update({"seed": seed})

        if isinstance(self._solver, SolverBase):
            Y_ = self._solver(**self.parameters)

        else:
            Y_ = self._solver(parameters=self.parameters, **self._signature)
        
        # TODO: Consider which elements may be abstracted from the solve methods 
        # implemented in mod.py below is an unsuccessful approach
        # params = self._signature["parameters"]["parameters"]
        # time = self._signature["coordinates"]["time"]
        
        # s_dim, s_idx = self._signature["indices"]["substance"]
        # pp_args = mappar(self.post_processing, params, exclude=["t", "results"])
        # pp_args = [np.array(a, ndmin=1)[s_idx] for a in pp_args]
        # Y_ = self.post_processing(Y_, time, *pp_args)
        self.Y = Y_

    @property
    def dimensionality(self):
        return {key: len(values) for key, values in self.coordinates.items()}

    @property
    def parameters(self) -> frozendict:
        return self._parameters
    
    @parameters.setter
    def parameters(self, value: Dict):
        if len(self._parameters) == 0:
            self._parameters = frozendict(value)
        elif value == self._parameters:
            pass
        else:
            raise ValueError(
                "It is unsafe to change the parameters of an evaluator "
                "After it has been created. Use 'sim.dispatch(theta=...)' "
                "to create a new evaluator and initialize it with a new set "
                "of parameters."
                "If you really need to do it, use evaluator._parameters to "
                "overwrite the parameters on your own risk."
            )



    @property
    def results(self):
        if isinstance(self.Y, dict):
            dataset = create_dataset_from_dict(
                Y=self.Y, 
                coordinates=self.coordinates,
                data_structure=self.data_structure,
                var_dim_mapper=self.var_dim_mapper
            )
        elif isinstance(self.Y, np.ndarray):
            dataset = create_dataset_from_numpy(
                Y=self.Y,
                Y_names=self.data_variables,
                coordinates=self.coordinates,
            )
        else:
            raise NotImplementedError(
                "Results returned by the solver must be of type Dict or np.ndarray."
            )
        
        # assign the coordinates from the indices to the dataset if available
        dataset = dataset.assign_coords({
            f"{idx}_index": data_array 
            for idx, data_array in self.indices.items()
        })
        return dataset
    
    def spawn(self):
        return deepcopy(self)