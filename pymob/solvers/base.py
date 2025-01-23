import numpy
import numpy as np
from numpy.typing import ArrayLike
from types import ModuleType
import xarray as xr
from typing import (
    Callable, Dict, List, Optional, Sequence, Literal, Tuple, Union
)
from frozendict import frozendict
from dataclasses import dataclass, field
import inspect
from scipy.ndimage import gaussian_filter1d
from diffrax import rectilinear_interpolation

@dataclass(frozen=True)
class SolverBase:
    """
    The idea of creating a solver as a class is that it is easier
    to pass on important arguments of the simulation relevant to the 
    Solver. Therefore a solver can access all attributes of an Evaluator
    """
    model: Callable
    dimensions: Tuple
    dimension_sizes: frozendict[str, int]
    parameter_dims: frozendict[str, Tuple[str, ...]]
    n_ode_states: int
    coordinates: frozendict[str, Tuple] = field(repr=False)
    coordinates_input_vars: frozendict[str, frozendict[str, frozendict[str, Tuple[Union[float,int,str], ...]]]]
    dims_input_vars: frozendict[str, frozendict[str, Tuple[str, ...]]]
    coordinates_indices: frozendict[str, tuple]
    data_variables: Tuple
    data_structure_and_dimensionality: frozendict[str, frozendict[str, int]]
    is_stochastic: bool
    post_processing: Callable
    solver_kwargs: frozendict = frozendict()
    indices: frozendict[str, Tuple] = field(repr=False, default=frozendict())

    x_dim: str = "time"
    batch_dimension: str = "batch_id"
    exclude_kwargs_model: Tuple[str, ...] = ("t", "x_in", "y", "X")
    exclude_kwargs_postprocessing: Tuple[str, ...] = ("t", "time", "interpolation", "results")

    # fields that are computed post_init
    x: Tuple[float] = field(init=False, repr=False)
    shapes_coordinates: Dict[str, int] = field(init=False, repr=False)
    shapes_parameter_coordinates: Dict[str, Tuple[int, ...]] = field(init=False, repr=False)
    shapes_input_vars_coordinates: frozendict[str, frozendict[str, Tuple[int, ...]]] = field(init=False, repr=False)
    len_batch_coordinate: int = field(init=False)
    x_shape_batched: Tuple[int, ...] = field(init=False)

    def __post_init__(self, *args, **kwargs):
        x = self.coordinates[self.x_dim]
        if not np.all(x[:-1] <= x[1:]):
            raise ValueError(
                f"x_dim '{self.x_dim}' must be sorted in ascending order."
            )
        object.__setattr__(self, "x", x)

        coord_dims = self._get_coordinate_unique_shapes()
        object.__setattr__(self, "shapes_coordinates", coord_dims)
        
        parameter_shapes = self._get_parameter_shapes()
        object.__setattr__(self, "shapes_parameter_coordinates", parameter_shapes)

        input_vars_shapes = self._get_input_vars_shapes()
        object.__setattr__(self, "shapes_input_vars_coordinates", input_vars_shapes)

        batch_length = self._get_batch_length()
        object.__setattr__(self, "len_batch_coordinate", batch_length)

        x_shape_batched = self._get_x_shape_batched()
        object.__setattr__(self, "x_shape_batched", x_shape_batched)

        # set extra attributes from solver_kwargs, which are specified through
        # the dispatch_constructor. Those don't receive post-processing

        self.test_matching_batch_dims()
        self.test_x_coordinates()
        

    def __call__(self, **kwargs):
        return self.solve(**kwargs)
    
    def _get_batch_length(self) -> int:
        batch_coordinates = self.coordinates.get(self.batch_dimension, [0])
        return len(batch_coordinates)

    def _get_x_shape_batched(self) -> Tuple[int, ...]:
        n_batch = self.len_batch_coordinate
        n_x = self.dimension_sizes[self.x_dim]
        return (n_batch, n_x)

    def _get_coordinate_unique_shapes(self, ) -> frozendict[str, int]:
        coordinate_shape_dict = {}
        for key, coords in self.coordinates.items():
            n_coords = len(coords)
            coordinate_shape_dict.update({key: n_coords})

        for key, coords in self.coordinates_indices.items():
            unique_coords = set(coords)
            n_coords = len(unique_coords)
            coordinate_shape_dict.update({key: n_coords})

        return frozendict(coordinate_shape_dict)

    def _get_parameter_shapes(self, ) -> frozendict[str, Tuple[int, ...]]:
        par_shape_dict = {}
        for par_name, par_dims in self.parameter_dims.items():
            dim_shape = []
            for d in par_dims:
                if d == self.batch_dimension and d not in self.dimension_sizes:
                    dim_size = 1
                else:
                    try:
                        dim_size = self.dimension_sizes[d]
                    except KeyError as err:
                        raise KeyError(
                            f"KeyError: Dimension '{d}' could not be found in "+
                            f"any dimension specified for the simulation "+
                            f"{self.dimension_sizes}. It must be either "+
                            "specified via the `sim.indices` dict."
                        )
                
                dim_shape.append(dim_size)
            par_shape_dict.update({par_name: tuple(dim_shape)})

        return frozendict(par_shape_dict)

    def _get_input_vars_shapes(self, ) -> frozendict[str, frozendict[str, Tuple[int, ...]]]:
        input_vars_shape_dict = {}
        for input_var_name, input_var_dict in self.dims_input_vars.items():
            data_var_shape_dict = {}
            for data_var_name, data_var_dims in input_var_dict.items():
                dim_shape = []
                for d in data_var_dims:
                    # look for the dimension in the coodinates of the input vars
                    coords_data_var = self.coordinates_input_vars[input_var_name][data_var_name]
                    coords = coords_data_var.get(d, None)
                    
                    # if it is found use its length as a dimension size
                    if coords is not None:
                        dim_size = len(coords)
                    
                    # if it is not found, see if the dim is the batch dimensions
                    # and handle the batch dimension size
                    # this is done, in cases where there is a batch dimension, 
                    # but it was not specified for the input, because it is 
                    # expected that it simply broadcasts to the batch dimension
                    elif d == self.batch_dimension:
                        if d not in self.dimension_sizes:
                            dim_size = 1
                        else:
                            dim_size = self.dimension_sizes[d]

                    # if the dimension is not in the coordinates of the input vars
                    # or not as a batch dimension in the dimension sizes dict
                    # something is severely wrong
                    else:
                        raise KeyError(
                            f"KeyError: Dimension: '{d}' for data_var: "+
                            f"'{data_var_name}' in input var: '{input_var_name}' "+
                            f"could not be found in the coordinate dimensions: "+
                            f"{list(coords_data_var.keys())}. "+
                            f"Make sure `sim.model_parameters['{input_var_name}']` "+
                            "is correctly specified. It is recommended to "+
                            "specify the model_input with `sim.config.simulation."+
                            f"{input_var_name} = ['...', '{data_var_name}=...', '...']` "+
                            f"and use `sim.parse_input(input='{input_var_name}', "+
                            f"reference_data=..., drop_dims=[...])` to parse the "+
                            "input."
                        )
                    
                    dim_shape.append(dim_size)
                data_var_shape_dict.update({data_var_name: tuple(dim_shape)})
            input_vars_shape_dict.update({
                input_var_name: frozendict(data_var_shape_dict)
            })
        return frozendict(input_vars_shape_dict)

    # def _get_input_vars_shapes(self, ) -> frozendict[str, frozendict[str, Tuple[int, ...]]]:
    #     frozen_shapes_input_vars = frozendict({
    #         k_input_var: frozendict({
    #             k_datavar: tuple([
    #                 len(v_coord)
    #                 for _, v_coord in v_datavar.items()
    #             ]) 
    #             for k_datavar, v_datavar in v_input_var.items()
    #         }) 
    #         for k_input_var, v_input_var in self.coordinates_input_vars.items()
    #     })

    #     return frozen_shapes_input_vars

    #     raise NotImplementedError(
    #         "Currently, it is expected that x_in is already in the correct "+
    #         "shape (minus the batch dimension), when passed to the Evaluator. "+
    #         "This is why no definite shapes can be defined on initialization. "+
    #         "The only broadcasting that can be sensibly made is to promote the "+
    #         "input vectors to the batch dimension, if this has not yet been done."
    #     )

    def test_matching_batch_dims(self):
        bc = self.coordinates.get(self.batch_dimension, None)

        if bc is not None:
            matching_batch_coords_if_present = [
                v[self.batch_dimension] == bc 
                for k, v in self.coordinates_input_vars.items() 
                if self.batch_dimension in v
            ]

            if not all(matching_batch_coords_if_present):
                raise IndexError(
                    f"Batch coordinates '{self.batch_dimension}' of input "+
                    "variables do not have the same size "+
                    "as the batch dimension of the observations."
                )

    def test_x_coordinates(self):
        x = self.coordinates[self.x_dim]
        if len(self.coordinates_input_vars["x_in"]) == 0:
            return
        
        x_xin = [
            np.max(v.get(self.x_dim, [0])) 
            for v in self.coordinates_input_vars["x_in"].values()
        ]

        if np.max(x) > np.max(x_xin):
            raise AssertionError(
                f"The {self.x_dim}-coordinate on the observations (sim.coordinates) "+
                f"goes to a higher {self.x_dim} than the {self.x_dim}-coordinate "+
                "of the model_parameters['x_in']. "+
                "Make sure to run the simulation only until the provided x_in "+
                "values, or extend the x_in values until the required time"
            )

    def preprocess_parameters(self, parameters, num_backend: ModuleType=numpy):
        ode_args = mappar(
            self.model, 
            parameters, 
            exclude=self.exclude_kwargs_model, 
            to="dict"
        )
        ode_args_broadcasted = self._broadcast_args(
            arg_dict=frozendict(ode_args), # type: ignore
            num_backend=num_backend
        )
        
        pp_args = mappar(
            self.post_processing, 
            parameters, 
            exclude=self.exclude_kwargs_postprocessing, 
            to="dict"
        )
        pp_args_broadcasted = self._broadcast_args(
            arg_dict=frozendict(pp_args), # type: ignore
            num_backend=num_backend
        )

        return ode_args_broadcasted, pp_args_broadcasted


    def _broadcast_args(self, arg_dict: frozendict[str, numpy.ndarray], num_backend: ModuleType=numpy):
        # simply broadcast the parameters along the batch dimension
        # if there is no other index provided
        n_batch = self.len_batch_coordinate

        args = []
        for arg_name, arg in arg_dict.items():
            # you can expect that any of the expected parameter_shapes have the
            # size of the batch_dimension in the first axis.
            # Note that this is the taget shape without any extended dimension.
            # Dimensions are only extended, if there is no batch dimension already
            # present
            target_shape = self.shapes_parameter_coordinates.get(arg_name, (n_batch, ))

            # make sure the argument is an array with one dimension
            # promoting to two dimension (including a batch dimension will be done)
            # in the different conditional branches
            arg_promoted = num_backend.array(arg, ndmin=1, dtype=float)
            
            # if the size of the 1st dimension of the argument array 
            # is identical to the size of the specified first dimension
            # we know that the input array was constructed to match the size.
            # The length of the batch coordinates is at least 1
            # There are weird cases in which the input arrays are square,
            # and the dimensions of the parameter have been specified in a 
            # different order (other than batch_dimension first)
            # This problem has been fixed, as the dimensional order is now
            # checked in the Evaluator.__init__
            if arg_promoted.shape[0] == target_shape[0]:
                # if the dimensionality of the argument array is 1
                # we add a dummy dimension at the end, in order
                # to harmonize it with arguments that are vectors
                if arg_promoted.ndim == 1:
                    arg_broadcasted = num_backend.expand_dims(arg_promoted, -1)
                else:
                    # if greater zero (zero dim not possible because of the 
                    # promotion to 1D arrays at the beginning of the loop)
                    # then it is assumed that the array correctly contains 
                    # more than one value for each id in the batch dimension
                    # i.e. vector, matrix or nd-array parameters in the ODE
                    # We leave the array as is
                    arg_broadcasted = arg_promoted

            elif (
                # this is when passed and expected shapes have the same number
                # of dimensions
                arg_promoted.shape[0] == 1 
                # this is when the shape of the later dimensions match.
                # I.e. the batch_dimension is not in the passed argument values
                # This happens frequently, when parameters are not broadcasted
                # to the batch dimension before being passed to the solver
                #
                # An example for the syntax below:
                # arg_promoted.shape >>> (2,5) 
                # target_shape >>> (10,2,5)
                # len(arg_promoted.shape) >>> 2
                # target_shape[-2:] >>> (2,5)
                # (2,5) == (2,5) >>> True
                or arg_promoted.shape == target_shape[-len(arg_promoted.shape):]
            ):
                # Note that this will also broadcast parameters across multiple
                # dimension e.g. (5,) -> (10,2,5)
                if arg_promoted.ndim > len(self.shapes_parameter_coordinates[arg_name]):
                    raise ValueError(
                        f"Parameter '{arg_name}' values have shape "+
                        f"{arg_promoted.shape}. This is in conflict with the "+
                        f"specified shape {self.shapes_parameter_coordinates[arg_name]} "+
                        f"from the dimensions {self.parameter_dims[arg_name]}."
                    )
                # the operation will broadcase argument arrays that have the
                # same number of dimensions (and the first dimension is )
                arg_broadcasted = num_backend.broadcast_to(
                    arg_promoted, shape=self.shapes_parameter_coordinates[arg_name]
                )

                # also here we apply broadcasting to the array if the result has 
                # only one dimension
                if arg_broadcasted.ndim == 1:
                    arg_broadcasted = num_backend.expand_dims(arg_broadcasted, -1)
            else:
                raise ValueError(
                    f"The values of parameter '{arg_name}' with the shape "+
                    f"{arg_promoted.shape} could not be broadcasted to the "+
                    f"specified shape {self.shapes_parameter_coordinates[arg_name]} "+
                    f"from the dimensions {self.parameter_dims[arg_name]}. "+
                    f"Make sure you add the missing dimensions in the "+
                    f"parameter specification Param(..., dims=(...,)) "+
                    f"and handle parameter coordinates appropriately. "+
                    "Parameter dimension coordinates can be specified either "+
                    f"1) Add the missing dimension in sim.config.parameters.{arg_name} "+
                    "and sim.coordinates['MISSING_DIM'] = [...]"+
                    "2) Add the missing dimension in "+
                    "sim.config.data_structure.ANY_DATA_VAR and sim.coordinates['MISSING_DIM'] "+
                    "3) dimension in sim.indices "+
                    f"the batch dimension {self.batch_dimension} is added "+
                    f"automatically."
                )
            
            args.append(arg_broadcasted)

        return tuple(args)


    def preprocess_x_in(self, x_in, num_backend:ModuleType=numpy):
        X_in_list = []
        for x_in_var, x_in_vals in x_in.items():
            # parse to array
            x_in_x = num_backend.array(
                self.coordinates_input_vars["x_in"][x_in_var][self.x_dim], 
                dtype=float,
                ndmin=1
            )
            x_in_y = num_backend.array(x_in_vals, ndmin=1, dtype=float)

            # broadcast x
            x_in_batched_shape = (self.len_batch_coordinate, *x_in_x.shape)
            x_in_x_broadcasted = num_backend.broadcast_to(x_in_x, x_in_batched_shape)

            # broadcast y
            y_in_batched_shape = self.shapes_input_vars_coordinates["x_in"][x_in_var]
            x_in_y_broadcasted = num_backend.broadcast_to(x_in_y, y_in_batched_shape) 

            # also here we apply broadcasting to the array if the result has 
            # only one dimension
            if x_in_y_broadcasted.ndim == 1:
                x_in_y_broadcasted = num_backend.expand_dims(
                    x_in_y_broadcasted, -1
                )

            # combine xs and ys to make them ready for interpolation
            X_in = [
                num_backend.array(v) for v in 
                [x_in_x_broadcasted, x_in_y_broadcasted]
            ]

            X_in_list.append(X_in)

        return X_in_list
    
    def preprocess_y_0(self, y0, num_backend:ModuleType=numpy):
        Y0 = []

        for y0_var, y0_vals in y0.items():
            y0_vals_promoted = num_backend.array(y0_vals, ndmin=1, dtype=float)
            
            # wrap y0 data in a dummy batch dim if the batch dim is not
            # included in the coordinates
            y0_var_shape = self.shapes_input_vars_coordinates["y0"][y0_var]
            y0_vals_broadcasted = num_backend.broadcast_to(
                y0_vals_promoted, 
                y0_var_shape
            ) 

            # also here we apply broadcasting to the array if the result has 
            # only one dimension
            if y0_vals_broadcasted.ndim == 1:
                y0_vals_broadcasted = num_backend.expand_dims(
                    y0_vals_broadcasted, -1
                )

            Y0.append(y0_vals_broadcasted)
        return Y0


    def solve(self):
        raise NotImplementedError("Solver must implement a solve method.")

def mappar(
    func, 
    parameters: Dict[str,float|int|List|Tuple], 
    exclude=[], 
    to:Literal["tuple","dict","names"]="tuple"
) -> Tuple|Dict:
    func_signature = inspect.signature(func).parameters.keys()
    model_param_signature = [p for p in func_signature if p not in exclude]
    if to == "tuple":
        model_args = [parameters.get(k) for k in model_param_signature]
        return tuple(model_args)
    elif to == "dict":
        return {k: parameters.get(k) for k in model_param_signature}
    elif to == "names":
        return tuple(model_param_signature)

    raise NotImplementedError(f"'to={to}' is not implemented for 'mappar'")


def jump_interpolation(
        x_in: xr.Dataset, 
        x_dim: str="time", 
        factor: float=0.001, 
        interpolation: Literal["fill-forward", "linear"] = "fill-forward",
    ) -> xr.Dataset:
    """Make the interpolation safe by adding a coordinate just before each 
    x-value (except the first vaue). The distance between the new and the next
    point are calculated as a fraction of the previous distance between
    neighboring points. The corresponding y-values are first set to NaN and then
    interpolated based on the interpolation method.

    Parameters
    ----------
    x_in : xr.Dataset
        The input dataset which contains a coordinate (x) and a data variable
        (y)
    x_dim : str, optional
        The name of the x coordinate, by default "time"
    factor : float, optional
        The distance between the newly added points and the following existing
        points on the x-scale, by default 1e-4
    interpolation : Literal["fill-forward", "linear"], optional
        The interpolation method. In addition to 'fill-forward' and 'linear',
        any method give in `xr.interpolate_na` can be chosen, by default
        "fill-forward"

    Returns
    -------
    xr.Dataset
        The interpolated dataset
    """
    xs = x_in.coords[x_dim]

    # calculate x values that are located just a little bit smaller than the xs
    # where "just a little bit" is defined by the distance to the previous x
    # and a factor. This way the scale of the observations should not matter
    # and even very differently sized x-steps should be interpolated correctly
    fraction_before_xs = (
        xs.isel({x_dim:range(1, len(xs))})
        - xs.diff(dim=x_dim) * factor
    )

    # create a sorted time vector
    xs = sorted([*fraction_before_xs.values, *xs.values])

    # add new time indices with NaN values 
    x_in_reindexed = x_in.reindex({x_dim:xs})

    if interpolation == "fill-forward":
        # then fill nan values with the previous value (forward-fill)
        x_in_interpolated = x_in_reindexed.ffill(dim=x_dim, limit=1)

    else:
        x_in_interpolated = x_in_reindexed.interpolate_na(dim=x_dim, method="linear")

    return x_in_interpolated


def smoothed_interpolation(
    x_in: xr.Dataset, 
    x_dim: str="time", 
    factor: float=0.001, 
    sigma: int = 20,
) -> xr.Dataset:
    """Smooth the interpolation by first creating a dense x vector and forward
    filling all ys. Following this the values are smoothed by a gaussian filter.

    Parameters
    ----------
    x_in : xr.Dataset
        The input dataset which contains a coordinate (x) and a data variable
        (y)
    x_dim : str, optional
        The name of the x coordinate, by default "time"
    factor : float, optional
        The distance between the newly added points and the following existing
        points on the x-scale, by default 1e-4

    Returns
    -------
    xr.Dataset
        The interpolated dataset
    """
    xs = x_in.coords[x_dim]
    assert factor > 0, "Factor must be larger than zero, to ensure correct ordering"
    
    xs_extra = np.arange(xs.values.min(), xs.values.max()+factor, step=factor)
    xs_ = np.sort(np.unique(np.concatenate([xs.values, xs_extra])))

    # add new time indices with NaN values 
    x_in_reindexed = x_in.reindex({x_dim:xs_})

    # then fill nan values with the previous value (forward-fill)
    x_in_interpolated = x_in_reindexed.ffill(dim=x_dim)

    # Apply Gaussian smoothing
    sigma = 20  # Adjust sigma for desired smoothness
    for k in x_in_interpolated.data_vars.keys():
        y = x_in_interpolated[k]
        y_smoothed = gaussian_filter1d(y.values, sigma, axis=list(y.dims).index(x_dim))

        x_in_interpolated[k].values = y_smoothed

    return x_in_interpolated


def radius_interpolation(
    x_in: xr.Dataset, 
    x_dim: str="time", 
    radius: float=0.1, 
    num_points: int=10,
    rectify=True
) -> xr.Dataset:
    """Smooth the interpolation by first creating a dense x vector and forward
    filling all ys. Following this the values are smoothed by a gaussian filter.

    WARNING! It is very pretty but does not work with diffrax

    Parameters
    ----------
    x_in : xr.Dataset
        The input dataset which contains a coordinate (x) and a data variable
        (y)
    x_dim : str, optional
        The name of the x coordinate, by default "time"
    radius : float, optional
        The radius of the quarter-circle to curve the jump transition. 
        By default 0.1
    num_points : int, optional
        The number of points to interpolate each jump with. Default: 10
    rectify : bool
        Whether the input should be converted to a stepwise pattern. Default 
        is True. This is typically applied if an unprocessed signal is included.
        E.g. the signal was observed y_i 
        
    Returns
    -------
    xr.Dataset
        The interpolated dataset
    """
    x = x_in.coords[x_dim] 
    assert radius <= np.diff(np.unique(x)).min() / 2

    if rectify:
        x = np.concatenate([[x[0]], *[[x_i-0.00, x_i] for x_i in x[1:]]])

    data_arrays = []
    for k in x_in.data_vars.keys():
        y = x_in[k]
        if rectify:
            yvals = np.concatenate([*[[y_i, y_i] for y_i in y[:-1]], [y[-1]]])
        else:
            yvals = y.values

        x_interpolated = [np.array(x[0],ndmin=1)]
        y_interpolated = [np.array(yvals[0], ndmin=2)]
        for i in range(0, len(x) - 1):
            x_, y_, = curve_jumps(x, yvals, i, r=radius, n=num_points)

            x_interpolated.append(x_)
            y_interpolated.append(y_)
    
        x_interpolated = np.concatenate(x_interpolated)
        x_uniques = np.where(np.diff(x_interpolated) != 0)
        x_interpolated = x_interpolated[x_uniques]

        y_interpolated = np.row_stack(y_interpolated)[x_uniques]
        
        coords = {x_dim: x_interpolated}
        coords.update({d: y.coords[d].values for d in y.dims if d != x_dim})

        y_reindexed = xr.DataArray(
            y_interpolated, 
            coords=coords,
            name=y.name
        )
        data_arrays.append(y_reindexed)

    x_in_interpolated = xr.combine_by_coords(data_objects=data_arrays)
    
    # there will be nans if the data variables have different steps
    # x_in_interpolated = x_in_interpolated.interpolate_na(
    #     dim="time", method="linear"
    # )

    return x_in_interpolated # type: ignore


def curve_jumps(x, y, i, r, n):
    x_i = x[i] # jump start
    y_i = y[i] # jump start
    y_im1 = y[i-1] # jump start
    y_ip1 = y[i+1] # jump end

    def circle_func(x, r, a): 
        # using different radii does not work, because this would also require different x_values
        arc = r**2 - (x - a)**2
        return np.sqrt(np.where(arc >= 0, arc, 0))
    
    # end of jump
    dy_im1 = y_i - y_im1 # jump difference to previous point
    if np.all(dy_im1 == 0):
        dyj = y_ip1 - y_i
        sj = np.where(np.abs(dyj) > r, np.sign(dyj), dyj / 2 / r) # direction and size of the jump, scaled by the radius
        # sj = np.clip((y_ip1 - y_i)/r/2, -1, 1) # direction and size of the jump, scaled by the radius
        xc = np.linspace(x_i-r, x_i-r*0.001, num=n)
        yc = y_i + (np.einsum("k,x->xk",  -sj, circle_func(x=xc, r=r, a=x_i-r)) + sj * r)
    else:
        dyj = y_i - y_im1
        sj = np.where(np.abs(dyj) > r, np.sign(dyj), dyj / 2 / r) # direction and size of the jump, scaled by the radius
        xc = np.linspace(x_i+ r*0.001, x_i+r, num=n)
        yc = y_i + (np.einsum("k,x->xk",  sj, circle_func(x=xc, r=r, a=x_i+r)) - sj * r)
    
    return xc, yc


def rect_interpolation(
    x_in: xr.Dataset, 
    x_dim: str="time", 
):  
    """Use diffrax' rectilinear_interpolation. To add values and interpolate
    one more step after the end of the timeseries
    """
    data_arrays = []
    data_vars = tuple(x_in.data_vars.keys())
    dataset_dims = tuple(x_in.dims.keys())

    for k in data_vars:
        v_orig = x_in[k]
        v = v_orig.transpose(x_dim, ...)
        x = v[x_dim].values
        y = v.values
        x_dim_loc = v.dims.index(x_dim)

        ts_ = np.concatenate([x, np.array(x[-1]+1, ndmin=1)]) 
        if y.ndim == 1:
            ys_ = np.concatenate([y, np.array(y[-1], ndmin=1)])
        elif y.ndim == 2:
            if x_dim_loc == 0:
                ys_ = np.row_stack([y, np.expand_dims(y[-1], axis=0)]) 
        elif y.ndim == 3:
            if x_dim_loc == 0:
                ys_ = np.row_stack([y, np.expand_dims(y[-1], axis=0)])
            else:
                raise ValueError(f"{x_dim_loc} should be the first dimension.")
        else:
            raise NotImplementedError(
                "Dimensions of interpolation > 2 or 0 are not implemented"
            )

        xs, ys = rectilinear_interpolation(ts=ts_, ys=ys_) # type:ignore
        # if y.ndim <= 2:
        #     xs = xs
        # else:
        #     xs = xs[0] # type:ignore

        coords = {x_dim: xs}
        coords.update({d: v.coords[d].values for d in v.dims if d != x_dim})
        # coords = {d:coords[d] for d in v.dims}

        y_reindexed = xr.DataArray(
            ys, 
            coords=coords,
            name=v.name
        )
        y_reindexed = y_reindexed.transpose(*v_orig.dims)

        data_arrays.append(y_reindexed)

    x_in_interpolated = xr.combine_by_coords(data_objects=data_arrays)
    x_in_interpolated = x_in_interpolated.transpose(*dataset_dims)


    return x_in_interpolated[[*dataset_dims, *data_vars]]