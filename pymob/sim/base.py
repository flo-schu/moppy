from typing import Callable, Dict, List, Optional, Union, List
import xarray as xr

def stack_variables(
    ds: xr.Dataset, 
    variables: List[str], 
    new_coordinates: List[str],
    new_dim: str,
    pattern: Callable = lambda var, coord: f"{var}_{coord}"
):
    """Combine data variables and coordinates into a new variable of 
    a higher dimension.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray Dataset.
    variables : List[str]
        List of variable bases to stack.
    new_coordinates : List[str]
        List of new coordinates for the higher dimension. Note that this must
        be the same for all variables you want to stack.
    new_dim : str
        The name of the new dimension.
    pattern : Callable, optional
        A function to generate names for new variables based on the variable and coordinate names.

    Returns
    -------
    xr.Dataset
        The modified xarray Dataset with stacked variables.


    Examples
    --------     
    A use case is the following. Consider a dataset that has the variables

        cext_A
        cext_B
        cext_C
        my_other_var_1
        my_other_var_2
        
    And you want to combine the variables with the same base `cext_` into
    a new variable that has the base as a dimension.

    >>> import xarray as xr
    >>> from typing import List, Callable
    >>> from pymob.sim.base import stack_variables
    >>> 
    >>> # Example usage:
    >>> an_xarray_dataset = xr.Dataset({
    ...     'cext_A': ([], 1.0),
    ...     'cext_B': ([], 2.0),
    ...     'cext_C': ([], 3.0),
    ...     'my_other_var_1': ([], 4.0),
    ...     'my_other_var_2': ([], 5.0),
    ... })
    >>> 
    >>> result_dataset = stack_variables(
    ...     ds=an_xarray_dataset,
    ...     variables=["cext"],
    ...     new_coordinates=["A", "B", "C"],
    ...     new_dim="letters",
    ... )
    >>> 
    >>> result_dataset
    <xarray.Dataset>
    Dimensions:         (letters: 3)
    Coordinates:
      * letters         (letters) <U1 'A' 'B' 'C'
    Data variables:
        my_other_var_1  float64 4.0
        my_other_var_2  float64 5.0
        cext            (letters) float64 1.0 2.0 3.0
    """
    for v in variables:
        v_names = [pattern(v, c) for c in new_coordinates]
        
        # convert coordinate if existing to a data_variable
        v_in_coords = False
        if all([vn in ds.coords for vn in v_names]):
            ds = ds.reset_coords(v_names)
            v_in_coords = True
        
        var: xr.Dataset = ds[v_names]
        array = var.to_array(dim=new_dim).assign_coords({new_dim:new_coordinates})
        ds[v] = array

        ds = ds.drop_vars(v_names)
        
        if v_in_coords:
            ds = ds.set_coords(v)

    return ds


def unlist_attrs(ds: xr.Dataset|xr.DataArray):
    """Transforms lists of variables to a comma separated string to 
    work around errors when storing the dataset or dataarray to disk"""
    for key, value in ds.attrs.items():
        if isinstance(value, list):
            ds.attrs[key] = ", ".join(value)

    return ds


def enlist_attr(ds: xr.Dataset|xr.DataArray, attr: str):
    """Transforms a string representation of a metadata attribute of an
    xarray dataset or datarray to a list"""
    attr_string = ds.attrs[attr]
    assert isinstance(attr_string, str)
    
    ds.attrs[attr] = attr_string.split(", ")

    return ds
