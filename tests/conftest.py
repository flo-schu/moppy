# content of conftest.py
import numpy
import xarray
import pytest

@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["xr"] = xarray
