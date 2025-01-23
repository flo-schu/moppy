from pymob.utils.errors import import_optional_dependency

from . import analysis
from . import base
from . import scipy_backend

pyabc = import_optional_dependency("pyabc", errors="ignore")
if pyabc is not None:
    from . import pyabc_backend

pymoo = import_optional_dependency("pymoo", errors="ignore")
if pymoo is not None:
    from . import pymoo_backend

numpyro = import_optional_dependency("numpyro", errors="ignore")
if numpyro is not None:
    from . import numpyro_backend
    from . import numpyro_dist_map