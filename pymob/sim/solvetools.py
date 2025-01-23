from typing import Literal
import xarray as xr


# legacy imports. These functions are now included in pymob.solvers.base
from pymob.solvers.base import mappar, smoothed_interpolation, jump_interpolation, rect_interpolation
from pymob.solvers.analytic import solve_analytic_1d
from pymob.solvers.scipy import solve_ivp_1d


