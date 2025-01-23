from functools import partial
from types import ModuleType
from collections import OrderedDict
from typing import Optional, List, Dict, Literal, Tuple, OrderedDict
from pymob.solvers.base import mappar, SolverBase
from frozendict import frozendict
from dataclasses import dataclass, field
import jax.numpy as jnp
from jax import Array
import jax
import diffrax
from diffrax.solver.base import _MetaAbstractSolver
from diffrax import (
    diffeqsolve, 
    Dopri5, 
    Tsit5,
    Kvaerno5,
    ODETerm, 
    SaveAt, 
    PIDController, 
    RecursiveCheckpointAdjoint,
    LinearInterpolation,
)

Mode = Literal['r', 'rb', 'w', 'wb']


@dataclass(frozen=True, eq=True)
class JaxSolver(SolverBase):
    """
    see https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
    to make thinks robust

    Parameters
    ----------

    throw_exceptions: bool
        Default is True. The JaxSolver will throw an exception if it runs into a problem
        with the step size and return infinity. This is done, because likelihood based inference
        algorithms can deal with infinity values and consider the tested parameter
        combination impossible. If used without caution, this can lead to severely
        biased parameter estimates.
    """

    diffrax_solver: _MetaAbstractSolver|str = field(default=Dopri5)
    rtol: float = 1e-6
    atol: float = 1e-7
    pcoeff: float = 0.0
    icoeff: float = 1.0
    dcoeff: float = 0.0
    max_steps: int = int(1e5)
    throw_exception: bool = True

    x_in_jumps: Tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self, *args, **kwargs):
        super().__post_init__(*args, **kwargs)

        # set the diffrax solver if it is a string instance
        dfx_solver = self.diffrax_solver
        if isinstance(dfx_solver, str):
            diffrax_solver = getattr(diffrax, dfx_solver)
            if not isinstance(diffrax_solver, _MetaAbstractSolver):
                raise TypeError(
                    f"Solver {diffrax_solver} must be {_MetaAbstractSolver}"
                )
            object.__setattr__(self, "diffrax_solver", diffrax_solver)

        x_in_jumps = self._get_x_in_jumps()
        object.__setattr__(self, "x_in_jumps", x_in_jumps)

        hash(self)

    def _get_x_in_jumps(self) -> Optional[Tuple[float, ...]]:
        """In case there are multiple interpolations present, this function
        ensures that all jumps are handled appropriately
        """
        xs_vars = []
        for _, data_var_coords in self.coordinates_input_vars["x_in"].items():
            x_var = data_var_coords[self.x_dim]

            xs_vars.append(jnp.array(x_var, dtype=float))

        if len(xs_vars) == 0:
            return None
        else:
            # unique returns a sorted and unique array, so it is perfect for 
            # the job
            x_in_jumps = jnp.unique(jnp.concatenate(xs_vars))

            # we need to cut off any jumps that are greater or equal to the
            # last observation, so that no infinities are returned for the
            # last x that is evaluated by the solver
            x_in_jumps_trimmed = x_in_jumps[x_in_jumps < self.x[-1]]
            # x_in[0][self.coordinates_input_vars["x_in"][self.x_dim] < self.x[-1]]
            
            return tuple(x_in_jumps_trimmed.tolist())



    @partial(jax.jit, static_argnames=["self"])
    def solve(self, parameters: Dict, y0:Dict={}, x_in:Dict={}):
        
        
        X_in = self.preprocess_x_in(x_in)
        x_in_flat = [x for xi in X_in for x in xi]

        Y_0 = self.preprocess_y_0(y0)

        ode_args, pp_args = self.preprocess_parameters(parameters)

        initialized_eval_func = partial(
            self.odesolve_splitargs,
            odestates = tuple(y0.keys()),
            n_odeargs=len(ode_args),
            n_ppargs=len(pp_args),
            n_xin=len(x_in_flat)
        )
        
        loop_eval = jax.vmap(
            initialized_eval_func, 
            in_axes=(
                *[0 for _ in range(self.n_ode_states)], 
                *[0 for _ in range(len(ode_args))],
                *[0 for _ in range(len(pp_args))],
                *[0 for _ in range(len(x_in_flat))], 
            )
        )
        result = loop_eval(*Y_0, *ode_args, *pp_args, *x_in_flat)

        # if self.batch_dimension not in self.coordinates:    
        # this is not yet stable, because it may remove extra dimensions
        # if there is a batch dimension of explicitly one specified

        # there is an extra dimension added if no batch dimension is present
        # this is added at the 0-axis
        # if parameters are scalars, the returned shape is 
        for v, val in result.items():
            if self.batch_dimension not in self.data_structure_and_dimensionality[v]:
                # otherwise it has a dummy dimension of length 1
                val_reduced = jnp.squeeze(val, 0)
            else:
                val_reduced = val

            expected_dims = tuple(self.data_structure_and_dimensionality[v].values())
            if len(expected_dims) != len(val_reduced.shape):
                # if the number of present dims is larger than the number of
                # expected dims, this is because the ODE "only" returned scalar
                # values. This is broadcasted to array of ndim=1
                val_reduced = jnp.squeeze(val_reduced, -1)
            else:
                pass

            # si = [
            #     s for dim, s in self.data_structure_and_dimensionality[v].items() 
            #     if dim != self.batch_dimension
            # ]
            
            # correct_shape = (s0, *si)
            
            # [i for i, vs in enumerate(val.shape) if vs not in expected_dims]
            # jnp.permute_dims(val, expected_dims)
            # val_reduced = val.permute_dims(expected_dims)
            result.update({v: val_reduced})

        return result

    @partial(jax.jit, static_argnames=["self"])
    def preprocess_parameters(self, parameters, num_backend: ModuleType = jnp):
        return super().preprocess_parameters(parameters, num_backend)
    
    @partial(jax.jit, static_argnames=["self"])
    def preprocess_x_in(self, x_in, num_backend: ModuleType = jnp):
        return super().preprocess_x_in(x_in, num_backend)

    @partial(jax.jit, static_argnames=["self"])
    def preprocess_y_0(self, y0, num_backend: ModuleType = jnp):
        return super().preprocess_y_0(y0, num_backend)

    @partial(jax.jit, static_argnames=["self"])
    def odesolve(self, y0, args, x_in):
        f = lambda t, y, args: self.model(t, y, *args)
        
        if len(x_in) > 0:
            if len(x_in) > 2:
                raise NotImplementedError(
                    "Currently only one interpolation is implemented, but "+
                    "it should be relatively simple to implement multiple "+
                    "interpolations. I assume, the interpolations could be "+
                    "passed as a list and expanded in the model. If you are "+
                    "dealing with this. Try pre-compute the interpolations. "+
                    "This should speed up the solver. "
                )
            interp = LinearInterpolation(ts=x_in[0], ys=x_in[1])
            args=(interp, *args)
            # jumps = x_in[0][self.coordinates_input_vars["x_in"][self.x_dim] < self.x[-1]]
            jumps = jnp.array(self.x_in_jumps, dtype=float)
        else:
            interp = None
            args=args
            jumps = None

        term = ODETerm(f)
        solver = self.diffrax_solver() # type: ignore (diffrax_solver is ensured
                                       # to be _MetaAbstractSolver type during 
                                       # post_init)
        saveat = SaveAt(ts=self.x)
        t_min = self.x[0]
        t_max = self.x[-1]
        # jump only those ts that are smaller than the last observations
        stepsize_controller = PIDController(
            rtol=self.rtol, atol=self.atol,
            pcoeff=self.pcoeff, icoeff=self.icoeff, dcoeff=self.dcoeff, 
            jump_ts=jumps,
        )
        
        sol = diffeqsolve(
            terms=term, 
            solver=solver, 
            t0=t_min, 
            t1=t_max, 
            dt0=0.1, 
            y0=tuple(y0), 
            args=args, 
            saveat=saveat, 
            stepsize_controller=stepsize_controller,
            adjoint=RecursiveCheckpointAdjoint(),
            max_steps=int(self.max_steps),
            # throw=False returns inf for all t > t_b, where t_b is the time 
            # at which the solver broke due to reaching max_steps. This behavior
            # happens instead of throwing an exception.
            throw=self.throw_exception
        )
        
        return tuple(sol.ys), interp

    @partial(jax.jit, static_argnames=["self", "odestates", "n_odeargs", "n_ppargs", "n_xin"])
    def odesolve_splitargs(self, *args, odestates, n_odeargs, n_ppargs, n_xin):
        n_odestates = len(odestates)
        y0 = args[:n_odestates]
        odeargs = args[n_odestates:n_odeargs+n_odestates]
        ppargs = args[n_odeargs+n_odestates:n_odeargs+n_odestates+n_ppargs]
        x_in = args[n_odestates+n_odeargs+n_ppargs:n_odestates+n_odeargs+n_ppargs+n_xin]
        sol, interp = self.odesolve(y0=y0, args=odeargs, x_in=x_in)
        
        res_dict = OrderedDict({v:val for v, val in zip(odestates, sol)})

        return self.post_processing(res_dict, jnp.array(self.x), interp, *ppargs)

