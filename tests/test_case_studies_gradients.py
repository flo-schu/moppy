import numpy as np
import pytest
from tests.fixtures import (
    init_simulation_casestudy_api,
    init_guts_casestudy_constant_exposure,
    init_guts_casestudy_variable_exposure,
    init_bufferguts_casestudy,
)
import jax
import jax.numpy as jnp

# this should raise an exception when nans are created the computation


def test_guts_constant_exposure():
    sim = init_guts_casestudy_constant_exposure()

    sim.use_jax_solver()
    sim.dispatch_constructor(rtol=1e-7, atol=1e-8)

    # test gradients
    def test_func(theta):
        k_d, h_b, b, z = theta
        e = sim.dispatch({"k_d": k_d, "h_b": h_b, "b":b, "z": z})
        e()
        return e.Y["S"][-1]

    val, grads = jax.value_and_grad(test_func)(jnp.array([1, 0.001, 0.2, 1.0]))

    assert all(grads != jnp.nan)

def test_guts_variable_exposure():
    sim = init_guts_casestudy_variable_exposure()

    sim.use_jax_solver()
    sim.dispatch_constructor(rtol=1e-7, atol=1e-8)

    # test gradients
    def test_func(theta):
        k_d, h_b, b, z = theta
        e = sim.dispatch({"k_d": k_d, "h_b": h_b, "b":b, "z": z})
        e()
        return e.Y["S"][-1]

    val, grads = jax.value_and_grad(test_func)(jnp.array([1, 0.001, 0.2, 1.0]))

    assert all(grads != jnp.nan)


def test_bufferguts_variable_exposure():
    # Can be fixed by replacing the bufferGUTS model with the improved
    # version from the jupyter notebook
    sim = init_bufferguts_casestudy()

    sim.use_jax_solver()
    sim.dispatch_constructor(rtol=1e-7, atol=1e-8, max_steps=1e6)

    # test gradients
    def test_func(theta):
        k_d, h_b, b, z = theta
        e = sim.dispatch({"k_d": k_d, "h_b": h_b, "b":b, "z": z})
        e()
        return e.Y["S"][-1]

    val, grads = jax.value_and_grad(test_func)(jnp.array([1, 0.001, 0.2, 1.0]))

    assert all(grads != jnp.nan)

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    pass
    # test_multiple_case_study_import()
    # test_variable_exposure()
    # test_bufferguts_variable_exposure()
    # test_bufferguts_hybrid_solution()
