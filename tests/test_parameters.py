from functools import partial

import jax
import pytest
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from numpyro.handlers import seed, trace

from pymob.simulation import SimulationBase
from pymob.sim.config import Datastructure, Modelparameters
from pymob.inference.scipy_backend import ScipyBackend
from pymob.inference.numpyro_backend import NumpyroBackend
from pymob.inference.numpyro_dist_map import inv_transform
from pymob.sim.parameters import Param, RandomVariable, Expression

from tests.fixtures import (
    init_simulation_casestudy_api, 
    create_composite_priors,
    create_composite_priors_wrong_order,
)

def test_casestudy_api():
    sim = init_simulation_casestudy_api("test_scenario")
    assert sim.model_parameter_dict == {'alpha': 0.5, 'beta': 0.02}


def test_truncated_priors_scipy():
    prior_mu = RandomVariable(distribution="lognormal", parameters={"loc": Expression("5"), "scale": Expression("2.0"), "high": Expression("10.0")})

    mu = Param(prior=prior_mu, hyper=True, dims=("experiment",))

    theta = Modelparameters() # type: ignore
    theta.mu = mu

    parsed_params = ScipyBackend.parse_model_priors(
        parameters=theta.free,
        dim_shapes={k:(100, 2) for k, _ in theta.all.items()},
        data_structure=Datastructure()
    )

    inferer = ScipyBackend(SimulationBase())
    inferer.prior = parsed_params

    # TODO: Does not work yet. The approach would be to also use transforms
    # from the truncated normal distribution to sample
    pytest.skip()
    samples = inferer.sample_distribution()


class TestNumpyro:

    @pytest.fixture
    def inferer(self) -> NumpyroBackend:
        prior_mu = RandomVariable(
            distribution="lognormal", 
            parameters={
                "scale": Expression("5"), 
                "s": Expression("2.0"), 
                "high": Expression("10.0")
            })

        mu = Param(prior=prior_mu, hyper=True, dims=("experiment",))

        theta = Modelparameters() # type: ignore
        theta.mu = mu

        parsed_params = NumpyroBackend.parse_model_priors(
            parameters=theta.free,
            dim_shapes={k:(100, 2) for k, _ in theta.all.items()},
            data_structure=Datastructure()
        )

        inferer = NumpyroBackend(SimulationBase())
        inferer.prior = parsed_params
        return inferer

    def test_truncation(self, inferer):
        d_constructed = inferer.prior["mu"].construct({})
        draws = d_constructed.sample(jax.random.key(1), (1000,))
        np.testing.assert_array_less(draws, 10)

    def test_sampling(self, inferer):
        model = inferer.parse_probabilistic_model()

        init_model = partial(model, solver=None, obs=None, masks=None, only_prior=True)
        trace_ = trace(seed(init_model, jax.random.key(1))).get_trace()

        dist = trace_["mu"]["fn"]
        
        np.testing.assert_equal(
            dist.base_dist.base_dist.high,
            inv_transform(dist.base_dist.transforms, 10)
        )

        mu = trace_["mu"]["value"]
        np.testing.assert_array_less(mu, 10)

        
    def test_sampling_gaussian_base(self, inferer):
        inferer.config.inference_numpyro.gaussian_base_distribution = True
        transforms = inferer.prior["mu"].construct({}).transforms
        model = inferer.parse_probabilistic_model()

        init_model = partial(model, solver=None, obs=None, masks=None, only_prior=True)
        trace_ = trace(seed(init_model, jax.random.key(1))).get_trace()

        base_dist = trace_["mu_normal_base"]["fn"]
        
        np.testing.assert_equal(
            base_dist.base_dist.high,
            inv_transform(transforms, 10)
        )

        mu = trace_["mu"]["value"]
        np.testing.assert_array_less(mu, 10)

def test_prior_parsing():
    params = create_composite_priors()
    parsed_params = ScipyBackend.parse_model_priors(
        parameters=params.free,
        dim_shapes={k:(100, 2) for k, _ in params.all.items()},
        data_structure=Datastructure()
    )

    inferer = ScipyBackend(SimulationBase())
    inferer.prior = parsed_params
    samples = inferer.sample_distribution()
    np.testing.assert_equal(samples["k"].shape, (100, 2))

def test_prior_parsing_error():
    params = create_composite_priors_wrong_order()
    try:
        parsed_params = ScipyBackend.parse_model_priors(
            parameters=params.free,
            dim_shapes={k:() for k, _ in params.all.items()},
            data_structure=Datastructure()
        )
        raise AssertionError("Parameter parsing should have failed.")
    except KeyError:
        pass


def test_prior_to_xarray():
    params = create_composite_priors()

    coords = {"experiment": (0, 1)}
    arr = params.k.to_xarray(coords)
    true = xr.DataArray([5,23.1], dims=("experiment",), coords={"experiment": [0,1]})

    assert np.all((arr == true).values)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())