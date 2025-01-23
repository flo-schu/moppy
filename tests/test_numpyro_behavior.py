import pytest
import numpy as np
import xarray as xr

from tests.fixtures import (
    create_simulation_for_test_numpyro_behavior,
    init_case_study_and_scenario
)

def replicate_obs(sim):
    sim.observations.expand_dims("id")
    obs1 = sim.observations.assign_coords({"id": 0})
    obs2 = sim.observations.assign_coords({"id": 1})
    obs = xr.combine_nested([obs1,obs2], concat_dim="id")
    sim.config.data_structure.rabbits.dimensions=["id", "time"]
    sim.config.data_structure.wolves.dimensions=["id", "time"]
    sim.config.simulation.batch_dimension = "id"
    sim.observations = obs

    sim.model_parameters["y0"] = sim.parse_input("y0", reference_data=obs, drop_dims=["time"])


# create config file for all following tests
create_simulation_for_test_numpyro_behavior()

def test_effect_handler_scaling():
    sim = init_case_study_and_scenario(
        case_study="lotka_volterra_case_study",
        scenario="test_numpyro_behavior"
    )
    sim.dispatch_constructor()
    sim.set_inferer("numpyro")
    idata = sim.inferer.prior_predictions(n=1)
    
    prior = idata.prior.sel(chain=0, draw=0)
    data_vars = sim.config.data_structure.observed_data_variables
    theta = {k: np.array(v.to_dict()["data"]) for k, v in prior.data_vars.items()}
    
    ll_sum_idata = idata.log_likelihood.sel(chain=0, draw=0)\
        .sum().to_array().sel(variable=data_vars)

    loglik_fn, _ = sim.inferer.create_log_likelihood(
        return_type="summed-by-site", scaled=False
    )

    loglik_fn_scaled, _ = sim.inferer.create_log_likelihood(
        return_type="summed-by-site", scaled=True
    )
    
    _, _, ll_sum_site = loglik_fn(theta=theta)
    _, _, ll_sum_site_scaled = loglik_fn_scaled(theta=theta)


    np.testing.assert_allclose(
        ll_sum_idata.values, 
        [ll_sum_site[f"{k}_obs"] for k in data_vars],
        atol=0.01,
        rtol=1e-7
    )

    # test if the scaled log-likelihood is equivalent to the scaled log-likelihood
    scales = {"rabbits": 2, "wolves": 1}
    np.testing.assert_allclose(
        ll_sum_idata.values, 
        [ll_sum_site_scaled[f"{k}_obs"] / scales[k] for k in data_vars],
        atol=0.01,
        rtol=1e-7
    )

def test_effect_handler_plate():
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    # Define the model with a multinomial inside a plate, using different n_trials for each batch
    def model(lethality, n_trials):
        batch_size, num_categories = lethality.shape  # Assumes probs has shape (batch_size, num_categories)
        probs = numpyro.sample("p", dist.Dirichlet(jnp.ones((batch_size,3))))
        
        # Define a plate for the batch dimension
        with numpyro.plate("batch", batch_size):
            # Sample from a Multinomial distribution with a different total_count for each batch item
            counts = numpyro.sample("lethality", dist.Multinomial(
                total_count=n_trials, probs=probs), obs=lethality)
        return counts

    # Define parameters for the multinomial distribution
    n_trials = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55]) * 100
    probs = np.array([
        [0.2, 0.5, 0.3],
        [0.3, 0.4, 0.3],
        [0.1, 0.7, 0.2],
        [0.25, 0.25, 0.5],
        [0.6, 0.2, 0.2],
        [0.3, 0.3, 0.4],
        [0.4, 0.4, 0.2],
        [0.2, 0.3, 0.5],
        [0.3, 0.3, 0.4],
        [0.1, 0.6, 0.3]
    ])  # Probability vector for each batch item

    lethality = jnp.array(list(map(lambda n, p: np.random.multinomial(n=n, pvals=p), n_trials, probs)))

    # Run MCMC sampling on the model
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    mcmc.run(jax.random.PRNGKey(0), lethality=lethality, n_trials=n_trials)

    # Get samples
    samples = mcmc.get_samples()["p"]

    # allow 5% deviation from any true probability value
    np.testing.assert_allclose(samples.mean(axis=0), probs, atol=0.05)


def test_effect_handler_reparam():
    pytest.skip()
    # TODO: Reparameterization can also be handled https://num.pyro.ai/en/stable/reparam.html#module-numpyro.infer.reparam

if __name__ == "__main__":
    pass
    # test_effect_handler_plate()
    # test_effect_handler_scaling()