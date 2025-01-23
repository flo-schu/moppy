import pytest
import numpy as np
from scipy.stats import norm

from pymob.inference.scipy_backend import ScipyBackend
from pymob.sim.parameters import Param
from pymob.solvers.diffrax import JaxSolver

from tests.fixtures import init_lotka_volterra_case_study_hierarchical_from_script

def test_parameter_parsing_different_priors_on_species():
    sim = init_lotka_volterra_case_study_hierarchical_from_script()
    
    sim.config.model_parameters.alpha_species = Param(
        value=0.5, free=True, hyper=True,
        dims=('rabbit_species','experiment'),
        # take good care to specify hyperpriors correctly. 
        # Dimensions are broadcasted following the normal rules of 
        # numpy. The below means, in dimension one, we have two different
        # assumptions 1, and 3. Dimension one is the dimension of the rabbit species.
        # The specification loc=[1,3] would be understood as [[1,3]] and
        # be understood as the experiment dimension. Ideally, the dimensionality
        # is so low that you can be specific about the priors. I.e.:
        # scale = [[1,1,1],[3,3,3]]. This of course expects you know about
        # the dimensionality of the prior (i.e. the unique coordinates of the dimensions)
        prior="norm(loc=[[1],[3]],scale=0.1)" # type: ignore
    )
    # prey birth rate
    # to be clear, this says each replicate has a slightly varying birth
    # rate depending on the valley where it was observed. Seems legit.
    sim.config.model_parameters.alpha = Param(
        value=0.5, free=True, hyper=False,
        dims=('id',),
        prior="lognorm(s=0.1,scale=alpha_species[rabbit_species_index, experiment_index])" # type: ignore
    )

    # re initialize the observation with
    sim.define_observations_replicated_multi_experiment(n=120) # type: ignore
    sim.coordinates["time"] = np.arange(12)
    y0 = sim.parse_input("y0", drop_dims=["time"])
    sim.model_parameters["y0"] = y0

    inferer = ScipyBackend(simulation=sim)


    theta = inferer.sample_distribution()

    alpha_samples_cottontail = theta["alpha"][sim.observations["rabbit_species"] == "Cottontail"]
    alpha_samples_jackrabbit = theta["alpha"][sim.observations["rabbit_species"] == "Jackrabbit"]

    alpha_cottontail = np.mean(alpha_samples_cottontail)
    alpha_jackrabbit = np.mean(alpha_samples_jackrabbit)
    
    # test if the priors that were broadcasted to the replicates 
    # match the hyperpriors
    np.testing.assert_array_almost_equal(
        [alpha_cottontail, alpha_jackrabbit], [1, 3], decimal=1
    )

    sim.solver = JaxSolver
    sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict
    sim.dispatch_constructor()
    e = sim.dispatch(theta=theta)
    e()

    # res_species = e.results.where(e.results.rabbit_species=="Cottontail", drop=True)
    # store simulated results

    # TODO: mark datavars as observed automatically
    # set observations and mark as observed. This 
    # could be automated by using length of data var > 0 to
    # trigger marking data vars as observed.
    rng = np.random.default_rng(1)
    
    
    obs = e.results
    obs.rabbits.values = rng.poisson(e.results.rabbits+1e-6)
    obs.wolves.values = rng.poisson(e.results.wolves+1e-6)
    obs.to_netcdf(
        f"{sim.data_path}/simulated_data_hierarchical_species_year.nc"
    )

    sim.observations = obs
    sim.config.data_structure.rabbits.observed = True
    sim.config.data_structure.wolves.observed = True

    # this is the conventional way to define error models. We center a lognormal
    # error model around the means of the distribution. The problem is: 
    # due to the large scale differences in rabbits and wolves, the log-likelihoods
    # end up very differently. Here the wolves data variable will basically
    # be meaningless, because the rabbits data variable is at such a high scale
    # Scaling alone also does not resolve this problem, because due to the dynamic
    # of the data variables, larger values will have a higher weight. This is not
    # right.
    sim.config.error_model.rabbits = "lognorm(scale=rabbits+EPS, s=0.1)"
    sim.config.error_model.wolves = "lognorm(scale=wolves+EPS, s=0.1)"
    sim.dispatch_constructor()
    # TODO: At this point, I should get a joint likelihood function and try
    # to fit the model with scipy minimize.


def test_parameter_parsing_different_priors_on_year():
    sim = init_lotka_volterra_case_study_hierarchical_from_script()
    
    sim.config.model_parameters.alpha_species = Param(
        value=0.5, free=True, hyper=True, dims=('rabbit_species','experiment'),
        prior="norm(loc=[[1,2,3]],scale=0.1)" # type: ignore
    )
    # prey birth rate
    sim.config.model_parameters.alpha = Param(
        value=0.5, free=True, hyper=True, dims=('id',),
        prior="lognorm(s=0.1,scale=alpha_species[rabbit_species_index, experiment_index])" # type: ignore
    )

    # re initialize the observation with
    sim.define_observations_replicated_multi_experiment(n=120) # type: ignore
    y0 = sim.parse_input("y0", drop_dims=["time"])
    sim.model_parameters["y0"] = y0

    inferer = ScipyBackend(simulation=sim)


    theta = inferer.sample_distribution()

    alpha_samples_2010 = theta["alpha"][sim.observations["experiment"] == "2010"]
    alpha_samples_2011 = theta["alpha"][sim.observations["experiment"] == "2011"]
    alpha_samples_2012 = theta["alpha"][sim.observations["experiment"] == "2012"]

    alpha_2010 = np.mean(alpha_samples_2010)
    alpha_2011 = np.mean(alpha_samples_2011)
    alpha_2012 = np.mean(alpha_samples_2012)
    
    # test if the priors that were broadcasted to the replicates 
    # match the hyperpriors
    np.testing.assert_array_almost_equal(
        [alpha_2010, alpha_2011, alpha_2012], [1, 2, 3], decimal=1
    )




if __name__ == "__main__":
    pass