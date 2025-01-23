from functools import partial
from typing import Dict, Tuple, Union

import numpy as np
from numpy.random import Generator, PCG64
from scipy.stats._distn_infrastructure import rv_continuous, rv_discrete

from pymob.sim.parameters import scipy_to_scipy
from pymob.inference.base import Distribution, Errorfunction, InferenceBackend
from pymob.simulation import SimulationBase
from pymob.utils.config import lookup

class ScipyDistribution(Distribution):
    distribution_map: Dict[str,Tuple[Union[rv_continuous,rv_discrete],Dict[str,str]]] = scipy_to_scipy
    parameter_converter = staticmethod(lambda x: np.array(x))
    
    @property
    def dist_name(self) -> str:
        return self.distribution.name
    

    
class ScipyBackend(InferenceBackend):
    _distribution = ScipyDistribution
    distribution: Union[rv_continuous,rv_discrete]

    def __init__(self, simulation: SimulationBase) -> None:
        super().__init__(simulation)
        self.random_state = Generator(PCG64(self.config.simulation.seed))

    def parse_deterministic_model(self):
        pass

    def parse_probabilistic_model(self):
        pass

    def posterior_predictions(self):
        pass

    def prior_predictions(self):
        pass

    def sample_distribution(self):
        prior_samples = {}

        observations = self.simulation.observations
        # prior is added here, so it is updated 
        context=[prior_samples,self.indices,observations]
        for name, prior in self.prior.items():

            dist = prior.construct(context=context)
            sample = dist.rvs(size=prior.shape, random_state=self.random_state)
            dist.rvs()

            prior_samples.update({name:sample})

        return prior_samples
    
    def create_log_likelihood(self) -> Tuple[Errorfunction,Errorfunction]:
        # TODO: define
        return 