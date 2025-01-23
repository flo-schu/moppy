import numpyro
import jax.numpy as jnp
from numpyro.distributions import Normal, TransformedDistribution
from numpyro.distributions import transforms

scipy_to_numpyro = {
    "deterministic": (numpyro.deterministic, {"value": "value"}),
    # Continuous Distributions
    "norm": (numpyro.distributions.Normal, {"loc": "loc", "scale": "scale", "low": "low", "high": "high"}),
    "normal": (numpyro.distributions.Normal, {"loc": "loc", "scale": "scale", "low": "low", "high": "high"}),
    "halfnorm": (numpyro.distributions.HalfNormal, {"scale": "scale", "high":"high"}),
    "halfnormal": (numpyro.distributions.HalfNormal, {"scale": "scale", "high":"high"}),
    "expon": (numpyro.distributions.Exponential, {"scale": "rate", "high": "high"}),  # rate = 1/scale
    "exponential": (numpyro.distributions.Exponential, {"scale": "rate", "high": "high"}),  # rate = 1/scale
    "uniform": (numpyro.distributions.Uniform, {"loc": "low", "scale": "high"}),  # high = loc + scale
    "beta": (numpyro.distributions.Beta, {"a": "concentration1", "b": "concentration0"}),
    "gamma": (numpyro.distributions.Gamma, {"a": "concentration", "scale": "rate", "low": "low", "high": "high"}),  # rate = 1/scale
    "lognorm": (numpyro.distributions.LogNormal, {"scale": "loc", "s": "scale", "loc": "loc", "low": "low", "high": "high"}),
    "lognormal": (numpyro.distributions.LogNormal, {"scale": "loc", "s": "scale", "loc": "loc", "low": "low", "high": "high"}),
    "chi2": (numpyro.distributions.Chi2, {"df": "df", "low": "low", "high": "high"}),
    "pareto": (numpyro.distributions.Pareto, {"b": "scale", "scale": "alpha", "low": "low", "high": "high"}),
    "t": (numpyro.distributions.StudentT, {"df": "df", "loc": "loc", "scale": "scale", "low": "low", "high": "high"}),
    "cauchy": (numpyro.distributions.Cauchy, {"loc": "loc", "scale": "scale", "low": "low", "high": "high"}),
    "gumbel_r": (numpyro.distributions.Gumbel, {"loc": "loc", "scale": "scale", "low": "low", "high": "high"}),
    "gumbel_l": (lambda loc, scale: numpyro.distributions.Gumbel(loc=-loc, scale=scale), {}), 
    "laplace": (numpyro.distributions.Laplace, {"loc": "loc", "scale": "scale", "low": "low", "high": "high"}),
    "logistic": (numpyro.distributions.Logistic, {"loc": "loc", "scale": "scale", "low": "low", "high": "high"}),
    "multivariate_normal": (numpyro.distributions.MultivariateNormal, {"mean": "loc", "cov": "covariance_matrix"}),
    "dirichlet": (numpyro.distributions.Dirichlet, {"alpha": "concentration"}),
    
    # Discrete Distributions
    "binom": (numpyro.distributions.Binomial, {"n": "total_count", "p": "probs"}),
    "binomial": (numpyro.distributions.Binomial, {"n": "total_count", "p": "probs"}),
    "bernoulli": (numpyro.distributions.Bernoulli, {"p": "probs"}),
    "geom": (numpyro.distributions.Geometric, {"p": "probs"}),
    "poisson": (numpyro.distributions.Poisson, {"mu": "rate"}),
    "nbinom": (numpyro.distributions.NegativeBinomialProbs, {"n": "total_count", "p": "probs"}),
    "multinomial": (numpyro.distributions.Multinomial, {"n": "total_count", "p": "probs"}),
    "categorical": (numpyro.distributions.Categorical, {"p": "probs"}),
    
}

import jax.numpy as jnp
from numpyro.distributions import Normal, TransformedDistribution, TruncatedNormal
from numpyro.distributions import transforms


def transform(transforms, x):
    for part in transforms:
        x = part(x)
    return x

def inv_transform(transforms, y):
    for part in transforms[::-1]:
        y = part.inv(y)
    return y



# LogNormal Transformation
def LogNormalTrans(loc, scale, low=None, high=None):
    _transforms = [
        transforms.AffineTransform(loc=jnp.log(loc), scale=scale),
        transforms.ExpTransform(),
    ]

    if high is not None:
        high = inv_transform(_transforms, high)

    if low is not None:
        low = inv_transform(_transforms, low)

    base_distribution=TruncatedNormal(loc=0, scale=1, low=low, high=high)
    return TransformedDistribution(
        base_distribution=base_distribution,
        transforms=_transforms
    )

# LogNormal Transformation
def NormalTrans(loc, scale):
    return TransformedDistribution(
        base_distribution=Normal(0, 1),
        transforms=[
            transforms.AffineTransform(loc=loc, scale=scale),
        ]
    )

# Exponential Transformation
def ExponentialTrans(rate):
    return TransformedDistribution(
        base_distribution=Normal(0, 1),
        transforms=[
            transforms.AffineTransform(loc=0, scale=1/rate),
            transforms.ExpTransform(),
        ]
    )

# Uniform Transformation
def UniformTrans(low, high):
    return TransformedDistribution(
        base_distribution=Normal(0, 1),
        transforms=[
            transforms.SigmoidTransform(),
            transforms.AffineTransform(loc=low, scale=high - low)
        ]
    )


# Half-Normal Transformation
def HalfNormalTrans(scale):
    return TransformedDistribution(
        base_distribution=Normal(0, 1),
        transforms=[
            transforms.AffineTransform(loc=0, scale=scale),
            transforms.AbsTransform()
        ]
    )

transformed_dist_map = {
    numpyro.distributions.LogNormal: LogNormalTrans,
    numpyro.distributions.Exponential: ExponentialTrans,
    numpyro.distributions.Uniform: UniformTrans,
    numpyro.distributions.HalfNormal: HalfNormalTrans,
    numpyro.distributions.Normal: NormalTrans,
}