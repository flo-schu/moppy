# Parameter inference

Parameterizing complex models is still surprisingly difficult. it is both fortunate
that many different tools exist today to approach model parameterization. This task
has a variety of names with slightly different associated meaning, but they all
address the same task. Namely, estimating the set of parameters that best describe
some data $Y$ under the (mathematical) assumptions of the model. This process can be called parameter inference, parameter estimation, optimization, calibration, fitting. While there are
some subtle differences between the terms, they can be used more or less interchangibly. Relevant differences arise from the tools being used. The terms could be classified based on the probabilistic nature of the approach. While calibration and optimization are rather associated with deterministic models, which should have a unique optimal solution; estimation and inference are rather associated with approaches where the data are assumed to originate from a probability distribution.

For simplicity we will use the term parameter **inference**, because we interpret by its meaning from the fields of logic and reasoning, where it describes the process from moving from premises to logical consequences. 

> Statistical inference uses mathematics to draw conclusions in the presence of uncertainty. This generalizes deterministic reasoning, with the absence of uncertainty as a special case. Statistical inference uses quantitative or qualitative (categorical) data which may be subject to random variations. 

(inference-backends)=
## Supported inference backends

* interactive (interactive backend in jupyter notebookswith parameter sliders)
* numpyro (bayesian inference and stochastic variational inference)
* pyabc (approximate bayesian inference)
* pymoo (experimental! multi-objective optimization)
* scipy (experimental! currently only used to generate artificial data for developing and testing models from the specified prior distributions and error models)

## Why use `pymob.inference`?

The goal of `pymob` is to reduce the overhead for exploring, parameterizing and comparing
models so that the focus can remain on model development. Because after all, 
we would like to use models to anwer questions.

`pymob.inference` only enforces minimal conventions, which allows the interoperability between different frameworks. At the same time it allows you to take the models you have already written and just plug them into the framework and use it to do inference on it. 

 + Observations/data have to be provided as one [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html).
 + Simulations have to be wrapped inside a function, which converts the simulation output to a dictionary of results, so that results can be converted to [`xarray.Dataset`s](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html)
 + It is recommended that priors are specified following the scipy syntax, because functionality is implemented to translate scipy priors to different probabilistic programming languages (PPL). This is no strict requirement, but using PPL specific notation, sacrifices the possibility of interoperability between frameworks

## Parameter inference

The components of parameter inference or optimization are usually the same or very similar. We need: 

- priors
- a deterministic function (which can be an identity)
- an error model (or error distribution)

If you are thinking I don't need any priors for an optimization task, you are somehow correct, but we could also say that the specified bounds (lb, ub) on a parameter act as a prior (with the corresponding distribution being $Uniform(lb, ub)$). Even an unconstrained parameter is a Uniform between $[-\infty, \infty]$. Despite `pymob` sticks to the convention and let's you specify bounds or priors as you like.

### Example: The hierarchical Lotka-Volterra case-study

The hierarchical Lotka-Volterra case study assumes that observations  were taken from two rabbit populations (species: Cottontail and Jackrabbit), observed in 2010, 2011 and 2012 in different valleys, which define the number of in-treatment replicates. 

The case study is designed to study the possibilities of recovering species level parameters from nested error structures.

We set up the case study like this:

```python
from pymob import Config

config = Config()
config.case_study.name = "lotka_volterra_case_study"
config.case_study.scenario = "test_hierarchical"
config.case_study.simulation = "HierarchicalSimulation"
config.import_casestudy_modules(reset_path=True)
Simulation = config.import_simulation_from_case_study()

sim = Simulation(config)
sim.setup()
```

#### Creating parameters and priors

A parameter in `pymob` for instance for the hierarchical Lotka-Volterra case study can be specified like this:

```python
from pymob.sim.parameters import Param
alpha_species = Param(
    value=0.5, free=True, hyper=True, min=None, max=None,
    prior="norm(loc=[[1],[3]],scale=0.1,dims=('rabbit_species','experiment'))"
)
```

The prior assumes that we have a lot of prior knowledge of the alpha parameter of the different species. 

```{admonition} Distribution shapes and dimensionality
:class: warning
Take good care to specify the shape of your priors correctly. Dimensions are broadcasted following the normal rules of numpy. The above means, in the 0-th dimension (axis), we have two different assumptions loc=1, and loc=3. The 0-th dimension is the dimension of the rabbit species. And the parameters are simply broadcasted to match the shapes of any other dimension.

The specification loc=[1,3] would be understood as [[1,3]] and be interpreted as the experiment dimension. If the dimensions have different shapes (i.e. different number of unique elements), this will result in an error. If the dimensions have the same size, this will lead to serious problems, because the above will work, but lead to incorrect results. 

Ideally, the dimensionality is so low that you can be specific about the priors. I.e.: loc=[[1,1,1],[3,3,3]]. This of course expects you know about the dimensional sizes of the prior (i.e. the unique elements of the dimensions). 
```

We can use the hyperprior and define a prior `alpha` which actually parameterizes our model.

```python
from pymob.sim.parameters import Param
alpha = Param(
    value=0.5, free=True, hyper=False,
    prior="lognorm(s=0.1,scale=alpha_species[rabbit_species_index, experiment_index],dims=('id',))"
)
```

This prior samples from a [Lognormal distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm), but the scale (log-mean) is parameterized by the hyper prior `alpha_species`. In addition, the indices `rabbit_species_index` and `experiment_index` are responsible for selecting the samples from the hyper-prior `alpha_species` so that they correctly map to the ids of the different observations. Pymob automatically creats these indices for you, if you provide the experimental layout. 


```{admonition} Parsing priors as strings
:class: hint
The prior is currently passed as a string, which is somewhat inconvenient, but the problem is that in more complex distributions, some variables are only known when they are sampled.

This means we have to postpone evaluation of these priors to a later point. `Pymob` handles this by parsing the string into an Abstract Syntax Tree, where only general syntax problems are detected. Any problems with the indices, dimensions will only occurr, when samples are taken from the prior.
```

#### Generating the experimental layout

```python
sim.define_observations_replicated_multi_experiment(n=12) 
y0 = sim.parse_input("y0", drop_dims=["time"])
sim.model_parameters["y0"] = y0
```

#### Sampling from the generated distribution

```python
from pymob.inference.scipy_backend import ScipyBackend

inferer = ScipyBackend(simulation=sim)
theta = inferer.sample_distribution()
```

```{admonition} TODO
:class: attention
- [ ] Create IO tools for importing existing models and data 
- [ ] Write documentation with examples
```