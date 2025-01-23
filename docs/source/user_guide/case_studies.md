# Case studies

Case studies are a principled approach to the *modelling* process. In essence, they are a simple template that contains building blocks for model and names and stores them in an intuitive and reproducible way.

Each case study consists of the following components:

```
lotka_volterra_case_study
  sim.py               # sets up the simulation
  mod.py               # (opt) outsources solver and model definitions
  data.py              # (opt) outsources data input
  plot.py              # (opt) outsources vizualizations

  scenarios
    scenario_A         # the scenario of the "lotka_volterra_case_study" 
      settings.cfg     # configuration for the case study and scenario
      simulation.cfg   # parameters of the simulation

  results
    scenario_A         # the results of "scenario_A"

  data                 # (optional) datafiles for data.py

  scripts              # (optional) evaluation scripts

  docs                 # (optional) documentation of the case study
```

While it is recommended to keep data, docs, results, scripts directories in each case study to keep a comprehensive and compact structure of the project, these can reside anywhere else. 

## Configuration


Settings files are created as conf `.cfg` files. These files are organized into the following sections.

```{admonition} Scripting API
:class: attention
Since `pymob-0.4.0` configurations can be specified in the scripting API and exported to config files from the Simulation instance. This makes configuration considerably more user friendly as the possible options are directly availble through type hints.
```

```conf
[case-study]
name = lotka_volterra_case_study
scenario = test_scenario
package = case_studies
modules = sim mod prob data plot
simulation = Simulation
observations = simulated_data.nc
logging = DEBUG

[simulation]
seed = 1

[data-structure]
rabbits = dimensions=[time] min=0 max=nan
wolves = dimensions=[time] min=0 max=nan

[inference]
objective_function = total_average
n_objectives = 1
EPS = 1e-8

[model-parameters]
alpha = value=0.5 min=0.1 max=5.0 prior=lognorm(s=0.1,scale=0.50) free=True
beta = value=0.02 min=0.005 max=0.2 prior=lognorm(s=0.1,scale=0.02) free=True

[error-model]
wolves = lognorm(scale=wolves+EPS,s=0.1)
rabbits = lognorm(scale=rabbits+EPS,s=0.1)

[multiprocessing]
cores = 1

[inference.pyabc]

[inference.pymoo]

[inference.numpyro]
gaussian_base_distribution = False
kernel = nuts
init_strategy = init_to_uniform
chains = 1
draws = 2000
warmup = 1000
```

### case-study configuration

Contains the information about the Simulation class to be used and the data and output directory. The paths are relative to the root directory (where the case study is launched)

### Simulation configuration

Contains the details about the simulation. If an ode model is used, it should be specified here. In that case also different solvers can be provided to the simulation. 

It is also possible to provide the solver directly which then executes the entire simulation. In this case the solver should be a wrapper around a possibly external simulation which returns the results as a dictionary with keys, corresponding to the data variables.

**Solver post processing** can be used if the results returned from the solver need some additional post processing before they can be compared with the observations. The method takes a dictionary as an input and potentially other arguments specified e.g. in the free-model-parameters or fixed-model-parameters. 

**y0** Initial values of the ODE model. Can be any list separated by whitespace that follows `sympy` syntax. E.g. `wolves=Array([2]) rabbits=rabbits`. This can be processed with `Simulation.parse_input(data=[observations], input="y0", drop_dims="time")`. This will create an array of the keys (wolves, rabbits) and broadcast the values along all coordinates of the observations, but retaining only the first value of the dropped dimensions if a variable (rabbits) was provided. This is very useful if some data_variables of the ODE model were observed and the initial value should be taken from the data.

**seed**. The seed is used to initate random processes for reproducibility. The behavior is still experimental.

### data-structure

This discribes the dimensions and dimensional order of the data. It optionally provides an interface for setting minima and maxima for data scaling for use in optimizers that perform better when working on scaled data.
In addition, also different dimensional order between observation and simulation results can be specified  with `dimensions_evaluator=[...]`

### model-parameters

Model parameters that are subject to parameter inference and can be varied in the parameter estimation process or in the interactive simulation. 
Model parmameters that remain fixed throughout the simulation.

### error-model

Error functions for comparing the simulation results to the data. These functions will also be parsed with `sympy` parsers. Still experimental

### inference

Options for adjusting the inference algorithms