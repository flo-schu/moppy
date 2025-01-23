# Simulation

The modelling toolkit offers a variety of algorithms that can be used on a large number of models. Recognizing that simulation, optimization/calibration, parameter inference, sensitivity analysis, validation, etc. require similar workflows and have common input and output datastreams allows the abstraction of shared processes into a generic Simulation class, which can be reused for the listed purposes.

```{admonition} Outdated documentation
:class: danger
The description of the simulation components is outdated. Especially the components `parameterize` and `set_coordinates` are now handled automatically, when parameters are defined, but they can still be used without problems.
```

## Simulation components

Any simulation has recurring components. Facilitating the transfer of information between those components is key to analyze a described model with different tools. In this framework, simulations are defined as classes which require the definition of methods that define the simulation.

```python
# sim.py

from timepath import SimulationBase
from mod import lotka_volterra, solve_jax
import plot  # imports the plot module which contains plotting functions
import prob  # imports custom probability models for bayesian inference


class Simulation(SimulationBase):
    solver: Callable = solve_jax  # a function f(t, x, θ) = y
    model: Callable = lotka_volterra  # an ODE term df/dt = ... to be used with the solver

    # import extra modules with outsourced code to keep the Simulation class
    # compact
    prob = prob   # probability models that go beyond automatically generated models
    mplot = plot  # plot methods can now be referenced with 
                  # self.mplot.my_beautiful_plot(self, ...)

    
    def initialize(self, input: List):
        # define the model parameters (can of course be also loaded from an 
        # input file)
        self.model_parameters["parameters"] = dict(
            alpha = 0.1,  # Prey birth rate
            beta = 0.02,  # Rate at which predators decrease prey population
            gamma = 0.3,  # Predator reproduction rate
            delta = 0.01,  # Predator death rate
        )

        # define the initial populations
        self.model_parameters["y0"] = np.array([40, 9])  # initial population of prey and predator

        # used to define observations and any other information needed by the solver
        self.observations = xr.load_dataset(input[1])
        self.observations["wolves"] = self.observations["wolves"] + 1e-8
        self.observations["rabbits"] = self.observations["rabbits"] + 1e-8

    @staticmethod
    def parameterize(free_parameters: Dict, model_parameters: Dict):
        # receives a dictionary of free_parameters from a simulation evaluation
        # and uses the free parameters to update the model_parameters dictionary,
        # which can contain additional information passed to the solver
        # can be used to modify parameters, before they are passed to the 
        # solver

        # Initial conditions and parameters
        y0 = model_parameters["y0"]
        parameters = model_parameters["parameters"]

        # mapping of parameters *theta* to the model parameters accessed by
        # the solver. This task is necessary for any model 
        parameters.update(free_parameters)
        
        return dict(y0=y0, parameters=parameters)



    def set_coordinates(self, input):
        # the model is integrated over the dimension time

        # e.g.
        t_start = 0
        t_end = 200
        t_step = 0.1
        time = np.arange(t_start, t_end, t_step)

        return time
    

    def plot(self, results):
        # while plot methods are not required to simply run a simulation they
        # are very helpful for using the interactive backend. These plots
        # are always updated when parameters of the plots are changed.
        # If a JAX based solver is used (compiled to highly efficient code)
        # then the updates can be in real time depending on the model complexity
        fig = self.mplot.plot_trajectory(results)
        fig.savefig(f"{self.output_path}/trajectory.png")
```

The `Simulation` class is not restricted to these methods, any number of methods can be added that customize the behavior, add plot methods, or any other function. 

Before the entire architecture and the different methods of the `Simulation` are explained one by one in greater detail, let's take a look at how to run a simulation and get started!

```python
from pymob.utils.store_file import prepare_casestudy

# load the configuration file of the case study by providing the name of the 
# case study directory and the name of the scenario directory. This assumes that
# all case studies are contained in a directory "case_studies". 
#
# In this example we have the following directory tree:
# project_directory (working directory)
#  ├─ case_studies
#  │   ├─ lotka_volterra_case_study
#  │   │   ├─ scenarios
#  │   │   │   ├─ test_scenario
#  │   │   │   │   └─ settings.cfg
#  │   |   |   └─ test_scenario_2
#  │   │   │   │   └─ settings.cfg
#  |   |   ├─ sim.py
#
config = prepare_casestudy(
    case_study=("lotka_volterra_case_study", "test_scenario"), 
    config_file="settings.cfg",
    pkg_dir="case_studies"
)

# loads the defined Simulation class from sim.py at the root of the case-study
# directory
from sim import Simulation

# create a Simulation object by passing the configuration file to the Simulation
# class defined in sim.py
sim = Simulation(config)

# execute a simulation with a 
func = sim.dispatch(theta={"alpha": 0.5})  # initiate the simulation
func()  # run the simulation (takes more time for the first run)
func.results  # obtain the results

# plot the results with the plotting function defined in the Simulation class
sim.plot(results=f.results)

```


### Model and the solver. Information flow in the Simulation class when using the dispatch API

The moment a simulation is dispatched, an instance of the `Evaluator` class is assembled with the specified parameters $\theta$. Next to the model, solver, and parameters, the `Evaluator` obtains a number additional attributes from the Simulation instance. These are:

+ `dimensions`: A list of the dimensions of the simulations
+ `n_ode_states`: The number of ODE states tracked
+ `var_dim_mapper`: A list of variables and their associated dimensions. This is relevant for simulations, where not all data variables have the same dimensional structure
+ `data_structure`: Similar to the var_dim_mapper, but additionally contains the coordinates
+ `coordinates`: The coordinates of each dimension in a dict
+ `data_variables`: The data variables of the simulation
+ `stochastic`: Whether the model is a stochastic or a deterministic model
+ `indices`: Indices, which should be used to map potentially nested parameters to a flat array for batch processing the simulations, by default {}
+ `post_processing`: A function that takes a dictionary of simulation results and  parameters as an input and adds new variables to the results,  by default None, meaning that no post processing of the ODE solution is performed

These additional input can be *magically* picked up by the solver if the terms
are used as function arguments of the solver function.

```python
def solver(model, parameters, coordinates, indices):
    ...
```

if extra arguments are provided to the solve function, then they must be passed to the dispatch call of the Evaluator.

```python
def solver(model, parameters, my_extra_argument):
    ...

...

evaluator = sim.dispatch(theta=..., my_extra_argument="foo")
```

Note that theoretically no ODE model is needed. If you have a deterministic solution to your problem, or you have an entirely different model (e.g. a numeric individual based model), than you can ignore the model and specify all you need for solving a problem in the solver

```python
def solver(parameters, coordinates):
    x = coordinates["x"]
    a = parameters["a"]
    b = parameters["b"]

    y = a * x + b

    return {"y": y}
```

Back to the `Evaluator` instance. It has a `__call__` method, which is invoked, when the evaluator instance is called `evaluator()`. This runs the solve method with all specified arguments and stores the raw results to an attribute `Y`. When `evaluator.results` is accessed, the results dictionary returned by the solver is reshaped to an `xarray.Dataset`, which maps 1:1 to the observations with respect to the dimensionality of the output.

Note that the coordinates may differ, unless the coordinates of the observations are used as simulation coordinates.

### Model parameters

Model parameters are a critical part of the model. Changing model parameters
changes the output of a model. Parameters are involved in many typical tasks
of model development:

- parameter calibration (or optimization)
- parameter inference (slightly different operation)
- senstivity analysis

In addition to the model parameters, the `parameterize` method also returns 
the initial conditions of the simulation. 

```python
@staticmethod
def parameterize(free_parameters, model_parameters):
    # Initial conditions and parameters

    y0 = model_parameters["y0"]
    parameters = model_parameters["parameters"]

    # mapping of parameters *theta* to the model parameters accessed by
    # the solver. This task is necessary for any model 
    parameters.update(free_parameters)

    return dict(y0=y0, parameters=parameters)
```

When the simulation is initialized, `model_parameters` may be pre-specified.
like this 

```python
def initialize(self, input):
    ...
    self.model_paramaters["y0"] = ...
    self.model_parameters["parameters"] = dict(fixed_param=2.34)
    ...
```

This attribute `model_parameters` is then locked to the parameterize method by
using the python functools function `partial`

```py
def __init__(self, ...):
    ...
    self.parameterize = partial(self.parameterize, model_parameters=self.model_parameters)
```

This means in the `initialize` function of your simulation you can set fixed
parameters to the model. Free parameters specified in the dispatch call 
`sim.dispatch(theta=dict(...)) will then arrive as free_parameters in the 
parameterize call. If this API seems a bit complicated, it is probably true.
At the time of writing this, it seemed like a good and easy option. This will
be re-evaluated in the future and the API might change.

### Simulation coordinates

The model coordinates describe the dimensions over which the model is solved.
In a simple model only one coordinate is present. This dimension can be the
time dimension.

A very simple example:

```python
def set_coordinates(self, input)
    time = np.linspace(0, 100, 50)
    return time
```

Models can also have a batch dimension, if it is part of the model that multiple
replicates are simulated as one trial of an experiment.

```python
def set_coordinates(self, input)
    time = np.linspace(0, 100, 50)
    sample = [1,2,3]
    return time, sample
```

Any number of dimension can be specified as long as those dimensions are also
named in the same order in the `settings.cfg` file

```conf
[simulation]
dimensions = time sample
```
### Observations

Observations must have the same form as the model output

## Input / Output 

As a uniform exchange format `netcdf` is enforced by the package. In python `netcdf` files are handled by the `xarray` package. For convenience, data_variables of interest can be saved as a number of output formats. However, higher dimesnional datasets with more coordinates than e.g. time, must be aggregated over the remaining dimensions, before they can be processed to .csv files
