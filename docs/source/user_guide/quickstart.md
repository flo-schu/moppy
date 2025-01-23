# Pymob quickstart

## Initialize a simulation

In pymob a Simulation object is initialized by calling the {class}`pymob.simulation.SimulationBase` class from the simulation module.


```python
from pymob.simulation import SimulationBase

sim = SimulationBase()
```

```{admonition} Configuring the simulation
:class: hint
Optionally, we can configure the simulation at this stage with 
`sim.config.case_study.name = "linear-regression"`, `sim.config.case_study.scenario = "test"`, and many more options. 
```

## Define a model 

Let's investigate a linear regression as the most simple task.


```python
def linreg(x, a, b):
    return a + x * b
```

So we assume that this model describes our data well. So we add it to the simulation


```python
sim.model = linreg
```


## Defining a solver

In our case the model gives the exact solution of the model.
Solvers in pymob are callables that need to return a dictionary of results mapped to the data variables



```python
from pymob.sim.solvetools import solve_analytic_1d
sim.solver = solve_analytic_1d
```


## Generate artificial data

In the real world, you will have measured a dataset. For demonstration, we define parameters $theta$, that we assume describe the true data generating process and generate observations $y$. Then we
generate data for $x$ on [-5, 5] and add random noise with a standard deviation of $\sigma_y = 1$.


```python
import numpy as np
rng = np.random.default_rng(1)

# define the coordinates of the x-dimension to generate data for
x = np.linspace(-5, 5, 50)

# define a set of parameters θ
theta = dict(a=0, b=1, sigma_y=1)

# then simulate some data and add some noise
y = linreg(x=x, a=theta["a"], b=theta["b"])
y_noise = rng.normal(loc=y, scale=theta["sigma_y"])
```

## The pymob magic 🪄

So far we have not done anythin special. Pymob exists, because wrangling dimensions of input and output data, nested data-structures, missing data is painful. We avoid most of the mess by using `xarray` as a common input/output format. So we have to transform our data into a `xarray.Dataset` and add it to the simulation.


```python
import xarray as xr

sim.observations = xr.DataArray(y_noise, coords={"x": x}).to_dataset(name="y")
```

    MinMaxScaler(variable=y, min=-5.690912333645177, max=5.891166954282328)


    /home/flo-schu/miniconda3/envs/damage-proxy/lib/python3.11/site-packages/pymob/simulation.py:211: UserWarning: `sim.config.data_structure.y = Datavariable(dimensions=['x'] min=-5.690912333645177 max=5.891166954282328 observed=True dimensions_evaluator=None)` has been assumed from `sim.observations`. If the order of the dimensions should be different, specify `sim.config.data_structure.y = DataVariable(dimensions=[...], ...)` manually.
      warnings.warn(


This worked 🎉 `sim.config.data_structure` will now give us some information about the layout of our data, which will handle the data transformations in the background.

We can give `pymob` additional information about the data structure of our observations and intermediate (unobserved) variables that are simulated. This can be done with `sim.config.data_structure.y = DataVariable(dimensions=["x"])`.
These information can be used to switch the dimensional order of the observations or provide data variables that have differing dimensions from the observations, if needed. But if the dataset is ordinary, simply setting `sim.observations` property with a `xr.Dataset` will be sufficient.

```{admonition} Scalers
:class: hint
We also notice a mysterious Scaler message. This tells us that our data variable has been identified and a scaler was constructed, which transforms the variable between [0, 1]. This has no effect at the moment, but it can be used later. Scaling can be powerful to help parameter estimation in more complex models.
```

## Parameterizing a model

Parameters are specified via the `FloatParam` or `ArrayParam` class. Parameters can be marked free or fixed depending on whether they should be variable during an optimization procedure.


```python
from pymob.sim.config import Param
sim.config.model_parameters.a = Param(value=0, free=False)
sim.config.model_parameters.b = Param(value=3, free=True)
# this makes sure the model parameters are available to the model.
sim.model_parameters["parameters"] = sim.config.model_parameters.value_dict
```


`sim.model_parameters` is a dictionary that holds the model input data. The keys it takes by default are `parameters`, `y0` and `x_in`. In our case, we have a analytic model and need only `parameters`. In situations, where initial values for variables are needed, they can be provided with `sim.model_parameters["y0"] = ...`. 

```{admonition} generating input for solvers
:class: hint
A helpful function to generate `y0` or `x_in` from observations is `SimulationBase.parse_input`, combined with settings of `config.simulation.y0`
```


## Running the model 🏃

The model is prepared with a parameter set and executed with 


```python
# put everything in place for running the simulation
sim.dispatch_constructor()

# run
evaluator = sim.dispatch(theta={"b":3})
evaluator()
evaluator.results
```

    /home/flo-schu/miniconda3/envs/damage-proxy/lib/python3.11/site-packages/pymob/simulation.py:426: UserWarning: The number of ODE states was not specified in the config file [simulation] > 'n_ode_states = <n>'. Extracted the return arguments ['a+x*b'] from the source code. Setting 'n_ode_states=1.
      warnings.warn(





<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:  (x: 50)
Coordinates:
  * x        (x) float64 -5.0 -4.796 -4.592 -4.388 ... 4.388 4.592 4.796 5.0
Data variables:
    y        (x) float64 -15.0 -14.39 -13.78 -13.16 ... 13.16 13.78 14.39 15.0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-18260049-f5d3-48a7-b557-c83d34c8b998' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-18260049-f5d3-48a7-b557-c83d34c8b998' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 50</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-4882e7e1-ee21-460e-a0e7-3db43f2c4912' class='xr-section-summary-in' type='checkbox'  checked><label for='section-4882e7e1-ee21-460e-a0e7-3db43f2c4912' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-5.0 -4.796 -4.592 ... 4.796 5.0</div><input id='attrs-eb777311-160c-478f-a6b1-ef06490e87c5' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-eb777311-160c-478f-a6b1-ef06490e87c5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1948d030-83f4-465b-87eb-7d6f16b2a390' class='xr-var-data-in' type='checkbox'><label for='data-1948d030-83f4-465b-87eb-7d6f16b2a390' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-5.      , -4.795918, -4.591837, -4.387755, -4.183673, -3.979592,
       -3.77551 , -3.571429, -3.367347, -3.163265, -2.959184, -2.755102,
       -2.55102 , -2.346939, -2.142857, -1.938776, -1.734694, -1.530612,
       -1.326531, -1.122449, -0.918367, -0.714286, -0.510204, -0.306122,
       -0.102041,  0.102041,  0.306122,  0.510204,  0.714286,  0.918367,
        1.122449,  1.326531,  1.530612,  1.734694,  1.938776,  2.142857,
        2.346939,  2.55102 ,  2.755102,  2.959184,  3.163265,  3.367347,
        3.571429,  3.77551 ,  3.979592,  4.183673,  4.387755,  4.591837,
        4.795918,  5.      ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-06d7afd9-c09a-47b2-bd3d-bf45d8c84bda' class='xr-section-summary-in' type='checkbox'  checked><label for='section-06d7afd9-c09a-47b2-bd3d-bf45d8c84bda' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>y</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-15.0 -14.39 -13.78 ... 14.39 15.0</div><input id='attrs-ed40b7e8-4d6d-45cb-a6a0-51c897719eaf' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ed40b7e8-4d6d-45cb-a6a0-51c897719eaf' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-73ea477c-5957-46ed-b7b9-253181037456' class='xr-var-data-in' type='checkbox'><label for='data-73ea477c-5957-46ed-b7b9-253181037456' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-15.        , -14.3877551 , -13.7755102 , -13.16326531,
       -12.55102041, -11.93877551, -11.32653061, -10.71428571,
       -10.10204082,  -9.48979592,  -8.87755102,  -8.26530612,
        -7.65306122,  -7.04081633,  -6.42857143,  -5.81632653,
        -5.20408163,  -4.59183673,  -3.97959184,  -3.36734694,
        -2.75510204,  -2.14285714,  -1.53061224,  -0.91836735,
        -0.30612245,   0.30612245,   0.91836735,   1.53061224,
         2.14285714,   2.75510204,   3.36734694,   3.97959184,
         4.59183673,   5.20408163,   5.81632653,   6.42857143,
         7.04081633,   7.65306122,   8.26530612,   8.87755102,
         9.48979592,  10.10204082,  10.71428571,  11.32653061,
        11.93877551,  12.55102041,  13.16326531,  13.7755102 ,
        14.3877551 ,  15.        ])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-04c4f0d1-f4fb-4bc7-8b80-31750e84b13a' class='xr-section-summary-in' type='checkbox'  ><label for='section-04c4f0d1-f4fb-4bc7-8b80-31750e84b13a' class='xr-section-summary' >Indexes: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>x</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-3014f92c-ff4a-43e6-a648-92f179318c97' class='xr-index-data-in' type='checkbox'/><label for='index-3014f92c-ff4a-43e6-a648-92f179318c97' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([               -5.0,  -4.795918367346939,  -4.591836734693878,
        -4.387755102040816,  -4.183673469387755,  -3.979591836734694,
       -3.7755102040816326,  -3.571428571428571,   -3.36734693877551,
        -3.163265306122449, -2.9591836734693877, -2.7551020408163263,
       -2.5510204081632653, -2.3469387755102042,  -2.142857142857143,
       -1.9387755102040813, -1.7346938775510203, -1.5306122448979593,
       -1.3265306122448979, -1.1224489795918364, -0.9183673469387754,
       -0.7142857142857144, -0.5102040816326525, -0.3061224489795915,
       -0.1020408163265305,  0.1020408163265305,  0.3061224489795915,
        0.5102040816326534,  0.7142857142857144,  0.9183673469387754,
        1.1224489795918373,  1.3265306122448983,  1.5306122448979593,
        1.7346938775510203,  1.9387755102040813,  2.1428571428571432,
        2.3469387755102042,  2.5510204081632653,   2.755102040816327,
         2.959183673469388,   3.163265306122449,    3.36734693877551,
         3.571428571428571,   3.775510204081632,   3.979591836734695,
         4.183673469387756,   4.387755102040817,   4.591836734693878,
         4.795918367346939,                 5.0],
      dtype=&#x27;float64&#x27;, name=&#x27;x&#x27;))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-3c366ba3-e2f3-4dc3-9365-c1e556e5f4bc' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-3c366ba3-e2f3-4dc3-9365-c1e556e5f4bc' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



This returns a dataset which is of the exact same shape as the observation dataset, plus intermediate variables that were created during the simulation, if they are tracked by the solver.

Although this API seems to be a bit clunky, it is necessary, to make sure that simulations that are executed in parallel are isolated from each other.


## Estimating parameters 

We are almost set infer the parameters of the model. We add another parameter to also estimate the error of the parameters, We use a lognormal distribution for it. We also specify an error model for the distribution. This will be 

$$y_{obs} \sim Normal (y, \sigma_y)$$


```python
sim.config.model_parameters.sigma_y = Param(free=True , prior="lognorm(scale=1,s=1)")
sim.config.model_parameters.b.prior = "lognorm(scale=1,s=1)"

sim.config.error_model.y = "normal(loc=y,scale=sigma_y)"
```

```{admonition} numpyro distributions
:class: warning
Currently only few distributions are implemented in the numpyro backend. This API will soon change, so that basically any distribution can be used to specifcy parameters. 
```

Finally, we let our inferer run the paramter estimation procedure with the numpyro backend and a NUTS kernel. This does the job in a few seconds


```python
sim.set_inferer("numpyro")
sim.inferer.config.inference_numpyro.kernel = "nuts"
sim.inferer.run()

sim.inferer.idata.posterior
```

    Jax 64 bit mode: False
    Absolute tolerance: 1e-07


    Trace Shapes:     
     Param Sites:     
    Sample Sites:     
           b dist    |
            value    |
     sigma_y dist    |
            value    |
       y_obs dist 50 |
            value 50 |


      0%|                                                                                                                                                                       | 0/3000 [00:00<?, ?it/s]

    warmup:   0%|                                                                                                           | 1/3000 [00:01<1:14:16,  1.49s/it, 1 steps of size 1.87e+00. acc. prob=0.00]

    warmup:  16%|████████████████▍                                                                                         | 465/3000 [00:01<00:06, 407.50it/s, 7 steps of size 6.86e-01. acc. prob=0.79]

    warmup:  32%|██████████████████████████████████▎                                                                       | 970/3000 [00:01<00:02, 924.27it/s, 3 steps of size 1.91e+00. acc. prob=0.79]

    sample:  49%|██████████████████████████████████████████████████▌                                                    | 1471/3000 [00:01<00:01, 1484.75it/s, 15 steps of size 8.76e-01. acc. prob=0.92]

    sample:  66%|████████████████████████████████████████████████████████████████████▍                                   | 1973/3000 [00:01<00:00, 2063.90it/s, 3 steps of size 8.76e-01. acc. prob=0.91]

    sample:  82%|█████████████████████████████████████████████████████████████████████████████████████▋                  | 2473/3000 [00:01<00:00, 2619.91it/s, 3 steps of size 8.76e-01. acc. prob=0.91]

    sample:  99%|███████████████████████████████████████████████████████████████████████████████████████████████████████▏| 2975/3000 [00:02<00:00, 3131.43it/s, 3 steps of size 8.76e-01. acc. prob=0.91]

    sample: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 3000/3000 [00:02<00:00, 1433.31it/s, 1 steps of size 8.76e-01. acc. prob=0.91]

    


    
                    mean       std    median      5.0%     95.0%     n_eff     r_hat
             b      0.97      0.04      0.97      0.90      1.05   1684.80      1.00
       sigma_y      0.91      0.09      0.90      0.76      1.07   1148.32      1.00
    
    Number of divergences: 0





<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:  (chain: 1, draw: 2000)
Coordinates:
  * chain    (chain) int64 0
  * draw     (draw) int64 0 1 2 3 4 5 6 7 ... 1993 1994 1995 1996 1997 1998 1999
Data variables:
    b        (chain, draw) float32 1.013 1.031 1.046 ... 0.9507 0.8844 0.8844
    sigma_y  (chain, draw) float32 0.8514 1.115 0.8677 ... 0.7915 0.9756 0.9756
Attributes:
    created_at:     2025-01-19T08:19:25.912817+00:00
    arviz_version:  0.20.0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-f0278ca3-5669-4f64-8faa-13765aa467fc' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-f0278ca3-5669-4f64-8faa-13765aa467fc' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 1</li><li><span class='xr-has-index'>draw</span>: 2000</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-131f7a21-0349-4608-9178-a6d57f3d1bb6' class='xr-section-summary-in' type='checkbox'  checked><label for='section-131f7a21-0349-4608-9178-a6d57f3d1bb6' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-9c0fffca-53b7-4b48-a7af-3675eef96ffe' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9c0fffca-53b7-4b48-a7af-3675eef96ffe' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e966e10c-0145-4580-b2b9-793e32b71eba' class='xr-var-data-in' type='checkbox'><label for='data-e966e10c-0145-4580-b2b9-793e32b71eba' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 ... 1996 1997 1998 1999</div><input id='attrs-9aba4da6-1666-4233-b232-4fe359295f13' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9aba4da6-1666-4233-b232-4fe359295f13' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5e017ecb-309a-430a-83fd-1305e503887c' class='xr-var-data-in' type='checkbox'><label for='data-5e017ecb-309a-430a-83fd-1305e503887c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([   0,    1,    2, ..., 1997, 1998, 1999])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-00f941d3-f532-414a-8a6e-f8e1b531a61b' class='xr-section-summary-in' type='checkbox'  checked><label for='section-00f941d3-f532-414a-8a6e-f8e1b531a61b' class='xr-section-summary' >Data variables: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>b</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>1.013 1.031 1.046 ... 0.8844 0.8844</div><input id='attrs-c8d6cc3b-24ed-4e5d-be0a-95cb1dbef86c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-c8d6cc3b-24ed-4e5d-be0a-95cb1dbef86c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-cb0848c9-5fbd-4fb5-a8fb-6c3ecb7e6bbf' class='xr-var-data-in' type='checkbox'><label for='data-cb0848c9-5fbd-4fb5-a8fb-6c3ecb7e6bbf' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[1.012963 , 1.0309023, 1.0455396, ..., 0.9506697, 0.8844173,
        0.8844173]], dtype=float32)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>sigma_y</span></div><div class='xr-var-dims'>(chain, draw)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>0.8514 1.115 ... 0.9756 0.9756</div><input id='attrs-17148c9d-5d4e-43c6-b432-9a2b546ca1e3' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-17148c9d-5d4e-43c6-b432-9a2b546ca1e3' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-584c9731-1600-4c8d-9c54-ef64f03f4439' class='xr-var-data-in' type='checkbox'><label for='data-584c9731-1600-4c8d-9c54-ef64f03f4439' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[0.8514427, 1.1151575, 0.8676896, ..., 0.791502 , 0.9755601,
        0.9755601]], dtype=float32)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-29ad48b2-325b-4f79-ad0e-6ae8f1965e80' class='xr-section-summary-in' type='checkbox'  ><label for='section-29ad48b2-325b-4f79-ad0e-6ae8f1965e80' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>chain</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-fc4616bd-bc1c-42fe-82f1-d3445223f2cb' class='xr-index-data-in' type='checkbox'/><label for='index-fc4616bd-bc1c-42fe-82f1-d3445223f2cb' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([0], dtype=&#x27;int64&#x27;, name=&#x27;chain&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>draw</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-8e6a3515-1733-421b-8bfe-769e10285697' class='xr-index-data-in' type='checkbox'/><label for='index-8e6a3515-1733-421b-8bfe-769e10285697' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
       ...
       1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999],
      dtype=&#x27;int64&#x27;, name=&#x27;draw&#x27;, length=2000))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-a47fb2f2-95e7-4f48-9643-e609ea7daab4' class='xr-section-summary-in' type='checkbox'  checked><label for='section-a47fb2f2-95e7-4f48-9643-e609ea7daab4' class='xr-section-summary' >Attributes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2025-01-19T08:19:25.912817+00:00</dd><dt><span>arviz_version :</span></dt><dd>0.20.0</dd></dl></div></li></ul></div></div>



We can inspect our estimates and see that the parameters are well esimtated by the model. Note that we only get an estimate for $b$. This is because earlier we set the parameter `a` with the flag `free=False` this effectively excludes it from estimation and uses the default value, which was set to the true value `a=0`.


## Exporting the simulation and running it via the case study API

After constructing the simulation, all settings of the simulation can be exported to a comprehensive configuration file, along with all the default settings. This is as simple as 


```python
import os
sim.config.case_study.name = "quickstart"
sim.config.case_study.scenario = "test"
sim.config.create_directory("scenario", force=True)
sim.config.create_directory("results", force=True)

# usually we expect to have a data directory in the case
os.makedirs(sim.data_path, exist_ok=True)
sim.save_observations(force=True)
sim.config.save(force=True)
```

    Scenario directory exists at '/home/flo-schu/projects/pymob/docs/source/user_guide/case_studies/quickstart/scenarios/test'.
    Results directory exists at '/home/flo-schu/projects/pymob/docs/source/user_guide/case_studies/quickstart/results/test'.


The simulation will be saved to the default path (`CASE_STUDY/scenarios/SCENARIO/settings.cfg`) or to a custom path spcified with the `fp` keyword. `force=True` will overwrite any existing config file, which is the reasonable choice in most cases.

From there on, the simulation is (almost) ready to be executable from the commandline.

### Commandline API

The commandline API runs a series of commands that load the case study, execute the {meth}`pymob.simulation.SimulationBase.initialize` method and perform some more initialization tasks, before running the required job.

+ `pymob-infer`: Runs an inference job e.g. `pymob-infer --case_study=quickstart --scenario=test --inference_backend=numpyro`. While there are more commandline options, these are the two required 
