# Pymob

![testing](https://github.com/flo-schu/pymob/actions/workflows/python-test.yml/badge.svg)
![build](https://github.com/flo-schu/pymob/actions/workflows/python-build.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pymob/badge/?version=latest)](https://pymob.readthedocs.io/en/latest/?badge=latest)

Pymob is a **Py**thon based **mo**del **b**uilding platform. It abstracts repetitive
tasks in the modeling process so that you can focus on building models, asking questions to the real world and learn from observations.

The idea of `pymob` originated from the frustration with fitting complex models to complicated datasets (missing observations, non-uniform data structure, non-linear models, ODE models). In such scenarios a lot of time is spent matching observations with model results.

The main strength of `pymob` is to provide a uniform interface for describing models and using this model to fit a variety of state-of-the-art optimization and inference algorithms on it.

Currently, supported inference backends are:
- interactive (interactive backend in jupyter notebookswith parameter sliders)
- numpyro (bayesian inference and stochastic variational inference)
- pyabc (approximate bayesian inference)
- pymoo (experimental! multi-objective optimization)

## A word on nomenclature

::note:: Parameter estimation, parameter inference and parameter optimization have the same meaning within the scope of this package. While there are distinctions between parameter optimization, inference and estimation, pymob ignores these differences when talking about the process of using observations to describe relationships between cause and effect. For consistency we borrow the term "infer" to describe this process. Whenever, parameters are inferred or parameter inference is done in the scope of this work, it refers to the process of fitting a model to data and thus identify the processes in the model, described by model parameters. 

Using the same term for similar processes, breaks down barriers between frequentist and bayesian approaches. Bayesian and frequentist ideologies are sometimes fiercly fought, but much of their reasoning is based on the same fundament. We acknowledge the differences, but appreciate that for some models, frequentist approaches are more suitable and for others bayesian approaches bring advantages. By using the same interface for both **inference** tools, the user can seamlessly switch between *both worlds* and explore the differences and advantages of either tool.

Because this is what they are. Tools. Tools, to learn from data and ask meaningful questions!

## Installation

To install pymob `pip install pymob`

Backends can be installed with e.g. `pip install pymob[numpyro]`

Pymob is under active development. It is used and developed within multiple projects simultaneously and in order to maintain a consistent release history, the main work is done in project-branches which contain the most cutting-edge features. These can always checked out locally, but may not be working correctly. Instead, it is recommended to install alpha-versions. 

E.g. `pip install pymob==0.3.0a5` which is the 5th alpha release of a project branch that was based on pymob v0.2.x.

::warning:: It may be possible that different projects release on the same minor version. In this case the release notes (https://github.com/flo-schu/pymob/releases) should be reviewed to see which project it refers to.


## Documentation

The documentation is available on https://pymob.readthedocs.io/en/latest/


## Roadmap

`pymob` is a work in progress. As it is actively used in more projects the package will be maturing and delivering a smoother user experience. The design principle of pymob is that it will incorporate features that are repeatedly needed in research projects in order to keep the API lean and focus on providing code that matters and helps.

The future plans for pymob can be viewed in https://github.com/users/flo-schu/projects/10/views/10

## Getting started

In 0.3.0 the use is still a bit bumpy. The easiest way is currently to copy the lotka_volterra_case_study and get started from there.

In 0.4.0 a new configuration backend based on `pydantic` will be added. This will considerably ease the use of `pymob` as the API will be supported with rich type hints and helpful error messages.