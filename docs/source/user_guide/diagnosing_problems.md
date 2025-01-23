# Diagnosing problems in models

For structural problems `pymob` should offer constructive error messages, before
problems occurr. Still during parameter inference and optimization many problems
can ocurr, the below is an unsorted list of typical problems and ways to diagnose
and fix such problems.

## NaN likelihoods with `JaxSolver`

The {class}`pymob.solvers.diffrax.JaxSolver` is highly efficient, but can be difficult to use.

+ In the {meth}`pymob.simulation.SimulationBase.dispatch_constructor` set `throw_exception=True` to make the solver fail if it runs into nans. This will make the solver fail loudly if there are problems. The default is that the solver returns infinities if there are problems. Numpyro can handle infinities in the probability functions consider the parameters impossible which can lead to severely biased estimates.
+ First make sure this problem is not related to the choice and the parameterization
  of the solver. In the {meth}`pymob.simulation.SimulationBase.dispatch_constructor`, try increasing `max_steps`, 













## Further reference

