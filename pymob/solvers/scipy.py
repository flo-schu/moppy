from scipy.integrate import solve_ivp

from pymob.solvers.base import mappar

def solve_ivp_1d(model, parameters, coordinates, data_variables):
    """Initial value problems always need the same number of recurrent arguments

    - parameters: define the model
    - y0: set the initial values of the ODE states
    - coordinates: are needed to know over which values to integrate
    - seed: In case stochastic processes take place inside the model this is necessary
    
    In order to make things explicit, all information which is needed by the
    model needs to be specified in the function signature. 
    This also makes the solvers functionally oriented, a feature that helps the
    usability of models accross inference frameworks. Where functions should not
    have side effects.

    Additionally, passing arguments via the signature makes it easier to write
    up models in a casual way and only later embed them into more regulated
    structures such as pymob

    """
    odeargs = mappar(model, parameters["parameters"], exclude=["t", "y"])
    t = coordinates["time"]
    results = solve_ivp(
        fun=model,
        y0=parameters["y0"],
        t_span=(t[0], t[-1]),
        t_eval=t,
        args=odeargs,
    )
    return {data_var:y for data_var, y in zip(data_variables, results.y)}
