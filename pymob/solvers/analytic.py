from pymob.solvers.base import mappar

def solve_analytic_1d(model, parameters, dimensions, coordinates, data_variables, seed=None):
    """Solves an anlytic function for all coordinates in the first data 
    dimension of the model

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
    
    model_args = mappar(
        model, 
        parameters["parameters"], 
        exclude=["t", "x"] + dimensions,
        to="dict"
    )
    
    x = coordinates[dimensions[0]]

    model_results = []
    results = model(x, **model_args)
    model_results.append(results)
    
    return {data_var:y for data_var, y in zip(data_variables, model_results)}
