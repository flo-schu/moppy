from typing import Iterable, Any
from typing_extensions import Protocol
import inspect
import json
import re
import sympy
import warnings
from configparser import ConfigParser

CONSTANTS = ConfigParser()
CONSTANTS.read("config/constants.cfg")

def read_config(config_file):
    with open(config_file, "r") as f:
        return json.load(f)

def simulation_io_adapter(input_config_file, input_events_file, output_path):
    """
    A simple adapter to adapt the configuration file to be usable with the
    case-study design for the example of experiment based simulation
    """
    
    cfg = read_config(config_file=input_config_file)

    cfg["experiment"]["eventfile"] = input_events_file
    cfg["simulation"]["output"] = output_path

    return cfg


def catch_patterns(expression_str):
    # AVOID USING THE DESING OF EXPRESSIONS. JUST USE SYMPY SYNTAX._FlagsType
    # THIS WILL BE MORE STABLE AND EASY TO ADAPT NEW CONCEPTS.
    # THE KEY IS NOT THE EXPRESSION BUT THE LOOKUP
    
    # tries to match array notation [0 1 2]
    pattern = r"\[(\d+(\.\d+)?(\s+\d+(\.\d+)?)*|\s*)\]"
    if re.fullmatch(pattern, expression_str) is not None:
        expression_str = expression_str.replace(" ", ",")
        return f"Array({expression_str})"

    # tries to match array notation [0,1,2]
    pattern = r'\[(\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?)*|\s*)\]'
    if re.fullmatch(pattern, expression_str) is not None:
        return f"Array({expression_str})"

    return expression_str

def lambdify_expression(expression_str):
    # check for parentheses in expression
    
    expression_str = catch_patterns(expression_str)

    # Parse the expression without knowing the symbol names in advance
    parsed_expression = sympy.sympify(expression_str, evaluate=False)
    free_symbols = tuple(parsed_expression.free_symbols)

    # Transform expression to jax expression
    args = [str(s) for s in free_symbols]
    func = sympy.lambdify(
        args, parsed_expression
    )

    return func, args

from typing import Mapping

def lookup_from(val: Any, collection: Iterable[Mapping]) -> Any:
    if isinstance(collection, dict):
        collection = list(collection.values())

    for obj in collection:
        if val in obj:
            return obj[val]
        else:
            continue

    return val

def lookup(val, *indexable_objects):
    for obj in indexable_objects:
        if val in obj:
            return obj[val]
        else:
            continue

    return val

def lookup_args(args, *objects_to_search):
    return {k: lookup(k, *objects_to_search) for k in args}

def get_return_arguments(func):
    ode_model_source = inspect.getsource(func)
    
    # extracts last return statement of source
    return_statement = ode_model_source.split("\n")[-2]

    # extract arguments returned by ode_func
    return_args = return_statement.split("return")[1]

    # strip whitespace and separate by comma
    return_args = return_args.replace(" ", "").split(",")

    return return_args

def dX_dt2X(expr: str):
    expr = expr.split("_dt", 1)[0]
    expr = expr.split("_dx", 1)[0]

    if len(expr) == 2:
        if expr[0] == "d":
            expr = expr[1]
            return expr

    raise NotImplementedError(
        "Derviatives returned by an ODE model should follow the form "
        "'dX_dt' or 'dX_dx' or 'dX' where 'X' denotes the variable."
    )