import ast
import warnings
import inspect
from typing import Optional, List, Dict, Tuple, Any, Union
from typing_extensions import Annotated

import scipy
import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, model_serializer, field_validator, model_validator
from pydantic.functional_validators import BeforeValidator
from numpydantic import NDArray, Shape
from nptyping import Float64, Int64

NumericArray = NDArray[Shape["*, ..."], (Float64,Int64)] # type:ignore

class Expression:
    """Random variables are context dependent. They may be dependent on other
    Variables, or datasets. In the config they represent an abstract structure,
    so they remain unevaluated expressions that must follow python syntax.
    Once, the context is available, the expressions can be evaluated by
    `Expression.evaluate(context={...})`.
    """
    def __init__(self, expression: Union[str, ast.Expression]):
        if isinstance(expression, str):
            self.expression = ast.parse(expression, mode="eval")
        elif isinstance(expression, ast.Expression):
            self.expression = expression
        else:
            self.expression = ast.Expression(expression)

        finder = UndefinedNameFinder()
        self.undefined_args = finder.find_undefined_names(self.expression)
        self.compiled_expression = compile(self.expression, '', mode='eval')

    def __repr__(self):
        return str(self)

    def __str__(self) -> str:
        return ast.unparse(self.expression)
    
    def evaluate(self, context: Dict = {}) -> Any:
        try:
            val = eval(self.compiled_expression, context)
            return val
        except NameError as err:
            raise NameError(
                f"{err}. Have you forgotten to pass a context?"
            )

class UndefinedNameFinder(ast.NodeVisitor):
    # powered by ChatGPT
    def __init__(self):
        self.defined_names = set()
        self.undefined_names = set()

    def visit_FunctionDef(self, node):
        # Function arguments are considered defined within the function
        for arg in node.args.args:
            self.defined_names.add(arg.arg)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            # If the name is being assigned to, add it to the defined names
            self.defined_names.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            # If the name is being used, check if it's defined
            if node.id not in self.defined_names:
                self.undefined_names.add(node.id)

    def find_undefined_names(self, expr):
        tree = ast.parse(expr, mode='exec')
        self.visit(tree)
        return self.undefined_names


class RandomVariable(BaseModel):
    """Basic infrastructure to parse priors into their components so that 
    they can be more easily parsed to other backends.
    Parsing to distributions needs more context:
    - other parameters (for hierarchical parameter structures)
    - data structure and coordinates to identify dimension sizes
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        validate_assignment=True,
        extra="forbid"
    )

    distribution: str
    parameters: Dict[str, Expression]
    obs: Optional[Expression] = None
    obs_inv: Optional[Expression] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RandomVariable):
            raise NotImplementedError("Only compare to Prior instance")
        
        return bool(
            self.distribution == other.distribution and
            self.parameters.keys() == other.parameters.keys() and
            all([str(self.parameters[k]) == str(other.parameters[k]) 
                for k in self.parameters.keys()])
        )

    @model_serializer(when_used="json", mode="plain")
    def model_ser(self) -> str:
        distribution = self.distribution
        parameters = dict_to_string(self.parameters, jstr=",")
        if self.obs is None:
            pass
        else:
            obs = str(self.obs).replace(" ","")
            parameters = parameters + f",obs={obs}"
        if self.obs_inv is None:
            pass
        else:
            obs_inv = str(self.obs_inv).replace(" ","")
            parameters = parameters + f",obs_inv={obs_inv}"
        return f"{distribution}({parameters})"

    @field_validator("distribution", mode="after")
    def check_distribution(cls, new_value, info, **kwargs):
        if new_value not in scipy_to_scipy:
            warnings.warn(
                f"The distribution '{new_value}' is not part of the scipy "+
                "distributions implemented in pymob. "+
                "This can lead to inconsistent behavior. "+
                "It is recommended to use the scipy distribution protocol "+
                "where possible. "+
                "https://docs.scipy.org/doc/scipy/reference/stats.html "+
                "It may also be possible that your distribution has not yet "+
                "been introduced into the pymob package "
            )
        return new_value.lower()

    @field_validator("parameters", mode="after")
    def check_parameters(cls, new_value, info, **kwargs):
        distribution = info.data.get("distribution")
        dist = scipy_to_scipy.get(distribution, (None, ))[0]
        if dist is None:
            return new_value
        
        if distribution == "deterministic":
            return new_value
        
        dist_args = () if dist.shapes is None else dist.shapes
        dist_params = ("loc", "scale", *dist_args)
        unmatched_params = [k for k in new_value.keys() if k not in dist_params]
        if len(unmatched_params) > 0:
            warnings.warn(
                f"The parameters '{unmatched_params}' did not follow the scipy "+
                "protocol. "+
                "This can lead to inconsistent behavior. "+
                "It is recommended to use the scipy distribution protocol "+
                "where possible. "+
                "https://docs.scipy.org/doc/scipy/reference/stats.html"
            )
        return new_value

def string_to_prior_dict(prior_str: str):
    # Step 1: Parse the string to extract the function name and its arguments.
    node = ast.parse(source=prior_str, mode='eval')
    
    if not isinstance(node, ast.Expression):
        raise ValueError("The input must be a valid Python expression.")
    
    # Extract the function name (e.g., 'ZeroSumNormal')
    func_name = node.body.func.id
    
    # Extract the keyword arguments
    kwargs = {}
    for kw in node.body.keywords:
        key = kw.arg  # Argument name (e.g., 'loc')
        # if this is a valid python expression it can be compiled, but
        # evaluated, it can only be if the respective arguments are present
        # value = compile(ast.Expression(kw.value), '', mode='eval')
        # value = eval(compile(ast.Expression(kw.value), '', mode='eval'), {"wolves": 2, "EPS": 2})  # Evaluate the value part
        value = Expression(kw.value)
        kwargs[key] = value
    
    obs = kwargs.pop("obs", None)
    obs_inv = kwargs.pop("obs_inv", None)

    # Step 3: Return the symbolic expression and the argument dictionary
    return {
        "distribution": func_name, 
        "parameters": kwargs, 
        "obs": obs,
        "obs_inv": obs_inv
    }

def to_rv(option: Union[str,RandomVariable,Dict]) -> RandomVariable:
    if isinstance(option, RandomVariable):
        return option
    elif isinstance(option, Dict):
        prior_dict = option
    else:
        prior_dict = string_to_prior_dict(option)

    return RandomVariable.model_validate(prior_dict, strict=False)


def dict_to_string(dct: Dict, jstr=" "):
    string_items = []
    for k, v in dct.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()

        expr = f"{k}={v}".replace(" ", "")
        string_items.append(expr)

    return jstr.join(string_items)



OptionRV = Annotated[
    RandomVariable, 
    BeforeValidator(to_rv), 
]

class Param(BaseModel):
    """This class serves as a Basic model for declaring parameters
    Including a distribution with optional depdendencies

    Parameters
    ----------

    value : float|NumericArray
        The parameter value. If it is not a 0-d array, float or 1-d array of 
        length one, it should be accompanied by a dimension for each axis in 
        the array. The array coordinates must be specified in the observation
        coordinates or in index coordinates

    dims : Tuple[str, ...]
        Dimensions of the parameter. If the batch dimension is not specified
        here, it will be automatically added in the Evaluator 
        (dispatch_constructor). If dims are specified here, they should be
        present in:
        1) dimension in sim.config.data_structure and sim.coordinates  
        2) dimension in sim.config.parameters (so here) and sim.coordinates  
        3) dimension in sim.indices 

    prior : Optional[RandomVariable]
        This is a string or pymob.sim.parameters.RandomVariable that specifies 
        the prior. Strings are automatically parsed to RandomVariables if the 
        syntax is correct. The prior should follow the specification of 
        scipy.stats
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        validate_assignment=True, 
        extra="forbid"
    )
    name: Optional[str] = None
    value: float|NumericArray = 0.0
    dims: Tuple[str, ...] = ()
    prior: Optional[OptionRV] = None
    min: Optional[float|NumericArray] = None
    max: Optional[float|NumericArray] = None
    step: Optional[float|NumericArray] = None
    hyper: bool = False
    free: bool = True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Param):
            raise NotImplementedError("Only compare to Param instance")
        
        return bool(
            self.name == other.name and
            self.dims == self.dims and
            np.all(self.value == other.value) and
            np.all(self.min == other.min) and
            np.all(self.max == other.max) and
            np.all(self.step == other.step) and
            self.prior == other.prior and
            self.hyper == other.hyper and
            self.free == other.free
        )
  
    @model_validator(mode="after")
    def post_update(self):
        shape_value = np.array(self.value, ndmin=1).shape
        if max(shape_value) > 1 and len(self.dims) == 0:
            warnings.warn(
                f"Declaring parameter values of a shapes {shape_value} > 1"+
                "without specifiying Param(..., dims=(...)) is dangerous."+
                "If the dimension should represent only the batch dimension (e.g.)"+
                "the replicate dimension, you can simply add it here."+
                "If the dimension is part of the model, you should absolutely"+
                "specify it along with one of the following options: "+
                "1) dimension in sim.config.data_structure and sim.coordinates "+
                "2) dimension in sim.config.parameters and sim.coordinates "+
                "3) dimension in sim.indices "+
                "an index or a coordinate and DataVariable."+
                "If this dimension is not somehow part of the datastructure of the"+
                "simulation consider if you really need it."
            )

        return self
    
    def to_xarray(self, coordinates):
        """Converts the parameter to an xarray based on the dimensional 
        structure and adds additional info (prior, starting values, ...) 
        as metadata.
        """

        coords = {d:list(coordinates[d]) for d in self.dims}
        shape = tuple([len(c) for _, c in coords.items()])
        value = np.broadcast_to(np.array(self.value), shape)
        arr = xr.DataArray(value, dims=self.dims, coords=coords)

        for key, value in self.model_dump().items():
            if value is None: 
                continue

            if key == "prior":
                arr.attrs["prior"] = self.prior.model_dump_json().strip('"') #type:ignore
            
            elif key in ["value", "dims"]:
                continue
            
            else:
                arr.attrs[key] = value

        return arr


scipy_to_scipy = {
    # Continuous Distributions
    "norm": (scipy.stats.norm, {}),
    "normal": (scipy.stats.norm, {}),
    "expon": (scipy.stats.expon, {}),
    "exponential": (scipy.stats.expon, {}),
    "uniform": (scipy.stats.uniform, {}),
    "beta": (scipy.stats.beta, {}),
    "gamma": (scipy.stats.gamma, {}),
    "lognorm": (scipy.stats.lognorm, {}),
    "lognormal": (scipy.stats.lognorm, {}),
    "chi2": (scipy.stats.chi2, {}),
    "pareto": (scipy.stats.pareto, {}),
    "t": (scipy.stats.t, {}),
    "cauchy": (scipy.stats.cauchy, {}),
    "weibull_min": (scipy.stats.weibull_min, {}),
    "weibull_max": (scipy.stats.weibull_max, {}),
    "gumbel_r": (scipy.stats.gumbel_r, {}),
    "gumbel_l": (scipy.stats.gumbel_l, {}),
    "exponweib": (scipy.stats.exponweib, {}),
    "exponpow": (scipy.stats.exponpow, {}),
    "gamma": (scipy.stats.gamma, {}),
    "logistic": (scipy.stats.logistic, {}),
    "norminvgauss": (scipy.stats.norminvgauss, {}),
    "kstwobign": (scipy.stats.kstwobign, {}),
    "halfnorm": (scipy.stats.halfnorm, {}),
    "halfnormal": (scipy.stats.halfnorm, {}),
    "fatiguelife": (scipy.stats.fatiguelife, {}),
    "nakagami": (scipy.stats.nakagami, {}),
    "wald": (scipy.stats.wald, {}),
    "gompertz": (scipy.stats.gompertz, {}),
    "genextreme": (scipy.stats.genextreme, {}),
    "levy": (scipy.stats.levy, {}),
    "levy_stable": (scipy.stats.levy_stable, {}),
    "laplace": (scipy.stats.laplace, {}),
    "loggamma": (scipy.stats.loggamma, {}),
    "vonmises": (scipy.stats.vonmises, {}),
    "pareto": (scipy.stats.pareto, {}),
    "powerlaw": (scipy.stats.powerlaw, {}),
    "rayleigh": (scipy.stats.rayleigh, {}),
    "rice": (scipy.stats.rice, {}),
    "semicircular": (scipy.stats.semicircular, {}),
    "triang": (scipy.stats.triang, {}),
    "truncexpon": (scipy.stats.truncexpon, {}),
    "truncnorm": (scipy.stats.truncnorm, {}),
    "truncnormal": (scipy.stats.truncnorm, {}),
    "tukeylambda": (scipy.stats.tukeylambda, {}),
    "uniform": (scipy.stats.uniform, {}),
    "wrapcauchy": (scipy.stats.wrapcauchy, {}),
    
    # Discrete Distributions
    "binom": (scipy.stats.binom, {}),
    "bernoulli": (scipy.stats.bernoulli, {}),
    "geom": (scipy.stats.geom, {}),
    "hypergeom": (scipy.stats.hypergeom, {}),
    "poisson": (scipy.stats.poisson, {}),
    "skellam": (scipy.stats.skellam, {}),
    "nbinom": (scipy.stats.nbinom, {}),
    "logser": (scipy.stats.logser, {}),
    "planck": (scipy.stats.planck, {}),
    "boltzmann": (scipy.stats.boltzmann, {}),
    "randint": (scipy.stats.randint, {}),
    "zipf": (scipy.stats.zipf, {}),
    "dlaplace": (scipy.stats.dlaplace, {}),
    "yulesimon": (scipy.stats.yulesimon, {}),
    "hypergeom": (scipy.stats.hypergeom, {}),
    "betabinom": (scipy.stats.betabinom, {}),
    "nbinom": (scipy.stats.nbinom, {}),
    "binom": (scipy.stats.binom, {}),
    "randint": (scipy.stats.randint, {}),
    "boltzmann": (scipy.stats.boltzmann, {}),
    "planck": (scipy.stats.planck, {}),
    "logser": (scipy.stats.logser, {}),
    "zipf": (scipy.stats.zipf, {}),
    "dlaplace": (scipy.stats.dlaplace, {}),
    "skellam": (scipy.stats.skellam, {}),
}
