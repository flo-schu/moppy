import json

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from pymob.simulation import SimulationBase
from pymob.inference.base import InferenceBackend
from pymob.utils.errors import import_optional_dependency

extra = "'pymoo' dependencies can be installed with pip install pymob[pymoo]"
pymoo = import_optional_dependency("pymoo", errors="warn", extra=extra)
if pymoo is not None:
    import pathos.multiprocessing as mp
    from pymoo.algorithms.moo.unsga3 import UNSGA3
    from pymoo.core.problem import ElementwiseProblem, Problem
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.termination.default import DefaultMultiObjectiveTermination


class PymooBackend:
    def __init__(
        self, 
        simulation: SimulationBase,
    ):
        self.simulation = simulation
        self.config = self.simulation.config
        self.pool: mp.Pool = None  # type: ignore
        self.distance_function = self.distance_function_parser()
        self.transform = self.variable_parser()
        self.problem = OptimizationProblem(backend=self)

    
    def distance_function_parser(self):
        def f(x):
            evaluator = self.simulation.dispatch(theta=x)
            evaluator()
            obj_name, obj_value = self.simulation.objective_function(
                results=evaluator.results
            )
            return obj_value
        return f

    def variable_mapper(self, x):
        names = self.simulation.model_parameter_names
        return {n:x_i for n, x_i in zip(names, x)}

    def variable_parser(self):
        variables = self.simulation.free_model_parameters

        bounds = []
        names = []
        for v in variables:
            if v.min is None or v.max is None:
                raise ValueError(
                    f"Bounds are not fully defined in Param({v}). Bounds (min, max) "
                    "must be defined when using the pymoo backend.")
            bounds.append([v.min, v.max])
            names.append(v.name)

        bounds = np.array(bounds).T
        scaler = MinMaxScaler().fit(bounds)

        # check that parameter values and names match, so variable_mapper 
        # can be safely called
        assert names == self.simulation.model_parameter_names

        def transform(X_scaled):
            X = scaler.inverse_transform(X_scaled)
            return map(self.variable_mapper, X)

        return transform
    
    def store_results(self, results):
        params = list(self.transform(X_scaled=np.array(results.X, ndmin=2)) )[0]

        res_dict = {
            "f": results.f,
            "cv": results.cv,
            "X": params
        }
        
        file = f"{self.config.case_study.output_path}/pymoo_params.json"
        with open(file, "w") as fp:
            json.dump(res_dict, fp, indent=4)

        print(f"written results to {file}")

    def run(self):
        """Implements the parallelization in pymoo"""
        n_cores = self.config.multiprocessing.n_cores
        print(f"Using {n_cores} CPU cores")

        if n_cores == 1:
            self.optimize()
        elif n_cores > 1:
            with mp.ProcessingPool(n_cores) as pool:
                self.pool = pool  # type: ignore # assign pool so it can be accessed by _evaluate
                self.optimize()


    def optimize(self):

        reference_directions = get_reference_directions(
            name="energy", 
            n_dim=self.config.inference.n_objectives, 
            n_points=self.config.inference_pymoo.population_size,
            seed=self.config.simulation.seed + 1
        )

        algorithm = UNSGA3(
            ref_dirs=reference_directions,
            pop_size=self.config.inference_pymoo.population_size
        )

        termination = DefaultMultiObjectiveTermination(
            xtol=self.config.inference_pymoo.xtol,
            cvtol=self.config.inference_pymoo.cvtol,
            ftol=self.config.inference_pymoo.ftol,
            n_max_gen=self.config.inference_pymoo.max_nr_populations,
        )

        # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
        algorithm.setup(
            problem=self.problem, 
            termination=termination,
            seed=self.config.simulation.seed, 
            verbose=True,
        )


        # until the algorithm has no terminated
        while algorithm.has_next():

            # ask the algorithm for the next solution to be evaluated
            pop = algorithm.ask()

            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            algorithm.evaluator.eval(self.problem, pop) # type: ignore

            # returned the evaluated individuals which have been evaluated or even modified
            algorithm.tell(infills=pop)

            # self.post_processing(pop=pop)

        # obtain the result objective from the algorithm
        res = algorithm.result()

        # save the results
        self.results = res
        # TODO: Store results to idata (although it seems pointless),
        # extra dimensions for individuals and generations, one draw nevertheless

        self.store_results(results=res)

    def post_processing(self, pop):
        F = pop.get("F")
        X = pop.get("X")

        f_min = F.min()
        x_min = X[np.where(F[:,0] == f_min)]
        # print(
        #     f"Generation {gen}: f_min={f_min}, x_min={x_min}",
        #     flush=True
        # )

    def load_results(self):
        with open(f"{self.config.case_study.output_path}/pymoo_params.json", "r") as fp:
            self.result = json.load(fp)
            
    def plot_predictions(
            self, data_variable: str, x_dim: str, ax=None, subset={}, 
            upscale_x=True
        ):
        obs = self.simulation.observations.sel(subset)
        x_old = self.simulation.coordinates[x_dim].copy()
        
        nan_obs = obs[data_variable].isnull().all(dim=x_dim)

        if x_dim is not None:    
            if upscale_x:
                self.simulation.coordinates[x_dim] = np.linspace(
                    x_old.min(),
                    x_old.max(),
                    1000
                )
            else:
                raise NotImplementedError("x_new must be a 1D np.ndarray")
            
        Y = self.simulation.evaluate(self.result["X"])
        results = self.simulation.results_to_df(results=Y)

        if ax is None:
            ax = plt.subplot(111)
        
        ax.plot(
            results[x_dim].values, 
            results[data_variable].values.T[:, ~nan_obs], 
            color="black", lw=.8
        )

        ax.plot(
            obs[x_dim].values, 
            obs[data_variable].values[:, ~nan_obs], 
            marker="o", ls="", ms=3
        )
        
        ax.set_ylabel(data_variable)
        ax.set_xlabel(x_dim)

        # restore old coordinates
        self.simulation.coordinates[x_dim] = x_old

        return ax

if pymoo is not None:
    class OptimizationProblem(Problem):
        def __init__(self, backend: PymooBackend, **kwargs):
            self.backend = backend
            self.simulation = backend.simulation
            
            super().__init__(
                n_var=self.simulation.n_free_parameters, 
                n_obj=self.simulation.n_objectives, 
                n_ieq_constr=0, 
                xl=0.0, 
                xu=1.0,
                elementwise_evaluation=False,
                **kwargs
            )

        def _evaluate(self, X, out, *args, **kwargs):
            X_original_scale = self.backend.transform(X_scaled=X)       

            if self.backend.pool is None:
                F = list(map(self.backend.distance_function, X_original_scale))
            else:
                F = self.backend.pool.map(self.backend.distance_function, X_original_scale)

            out["F"] = np.array(F)