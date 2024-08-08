from functools import partial
from typing import Literal
import xarray as xr
import numpy as np

from pymob import SimulationBase, Config
from pymob.solvers.diffrax import JaxSolver
from pymob.sim.config import DataVariable

# load the basic TKTD RNA Pulse case study and use as a parent class for the
# hierarchical model
config = Config()
config.case_study.name = "tktd_rna_pulse"
config.case_study.package = "case_studies"
config.import_casestudy_modules(reset_path=True)
from tktd_rna_pulse.sim import SingleSubstanceSim2

class NomixHierarchicalSimulation(SingleSubstanceSim2):
    def initialize(self, input):
        super().initialize(input)
        self.set_fixed_parameters(None)
        
        # configure JaxSolver
        self.solver = JaxSolver
        self.config.jaxsolver.batch_dimension = "id"

        # use_numpyro_backend can be written into the intiialize method
        # self.use_numpyro_backend()
        
    def use_numpyro_backend(self):
        # configure the Numpyro backend
        self.set_inferer("numpyro")
        self.config.inference_numpyro.user_defined_preprocessing = None
        self.inferer.preprocessing = partial(         # type: ignore
            sim.inferer.preprocessing,                # type: ignore
            ci_max=self.config.model_parameters.fixed_value_dict["ci_max"] 
        ) 

        # set the fixed parameters
        self.model_parameters["parameters"] = self.config.model_parameters\
            .fixed_value_dict

    def setup_data_structure_from_observations(self):
        self.setup()

        # select observations
        obs = [0]
        sim.observations = sim.observations.isel(id=obs)
        sim.observations.attrs["substance"] = list(
            np.unique(sim.observations.substance)
        )
        sim.set_y0()
        sim.indices["substance"] = sim.indices["substance"].isel(id=obs)


    def setup_data_structure_manually(
        self, 
        scenario: Literal[
            "data_structure_01_single_observation",
            "data_structure_02_replicated_observation",
            "data_structure_03_gradient_observation",
        ] = "data_structure_01_single_observation"
    ):
        self.config.case_study.scenario = scenario
        self.config.create_directory("results")
        self.config.create_directory("scenario")

        # mark existing data variables as unobserved
        for _, datavar in self.config.data_structure.all.items():
            datavar.observed = False
            datavar.min = np.nan
            datavar.max = np.nan

        # copy data structure from survival to lethality
        self.config.data_structure.lethality =\
            self.config.data_structure.survival # type:ignore

        if scenario == "data_structure_01_single_observation":
            self.define_observations_unreplicated()

        elif scenario == "data_structure_02_replicated_observation":
            self.define_observations_replicated()

        elif scenario == "data_structure_03_gradient_observation":
            self.define_observations_replicated_gradient()


        # set up coordinates
        self.coordinates["time"] = np.arange(0, 120)
        # self.coordinates["substance"] = "diuron"

        # define starting values
        self.config.simulation.y0 = [
            "cext=cext_nom", 
            "cint=Array([0])", 
            "nrf2=Array([1])", 
            "P=Array([0])", 
        ]

        y0 = self.parse_input("y0", reference_data=self.observations, drop_dims=["time"])
        self.model_parameters["y0"] = y0

        # define parameters

        # set the fixed parameters
        self.model_parameters["parameters"] = self.config.model_parameters\
            .fixed_value_dict

        # set up the solver
        self.config.simulation.solver = "JaxSolver"
        self.config.jaxsolver.batch_dimension = "id"

        self.validate()
        self.config.save(force=True)

    def decorate_results(self, results):
        """Convenience function to add attributes and coordinates to simulation
        results needed for other post-processing tasks (e.g. plotting)
        """
        results.attrs["substance"] = np.unique(results.substance)
        results = results.assign_coords({
            "cext_nom": self.model_parameters["y0"]["cext"]
        })
        return results

    def plot(self, results: xr.Dataset):
        if "substance" not in results.coords:
            results = results.assign_coords({"substance": self.observations.substance})
        if "cext_nom" not in results.coords:
            results = results.assign_coords({"cext_nom": self.observations.cext_nom})
        fig = self._plot.plot_simulation_results(results)


    def define_observations_unreplicated(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      ], dims=("id"), coords={"id": [0]}),
            "cext_nom":  xr.DataArray([1000    ], dims=("id"), coords={"id": [0]}),
            "substance": xr.DataArray(["diuron"], dims=("id"), coords={"id": [0]})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0],
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }

    def define_observations_replicated(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      ] * 5, dims=("id"), coords={"id": np.arange(5)}),
            "cext_nom":  xr.DataArray([1000    ] * 5, dims=("id"), coords={"id": np.arange(5)}),
            "substance": xr.DataArray(["diuron"] * 5, dims=("id"), coords={"id": np.arange(5)})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0] * 5,
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }

    def define_observations_replicated_gradient(self):
        # set up the observations with the number of organisms and exposure 
        # concentrations. This is an observation frame for indexed data with 
        # substance provided as an index
        n = 5
        self.observations = xr.Dataset().assign_coords({
            "nzfe":      xr.DataArray([10      ] * n, dims=("id"), coords={"id": np.arange(n)}),
            "cext_nom":  xr.DataArray(np.logspace(2,4, n), dims=("id"), coords={"id": np.arange(n)}),
            "substance": xr.DataArray(["diuron"] * n, dims=("id"), coords={"id": np.arange(n)})
        })

        # set up the corresponding index
        self.indices = {
            "substance": xr.DataArray(
                [0] * n,
                dims=("id"), 
                coords={
                    "id": self.observations["id"], 
                    "substance": self.observations["substance"]
                }, 
                name="substance_index"
            )
        }


if __name__ == "__main__":
    cfg = "case_studies/hierarchical_ode_model/scenarios/testing/settings.cfg"
    # cfg = "case_studies/tktd_rna_pulse/scenarios/rna_pulse_3_6c_substance_specific/settings.cfg"
    sim = NomixHierarchicalSimulation(cfg)
    
    # TODO: this will become a problem once I try to load different extra
    # modules. The way to deal with this is to load modules as a list and try
    # to get them in hierarchical order
    sim.config.import_casestudy_modules()
    
    # sim.setup_data_structure_from_observations()
    sim.setup_data_structure_manually(
        scenario="data_structure_01_single_observation"
    )

    # run a simulation
    sim.dispatch_constructor()
    e = sim.dispatch(theta=sim.model_parameter_dict)
    e()
    sim.plot(e.results)

    # generate artificial data
    sim.dispatch_constructor()
    res = sim.generate_artificial_data(nan_frac=0.0)
    sim.plot(res)

    # perform inference
    sim.use_numpyro_backend()
    sim.config.inference_numpyro.kernel = "map"
    sim.config.inference_numpyro.draws = 1000
    sim.inferer.run()



    from tktd_rna_pulse import mod as trpmod

    trpmod.tktd_rna_3_6c

    # define a hierarchical error structure
    # check out murefi for this

    # the long form should always be used for the actual model calculations
    # unless wide form is actually required (i.e. vectors or matrices need)
    # to enter the ODE
    
    # currently I use the substance as an index for broadcasting the parameters
    # from a substance index to the long form.
    # multilevel index or something along these lines would be needed to 
    # bring a multilevel index into the long form.


    # currently parameters are at least broadcasted in the JaxSolver, but this
    # is not happening with the other solvers. 
    # Approach:
    # + Define a module that can handle parameter broadcasting automatically 
    #   during dispatch. This can be adapted from the JaxSolver.
    # + Solvers themselves should only handle the casting of the data to types
    #   they require.
    # + This would mean that it is ensured that parameter, y_0 and x_in shapes
    #   can be handled by the solver, because they have been broadcasted, and
    #   can be vectorized or iterated over.
    #