from typing import Literal, Dict, Optional, List, Callable, TypeAlias
import warnings

import xarray as xr
import arviz as az
import numpy as np
import numpy.typing as npt

from pymob.sim.config import Config
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

ALLOWED_IDATA_GROUPS: TypeAlias = Literal[
    "prior_predictive", 
    "prior_model_fits",
    "posterior_predictive", 
    "posterior_model_fits"
]


class SimulationPlot:
    """
    Parameters
    ----------

    observations : xr.Dataset
        Simulation observations dataset 

    idata : az.InferenceData
        Arviz InferenceData created from prior or posterior predictions.

    coordinates : Dict
        Coordinates of the simulation

    config : Config
        Simulation configuration

    rows : List[str]
        Optional. List of variables to plot in the rows of the figure. Defaults
        to the names of the data variables

    columns : str|Dict[str,List[str|int]]
        Optional. Group identifiers. Can be a single string (in this case 
        should be a dimension of the data structure). Can be a dictionary that
        has a single key (the dimension) and a list of values (a subset of)
        the dimension coordinates.
        
    idata_groups : List[str]
        Optional. The idata groups to plot for predictions. Only works for groups
        that have the same data structure as the observations. These are
        
    obs_idata_map : Dict[str|str|callable]
        Optional. Maps the data variables of the observation to the data variables
        of the inference data. This helps in cases, where the data variable of 
        the idata do not have the same name as the data variable of the observations.
        It can also be a callable that extracts the data variable from the 
        idataset.
    """
    figure: Figure
    axes_map: Dict[str,Dict[str,Axes]]
    def __init__(
        self, 
        observations: xr.Dataset,
        idata: az.InferenceData,
        coordinates: Dict,
        config: Config,
        rows: Optional[List[str]] = None,
        columns: Optional[str|Dict[str,List[str|int]]] = None,
        obs_idata_map: Dict[str,str|Callable] = {},
        idata_groups: Optional[ALLOWED_IDATA_GROUPS] = None,
        pred_mode: str = "mean+hdi",
        hdi_interval: float = 0.95,
        obs_style: Dict = {},
        pred_mean_style: Dict = {},
        pred_hdi_style: Dict = {},
        pred_draws_style: Dict = {},
        sharex=True,
        sharey="row",
    ):
        self.observations = observations
        self.idata = idata
        self.coordinates: Dict = coordinates
        self.config: Config = config

        if rows is None:
            self.rows = self.config.data_structure.observed_data_variables
        else:
            self.rows = rows

        if columns is None:
            self.columns = ["all"]
        else:
            self.columns = columns

        self.idata_groups = idata_groups

        self.obs_idata_map = obs_idata_map

        self.sharex = sharex
        self.sharey = sharey

        self.obs_style = dict(marker="o", ls="", ms=3, color="tab:blue")
        self.obs_style.update(obs_style)

        self.pred_mode = pred_mode
        self.hdi_interval = hdi_interval
        
        batch_idx = self.coordinates.get(self.config.simulation.batch_dimension, None)
        if batch_idx is None:
            N = 1
        else:
            N = len(batch_idx)

        self.pred_mean_style = dict(color="black", alpha=0.1)
        self.pred_mean_style.update(pred_mean_style)

        self.pred_hdi_style = dict(color="black", lw=1, alpha=max(1/N, 0.05))
        self.pred_hdi_style.update(pred_hdi_style)

        self.pred_draws_style = dict(color="black", lw=1, alpha=max(1/N, 0.05))
        self.pred_draws_style.update(pred_draws_style)
        
        self.create_figure()
        self.inf_preds: Dict[str: xr.DataArray] = {}


    def create_figure(self):
        r = len(self.rows)
        c = len(self.columns)
        self.figure, axes = plt.subplots(
            nrows=r, ncols=c,
            sharex=self.sharex,
            sharey=self.sharey,
            squeeze=False,
            figsize=(5+(c-1*2), 3+(r-1)*2),
        )

        self.axes_map = {}
        for i, row in enumerate(self.rows):
            self.axes_map[row] = {}
            for j, col in enumerate(self.columns):
                self.axes_map[row][col] = axes[i,j]

    def clean_idata_group(self, idata_group: str):
        # this mask is rigid. It will eliminate each draw where any of the 
        # variables contained an inf value
        group = self.idata[idata_group]
        
        stack_dims = [d for d in group.dims if d not in ["chain", "draw"]]

        mask = (group == np.inf).sum(stack_dims).to_array().sum("variable") > 0
        n_inf = int(mask.sum())
        if n_inf > 0:
            warnings.warn(
                f"There were {n_inf} NaN or Inf values in the idata group "+
                f"'{idata_group}'. See "+
                "Simulation.inf_preds for a mask with the coordinates.",
                category=UserWarning
            )

        self.inf_preds.update({idata_group: mask})

        return group.where(~mask, drop=True)

    def plot_data_variables(self):
        for i, row in enumerate(self.rows):
            for j, col in enumerate(self.columns):
                ax = self.axes_map[row][col]
                for igroup in self.idata_groups:
                    self.plot_predictions(idata_group=igroup, data_variable=row, column=col, ax=ax)

                if self.config.data_structure.all[row].observed:
                    self.plot_observations(data_variable=row, column=col, ax=ax)

        self.figure.tight_layout()
        
    def plot_observations(
        self,
        data_variable: str,
        column: str,
        ax: Axes,
    ):
        x_dim = self.config.simulation.x_dimension
        observations = self.observations[data_variable].copy()
        # stack all dims that are not in the time dimension
        if len(observations.dims) == 1:
            # add a dummy batch dimension
            observations = observations.expand_dims("batch")
            observations = observations.assign_coords(batch=[0])


        stack_dims = [d for d in observations.dims if d not in [x_dim]]
        observations = observations.stack(i=stack_dims)
        N = len(observations.coords["i"])
            
        for i in observations.i:
            obs = observations.sel(i=i)
            if obs.isnull().all():
                # skip plotting combinations, where all values are NaN
                continue
            
            self.plot_single_observation(obs, ax)

        ax.set_xlabel(x_dim)
    
    def plot_predictions(
        self,
        idata_group: str,
        data_variable: str,
        column: str,
        ax: Axes, 
    ):
        x_dim = self.config.simulation.x_dimension
        idata_dims = ["chain", "draw"]

        # get idata dataset and remove inf draws from the prior/posterior
        idata_dataset = self.clean_idata_group(idata_group)
        idata_dataset.attrs.update({"group": idata_group})

        prediction_data_variable = self.obs_idata_map.get(
            data_variable, data_variable
        )

        # this passes the idata group to the callable in the obs_idata_map
        # and returns it. This is very helpful if the prediction data variable
        # needs to be transformed prior to plotting 
        if callable(prediction_data_variable):
            predictions = prediction_data_variable(idata_dataset)
            prediction_data_variable = predictions.name
        else:
            if prediction_data_variable in idata_dataset:
                predictions = idata_dataset[prediction_data_variable].copy()
            else:
                return

        # stack all dims that are not in the time dimension
        if len([d for d in predictions.dims if d not in idata_dims]) == 1:
            # add a dummy batch dimension
            predictions = predictions.expand_dims("batch")
            predictions = predictions.assign_coords(batch=[0])


        stack_dims = [d for d in predictions.dims if d not in [x_dim] + idata_dims]
        predictions = predictions.stack(i=stack_dims)
        N = len(predictions.coords["i"])
            


        for i in predictions.i:
            # if obs.sel(i=i).isnull().all() and not plot_preds_without_obs:
            #     # skip plotting combinations, where all values are NaN
            #     continue
            preds = predictions.sel(i=i)

            self.plot_single_prediction(predictions=preds, ax=ax)
            
        
        ax.set_ylabel(data_variable)
        ax.set_xlabel(x_dim)


    def plot_single_prediction(self, predictions: xr.DataArray, ax: Axes):

        modes = self.pred_mode.split("+")

        for mode in modes:
            if mode == "hdi":
                self.plot_pred_hdi(predictions, ax)

            elif mode == "mean":
                self.plot_pred_mean(predictions, ax)

            elif mode == "draws":
                self.plot_pred_draws(predictions, ax)

            else:
                raise NotImplementedError(
                    f"Mode '{self.pred_mode}' not implemented. "+
                    "Choose 'mean', 'hdi', 'draws' or any combination of them, "+
                    "e.g., 'hdi+mean' (in this case the order determines the "+
                    "order of plots)"
                )

    def plot_single_observation(self, observations: xr.DataArray, ax: Axes):
        x_dim = self.config.simulation.x_dimension
        ax.plot(
            observations[x_dim].values, observations.values, 
            **self.obs_style
        )

    def plot_pred_hdi(self, predictions: xr.DataArray, ax: Axes):
        x_dim = self.config.simulation.x_dimension
        hdi = az.hdi(predictions, self.hdi_interval)[predictions.name]
        ax.fill_between(
            predictions[x_dim].values, *hdi.values.T, # type: ignore
            **self.pred_hdi_style
        )
    
    def plot_pred_mean(self, predictions: xr.DataArray, ax: Axes):
        x_dim = self.config.simulation.x_dimension
        y_mean = predictions.mean(dim=("chain", "draw"))
        ax.plot(
            predictions[x_dim].values, y_mean.values, 
            **self.pred_mean_style
        )

    def plot_pred_draws(self, predictions: xr.DataArray, ax: Axes):
        x_dim = self.config.simulation.x_dimension
        ys = predictions.stack(sample=("chain", "draw"))
        ax.plot(
            predictions[x_dim].values, ys.values, 
            **self.pred_draws_style
        )
    
    def set_titles(self, title: Callable):
        for c in self.columns:
            ax = self.axes_map[self.rows[0]][c]
            ax.set_title(label=title(self, c))

        self.figure.tight_layout()

    def close(self):
        plt.close(self.figure)

    def save(self, filename):
        self.figure.savefig(
            f"{self.config.case_study.output_path}/{filename}"
        )

        