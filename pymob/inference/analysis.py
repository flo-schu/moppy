import warnings

import numpy as np
import xarray as xr
import arviz as az
from arviz.sel_utils import xarray_var_iter
from matplotlib import pyplot as plt
import matplotlib as mpl

from pymob.utils.plot_helpers import plot_hist, plot_loghist

def cluster_chains(posterior, deviation="std"):
    assert isinstance(posterior, (xr.DataArray, xr.Dataset))
    chain_means = posterior.mean(dim="draw")
    if deviation == "std":
        chain_dev = posterior.std(dim="draw")
    elif "frac:" in deviation:
        _, frac = deviation.split(":")
        chain_dev = chain_means * float(frac)
    else:
        raise ValueError("Deviation method not implemented.")    

    global cluster_id
    cluster_id = 1
    cluster = [cluster_id] * len(posterior.chain)
    unclustered_chains = posterior.chain.values

    def recurse_clusters(unclustered_chains):
        global cluster_id
        compare = unclustered_chains[0]
        new_cluster = []
        for i in unclustered_chains[1:]:
            a = chain_means.sel(chain=compare) + chain_dev.sel(chain=compare) > chain_means.sel(chain=i)
            b = chain_means.sel(chain=compare) - chain_dev.sel(chain=compare) < chain_means.sel(chain=i)
            isin_dev = (a * b).all()

            if isinstance(isin_dev, xr.Dataset):
                isin_dev = isin_dev.to_array().all()

            if not isin_dev:
                cluster[i] = cluster_id + 1
                new_cluster.append(i)

        cluster_id += 1
        if len(new_cluster) == 0:
            return

        recurse_clusters(new_cluster)
    
    recurse_clusters(unclustered_chains)

    return cluster


def rename_extra_dims(df, extra_dim_suffix="_dim_0", new_dim="new_dim", new_coords=None):
    # TODO: COuld be used for numypro backend for fixing posterior indexes
    df_ = df.copy()
    data_vars = list(df_.data_vars.keys())

    # swap dimension names for all dims that have the suffix 
    new_dims = {}
    for dv in data_vars:
        old_dim = f"{dv}{extra_dim_suffix}"
        if df_.dims[old_dim] == 1:
            df_[dv] = df_[dv].squeeze(old_dim)
        else:
            new_dims.update({old_dim: new_dim})

    df_ = df_.swap_dims(new_dims)

    # assign coords to new dimension
    df_ = df_.assign_coords({new_dim: new_coords})
    
    # drop renamed coords
    df_ = df_.drop([f"{dv}{extra_dim_suffix}" for dv in data_vars])

    return df_



# plot loghist
def plot_posterior_samples(posterior, col_dim=None, log=True, hist_kwargs={}):
    if log:
        hist = plot_loghist
    else:
        hist = plot_hist

    parameters = list(posterior.data_vars.keys())
    samples = posterior.stack(sample=("chain", "draw"))

    fig = plt.figure(figsize=(5, len(parameters)*2))
    fig.subplots_adjust(right=.95, top=.95, hspace=.25)

    gs = fig.add_gridspec(len(parameters), 1)

    for i, key in enumerate(parameters):
        postpar = samples[key]
        if col_dim in postpar.dims:
            col_coords = postpar[col_dim]
            gs_par = gs[i, 0].subgridspec(1, len(col_coords))
            axes = gs_par.subplots()

            for ax, coord in zip(axes, col_coords):
                hist(
                    x=postpar.sel({col_dim: coord}), 
                    name=f"${key}$ {str(coord.values)}",
                    ax=ax,
                    **hist_kwargs
                )

        else:
            gs_par = gs[i, 0].subgridspec(1, 1)
            ax = gs_par.subplots()

            hist(
                x=postpar, 
                name=f"${key}$",
                ax=ax,
                **hist_kwargs
            )


    return fig


def bic(idata: az.InferenceData):
    """calculate the BIC for az.InferenceData. The function will average over
    all samples from the markov chain
    """
    log_likelihood = idata.log_likelihood.mean(("chain", "draw")).sum().to_array().sum()
    k = idata.posterior.mean(("chain", "draw")).count().to_array().sum()

    vars = [i.split("_")[0] for i in list(idata.log_likelihood.data_vars.keys())]
    n = 0
    for v in vars:
        if v in idata.observed_data:
            n += (~idata.observed_data[v].isnull()).sum()
        elif v + "_obs" in idata.observed_data:
            n += (~idata.observed_data[v + "_obs"].isnull()).sum()
        else:
            raise IndexError(f"Variable {v} or {v+'_obs'} not in idata")

    # n = (~idata.observed_data[vars].isnull()).sum().to_array().sum()

    bic = float(k * np.log(n) - 2 * log_likelihood)
    msg = str(
        "Bayesian Information Criterion (BIC)"
        "\n===================================="
        f"\nParameters: {int(k)}"
        f"\nData points: {int(n)}"
        f"\nLog-likelihood: {float(log_likelihood)}"
        f"\nBIC: {bic}"
    )

    return msg, bic


def add_cluster_coordinates(idata, deviation="std"):
    cluster = cluster_chains(idata.posterior, deviation=deviation)
    idata = idata.assign_coords(cluster=("chain", cluster))
    return idata


def format_parameter(par, subscript_sep="_", superscript_sep="__", textwrap="\\text{}"):
    super_pos = par.find(superscript_sep)
    sub_pos = par.find(subscript_sep)
    
    scripts = sorted(zip([sub_pos, super_pos], [subscript_sep, superscript_sep]))
    scripts = [sep for pos, sep in scripts if pos > -1]


    def wrap_text(substr):
        if len(substr) == 1:
            substring_fmt = f"{substr}"
        else:
            substring_fmt = textwrap.replace("{}", "{{{}}}").format(substr)
    
        return f"{{{substring_fmt}}}"

    formatted_string = "$"
    for i, sep in enumerate(scripts):
        substr, par = par.split(sep, 1)
        substring_fmt = wrap_text(substr=substr)

        math_sep = "_" if sep == subscript_sep else "^"

        formatted_string += substring_fmt + math_sep

    formatted_string += wrap_text(par) + "$"

    return formatted_string


def create_table(posterior, error_metric="hdi", vars={}, nesting_dimension=None):
    """The function is not ready to deal with any nesting dimensionality
    and currently expects the 2-D case
    """
    parameters = list(posterior.data_vars.keys())

    tab = az.summary(posterior, fmt="xarray", kind="stats", stat_focus="mean", hdi_prob=0.94)
    if len(vars) > 0:
        tab = tab[vars.keys()]

    tab = tab.rename(vars)

    if error_metric == "sd":
        arrays = []
        for par in parameters:
            par_formatted = tab.sel(metric=["mean", "sd"])[par]\
                .round(3)\
                .astype(str).str\
                .join("metric", sep=" ±")
            arrays.append(par_formatted)

        formatted_tab = xr.combine_by_coords(arrays).to_dataframe().T

        formatted_parameters = []
        for idx in formatted_tab.index:
            formatted_parameters.append(format_parameter(idx))


        formatted_tab.index = formatted_parameters
        return formatted_tab    
    

    elif error_metric == "hdi":
        stacked_tab = tab.sel(metric=["mean", "hdi_3%", "hdi_97%"])\
            .assign_coords(metric=["mean", "hdi 3%", "hdi 97%"])\
            .stack(col=(nesting_dimension, "metric"))\
            .round(2)
        formatted_tab = stacked_tab.to_dataframe().T.drop(index=[nesting_dimension, "metric"])

        return formatted_tab
    
    else:
        raise NotImplementedError("Must use one of 'sd' or 'hdi'")
    

def filter_not_converged_chains(idata, deviation=1.05):
    posterior = idata.posterior
    log_likelihood = idata.log_likelihood
    log_likelihood_summed = log_likelihood.to_array("obs")
    log_likelihood_summed = log_likelihood_summed.sum(("time", "id", "obs"))

    # filter non-converged parameter estimates
    likelihood_mask = (
        # compares the mean of the summed log likelihood of a given chain
        log_likelihood_summed.mean("draw") > 
        # to the maximum of all chain means times a factor
        log_likelihood_summed.mean("draw").max() * deviation
    )
    posterior_filtered = posterior.where(likelihood_mask, drop=True)
    log_likelihood_filtered = log_likelihood.where(likelihood_mask, drop=True)

    idata = az.InferenceData(
        posterior=posterior_filtered, 
        log_likelihood=log_likelihood_filtered,
        observed_data=idata.observed_data,
    )

    return idata


def log(msg, out, newlines=1, mode="a"):
    with open(out, mode) as f:
        print(msg, file=f, end="\n")
        for _ in range(newlines):
            print("", file=f, end="\n")


def evaluate_posterior(sim, nesting_dimension, n_samples=10_000, vars_table={}, 
                       seed=1, save=True, show=False):
    """The function is not ready to deal with any nesting dimensionality 
    and currently expects the 2-D case
    """
    rng = np.random.default_rng(seed)
    idata = sim.inferer.idata
    # idata.posterior = idata.posterior.chunk(chunks={"draw":100}).load()
    # idata.log_likelihood = idata.log_likelihood.chunk(chunks={"draw":100})
    n_subsample = min(
        int(n_samples / idata.posterior.sizes["chain"]), 
        idata.posterior.sizes["draw"]
    )

    if n_subsample < 250:
        warnings.warn(
            "The number of samples drawn from each chain for the pairplot "
            f"({n_subsample}) may be too small to be representative. "
            "Consider increasing n_samples."
        )

    subsamples = rng.choice(idata.posterior.draw, n_subsample, replace=False)
    idata.posterior = idata.posterior.sel(draw=subsamples)
    idata.log_likelihood = idata.log_likelihood.sel(draw=subsamples)

    log_likelihood_summed = idata.log_likelihood.to_array("obs")
    log_likelihood_summed = log_likelihood_summed.sum(("time", "id", "obs"))

    print(az.summary(idata.posterior))

    table = create_table(
        posterior=idata.posterior, 
        error_metric="hdi",
        vars=vars_table,
        nesting_dimension=nesting_dimension,
    )
    table = table.rename(columns={"hdi 3%": "hdi 3\\%", "hdi 97%": "hdi 97\\%"})
    table = table.rename(columns={c: c.capitalize() for c in table.columns.levels[0]})
    table.columns = table.columns.set_names(["Parameters", ""])
    table.index = [format_parameter(i) for i in list(table.index)]
    table_latex = table.to_latex(
        float_format="%.2f",
        caption=(
            "Parameter estimates and posterior highest densitiy intervals (HDI) "+
            f"of the {sim.case_study}__{sim.scenario} model. The HDI "+
            "contains 94\% of the probable parameter values given the data."
        ),
        label=f"tab:parameters-{sim.case_study}__{sim.scenario}"
    )

    # bic 
    msg, _ = bic(idata)
    if save:
        log(table_latex, out=f"{sim.output_path}/parameter_table.tex", mode="w")
        log(msg=msg, out=f"{sim.output_path}/bic.md", mode="w")

    if show:
        print(table)
        print(msg)

    fig_param = plot_posterior_samples(
        idata.posterior, 
        col_dim=nesting_dimension, 
        log=True,
        hist_kwargs = dict(hdi=True, bins=20)
    )
    fig_param.set_size_inches(12, 30)

    if save:
        fig_param.savefig(f"{sim.output_path}/multichain_parameter_estimates.jpg")

    if show:
        plt.show()
    else:
        plt.close()

    def plot_pairs(posterior, likelihood):
        parameters = list(posterior.data_vars.keys())

        N = len(parameters)
        parameters_ = parameters.copy()
        fig = plt.figure(figsize=(3*N, 3*(N+1)))
        gs = fig.add_gridspec(N, N+1, width_ratios=[1]*N+[0.2])
        

        i = 0
        while len(parameters_) > 0:
            par_x = parameters_.pop(0)
            hist_ax = gs[i,i].subgridspec(1, 1).subplots()
            plot_hist(
                posterior[par_x].stack(sample=("chain", "draw")), 
                ax=hist_ax, decorate=False, bins=20
            )
            hist_ax.set_title(par_x)
            for j, par_y in enumerate(parameters_, start=i+1):
                ax = gs[j,i].subgridspec(1, 1).subplots()

                scatter = ax.scatter(
                    posterior[par_x], 
                    posterior[par_y], 
                    c=likelihood, 
                    alpha=0.25,
                    s=10,
                    cmap=mpl.colormaps["plasma_r"]
                )

                if j != len(parameters)-1:
                    ax.set_xticks([])
            
                ax.set_xlabel(par_x)            
                ax.set_ylabel(par_y)

            i += 1

        # ax_colorbar = gs[:,N].subgridspec(1, 1).subplots()
        # fig.colorbar(scatter, cax=ax_colorbar)
        return fig

    for coord in idata.posterior[nesting_dimension].values:
        print("=" * len(coord))
        print(coord.capitalize())
        print("=" * len(coord))
        az.plot_trace(idata.posterior.sel({nesting_dimension:coord}))
        
        if save:
            plt.savefig(f"{sim.output_path}/multichain_pseudo_trace_{coord}.jpg")

        if show:
            plt.show()
        else:
            plt.close()

        fig = plot_pairs(
            posterior=idata.posterior.sel({nesting_dimension:coord}), 
            likelihood=log_likelihood_summed,
        )

        if save:
            fig.savefig(f"{sim.output_path}/multichain_pairs_{coord}.jpg")

        if show:
            plt.show()
        else:
            plt.close()



def plot_pair(posterior, likelihood, parameters, axes=None):

    likelihood_vars = list(likelihood.data_vars.keys())
    N = len(likelihood_vars)

    if axes is None:
        fig, axes = plt.subplots(1, N, figsize=(3+(2*(N-1)), 3), sharey=True, sharex=True)
    
    if isinstance(likelihood, xr.DataArray):
        if likelihood.name is None:
            likelihood.name = "joint_likelihood"
        likelihood = likelihood.to_dataset()
    

    par_a, par_b = parameters

    post_a = posterior[par_a]
    post_b = posterior[par_b]

    # always put the smaller parameter on the x-axis
    if post_a.ndim <= post_b.ndim:
        post_x = post_a
        post_y = post_b
    else:
        post_x = post_b.stack(sample=("chain", "draw"))
        post_y = post_a.stack(sample=("chain", "draw"))

    loglik = likelihood.stack(sample=("chain","draw"))
    extra_dims_y = [d for d in post_y.dims if d not in post_x.dims]
    norm = mpl.colors.Normalize(vmin=(likelihood).to_array().min(), vmax=(likelihood).to_array().max())
    cmap = mpl.colormaps["plasma"]

    for likvar, ax in zip(likelihood_vars, axes):
        ll = loglik[likvar]
        ax.set_title(likvar)
        ax.set_xlabel(post_x.name)
        ax.set_ylabel(post_y.name)
        for dim in extra_dims_y:
            for coord in post_y.coords[dim].values:
                if dim in ll.dims:
                    ll_coord = ll.sel({dim:coord})
                else:
                    ll_coord = ll

                scatter = ax.scatter(
                    post_x, 
                    post_y.sel({dim:coord}), 
                    c=ll_coord, 
                    alpha=0.25,
                    marker=f"${coord}$",
                    s=40,
                    cmap=cmap,
                    norm=norm
                )

                # ax.annotate()
        
    cbar = fig.colorbar(
        mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cmap), 
        # if the same axis is used it is plot next to
        # the plot depending on the location option
        # ax=ax,
        # location="right",
        orientation="vertical",

        # if another axis is used, the fraction should be increased to 1.0
        ax=ax,
        fraction=0.1,
        pad=0.05,
        # aspect=20, # parameter is not so important. It thins the colorbar
        label="negative log-likelihood"
    )
    fig.tight_layout()

    # ax_colorbar = gs[:,N].subgridspec(1, 1).subplots()
    return fig