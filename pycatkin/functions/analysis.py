import os
from collections.abc import Iterable
from typing import Union, Sequence, TypeAlias
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors
from matplotlib.ticker import MultipleLocator, StrMethodFormatter
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pycatkin.classes.system import SteadyStateResults, System
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

# Aesthetics
loc = MultipleLocator(base=1, offset=0)
formatter = StrMethodFormatter("{x:.0f}")
cb_formatter = StrMethodFormatter("{x:.2f}")

# Type alias for the list of descriptor indices
idx_list: TypeAlias = list[tuple[int,int]]
log_dict: TypeAlias = dict[tuple[int,int],SteadyStateResults]

# Check for failed cases
def check_convergence(log:log_dict, sim_system:System, C_range:Sequence, O_range:Sequence) -> tuple[idx_list, idx_list]:
    """Checks if calculations converged and stores the converged/failed indices in lists

    Args:
        log (log_dict): Results dictionary
        sim_system (System): System class (system not already built)
        C_range (Sequence): EC energy array
        O_range (Sequence): EO energy array

    Returns:
        tuple[idx_list, idx_list]: Misfit and worked list of indices
    """
    # Safety copy to avoid populating attributes
    sis_use = deepcopy(sim_system)

    # Store results
    misfit_list = []
    worked_list = []

    # Iterate over log terms
    for k,v in log.items():
        if not v.success:
            # Add to list
            misfit_list.append(k)

            # Get the sis_use instance
            # Set the descriptors' energies
            sis_use.reactions["C_ads"].dErxn_user = C_range[k[0]]
            sis_use.reactions["O_ads"].dErxn_user = O_range[k[1]]
            # Manually set the adsorption energies
            sis_use.states["sC"].Gelec = C_range[k[0]]
            sis_use.states["sO"].Gelec = O_range[k[1]]

            # Build system
            sis_use.build()

            # Get the concentrations
            y = np.concat((sis_use.initial_system[len(sis_use.gas_indices):],v.x))

            # Check if the surface coverage makes sense
            surf_sum = [sum(y[list(surf_indices)]) for surf_indices in sis_use.coverage_map.values()]
            if np.any(np.abs(np.array(surf_sum) - 1) > 0.05):
                print(f"{k} : SURF SUM FAILED: {' , '.join(str(x)[:8] for x in surf_sum)}")
            
            # Check if the rates make sense
            elif np.any(np.abs(sis_use.get_dydt(y)) > 1e-6):
                print(f"{k} : RATE FAILED: {max(sis_use.get_dydt(y)):.4e}")          
        else:
            worked_list.append(k)
    return misfit_list, worked_list

# Average the surrounding of failed cases
def average_neighborhood(misfit_list: idx_list, worked_list: idx_list, log: log_dict) -> log_dict:
    """aproximates the coverage of a failed point by the average of its successful neighbors

    Args:
        misfit_list (idx_list): List of failed index tuples
        worked_list (idx_list): List of successful index tuples
        log (log_dict): Log of results

    Returns:
        log_dict: Dictionary log copy with updated (averaged) values
    """
    # Safety copy
    new_log = deepcopy(log)

    # Iterate over failed pairs
    for mpair in misfit_list:
        iC,iO = mpair
        
        # Datapoints surrounding the failed calculation
        neighborhood = [
            (iC + k, iO + j) 
                for k in [-1,0,1] 
                for j in [-1,0,1] 
                if (k,j) != (0,0) and 
                (iC + k, iO + j) in worked_list
            ]
        
        if len(neighborhood) < 2:
            print(f"FAILED FINDING SURROUNDINGS FOR {iC,iO}")
            continue

        # Get neighboring points that worked out
        L=[new_log[pair].x for pair in neighborhood]

        # Update dictionary with average coverages
        new_log[mpair] = SteadyStateResults(x=np.mean(L, axis=0), success=False)
        
        return new_log


# Visualizing failed cases
def convergence_heatmap(C_range: Sequence, O_range: Sequence, misfit_list: idx_list) -> Axes:
    """Generates a heatmap for the convergence (so we can see failed cases on a 2D grid)

    Args:
        C_range (Sequence): Array of investigated EC values
        O_range (Sequence): Array of investigated EO values
        misfit_list (list[tuple[int,int]]): List of misfit indices

    Returns:
        Axes: axes with heatmap
    """
    work_map = np.ones(shape=(len(C_range), len(O_range)))
    for iC, _ in enumerate(C_range):
        for iO, _ in enumerate(O_range):
            if (iC,iO) in misfit_list:
                work_map[(iC,iO)] = 0

    ax = sns.heatmap(work_map.T, linewidth=1, cmap="Pastel1")
    ax.set_ylabel("EO (eV)")
    ax.set_xlabel("EC (eV)")
    return ax

# Create heatmaps of activity or selectivity
def _custom_heatmap(
    fig: Figure, ax: Axes, C_range: Sequence, O_range: Sequence, Z: np.ndarray, norm: colors.Normalize = None, 
    y_label: str = 'log(TOF[1/s])', sigma: float = 0.75, shrink: float = 0.7
    ):
    """Generates a heatmap

    Args:
        fig (Figure): Matplotlib figure to use
        ax (Axes): Matplotlib ax to use
        C_range (Sequence): EC energies
        O_range (Sequence): EO energies
        Z (np.ndarray): Matrix of dimensions (EC,EO) with value to visualize
        norm (colors.Normalize, optional): Normalize object with min and max values of Z. Defaults to None.
        y_label (str, optional): Y axis label. Defaults to 'log(TOF[1/s])'.'
        sigma (float, optional): Smoothing factor for the heatmap. Defaults to 0.75
        shrink (float, optional): How much to shrink the colorbar. Defaults to 0.70 (70% of original size)
    """
    n_levels = 30
    levels = n_levels if norm is None else np.linspace(norm.vmin, norm.vmax, n_levels, endpoint=True)
    Z = ndimage.gaussian_filter(Z,sigma)
    CS = ax.contourf(C_range, O_range, Z, levels=levels, cmap=plt.get_cmap("RdYlBu_r"), norm=norm)

    # Format the colorbar
    fig.colorbar(CS, ax=ax, format=cb_formatter, label=y_label, shrink=shrink)

    # Main ax editing
    ax.set(xlabel=r'$E_{\mathsf{C}}$ (eV)', ylabel=r'$E_{\mathsf{O}}$ (eV)')
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_formatter(formatter)
    

def make_heatmap(
    labels: Union[Sequence[str], str], 
    results: log_dict, 
    C_range: Sequence, 
    O_range: Sequence, 
    use_log: bool = True, 
    panel_size = (3,3),
    figname: str = None,
    y_label: str = 'log(TOF[1/s])',
    sigma: float = 0.75, 
    shrink: float = 0.7
    ) -> (Figure, Axes):
    """Makes heatmap of a given quantity

    Args:
        labels (Union[Iterable[str], str]): Which species to consider. Could be a list or a string. e.g., "CH3OH" or ["CH2OH","CH3OH"]
        results (log_dict): Results dict with quantity of interest
        C_range (Sequence): EC values
        O_range (Sequence): EO values
        use_log (bool, optional): Use log scale for y. Defaults to True.
        panel_size (tuple, optional): Size of individual axes. Defaults to (3,3).
        figname (str, optional): Figname to save. Defaults to None.
        y_label (str, optional): Y label. Defaults to 'log(TOF[1/s])'.
        sigma (float, optional): Diffusion factor for heatmap (smoothing). Defaults to 0.75.
        shrink (float, optional): Shrinking factor for colorbar. Defaults to 0.7 (70% of colorbar size).

    Returns:
        Figure: Matplotlib figure with the heatmaps
        Axes: Matplotlib axes with the heatmaps
    """

    # Defensive duck typing and sorting
    labels = [labels] if isinstance(labels, str) else list(labels)
    n_labels = len(labels)

    # Store the solutions
    scores = np.zeros((n_labels, len(C_range), len(O_range)))

    # Iterate over the solutions dictionary and store in the np.array defined before
    for idx, case in enumerate(labels):
        for k,v in results.items():
            store = np.log(np.abs(v[case])) if use_log else np.abs(v[case])
            scores[(idx,*k)] = store

    ## Generate graphs
    # Figure outline
    if n_labels > 1:
        ncols = 2
        nrows = int(np.ceil(n_labels/ncols))
        figsize = (panel_size[0]*ncols, panel_size[1]*nrows)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(figsize=panel_size)

    # Generate heatmaps
    if use_log:
        scores[np.where(scores < -25)] = -25
    vmin = np.round(np.min(scores), 2)
    vmax = np.round(np.max(scores), 2)
    norm = colors.Normalize(vmin=vmin, vmax=vmax) #normalization object for cbars

    if n_labels > 1:
        for idx, case in enumerate(labels):
            # Get scores for that particular label
            Z = scores[idx, :, :]
            _custom_heatmap(fig, axes[idx], C_range, O_range, Z, norm, y_label, sigma, shrink)
            axes[idx].set_title(case)
    else:
        Z = scores[0, :, :]
        _custom_heatmap(fig, axes, C_range, O_range, Z, norm, y_label, sigma, shrink)
        axes.set_title(case)

    # Edit colorbar axes (had issues trying to accomplish this feat above)
    for cbar_ax in fig.axes[-len(axes):]:
        cbar_ax.set_ylabel(y_label)
        custom_ticks = np.round(np.linspace(norm.vmin, norm.vmax, 5, endpoint=True), 2)
        cbar_ax.set_yticks(custom_ticks, custom_ticks)
    
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')

    # save or return
    if figname is not None:
        # Directory to store figures
        if not os.path.isdir('figures'):
            os.mkdir('figures')
        plt.tight_layout()
        plt.savefig(f"figures/{figname}", dpi=600, format='png')

    else:
        return fig, axes
