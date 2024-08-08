import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import norm

from scripts.ellipse import CosmoFisher
from scripts.config import cosmo_params, recon_colors

rc('text', usetex=True)
rc('font', family='serif')
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)


def contour_plot(p1, p2, fisher_matrix_list, labels_list, save_fig=False, fig_name='contour_plot.pdf'):
    """
    Plot the 2D contour plot of the Fisher matrix for the two parameters p1 and p2.

    Parameters
    ----------
    p1 : int
        Index of the first parameter in the Fisher matrix.
    p2 : int
        Index of the second parameter in the Fisher matrix.
    fisher_matrix_list : list
        List of Fisher matrices.
    labels_list : list
        List of labels for the Fisher matrices.
    save_fig : bool, optional
        Save the figure. The default is False.
    fig_name : str, optional
        Name of the figure. The default is 'contour_plot.pdf'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the contour plot.
    """

    if len(fisher_matrix_list) != len(labels_list):
        raise ValueError('fisher_matrix_list and labels_list must have the same length')

    fig, axs = None, None
    for index, fisher_matrix in enumerate(fisher_matrix_list):
        if index==0:
            fig, axs = plot_ellipse(p1, p2, fisher_matrix, cosmo_params, color=recon_colors[index], label=labels_list[index])
        else:
            plot_ellipse(p1, p2, fisher_matrix, cosmo_params, axs=axs, color=recon_colors[index], label=labels_list[index])

    if save_fig and fig is not None:
        fig.savefig(fig_name)

    return fig


def plot_ellipse(p1, p2, fisher_matrix, cosmo_params, axs=None, color=None, label=None, dash=False, shade=False):
    '''
    Plot the 1-sigma contour for two parameters.

    Parameters
    ----------
    p1 : str
        Name of the first cosmological parameter
    p2 : str
        Name of the second cosmological parameter
    fisher_matrix : array
        Fisher matrix
    cosmo_params : dict
        Dictionary of cosmological parameters with keys 'value' and 'label'
    axs : array (optional)
        Array of matplotlib axes
    color : array (optional)
        Color of the ellipse   
    label : str (optional)
        Label for the ellipse
    dash : bool (optional)
        Whether to plot the ellipse as dashed
    shade : bool (optional)
        Whether to shade the ellipse
    
    Returns
    -------
    axs : array
        Array of matplotlib axes
    '''

    if p1 == 'Om' and p2 == 'S8':
        ellip, center, stdv = CosmoFisher(p1, p2, fisher_matrix, cosmo_params).S8_ellipse()
    else:
        ellip, center, stdv = CosmoFisher(p1, p2, fisher_matrix, cosmo_params).ellipse()

    fig = None
    if axs is None:

        p1_min = center[0] - 3*stdv[0]
        p1_max = center[0] + 3*stdv[0]
        p2_min = center[1] - 3*stdv[1]
        p2_max = center[1] + 3*stdv[1]

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.delaxes(axs[0,1])
        plt.subplots_adjust(hspace=0, wspace=0)
        axs[1,0].set_xlim(p1_min, p1_max)
        axs[1,0].set_ylim(p2_min, p2_max)
        axs[1,0].set_xlabel(cosmo_params[p1]['label'], fontsize=25)
        axs[1,0].set_ylabel(cosmo_params[p2]['label'], fontsize=25)

        # Plot the center of the ellipse
        axs[1,0].plot(center[0], center[1], 'o', alpha=0.0001)
        axs[1,0].axvline(center[0], ls='--', color='lightgray')
        axs[1,0].axhline(center[1], ls='--', color='lightgray')

        # set x and y limits for histograms
        axs[0,0].set_xlim(p1_min, p1_max)
        mu = center[0]
        sig = stdv[0]
        x = np.linspace(mu - 10 * sig, mu + 10 * sig, 100)

        axs[1,1].set_xlim(p2_min, p2_max)
        mu = center[1]
        sig = stdv[1]
        x = np.linspace(mu - 10 * sig, mu + 10 * sig, 100)

    # Modify properties of the ellipse
    ellip.set_edgecolor(color + [1.0])
    ellip.set_linewidth(2) 
    ellip.set_facecolor('none')
    if shade:
        ellip.set_facecolor(color + [0.3])
    if dash:
        ellip.set_linestyle('dashed')

    # Plot the perimeter of the ellipse on ax2
    axs[1,0].add_patch(ellip)

    # plot Gaussian histogram above ellipse for p1
    mu = center[0]
    sig = stdv[0]
    x = np.linspace(mu - 10 * sig, mu + 10 * sig, 1000)
    y = norm.pdf(x, mu, sig)
    axs[0,0].plot(x, y,color=color,label=label)
    axs[0,0].fill_between(x,y, where=(x >= mu - sig) & (x <= mu + sig), color=color, alpha=0.2) # shade region within 1 sigma
    
    # plot Gaussian histogram to the right of ellipse for p2
    mu = center[1]
    sig = stdv[1]
    x = np.linspace(mu - 10 * sig, mu + 10 * sig, 1000)
    y = norm.pdf(x, mu, sig)
    axs[1,1].plot(x,norm.pdf(x, mu, sig),color=color)
    axs[1,1].fill_between(x,y, where=(x >= mu - sig) & (x <= mu + sig), color=color, alpha=0.2) # shade region within 1 sigma

    # remove axis labels from hist plots
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    # legend
    axs[0, 0].legend(loc='upper right',bbox_to_anchor=(2, 1), borderaxespad=0., fontsize=16)

    return axs
