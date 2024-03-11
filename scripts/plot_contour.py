import numpy as np
import matplotlib.pyplot as plt
from ellipse import CosmoFisher
from scipy.stats import norm


def plot_ellipse(p1, p2, fisher_matrix, cosmo_params, params_hist_lim, axs=None, color=None, label=None, dash=False):

    ellip, center, stdv = CosmoFisher(p1, p2, fisher_matrix, cosmo_params).ellipse()

    if axs is None:

        # design plot
        box_width = cosmo_params[p1]['box_range']
        box_height = cosmo_params[p2]['box_range']

        hist_lims = params_hist_lim.get((p1, p2), {})
        hist_p1_lim = hist_lims.get('axs_00_ylim', None)
        hist_p2_lim = hist_lims.get('axs_11_ylim', None)

        fig, axs = plt.subplots(2,2,figsize=(10,10))
        fig.delaxes(axs[0,1])
        plt.subplots_adjust(hspace=0, wspace=0)
        axs[1,0].set_xlim(center[0] - box_width, center[0] + box_width)
        axs[1,0].set_ylim(center[1] - box_height, center[1] + box_height)
        axs[1,0].set_xlabel(cosmo_params[p1]['label'], fontsize=25)
        axs[1,0].set_ylabel(cosmo_params[p2]['label'], fontsize=25)

        # Plot the center of the ellipse
        axs[1,0].plot(center[0], center[1], 'o',alpha=0.0001)
        axs[1,0].axvline(center[0], ls='--', color='lightgray')
        axs[1,0].axhline(center[1], ls='--', color='lightgray')

        # set x and y limits for histograms
        axs[0,0].set_xlim(center[0] - box_width, center[0] + box_width)
        mu = center[0]
        sig = stdv[0]
        x = np.linspace(mu - 10 * sig, mu + 10 * sig, 100)
        axs[0,0].set_ylim(0, max(norm.pdf(x, mu, sig)) + hist_p1_lim)

        axs[1,1].set_xlim(center[1] - box_height, center[1] + box_height)
        mu = center[1]
        sig = stdv[1]
        x = np.linspace(mu - 10 * sig, mu + 10 * sig, 100)
        axs[1,1].set_ylim(0, max(norm.pdf(x, mu, sig)) + hist_p2_lim)

    # Modify properties of the ellipse
    ellip.set_edgecolor(color+[1.0])
    ellip.set_facecolor('none')#color+[0.3]) # inside shading
    ellip.set_linewidth(2) 
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





def plot_ellipse_S8(p1,p2,ellip,center,stdv,axs=None,color=None,label=None,dash=False,legend=False):

    if axs is None:

        # design plot
        box_width = cosmo_params[p1]['box_range']
        box_height = 0.012
        fig, axs = plt.subplots(2,2,figsize=(10,10))
        fig.delaxes(axs[0,1])
        plt.subplots_adjust(hspace=0, wspace=0)
        axs[1,0].set_xlim(center[0] - box_width, center[0] + box_width)  # Set x-axis limits
        axs[1,0].set_ylim(center[1] - box_height, center[1] + box_height)  # Set y-axis limits
        axs[1,0].set_xlabel(cosmo_params[p1]['label'], fontsize=25)
        axs[1,0].set_ylabel('$S_8$', fontsize=25)

        # Plot the center of the ellipse
        axs[1,0].plot(center[0], center[1], 'o',alpha=0.0001)
        axs[1,0].axvline(center[0], ls='--', color='lightgray')
        axs[1,0].axhline(center[1], ls='--', color='lightgray')

        # set x and y limits for histograms
        axs[0,0].set_xlim(center[0] - box_width, center[0] + box_width)
        axs[1,1].set_xlim(center[1] - box_height, center[1] + box_height)
        mu = center[0]
        sig = stdv[0]
        x = np.linspace(mu - 10 * sig, mu + 10 * sig, 100)
        mu = center[1]
        sig = stdv[1]
        x = np.linspace(mu - 10 * sig, mu + 10 * sig, 100)
        axs[0,0].set_ylim(0, max(norm.pdf(x, mu, sig)) + 30)
        axs[1,1].set_ylim(0, max(norm.pdf(x, mu, sig)) + 10)

    # Modify properties of the ellipse
    ellip.set_edgecolor(color+[1.0]) # border
    ellip.set_facecolor('none')#color+[0.3]) # inside shading
    ellip.set_linewidth(2) 
    if dash:
        ellip.set_linestyle('dashed')

    # Plot the perimeter of the ellipse on ax2
    axs[1,0].add_patch(ellip)

    # plot Gaussian histogram above ellipse
    mu = center[0]
    sig = stdv[0]

    x = np.linspace(mu - 10 * sig, mu + 10 * sig, 1000)
    y = norm.pdf(x, mu, sig)
    axs[0,0].plot(x, y,color=color,label=label)
    axs[0,0].fill_between(x,y, where=(x >= mu - sig) & (x <= mu + sig), color=color, alpha=0.2) # shade region within 1 sigma
    
    # plot Gaussian histogram to the right of ellipse
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
    if legend:
        axs[0, 0].legend(loc='upper right',bbox_to_anchor=(2, 1), borderaxespad=0., fontsize=16)

    return axs

