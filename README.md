# fisher_to_contour
Make a contour plot from fisher matrices (as seen in [Grewal et al 2024](https://arxiv.org/pdf/2402.13912.pdf))

<img src="contour_plot_Om_s8.pdf" alt="Contour Plot" width="600" height="600">

## Installation

Clone the repository: 
```
git clone https://github.com/nishagrewal/fisher_to_contour.git
```

Tip: If you are using VSCode, you can enable autosaving by going to *File* -> *Auto Save*.


## Usage

1. Update cosmological parameters in `config.py`
2. Make a list of fisher matrices and labels in `Example_notebook.ipynb`
3. Select parameters `p1` and `p2` to plot on contour axes
4. Run `contour_plot function`, saving if desired


**Optional:**

* To make an $\Omega_m$ - $S_8$ contour plot:
    * ensure $\Omega_m$ and $\sigma_8$ are in the `cosmo_params` dictionary
    * set `p1='Om'` and `p2='S8'`
* Contour colours can be reconfigured in config.py


## Contributing

Feel free to contribute by opening an issue or submitting a pull request.


## References

The ellipse calculation follows in [Coe 2009](https://arxiv.org/pdf/0906.4123.pdf), with one exception in the marginalisation procedure in Section 3.1: the inverse of the Fisher matrix is taken, ***then*** marginalised parameters are removed.

The $S_8$ calculation is done following Appendix A from [Euclid preparation XXVIII. Forecasts for ten different higher-order weak lensing statistics](https://www.aanda.org/articles/aa/full_html/2023/07/aa46017-23/aa46017-23.html#R20).
