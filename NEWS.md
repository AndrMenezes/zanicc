# `zanicc` 0.1.0

- Added a vignette to demonstrate package's usability for regression analysis.
- Added a vignette to demonstrate package's usability for pollen-based palaeoclimate reconstruction using the ZANIM-LN-BART model.
- Improved the function `plot_ppc()` with plots using `ggplot2`, and 
an option to supply a list of posterior predictive distributions for different models.  
- Improved documentation of the functions, including all `R6` classes.
- Modified DESCRIPTION with more details on the package. 
- Added two different methods to estimate the multivariate density function within 
the function `compute_prediction_metrics()`: `kde:ks()` for kernel-density estimation or
`rvinecopulib::vine()` for vine copula methods.
- Improved function `compute_prediction_metrics()` for computing joint 
coverage using Hyndman (1996) density-quantile method. 
- Fixed initialisation warnings in the C++ class of DMRregression.
- Added examples with `dontrun{}` for the functions `runifconvexhull()`,  `create_rectangular_grid()`
and `compute_prediction_metrics()`.
- Added CITATION.
- Removed inst folder from .`Rbuildignore` file.

# `zanicc` 0.0.2

- Added new function `inverse_posterior_zanimlnbart()` for sampling from the 
inverse posterior distribution to perform pollen-based palaeoclimate reconstruction
using the ZANIM-LN-BART model.
- Updated the documentation for the `zanicc()` wrapper function.
- Updated the documentation for the `predict()` method.
- Added the `pollen_monticchio` data set with compositional pollen counts from the Lago di Grande Monticchio site.
- Renamed the data sets: `microbome_data` to `microbiome_gut` and `pollen_data` to `pollen_climate`. 
- Added diagnostic plotting functions: `plot_ppc()` and `plot_qqplots_ppd()`.
- Consolidated documentation for functions related to empirical count-compositional indices,
recovery metrics, MCMC posterior metrics, and prediction and classification metrics.


