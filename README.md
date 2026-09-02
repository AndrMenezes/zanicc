
<!-- README.md is generated from README.Rmd. Please edit that file -->

# `zanicc` R package

<!-- badges: start -->

<!-- badges: end -->

The `zanicc` R package implements Bayesian nonparametric models for the
analysis of zero-inflated count count-compositional data. Its first main
contribution is an implementation of the models introduced in the paper
[“Bayesian nonparametric models for zero-inflated count-compositional
data using ensembles of regression
trees”](https://arxiv.org/abs/2601.08067) by André F. B. Menezes, Andrew
C. Parnell, and Keefe Murphy, along with other related models from the
literature on count-compositional modelling such as the multinomial
logistic BART and parametric regression models based on the
Dirichlet-multinomial (DM) and zero-and-N-inflated DM distributions.

Further functionalities are provided which implement the Bayesian
modular framework for pollen-based palaeoclimate reconstruction
following the paper “Bayesian palaeoclimate reconstruction from
zero-inflated count-compositional pollen data: A case study of Lago
Grande di Monticchio in southern Italy” (*to appear*), by the same
authors.

You can install the development version of `zanicc` from GitHub with:

``` r
remotes::install_github("AndrMenezes/zanicc")
```

> Code to reproduce the simulations and the real data analyses presented
> in the first paper can be found in
> [AndrMenezes/reproduce\_\_zanim_ln_bart](https://github.com/AndrMenezes/reproduce__zanim_ln_bart).

> Code to reproduce the simulations and the Lago Grande di Monticchio
> case study presented in the second paper can be found in
> [AndrMenezes/reproduce\_\_palaeoclimate_zanim_ln_bart](https://github.com/AndrMenezes/reproduce__palaeoclimate_zanim_ln_bart).
