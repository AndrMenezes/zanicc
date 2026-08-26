
<!-- README.md is generated from README.Rmd. Please edit that file -->

# `zanicc` R package

<!-- badges: start -->
<!-- badges: end -->

The `zanicc` R package implements Bayesian nonparametric models for
analysis of zero-inflated count count-compositional data. Its main
contribution is an implementation of the models introduced in the paper
[“Bayesian nonparametric models for zero-inflated count-compositional
data using ensembles of regression
trees”](https://arxiv.org/abs/2601.08067) by André F. B. Menezes, Andrew
C. Parnell, and Keefe Murphy, along with other related models from the
literature such as the multinomial logistic BART and the parametric
regression model based on the Dirichet-multinomial (DM) and the
zero-and-N-inflated DM distributions.

You can install the development version of `zanicc` from GitHub with:

``` r
remotes::install_github("AndrMenezes/zanicc")
```

> Code to reproduce the simulations and the real data analyses presented
> in the paper can be found in
> [AndrMenezes/zanicc_paper](https://github.com/AndrMenezes/zanicc_paper).
