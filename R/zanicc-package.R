if (getRversion() >= "2.15.1") utils::globalVariables(c("self", "private", "x",
                                                        "theta", "ci_lower", "ci_upper",
                                                        "label_model"))
#' @name zanicc-package
#' @aliases zanicc-package
#'
#' @title \pkg{zanicc}: zero-and-N-inflated count-compositional models
#'
#' @description
#'
#' The \pkg{zanicc} package provides tools for Bayesian (non)parametric regression
#' analysis of zero-inflated count-compositional data.
#'
#' Its primary contribution is the implementation of Bayesian inference schemes
#' for the ZANIM-BART and ZANIM-LN-BART models developed by Menezes et al. (2026a).
#' The package also implements other models, including the ML-BART model of Murray (2021) and the
#' ZANIDM (ZIDM) linear regression model of Koslovsky (2023).
#'
#' In addition to these regression models, \pkg{zanicc} provides efficient
#' implementations of the finite-mixture probability mass functions (PMFs) for
#' the ZANIM and ZANIDM distributions proposed by Menezes et al. (2025).
#' Finally, it implements Bayesian modular framework proposed by Menezes et al. (2026b)
#' for performing pollen-based palaeoclimate reconstruction.
#'
#' For efficient computing, the core implementation of the MCMC algorithms and
#' related functions are written in C++.
#'
#' The terminology `zani` ("zero-and-N-inflation") comes from the fact that, in
#' multivariate count-compositional settings, it can happen that zeros co-occur in one or more
#' categories. In the extreme case of N-inflation, when all but one category exhibits a count of
#' zero, the count for the remaining category coincides with the number of trials.
#'
#' @section Usage:
#'
#' The most important functions of the package are [zanicc()], which performs
#' Bayesian inference for several (non)parametric count compositional models, and
#' [inverse_posterior_zanimlnbart()] for performing pollen-based palaeoclimate
#' reconstruction using a modular framework built on a previously fitted
#' ZANIM-LN-BART model.
#'
#' The package also provides other relevant functions for analysing the output
#' of [zanicc()] are.
#' These include [ppd()], which generates draws from the posterior predictive distribution
#' of the fitted model;
#' [plot_ppc()], which computes and displays posterior predictive checks tailored to
#' count-compositional data;
#' [plot_qqplots_ppd()], which generates Q-Q plots based on the posterior predictive
#' distribution, and
#' [predict()], which computes posterior predictions for several models, including
#' those with BART prior.
#'
#' @details
#'
#' \describe{
#' \item{Type: }{Package}
#' \item{Package: }{\pkg{zanicc}}
#' \item{Version: }{0.0.2}
#' \item{Date: }{XXXX-XX-XX (this version), XXXX-XX-XX (original release)}
#' \item{Licence: }{GPL (>= 3)}
#' }
#'
#'
#' @references
#'
#' Koslovsky, M. D. (2023), A Bayesian zero-inflated Dirichlet-multinomial regression model for multivariate
#' compositional count data. \emph{Biometrics} \strong{79(4)}, 3239–3251.
#'
#' Menezes, A. F. B., Parnell, A. C., and Murphy, K. (2025), Finite mixture representations of
#' zero-and-N-inflated distributions for count-compositional data. \emph{Journal of Multivariate Analysis},
#' \strong{210:105492}. <https://doi.org/10.1016/j.jmva.2025.105492>.
#'
#' Menezes, A. F. B., Parnell, A. C. and Murphy, K. (2026a),
#' Bayesian nonparametric models for zero-inflated count-compositional data using
#' ensembles of regression trees. \emph{arXiv preprint}, \strong{arXiv:2601.08067}
#' <https://arxiv.org/abs/2601.08067v2>.
#'
#' Menezes, A. F. B., Parnell, A. C., Huntley, B., and Murphy, K. (2026b),
#' Bayesian palaeoclimate reconstruction from zero-inflated count-compositional pollen data:
#' A case study of Lago Grande di Monticchio in southern Italy. \emph{arXiv preprint},
#' \strong{arXiv:2609.08866} <https://arxiv.org/abs/2609.08866>.
#'
#' Murray, J. S. (2021), Log-linear Bayesian additive regression trees for
#' multinomial logistic and count regression models.
#' \emph{Journal of the American Statistical Association}, \strong{116} (534), 756--769.
#'
#' @author
#' André F. B. Menezes \email{andrefelipemaringa@gmail.com}
#'
#' Keefe Murphy \email{keefe.murphy@mu.ie}
#'
#' @useDynLib zanicc, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL
