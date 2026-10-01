if (getRversion() >= "2.15.1") utils::globalVariables(c("self", "private", "x", "theta", "ci_lower", "ci_upper"))
#' @name zanicc-package
#' @aliases zanicc-package
#'
#' @title \pkg{zanicc}: zero-and-N-inflated count-compositional models
#'
#' @description
#' The \pkg{zanicc} is an R package for Bayesian (non)parametric regression analysis of
#' zero-inflated count-compositional data.
#' Most notably, the package implements the Bayesian inference schemes for the
#' ZANIM-BART and ZANIM-LN-BART models developed by Menezes et al. (2026a).
#' Moreover, it provides efficient implementation of the finite mixture probability
#' mass functions (PMFs) of ZANIM and ZANIDM distributions as proposed by Menezes et al. (2025).
#' Finally, it also provides tools for performing pollen-based palaeoclimate reconstruction
#' using the Bayesian modular framework proposed by Menezes et al. (2026b).
#' For efficient computing, the core implementation of MCMC algorithms and related functions
#' are written in C++.
#'
#' @details
#'
#' Count-compositional data are multivariate count data constrained by sample-specific
#' totals.
#'
#'
#'
#'
#' Probability mass functions and random number generation functions are also
#' provided for many of the multivariate count-compositional distributions.
#' Notably, C++ implementations are provided for the finite mixture PMFs
#' and marginal distributions of the ZANIM and ZANIDM distributions proposed
#' in Menezes et al. (2025), along with functions to compute the moments and
#' other properties derived therein.
#'
#' In addition, the package \pkg{zanicc} implements the Bayesian modular
#' framework introduced in Menezes et al. (2026b) for performing
#' pollen-based palaeoclimate reconstruction using any of the three proposed
#' inverse posterior sampling schemes.
#'
#' The terminology `zani` ("zero-and-N-inflation") comes from the fact that, in
#' multivariate count-compositional settings, it can happen that zeros co-occur in one or more
#' categories. In the extreme case of N-inflation, when all but one category exhibits a count of
#' zero, the count for the remaining category coincides with the number of trials.
#'
#' @references
#'
#' Menezes, A. F. B., Parnell, A. C., and Murphy, K. (2025), Finite mixture representations of
#' zero-and-N-inflated distributions for count-compositional data. \emph{Journal of Multivariate Analysis},
#' \strong{210:105492}. <https://doi.org/10.1016/j.jmva.2025.105492>.
#'
#' Menezes, A. F. B., Parnell, A. C. and Murphy, K. (2026a),
#' Bayesian nonparametric models for zero-inflated count-compositional data using
#' ensembles of regression trees. \emph{arXiv preprint}, \strong{arXiv:2601.08067}
#' <https://arxiv.org/abs/2601.08067v2>
#'
#' Menezes, A. F. B., Parnell, A. C., Huntley, B., and Murphy, K. (2026b),
#' Bayesian palaeoclimate reconstruction from zero-inflated count-compositional pollen data:
#' A case study of Lago Grande di Monticchio in southern Italy \emph{arXiv preprint},
#' \strong{arXiv:2609.08866} <https://arxiv.org/abs/2609.08866>
#'
#'
#' @author
#' André F. B. Menezes \email{andrefelipemaringa@gmail.com}
#'
#' Keefe Murphy \email{keefe.murphy@mu.ie}
#'
#' @useDynLib zanicc, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL
