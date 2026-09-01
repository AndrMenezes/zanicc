if (getRversion() >= "2.15.1")  utils::globalVariables(c("self", "private"))
#' @name zanicc-package
#' @aliases zanicc-package
#'
#' @title Overview of the \pkg{zanicc} package
#'
#' @description
#' The \pkg{zanicc} R package provides functions for the analysis of zero-inflated
#' count-compositional data.
#'
#' Count-compositional data are multivariate count data constrained by sample-specific
#' totals.
#'
#' The terminology `zani` ("zero-and-N-inflation") comes from the fact that, in
#' multivariate count-compositional settings, it can happen that zeros co-occur in one or more
#' categories. In the extreme case of N-inflation, when all but one category exhibits a count of
#' zero, the count for the remaining category coincides with the number of trials.
#'
#' @author
#' André F. B. Menezes \email{andrefelipemaringa@gmail.com}
#'
#' Keefe Murphy \email{keefe.murphy@mu.ie}
#'
#' @useDynLib zanicc, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL
