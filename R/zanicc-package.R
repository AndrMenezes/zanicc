if (getRversion() >= "2.15.1")  utils::globalVariables(c("self", "private"))
#' @name zanicc-package
#' @aliases zanicc-package
#'
#' @title Overview of the \pkg{zanicc} package
#'
#' @description
#' The \pkg{zanicc} R package provides functions to analysis zero-inflated
#' count-compositional data.
#'
#' Count-compositional data are multivariate count data constrained by sample-specific
#' totals.
#' This
#'
#' The terminology `zani` comes from the fact that in
#' count-compositional settings it can happen
#'
#' @author
#' André F. B. Menezes \email{andrefelipemaringa@gmail.com}
#'
#' Keefe Murphy \email{keefe.murphy@mu.ie}
#'
#' @useDynLib zanicc, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL
