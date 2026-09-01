.predictMLNBART <- function(object, newdata, ndpost, output_dir, load, verbose) {
  if (!object$save_trees) stop("You didn't save the trees. You need to run again using {save_trees=TRUE}.")
  n <- nrow(newdata)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  cat("Saving posterior predictions in", output_dir, "\n")
  # Perform the predictions
  object$cpp_obj$Predict(newdata, as.integer(ndpost), as.integer(object$ntrees),
                         object$forests_dir, output_dir, as.integer(verbose))
  # load results
  if (load) {
    pred <- load_bin_predictions(fname = file.path(output_dir, "theta_ij.bin"),
                                 n = n, d = object$d, m = ndpost)
    return(pred)
  }
  return(invisible())
}
.predictZANIBART <- function(object, newdata, type, ndpost, output_dir, load,
                             verbose) {

  if (!object$save_trees) stop("You didn't save the forests/trees. You need to run again using {save_trees=TRUE}.")

  n <- nrow(newdata)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  cat("Saving posterior predictions in", output_dir, "\n")

  # Perform the predictions
  switch(type,
         "theta" = object$cpp_obj$ComputePredictProb(newdata, as.integer(ndpost),
                                                     as.integer(object$ntrees_theta),
                                                     object$forests_dir, output_dir,
                                                     as.integer(verbose)),
         "zeta" = object$cpp_obj$ComputePredictProbZero(newdata, as.integer(ndpost),
                                                        as.integer(object$ntrees_zeta),
                                                        object$forests_dir, output_dir,
                                                        as.integer(verbose)))
  # load results
  if (load) {
    pred <- load_bin_predictions(fname = file.path(output_dir, paste0(type, "_ij.bin")),
                                 n = n, d = object$d, m = ndpost)
    return(pred)
  }

  return(invisible())
}

# Dispatch methods for different models

#' Posterior predictions
#'
#' Compute the posterior predictions for the ML-BART, MLN-BART, ZANIM-BART,
#' ZANIM-LN-BART, DM-reg and ZANIDM-reg models for a given newdata across all the
#' posterior draws of model parameters.
#'
#' For BART-based models, the predictions can be written to binary files on
#' disk. The `load` argument controls whether these draws are loaded into
#' R memory.
#'
#' @param object A fitted model object. Supported classes include
#' `MultinomialBART`, `MultinomialLNBART`, `ZANIMBART`,
#' `ZANIMLNBART`, `DMRegression`, and  `ZANIDMRegression`.
#' @param newdata A matrix or data frame containing the covariates for which
#' predictions are required. The columns must correspond to the predictors
#' used when fitting the model.
#' @param type The type of prediction to return. For `ZANIMBART` and
#' `ZANIMLNBART`, choices are `"theta"` and `"zeta"`. For
#' `DMRegression`, choices are `"alpha"` and `"theta"`. For
#' `ZANIDMRegression`, choices are `"alpha"`, `"zeta"`, and
#' `"theta"`.
#' @param ndpost Number of posterior draws to use for prediction. Default is
#' \code{object$ndpost}.
#' @param output_dir Directory in which the predictions across all posterior draws
#' are written. The resulting binary files are named according to the predicted
#' quantity, for example `theta_ij.bin` for `type="theta"` or
#' `zeta_ij.bin` for `type="zeta"`. The default is [tempdir()].
#' @param load Logical indicating whether prediction draws written to binary
#' files should be loaded into R memory. Defaults to `TRUE`. Set to
#' `FALSE` when predictions are too large to comfortably fit in memory.
#' @param verbose Logical indicating whether progress should be displayed.
#' The default is `TRUE`.
#' @param ... Catches unused arguments.
#'
#' @return Posterior prediction draws. When `load = TRUE`, the draws are
#'   loaded into R memory. When `load = FALSE`, the prediction draws are
#'   written to the files in `output_dir`.
#' @rdname predict.cc
#' @export
predict.MultinomialBART <- function(object, newdata, ndpost = object$ndpost,
                                    output_dir = tempdir(), load = TRUE,
                                    verbose = TRUE, ...) {
  .predictMLNBART(object = object, newdata = newdata, ndpost = ndpost,
                  output_dir = output_dir, load = load, verbose = verbose)
}

#' @export
#' @rdname predict.cc
predict.MultinomialLNBART <- function(object, newdata, ndpost = object$ndpost,
                                      output_dir = tempdir(), load = TRUE,
                                      verbose = TRUE, ...) {
  .predictMLNBART(object = object, newdata = newdata, ndpost = ndpost,
                  output_dir = output_dir, load = load, verbose = verbose)
}

#' @export
#' @rdname predict.cc
predict.ZANIMBART <- function(object, newdata, type = c("theta", "zeta"),
                              ndpost = object$ndpost, output_dir = tempdir(),
                              load = TRUE, verbose = TRUE, ...) {
  type <- match.arg(type)
  .predictZANIBART(object, newdata = newdata, type = type, ndpost = ndpost,
                   output_dir = output_dir, load = load, verbose = verbose)
}
#' @export
#' @rdname predict.cc
predict.ZANIMLNBART <- function(object, newdata, type = c("theta", "zeta"),
                                ndpost = object$ndpost, output_dir = tempdir(),
                                load = TRUE, verbose = TRUE, ...) {
  type <- match.arg(type)
  .predictZANIBART(object, newdata = newdata, type = type, ndpost = ndpost,
                   output_dir = output_dir, load = load, verbose = verbose)
}

#' @export
#' @rdname predict.cc
predict.DMRegression <- function(object, newdata, type = c("alpha", "theta"),
                                 ndpost = object$ndpost, verbose = TRUE, ...) {
  type <- match.arg(type)
  if (!is.null(object$draws_betas)) {
    predictions <- apply(object$draws_betas, 3,
                         function(b) newdata %*% b, simplify = "array")
  } else {
    ff <- file.path(object$dir_draws, "draws_betas.bin")
    if (!file.exists(ff))
      stop("File {draws_betas.bin} with the posterior draws of the regression coefficient doesn't exist.")
    draws_betas <- load_bin_coefficients(fname = ff, p = object$p, d = object$d, m = ndpost)
    predictions <- apply(draws_betas, 3,
                         function(b) newdata %*% b, simplify = "array")
  }
  predictions <- exp(simplify2array(predictions))
  if (type == "theta") return(sweep(predictions, MARGIN = c(1, 3), apply(predictions, c(1, 3), sum), "/"))
  return(predictions)
}

#' @export
#' @rdname predict.cc
predict.ZANIDMRegression <- function(object, newdata, type = c("alpha", "zeta", "theta"),
                                     ndpost = object$ndpost, verbose = TRUE, ...) {
  type <- match.arg(type)
  if (!is.null(object$draws_betas_alpha) && !is.null(object$draws_betas_alpha)) {
    res <- switch(type,
      "alpha" = apply(object$draws_betas_alpha, 3,
                      function(b) newdata %*% b, simplify = "array"),
      "theta" = apply(object$draws_betas_alpha, 3,
                      function(b) newdata %*% b, simplify = "array"),
      "zeta" = apply(object$draws_betas_zeta, 3,
                     function(b) newdata %*% b, simplify = "array")
    )
  } else {
    if (type == "zeta") {
      ff <- file.path(object$dir_draws, "draws_betas_zeta.bin")
      if (!file.exists(ff))
        stop("File {draws_betas_zeta.bin} with the posterior draws of the zero-inflation components regression coefficient doesn't exist.")
      draws <- load_bin_coefficients(fname = ff, p = object$p_zeta, d = object$d,
                                     m = ndpost)
      res <- apply(draws, 3, function(b) newdata %*% b, simplify = "array")
    } else {
      ff <- file.path(object$dir_draws, "draws_betas_alpha.bin")
      if (!file.exists(ff))
        stop("File {draws_betas_zeta.bin} with the posterior draws of the count components regression coefficient doesn't exist.")
      draws <- load_bin_coefficients(fname = ff, p = object$p_alpha, d = object$d,
                                     m = ndpost)
      res <- apply(draws, 3, function(b) newdata %*% b, simplify = "array")
    }
  }
  res <- simplify2array(res)
  res <- if (type == "zeta") stats::pnorm(res) else exp(res)
  if (type == "theta")
    return(sweep(res, MARGIN = c(1, 3), apply(res, c(1, 3), sum), "/"))
  return(res)
}

