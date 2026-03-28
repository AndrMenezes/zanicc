.predictMLNBART <- function(object, newdata, ndpost, output_dir, load, verbose) {
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

#' @export
predict.MultinomialBART <- function(object, newdata, ndpost = object$ndpost,
                                    output_dir = tempdir(), load = TRUE,
                                    verbose = TRUE) {
  .predictMLNBART(object = object, newdata = newdata, ndpost = ndpost,
                  output_dir = output_dir, load = load, verbose = verbose)
}

#' @export
predict.MultinomialLNBART <- function(object, newdata, ndpost = object$ndpost,
                                      output_dir = tempdir(), load = TRUE,
                                      verbose = TRUE) {
  .predictMLNBART(object = object, newdata = newdata, ndpost = ndpost,
                  output_dir = output_dir, load = load, verbose = verbose)
}

#' @export
predict.ZANIMBART <- function(object, newdata, type = c("theta", "zeta"),
                              ndpost = object$ndpost, output_dir = tempdir(),
                              load = TRUE, verbose = TRUE) {
  type <- match.arg(type)
  .predictZANIBART(object, newdata = newdata, type = type, ndpost = ndpost,
                   output_dir = output_dir, load = load, verbose = verbose)
}
#' @export
predict.ZANIMLNBART <- function(object, newdata, type = c("theta", "zeta"),
                                ndpost = object$ndpost, output_dir = tempdir(),
                                load = TRUE, verbose = TRUE) {
  type <- match.arg(type)
  .predictZANIBART(object, newdata = newdata, type = type, ndpost = ndpost,
                   output_dir = output_dir, load = load, verbose = verbose)
}

#' @export
predict.DMRegression <- function(object, newdata, type = c("alpha", "theta"),
                                 ndpost = object$ndpost, verbose = TRUE) {
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
predict.ZANIDMRegression <- function(object, newdata, type = c("alpha", "zeta", "theta"),
                                     ndpost = object$ndpost, verbose = TRUE) {
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
  if (type == "theta") return(sweep(res, MARGIN = c(1, 3), apply(res, c(1, 3), sum),
                                    "/"))
  return(res)
}

