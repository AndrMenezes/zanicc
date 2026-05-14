#' Compute the volume (proportional to determinant) of each simplex
#' @param X matrix with p columns
#' @param i matrix with p+1 columns indicating the simplicies
#' Returns the determinants () of the simplices.
.volume_simplex <- function(i, X) {
  if (!is.matrix(i)) i <- matrix(i, nrow = 1)
  apply(i, 1, \(j) {
    S <- X[j, ]
    X0 <- apply(S[-1, ], 1, `-`, y = S[1, ])
    det(X0)
  })
}
#' Generate n samples from the convex hull of X
#' @param n integer for the number of samples.
#' @param X an n\times p matrix.
#' @description
#' This function was written by the user of Stack Exchange "whuber" for the following
#' post:
#' https://stats.stackexchange.com/questions/663559/transform-convex-hull-vertex-weights-for-uniform-distribution/663594#663594
#' @export
rconvexhull <- function(n, X) {
  # Triangulate the points. Indexes into simplices, by row.
  V <- geometry::delaunayn(X)
  # Sample the component simplices of the triangulation in p
  # to their volumes
  probs <- abs(.volume_simplex(V, X))
  probs <- probs / sum(probs)
  i <- sample.int(nrow(V), n, replace = TRUE, prob = probs)
  # Now, sample uniformly within each selected simplex.
  wts <- matrix(stats::rexp(n * (dim(X)[2] + 1)), nrow = n)
  wts <- wts / rowSums(wts)
  # Apply the weights to obtain Cartesian point coordinates.
  pts <- sapply(seq_len(n), \(j) {
    wts[j, ] %*% X[V[i[j], ], ]
  })
  t(pts)
}




#' Inverse posterior using the ZANIM-LN-BART model
#' @export
inverse_posterior_zanimlnbart <- function(object, Y, dir_posterior_fx,
                                          x_proposal = NULL,
                                          method = c("sir", "ess", "c_ess"),
                                          ndpost = object$ndpost, nburnin = 10L,
                                          mean_prior = NULL, S_prior = NULL,
                                          X_ini = NULL, Amat = NULL, bvec = NULL,
                                          eta = 50.0, mc = 10L) {
  # Some checks
  method <- match.arg(method)
  if (object$d != ncol(Y)) stop("Dimension of Y does not match with forward model")
  if (ndpost > object$ndpost) {
    warning("{ndpost} should be at least {object$ndpost}. Setting {ndpost} to  {object$ndpost}")
    ndpost <- object$ndpost
  }
  if (method == "sir" && is.null(x_proposal))
    stop("You should given the {x_proposal} for sir!")

  # Some common arguments
  B <- t(object$Bt)
  n <- nrow(Y)
  p <- object$p_theta

  # Initialise C++ class
  ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
  cpp_obj <- new(ml$InversePosterior, object$d, object$ntrees_theta,
                 object$ntrees_zeta, object$forests_dir)

  # Check which method to dispatch
  if (method == "sir") {
    n_proposal <- nrow(x_proposal)
    # Check if the files with parameter predictions exist
    ftheta <- file.path(dir_posterior_fx, "theta_ij.bin")
    fzeta <- file.path(dir_posterior_fx, "zeta_ij.bin")
    do_predict <- FALSE
    if (!file.exists(ftheta) || !file.exists(fzeta)) {
      do_predict <- TRUE
      cat("files {theta_ij.bin} and {zeta_ij.bin} with the posterior distribution of f^{(c)}(x*) and f^{(0)}(x*) do not exist in the folder {dir_posterior_fx}. Computing such predictions...\n")
      # Compute the posterior distribution of f^{(c)}_j(x*) and f^{(0)}_j(x*) for x*~\pi(x*) and j=1,...,d
      ini <- proc.time()
      predict(object, newdata = x_proposal, load = FALSE, output_dir = dir_posterior_fx,
              type = "theta")
      predict(object, newdata = x_proposal, load = FALSE, output_dir = dir_posterior_fx,
              type = "zeta")
      end_predict <- proc.time() - ini
    }
    # For each posterior draw of f's run SIR
    ini <- proc.time()
    res <- lapply(seq_len(n), function(i) {
      cat("Observation: ", i, "of", n, "\n")
      indices <- cpp_obj$SIRZANIMLNBART(Y[i, ], n_proposal, ndpost, B,
                                        dir_posterior_fx, mc)
      x_proposal[indices + 1L, , drop = FALSE] # C++ indices starts at 0
    })
    elapsed <- proc.time() - ini
    res <- simplify2array(res)
    if (do_predict) attr(res, "elapsed_time_predict") <- end_predict
  } else {
    if (is.null(mean_prior)) mean_prior <- rep(0.0, object$p_theta)
    if (is.null(S_prior)) S_prior <- diag(1.0, object$p_theta, object$p_theta)

    # If there is no initial value sample from the prior
    if (is.null(X_ini)) {
      X_ini <- matrix(nrow = n, ncol = p)
      cS <- chol(S_prior)
      for (i in seq_len(n)) X_ini[i, ] <- stats::rnorm(p) %*% cS + mean_prior
    }
    ini <- proc.time()
    xx <- switch(method,
      "ess" = cpp_obj$SamplerZANIMLNBARTeSS(Y, X_ini, ndpost, nburnin, mean_prior,
                                            S_prior, B, mc),
      "c_ess" = cpp_obj$SamplerZANIMLNBARTceSS(Y, X_ini, ndpost, nburnin,
                                               mean_prior, S_prior, B, Amat,
                                               bvec, eta)
    )
    elapsed <- proc.time() - ini
    res <- array(xx, dim = c(ndpost, p, n))
  }
  attr(res, "elapsed_time") <- elapsed
  res
}

#' Inverse posterior using the ZANIM-BART model
#' @export
inverse_posterior_zanimbart <- function(object, Y, dir_posterior_fx,
                                        x_proposal = NULL,
                                        method = c("sir", "ess"),
                                        ndpost = object$ndpost, nburnin = 10L,
                                        mean_prior = NULL, S_prior = NULL,
                                        X_ini = NULL, conditional_loglik = FALSE) {
  # Some checks
  method <- match.arg(method)
  if (object$d != ncol(Y)) stop("Dimension of Y does not match with forward model")
  if (ndpost > object$ndpost) {
    warning("{ndpost} should be at least {object$ndpost}. Setting {ndpost} to  {object$ndpost}")
    ndpost <- object$ndpost
  }
  if (method == "sir" && is.null(x_proposal))
    stop("You should given the {x_proposal} for sir!")

  # Some common arguments
  n <- nrow(Y)
  p <- object$p_theta

  # Initialise C++ class
  ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
  cpp_obj <- new(ml$InversePosterior, object$d, object$ntrees_theta,
                 object$ntrees_zeta, object$forests_dir)

  # Check which method to dispatch
  if (method == "sir") {
    n_proposal <- nrow(x_proposal)
    # Check if the files with parameter predictions exist
    ftheta <- file.path(dir_posterior_fx, "theta_ij.bin")
    fzeta <- file.path(dir_posterior_fx, "zeta_ij.bin")
    do_predict <- FALSE
    if (!file.exists(ftheta) || !file.exists(fzeta)) {
      do_predict <- TRUE
      cat("files {theta_ij.bin} and {zeta_ij.bin} with the posterior distribution of f^{(c)}(x*) and f^{(0)}(x*) do not exist in the folder {dir_posterior_fx}. Computing such predictions...\n")
      # Compute the posterior distribution of f^{(c)}_j(x*) and f^{(0)}_j(x*) for x*~\pi(x*) and j=1,...,d
      ini <- proc.time()
      predict(object, newdata = x_proposal, load = FALSE, output_dir = dir_posterior_fx,
              type = "theta")
      predict(object, newdata = x_proposal, load = FALSE, output_dir = dir_posterior_fx,
              type = "zeta")
      end_predict <- proc.time() - ini
    }
    # For each posterior draw of f's run SIR
    ini <- proc.time()
    res <- lapply(seq_len(n), function(i) {
      cat("Observation: ", i, "of", n, "\n")
      indices <- cpp_obj$SIRZANIMBART(Y[i, ], n_proposal, ndpost,
                                      dir_posterior_fx, as.integer(conditional_loglik))
      x_proposal[indices + 1L, , drop = FALSE] # C++ indices starts at 0
    })
    elapsed <- proc.time() - ini
    res <- simplify2array(res)
    if (do_predict) attr(res, "elapsed_time_predict") <- end_predict
  } else {
    if (is.null(mean_prior)) mean_prior <- rep(0.0, p)
    if (is.null(S_prior)) S_prior <- diag(1.0, p, p)

    # If there is no initial value sample from the prior
    if (is.null(X_ini)) {
      X_ini <- matrix(nrow = n, ncol = p)
      cS <- chol(S_prior)
      for (i in seq_len(n)) X_ini[i, ] <- stats::rnorm(p) %*% cS + mean_prior
    }
    ini <- proc.time()
    xx <- cpp_obj$SamplerZANIMBARTeSS(Y, X_ini, ndpost, mean_prior,
                                      S_prior, nburnin, as.integer(conditional_loglik))
    elapsed <- proc.time() - ini
    res <- array(xx, dim = c(ndpost, p, n))
  }
  attr(res, "elapsed_time") <- elapsed
  res
}

#' Inverse posterior using the ML-BART model
#' @export
inverse_posterior_mlbart <- function(object, Y, dir_posterior_fx,
                                          x_proposal = NULL,
                                          method = c("sir", "ess"),
                                          ndpost = object$ndpost, nburnin = 10L,
                                          mean_prior = NULL, S_prior = NULL,
                                          X_ini = NULL) {
  # Some checks
  method <- match.arg(method)
  if (object$d != ncol(Y)) stop("Dimension of Y does not match with forward model")
  if (ndpost > object$ndpost) {
    warning("{ndpost} should be at least {object$ndpost}. Setting {ndpost} to  {object$ndpost}")
    ndpost <- object$ndpost
  }
  if (method == "sir" && is.null(x_proposal))
    stop("You should given the {x_proposal} for sir!")
  # Some common arguments
  n <- nrow(Y)
  p <- object$p
  # Initialise C++ class
  ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
  cpp_obj <- new(ml$InversePosterior, object$d, object$ntrees,
                 object$ntrees, object$forests_dir)
  # Check which method to dispatch
  if (method == "sir") {
    n_proposal <- nrow(x_proposal)
    # Check if the files with parameter predictions exist
    ftheta <- file.path(dir_posterior_fx, "theta_ij.bin")
    do_predict <- FALSE
    if (!file.exists(ftheta)) {
      do_predict <- TRUE
      cat("files {theta_ij.bin} with the posterior distribution of f_j(x*) does not exist in the folder {dir_posterior_fx}. Computing such predictions...\n")
      # Compute the posterior distribution of f_j(x*) for x*~\pi(x*) and j=1,...,d
      ini <- proc.time()
      predict(object, newdata = x_proposal, load = FALSE, output_dir = dir_posterior_fx)
      end_predict <- proc.time() - ini
    }
    # For each posterior draw of f's run SIR
    ini <- proc.time()
    res <- lapply(seq_len(n), function(i) {
      cat("Observation: ", i, "of", n, "\n")
      indices <- cpp_obj$SIRMLBART(Y[i, ], n_proposal, ndpost, dir_posterior_fx)
      x_proposal[indices + 1L, , drop = FALSE] # C++ indices starts at 0
    })
    elapsed <- proc.time() - ini
    res <- simplify2array(res)
    if (do_predict) attr(res, "elapsed_time_predict") <- end_predict
  } else {
    if (is.null(mean_prior)) mean_prior <- rep(0.0, p)
    if (is.null(S_prior)) S_prior <- diag(1.0, p, p)
    # If there is no initial value sample from the prior
    if (is.null(X_ini)) {
      X_ini <- matrix(nrow = n, ncol = p)
      cS <- chol(S_prior)
      for (i in seq_len(n)) X_ini[i, ] <- stats::rnorm(p) %*% cS + mean_prior
    }
    ini <- proc.time()
    xx <- cpp_obj$SamplerMLBARTeSS(Y, X_ini, ndpost, mean_prior, S_prior, nburnin)
    elapsed <- proc.time() - ini
    res <- array(xx, dim = c(ndpost, p, n))
  }
  attr(res, "elapsed_time") <- elapsed
  res
}
