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

#' Importance sampling approximate the inverse posterior under ML-BART models
#' @param y observed count vector.
#' @param posterior_fx array with the posterior draws of f_j(x_i*).
.prob_x_mlbart <- function(y, posterior_fx) {
  d <- length(y)
  ndpost <- ncol(posterior_fx)
  ngrid <- nrow(posterior_fx)
  # Expand the Y's to evaluate the density in vectorise way
  Y_expanded <- matrix(rep(y, times = ngrid), nrow = ngrid, ncol = d,
                       byrow = TRUE)
  # For each draw k compute the probability over the `ngrid` observations
  prob_x <- matrix(nrow = ngrid, ncol = ndpost)
  for (k in seq_len(ndpost)) {
    log_w <- .dmultinomial(x = Y_expanded, prob = posterior_fx[,,k])
    w <- exp(log_w - max(log_w))
    prob_x[, k] <- w / sum(w)
  }
  rowMeans(prob_x)
}

.prob_x_zanimbart <- function(y, posterior_fx_theta, posterior_fx_zeta) {
  d <- length(y)
  ndpost <- ncol(posterior_fx_theta)
  ngrid <- nrow(posterior_fx_theta)
  seqn <- seq_len(ngrid)
  # For each draw k compute the probability over the `ngrid` observations
  prob_x <- matrix(nrow = ngrid, ncol = ndpost)
  for (k in seq_len(ndpost)) {
    log_w <- lapply(seqn, function(i) {
      log_pmf_zanim(x = y, prob = posterior_fx_theta[i, ,k],
                    zeta = posterior_fx_zeta[i, ,k])
    })
    log_w <- unlist(log_w)
    w <- exp(log_w - max(log_w))
    prob_x[, k] <- w / sum(w)
  }
  rowMeans(prob_x)
}


#' Compute the posterior-predictive distribution of the latent field over the proposal
#' distribution of \eqn{x^\ast \sim q(x^\ast)}.
#' @param object an object of class MultinomialBART
#' @param proposal_parms,n_proposal proposal parameters and number of points to draw
#' from the proposal distribution.
#' @param load logical. Whether the posterior-predictive distribution of \eqn{f(x^\ast)}.
#' should be loaded in memory.
#' @param save logical, whether to save the simulated proposal values for \eqn{x^\ast} in the `output_dir` folder.
#' @param output_dir where to save the predictions, \eqn{f(x^\ast)} and the proposal \eqn{x^\ast}.
#' @param x_proposal matrix with proposal values for \eqn{x^\ast}. Default is `NULL`, which then
#' generate uniform random values as the proposal distribution of x*.
#' @export
compute_proposal_fx_mlbart <- function(object, proposal_parms, n_proposal,
                                        load = TRUE, save = FALSE,
                                        output_dir = tempdir(), x_proposal = NULL) {

  if (is.null(x_proposal)) {
    # Simulate x* ~ q(x*) from uniform proposal
    if (object$p == 1L) {
      # x_proposal <- matrix(stats::runif(n_proposal, min = a, max = b), ncol = 1L)
      x_proposal <- matrix(seq(from = proposal_parms$min_x,
                               to = proposal_parms$max_x, length.out = n_proposal),
                           ncol = 1L)
    } else {
      x_proposal <- rconvexhull(n = n_proposal, X = proposal_parms$X)
    }
  }

  # Compute the posterior distribution of f_j(x*) for x*~\pi(x*) for j=1,...,d
  posterior_fx <- predict(object, newdata = x_proposal, load = load,
                          output_dir = output_dir)
  if (save)
    saveRDS(object = x_proposal, file = file.path(output_dir, "x_proposal.rds"))
  if (load) return(list(x_proposal = x_proposal, posterior_fx = posterior_fx))
  return(invisible())
}

#' Compute the posterior distribution of individual-level probabilities under the ZANIM-BART model
#' @param thetas array with \eqn{(n \times d \times r)} dimension for the posterior distribution of \eqn{\theta_{ij}^{(r)}}.
#' @param zetas array with \eqn{(n \times d \times r)} dimension for the posterior distribution of \eqn{\zeta_{ij}^{(r)}}.
#' @param verbose logical to keep track of the posterior draws.
#' @param printevery integer to print the posterior draws.
compute_vartheta_zanimbart <- function(thetas, zetas, verbose = FALSE,
                                       printevery = 100L)  {
  n_sample <- dim(thetas)[1L]
  d <- dim(thetas)[2L]
  ndpost <- dim(thetas)[3L]
  seqn <- seq_len(n_sample)
  draws <- array(data = NA_real_, dim = c(n_sample, d, ndpost))
  for (k in seq_len(ndpost)) {
    if (verbose && (k %% printevery == 0L)) cat(k, "\n")
    # Generate the z's
    tmp <- lapply(seqn, function(i) {
      z <- stats::rbinom(n = d, size = 1, prob = 1 - zetas[i,,k])
      is_zero <- z == 0L
      if (all(is_zero)) {
        vt <- rep(0.0, d)
      }
      else if (sum(is_zero) == d - 1L) {
        vt <- rep(0.0, d)
        vt[!is_zero] <- 1.0
      } else {
        vt <- thetas[i,,k] * z
        vt <- vt / sum(vt)
      }
      vt
    })
    draws[,,k] <- do.call(rbind, tmp)
  }
  draws
}

compute_proposal_fx_zanimbart <- function(object, proposal_parms, n_proposal,
                                          load = TRUE, save = FALSE,
                                          output_dir = tempdir(),
                                          x_proposal = NULL,
                                          conditional = FALSE) {

  if (is.null(x_proposal)) {
    # Simulate x* ~ q(x*) from uniform proposal
    if (object$p_theta == 1L) {
      x_proposal <- matrix(seq(from = proposal_parms$min_x,
                               to = proposal_parms$max_x, length.out = n_proposal),
                           ncol = 1L)
    } else {
      x_proposal <- rconvexhull(n = n_proposal, X = proposal_parms$X)
    }
  }

  # Compute the posterior distribution of f^{(c)}_j(x*) and f^{(0)}_j(x*)
  # for x*~\pi(x*) and j=1,...,d
  posterior_fx_theta <- predict(object, newdata = x_proposal, load = load,
                                output_dir = output_dir, type = "theta")
  posterior_fx_zeta <- predict(object, newdata = x_proposal, load = load,
                               output_dir = output_dir, type = "zeta")

  if (conditional) {
    vartheta <- compute_vartheta_zanimbart(thetas = posterior_fx_theta,
                                           zetas = posterior_fx_zeta, verbose = FALSE)
    if (save)
      saveRDS(object = vartheta, file = file.path(output_dir, "vartheta.rds"))
    if (load) return(list(x_proposal = x_proposal, posterior_fx = vartheta))
  }

  if (save)
    saveRDS(object = x_proposal, file = file.path(output_dir, "x_proposal.rds"))
  if (load) return(list(x_proposal = x_proposal,
                        posterior_fx_theta = posterior_fx_theta,
                        posterior_fx_zeta = posterior_fx_zeta))
  return(invisible())
}

#' Importance sampling for approximate the inverse posterior distribution using the
#' ML-BART model.
.is_mlbart <- function(object, Y, proposal_parms, n_proposal = 5000L,
                       sir = FALSE, n_resampling = floor(n_proposal/2),
                       output_dir = tempdir(), save = FALSE,
                       dir_posterior_fx = NULL, x_proposal = NULL) {

  n_samples <- nrow(Y)

  # Load the proposal and the posterior fx
  if (!is.null(dir_posterior_fx)) {
    if (!file.exists(file.path(dir_posterior_fx, "theta_ij.bin")))
      stop("file {theta_ij.bin} with the posterior distribution of f(x*) doesn't exist in the folder {dir_posterior_fx}")
    if (!file.exists(file.path(dir_posterior_fx, "x_proposal.rds")))
      stop("file {x_proposal.rds} with the proposal distribution x* doesn't exist in the folder {dir_posterior_fx}")

    cat("Loading the proposal for x* and the posterior-predictive of f(x*) \n")
    x_proposal <- readRDS(file = file.path(dir_posterior_fx, "x_proposal.rds"))
    n_proposal <- nrow(x_proposal)
    posterior_fx <- load_bin_predictions(fname = file.path(dir_posterior_fx, "theta_ij.bin"),
                                         n = n_proposal, d = object$d, m = object$ndpost)
    cache_proposal_fx <- list(posterior_fx = posterior_fx, x_proposal = x_proposal)
  } else {
    cat("Generaing the proposal for x* and computing the posterior-predictive of f_j(x*) \n")
    # Generate proposal
    cache_proposal_fx <- compute_proposal_fx_mlbart(object, proposal_parms, n_proposal,
                                                     load = TRUE, save = save,
                                                     output_dir = output_dir,
                                                     x_proposal = x_proposal)
  }
  cat("Computing the probabilities using importance sampling...\n")
  # Estimated importance sampling probabilities
  probs <- apply(Y, 1, function(y) {
    .prob_x_mlbart(y = y, posterior_fx = cache_proposal_fx$posterior_fx)
  })

  # If SIR then return a list with the resampling x_proposal
  if (sir) {
    cat("Resampling...\n")
    x_sir <- apply(probs, 2, function(p) {
      ids <- sample.int(n = n_proposal, size = n_resampling, replace = TRUE, prob = p)
      cache_proposal_fx$x_proposal[ids, , drop = FALSE]
    }, simplify = FALSE)
    attr(x_sir, "is_probs") <- probs
    return(x_sir)
  }
  # Otherwise return a matrix with first column the probabilities a
  return(cbind(probs = probs, cache_proposal_fx$x_proposal))
}

#' Importance sampling for approximate the inverse posterior distribution using the
#' ZANIM-BART model.
.is_zanimbart <- function(object, Y, proposal_parms, n_proposal = 5000L,
                          sir = FALSE, n_resampling = floor(n_proposal/2),
                          conditional = FALSE,
                          output_dir = tempdir(), save = FALSE,
                          dir_posterior_fx = NULL, x_proposal = NULL) {

  n_samples <- nrow(Y)

  # Load the proposal and the posterior fx
  if (!is.null(dir_posterior_fx)) {
    cat("Loading the proposal for x* and the posterior-predictive of f(x*) \n")
    fproposal <- file.path(dir_posterior_fx, "x_proposal.rds")
    if (!file.exists(fproposal))
      stop("file {x_proposal.rds} with the proposal distribution x* doesn't exist in the folder {dir_posterior_fx}")
    x_proposal <- readRDS(file = fproposal)
    n_proposal <- nrow(x_proposal)

    if (conditional) {
      fvartheta <- file.path(dir_posterior_fx, "vartheta.rds")
      if (!file.exists(fvartheta))
        stop("file {vartheta.rds} with the posterior distribution of \vartheta_ij doesn't exist in the folder {dir_posterior_fx}")
      vartheta <- readRDS(file = fvartheta)
      cache_proposal_fx <- list(posterior_fx = vartheta, x_proposal = x_proposal)
    } else {
      ftheta <- file.path(dir_posterior_fx, "theta_ij.bin")
      fzeta <- file.path(dir_posterior_fx, "zeta_ij.bin")
      if (!file.exists(ftheta))
        stop("file {theta_ij.bin} with the posterior distribution of f^{(c)}(x*) doesn't exist in the folder {dir_posterior_fx}")
      if (!file.exists(fzeta))
        stop("file {zeta_ij.bin} with the posterior distribution of f^{(0)}(x*) doesn't exist in the folder {dir_posterior_fx}")
      posterior_fx_theta <- load_bin_predictions(fname = ftheta, n = n_proposal,
                                                 d = object$d, m = object$ndpost)
      posterior_fx_zeta <- load_bin_predictions(fname = fzeta, n = n_proposal,
                                                d = object$d, m = object$ndpost)
      cache_proposal_fx <- list(posterior_fx_theta = posterior_fx_theta,
                                posterior_fx_zeta = posterior_fx_zeta,
                                x_proposal = x_proposal)
    }
  } else {
    cat("Generaing the proposal for x* and computing the posterior-predictive of f^{(c)}_j(x*) and f^{(0)}_j(x*) \n")
    # Generate proposal
    cache_proposal_fx <- compute_proposal_fx_zanimbart(object,
                                                       proposal_parms = proposal_parms,
                                                       n_proposal = n_proposal,
                                                       load = TRUE, save = save,
                                                       output_dir = output_dir,
                                                       x_proposal = x_proposal,
                                                       conditional = conditional)
  }
  cat("Computing the probabilities using importance sampling...\n")
  # Estimated importance sampling probabilities
  if (conditional) {
    probs <- apply(Y, 1, function(y) {
      .prob_x_mlbart(y = y, posterior_fx = cache_proposal_fx$posterior_fx)
    })
  } else {
    probs <- apply(Y, 1, function(y) {
      .prob_x_zanimbart(y = y,
                        posterior_fx_theta = cache_proposal_fx$posterior_fx_theta,
                        posterior_fx_zeta = cache_proposal_fx$posterior_fx_zeta)
    })
  }

  # If SIR then return a list with the resampling x_proposal
  if (sir) {
    cat("Resampling...\n")
    x_sir <- apply(probs, 2, function(p) {
      ids <- sample.int(n = n_proposal, size = n_resampling, replace = TRUE,
                        prob = p)
      cache_proposal_fx$x_proposal[ids, , drop = FALSE]
    }, simplify = FALSE)
    attr(x_sir, "is_probs") <- probs
    return(x_sir)
  }
  # Otherwise, return a matrix with first column with the probabilities and the
  # remaining ones the proposal x
  return(cbind(probs = probs, cache_proposal_fx$x_proposal))
}


#' Two-stage Gibbs sampler for the inverse posterior using the elliptical slice sampling.
.gibbs_mlbart <- function(Y, mean_prior, S_prior, X_ini = NULL, forests_dir,
                           ntrees, ndpost, n_rep = 2L) {
  # Define new module
  ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
  cpp_obj <- new(ml$InversePosterior, ncol(Y), ntrees, ntrees, "ml_bart",
                 forests_dir)
  # If there is no initial value sample from the prior
  n <- nrow(Y)
  p <- length(mean_prior)
  if (is.null(X_ini)) {
    X_ini <- matrix(nrow = n, ncol = p)
    cS <- chol(S_prior)
    for (i in seq_len(n)) X_ini[i, ] <- stats::rnorm(p) %*% cS + mean_prior
  }
  xx <- cpp_obj$SamplerMLBARTeSS(Y, as.matrix(X_ini), as.integer(ndpost),
                                 mean_prior, S_prior, n_rep)
  array(xx, dim = c(ndpost, p, n))
}



#' Sample the inverse posterior distribution using the ML-BART model.
#' @param object an object of class MultinomialBART.
#' @param Y matrix with the compositional counts to compute the inverse posterior of x*.
#' @param method which sampler method to use. Options are `is` (importance sampling)
#'  and `gibbs` (two-stage Gibbs sampler).
#' @param proposal_parms list with the parameters to simulate from a uniform proposal
#' when performing the importance sampling.
#' If X is one-dimensional the list should contain the range with entries `min_x` and
#' `max_x`.
#' Otherwise the list should contain an entry `X` with a matrix n by p of
#' representing the observed covariates.
#' Then, the proposal distribution is an uniform over the convex hull of
#' observed `X`.
#' @param ess_parms list with parameters that control the elliptical slice sampling (eSS).
#' using for the `gibbs` method. The list should contain the following entries.
#' `mean_prior`: prior mean vector using in the eSS.
#' `S_prior`: prior covariance matrix using in the eSS.
#' `n_rep`:  number of times to repeat one iteration of eSS.
#' `X_ini`: (optional) matrix with the initial value for the eSS. If not passed, then
#' sample from the multivariate normal prior.
#' @param n_proposal integer indicating the number samples from the proposal distribution.
#' @param sir logical. Whether perform sampling importance resampling with replacement.
#' @param n_resampling integer. Number of samples to resample. Default is `floor(n_proposal/2)`.
#' @param output_dir valid character folder to write the posterior-predictive distribution \eqn{f(x^\ast)}.
#' Default is `tempdir()`.
#' @param save logical, whether to save the simulated proposal values for \eqn{x^\ast} in the `output_dir` folder.
#' @param dir_posterior_fx valid character folder to read both the simulated
#' proposal values \eqn{x^\ast} and the posterior-predictive distribution
#' \eqn{f(x^\ast)}. Default is `NULL`.
#' @param x_proposal matrix with the proposal distribution, if the user wants to
#' give a different proposal distribution. Default is `NULL`.
#' @export
inverse_posterior_mlbart <- function(object, Y, method = c("is", "gibbs"),
                                     proposal_parms = NULL, ess_parms = NULL,
                                     n_proposal = 5000L, sir = FALSE,
                                     n_resampling = floor(n_proposal / 2.0),
                                     dir_posterior_fx = NULL, save = FALSE,
                                     output_dir = tempdir(), x_proposal = NULL) {
  method <- match.arg(method)
  if (method == "is") {
    res <- .is_mlbart(object = object, Y = Y, proposal_parms = proposal_parms,
                      n_proposal = n_proposal, sir = sir, n_resampling = n_resampling,
                      dir_posterior_fx = dir_posterior_fx, save = save,
                      output_dir = output_dir, x_proposal = x_proposal)
  } else {
    res <- .gibbs_mlbart(Y = Y, mean_prior = ess_parms$mean_prior, ess_parms$S_prior,
                          X_ini = ess_parms$X_ini, forests_dir = object$forests_dir,
                          ntrees = object$ntrees, ndpost = object$ndpost,
                          n_rep = ess_parms$n_rep)
  }
  res
}




