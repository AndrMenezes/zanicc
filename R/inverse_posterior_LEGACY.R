#' Importance sampling approximate the inverse posterior under ML-BART models
#' @param y observed count vector.
#' @param posterior_fx array with the posterior draws of f_j(x_i*).
.prob_x_mlbart <- function(y, posterior_fx) {
  d <- length(y)
  ndpost <- dim(posterior_fx)[3L]
  nproposal <- dim(posterior_fx)[1L]
  # Expand the Y's to evaluate the density in vectorise way
  Y_expanded <- matrix(rep(y, times = ndpost), nrow = ndpost, ncol = d,
                       byrow = TRUE)
  # For each proposal value (generate from the prior) compute the unnormalised target
  log_probs <- numeric(nproposal)
  for (i in seq_len(nproposal)) {
    log_w <- .dmultinomial(x = Y_expanded, prob = t(posterior_fx[i,,]))
    log_probs[i] <- zanicc:::.log_sum_exp(log_w)
  }
  # Self-normalise
  probs <- exp(log_probs)
  probs / sum(probs)
}

.prob_x_zanimbart <- function(y, posterior_fx_theta, posterior_fx_zeta) {
  d <- length(y)
  ndpost <- dim(posterior_fx_theta)[3L]
  nproposal <- dim(posterior_fx_theta)[1L]
  seqn <- seq_len(ndpost)
  # For each proposal value (generate from the prior) compute the unnormalised target
  log_probs <- numeric(nproposal)

  for (i in seq_len(nproposal)) {
    log_w <- lapply(seqn, function(k) {
      log_pmf_zanim(x = y,
                    prob = posterior_fx_theta[i, ,k],
                    zeta = posterior_fx_zeta[i, ,k])
    })
    log_w <- unlist(log_w)
    log_probs[i] <- zanicc:::.log_sum_exp(log_w)
  }
  # Self-normalise
  probs <- exp(log_probs)
  probs / sum(probs)
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
                                       output_dir = tempdir(), x_proposal = NULL,
                                       verbose = TRUE, printevery = 100L) {

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
  load_ <- load
  if (is(object, "MultinomialLNBART")) load <- TRUE
  # Compute the posterior distribution of f_j(x*) for x*~\pi(x*) for j=1,...,d
  posterior_fx <- predict(object, newdata = x_proposal, load = load,
                          output_dir = output_dir)
  load <- load_

  if (is(object, "MultinomialLNBART")) {
    dm1 <- object$d - 1L
    ndpost <- dim(posterior_fx)[3L]
    Bt <- object$Bt
    vartheta <- array(dim = dim(posterior_fx))
    if (object$keep_draws) draws_chol_Sigma_V <- object$draws_chol_Sigma_V
    else {
      ff_chol_Sigma_V <- file.path(object$forests_dir, "chol_Sigma_V.bin")
      draws_chol_Sigma_V <- load_bin_coefficients(fname = ff_chol_Sigma_V,
                                                  p = dm1, d = dm1, m = ndpost)
    }
    # Compute the individual-level probabilities
    cat("Computing the individual-level probabilities, \vartheta_ij ...\n")
    for (k in seq_len(ndpost)) {
      if (verbose && (k %% printevery == 0L))
        cat("k =", k, "out of", ndpost, "for vartheta^{(k)}_ij...\n")
      for (i in seq_len(n_proposal)) {
        v <- stats::rnorm(dm1) %*% draws_chol_Sigma_V[,,k]
        u <- drop(v %*% Bt)
        p <- posterior_fx[i,,k] * exp(u)
        vartheta[i,,k] <- p / sum(p)
      }
    }
    if (save) {
      saveRDS(object = x_proposal, file = file.path(output_dir, "x_proposal.rds"))
      saveRDS(object = vartheta, file = file.path(output_dir, "vartheta.rds"))
    }
    if (load) return(list(x_proposal = x_proposal, posterior_fx = vartheta))
  } else {
    if (save)
      saveRDS(object = x_proposal, file = file.path(output_dir, "x_proposal.rds"))
    if (load) return(list(x_proposal = x_proposal, posterior_fx = posterior_fx))
  }

  return(invisible())
}

#' Compute the posterior distribution of individual-level probabilities under the ZANIM-BART model
#' @param thetas array with \eqn{(n \times d \times r)} dimension for the posterior distribution of \eqn{\theta_{ij}^{(r)}}.
#' @param zetas array with \eqn{(n \times d \times r)} dimension for the posterior distribution of \eqn{\zeta_{ij}^{(r)}}.
#' @param verbose logical to keep track of the posterior draws.
#' @param printevery integer to print the posterior draws.
#' TODO: These two functions aren't precise because they are not condition on Y* to
#' generate the latent structural zero z_{ij}, though we use the posterior draws.
#'
compute_vartheta_zanimbart <- function(thetas, zetas, verbose = FALSE,
                                       printevery = 100L)  {
  n_sample <- dim(thetas)[1L]
  d <- dim(thetas)[2L]
  ndpost <- dim(thetas)[3L]
  seqn <- seq_len(n_sample)
  draws <- array(data = NA_real_, dim = c(n_sample, d, ndpost))
  for (k in seq_len(ndpost)) {
    if (verbose && (k %% printevery == 0L)) cat(k, "of", ndpost, "\n")
    # Generate the z's
    tmp <- lapply(seqn, function(i) {
      z <- stats::rbinom(n = d, size = 1, prob = 1.0 - zetas[i,,k])
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
compute_vartheta_zanimlnbart <- function(thetas, zetas, chol_Sigma_V, Bt,
                                         verbose = FALSE, printevery = 100L)  {
  n_sample <- dim(thetas)[1L]
  d <- dim(thetas)[2L]
  ndpost <- dim(thetas)[3L]
  dm1 <- d - 1L
  seqn <- seq_len(n_sample)
  draws <- array(data = NA_real_, dim = c(n_sample, d, ndpost))
  for (k in seq_len(ndpost)) {
    if (verbose && (k %% printevery == 0L)) cat(k, "of", ndpost, "\n")
    # Generate the z's
    tmp <- lapply(seqn, function(i) {
      v <- stats::rnorm(dm1) %*% chol_Sigma_V[,,k]
      u <- drop(v %*% Bt)
      z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - zetas[i,,k])
      if (all(z == 0L)) p <- rep(0.0, d)
      else {
        p <- z * thetas[i,,k] * exp(u)
        p <- p / sum(p)
      }
      p
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
  load_ <- load
  if (conditional) load <- TRUE
  posterior_fx_theta <- predict(object, newdata = x_proposal, load = load,
                                output_dir = output_dir, type = "theta")
  posterior_fx_zeta <- predict(object, newdata = x_proposal, load = load,
                               output_dir = output_dir, type = "zeta")
  load <- load_
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

compute_proposal_fx_zanimlnbart <- function(object, proposal_parms, n_proposal,
                                            load = TRUE, save = FALSE,
                                            output_dir = tempdir(),
                                            x_proposal = NULL, verbose = TRUE) {

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

  if (object$keep_draws) draws_chol_Sigma_V <- object$draws_chol_Sigma_V
  else {
    ff_chol_Sigma_V <- file.path(object$forests_dir, "chol_Sigma_V.bin")
    draws_chol_Sigma_V <- load_bin_coefficients(fname = ff_chol_Sigma_V,
                                                p = object$d - 1L, d = object$d - 1L,
                                                m = object$ndpost)
  }

  # Compute the posterior distribution of f^{(c)}_j(x*) and f^{(0)}_j(x*)
  # for x*~\pi(x*) and j=1,...,d
  posterior_fx_theta <- predict(object, newdata = x_proposal, load = TRUE,
                                output_dir = output_dir, type = "theta")
  posterior_fx_zeta <- predict(object, newdata = x_proposal, load = TRUE,
                               output_dir = output_dir, type = "zeta")

  cat("Computing individual-level probabilities...\n")
  vartheta <- compute_vartheta_zanimlnbart(thetas = posterior_fx_theta,
                                           zetas = posterior_fx_zeta,
                                           chol_Sigma_V = draws_chol_Sigma_V,
                                           Bt = object$Bt,
                                           verbose = verbose)

  if (save) {
    saveRDS(object = vartheta, file = file.path(output_dir, "vartheta.rds"))
    saveRDS(object = x_proposal, file = file.path(output_dir, "x_proposal.rds"))
  }
  if (load) return(list(x_proposal = x_proposal, posterior_fx = vartheta))

  return(invisible())
}

#' Importance sampling for approximate the inverse posterior distribution using the
#' ML-BART or MLN-BART models.
.is_mlbart <- function(object, Y, proposal_parms, n_proposal = 5000L,
                       sir = FALSE, n_resampling = floor(n_proposal/2),
                       output_dir = tempdir(), save = FALSE,
                       dir_posterior_fx = NULL, x_proposal = NULL) {
  # Load the proposal and the posterior fx
  if (!is.null(dir_posterior_fx)) {
    if (!file.exists(file.path(dir_posterior_fx, "theta_ij.bin")))
      stop("file {theta_ij.bin} with the posterior distribution of f(x*) doesn't exist in the folder {dir_posterior_fx}")
    if (!file.exists(file.path(dir_posterior_fx, "x_proposal.rds")))
      stop("file {x_proposal.rds} with the proposal distribution x* doesn't exist in the folder {dir_posterior_fx}")

    cat("Loading the proposal for x* and the posterior-predictive of f_j(x_i*) \n")
    x_proposal <- readRDS(file = file.path(dir_posterior_fx, "x_proposal.rds"))
    n_proposal <- nrow(x_proposal)

    if (is(object, "MultinomialBART")) {
      posterior_fx <- load_bin_predictions(fname = file.path(dir_posterior_fx, "theta_ij.bin"),
                                           n = n_proposal, d = object$d, m = object$ndpost)

    } else {
      if (!file.exists(file.path(dir_posterior_fx, "vartheta.rds")))
        stop("file {vartheta.rds} with the posterior distribution of f_j(x_i*)e^{u_ij} doesn't exist in the folder {dir_posterior_fx}")
      posterior_fx <- readRDS(file = file.path(dir_posterior_fx, "vartheta.rds"))
    }

    cache_proposal_fx <- list(posterior_fx = posterior_fx, x_proposal = x_proposal)
  } else {
    cat("Generating the proposal for x* and computing the posterior-predictive of f_j(x_i*) \n")
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
  # Load the proposal and the posterior fx
  if (!is.null(dir_posterior_fx)) {
    cat("Loading the proposal for x* and the posterior-predictive of f(x*) \n")
    fproposal <- file.path(dir_posterior_fx, "x_proposal.rds")
    if (!file.exists(fproposal))
      stop("file {x_proposal.rds} with the proposal distribution x* doesn't exist in the folder {dir_posterior_fx}")
    x_proposal <- readRDS(file = fproposal)
    n_proposal <- nrow(x_proposal)
    n_resampling <- floor(n_proposal/2)

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
    cat("Generating the proposal for x* and computing the posterior-predictive of f^{(c)}_j(x*) and f^{(0)}_j(x*) \n")
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

#' Importance sampling for approximate the inverse posterior distribution using the
#' ZANIM-BART model.
.is_zanimlnbart <- function(object, Y, x_proposal, dir_posterior_fx,
                            ndpost = object$ndpost, sir = FALSE,
                            n_resampling = floor(nrow(x_proposal)/2)) {

  n_proposal <- nrow(x_proposal)
  # Check if the files with parameter predictions exist
  ftheta <- file.path(dir_posterior_fx, "theta_ij.bin")
  fzeta <- file.path(dir_posterior_fx, "zeta_ij.bin")
  if (!file.exists(ftheta) || !file.exists(fzeta)) {
    cat("files {theta_ij.bin} and {zeta_ij.bin} with the posterior distribution of f^{(c)}(x*) and f^{(0)}(x*) do exist in the folder {dir_posterior_fx}. Computing such predictions...\n")
    # Compute the posterior distribution of f^{(c)}_j(x*) and f^{(0)}_j(x*) for x*~\pi(x*) and j=1,...,d
    predict(object, newdata = x_proposal, load = FALSE, output_dir = dir_posterior_fx,
            type = "theta")
    predict(object, newdata = x_proposal, load = FALSE, output_dir = dir_posterior_fx,
            type = "zeta")
  }

  # Estimated importance sampling probabilities, call C++
  ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
  cpp_obj <- new(ml$InversePosterior, ncol(Y), object$ntrees_theta, object$ntrees_zeta,
                object$forests_dir)
  n <- nrow(Y)
  B <- t(object$Bt)
  probs <- lapply(seq_len(n), function(i) {
    cat("Observation: ", i, "of", n, "\n")
    cpp_obj$GetZANIMLNBARTWeightsIS(Y[i, ], n_proposal, ndpost, B, dir_posterior_fx)
  })
  probs <- do.call(cbind, probs)
  # Effective sample size
  ess <- apply(probs, 2, function(x) 1 / sum(x*x))

  # If SIR then return a list with the resampling x_proposal
  if (sir) {
    cat("Resampling...\n")

    x_sir <- apply(probs, 2, function(p) {
      ids <- sample.int(n = n_proposal, size = n_resampling, replace = TRUE,
                        prob = p)
      x_proposal[ids, , drop = FALSE]
    }, simplify = FALSE)
    attr(x_sir, "probs") <- probs
    attr(x_sir, "ess") <- ess
    return(x_sir)
  }
  # Otherwise, return a matrix with first column with the probabilities and the
  # remaining ones the proposal x
  out <- cbind(probs = probs, x_proposal)
  attr(out, "ess") <- ess
  return(out)
}


#' Perform resampling of vector/matrix given the probabilites
#' @export
resampling <- function(x_proposal, probs, size = floor(nrow(x_proposal)/2),
                       replace = TRUE) {
  n_proposal <- nrow(x_proposal)
  apply(probs, 2, function(p) {
    ids <- sample.int(n = n_proposal, size = size, replace = replace, prob = p)
    x_proposal[ids, , drop = FALSE]
  }, simplify = FALSE)
}
