#' Posterior predictive distribution under a multinomial model
#'
#' Generate posterior predictive samples from a multinomial model given
#' posterior draws of the individual-level count probabilities.
#'
#' For each posterior draw \eqn{k = 1,\dots,ndpost} and observation
#' \eqn{i = 1,\dots,n}, a count vector is generated as
#'
#' \deqn{Y^{(k)}_{i} \sim \text{Multinomial}(N_i, \boldsymbol{\vartheta}^{k}_{i})}
#'
#' where \eqn{N_i} is the number of trials for observation \eqn{i}
#' and \eqn{\boldsymbol{\vartheta}^{k}_{i}} are the posterior draws of the multinomial
#' probabilities.
#'
#' @param n_trials Integer vector of length \eqn{n} giving the total number of
#' trials for each observation.
#' @param draws_prob A 3D array of posterior draws of the multinomial probabilities
#' with dimensions \eqn{n \times d \times ndpost}, where \eqn{n} is the number of
#' observations, \eqn{d} is the number of categories, and \eqn{ndpost} is the
#' number of posterior samples.
#' @param relative Logical. If \code{TRUE}, the simulated posterior-predictive
#' distribution are given as compositional vectors, i.e., \eqn{y^{(k)}_{ij}/N_i}.
#' @param printevery Integer specifying how often progress is printed during sampling.
#'
#' @return
#' A 3D array of posterior predictive samples with dimensions
#' \eqn{ndpost \times n \times d}. If \code{relative = TRUE}, the array
#' contains relative abundances instead of counts.
#'
#' @details
#' This function is typically used internally to generate posterior
#' predictive samples for multinomial-based models. It can be used to
#' perform posterior predictive checks or to simulate replicated datasets
#' from the fitted model.
#'
#' @examples
#' n <- 5
#' d <- 3
#' ndpost <- 10
#'
#' probs <- array(runif(n * d * ndpost), dim = c(n, d, ndpost))
#' probs <- sweep(probs, c(1, 3), apply(probs, c(1, 3), sum), "/")
#'
#' n_trials <- sample(20:50, n, replace = TRUE)
#'
#' y_rep <- ppd_multinomial(n_trials, probs)
.ppd_multinomial <- function(n_trials, draws_prob, relative, printevery) {
  n_sample <- dim(draws_prob)[1L]
  d <- dim(draws_prob)[2L]
  ndpost <- dim(draws_prob)[3L]
  y_rep <- array(data = NA_integer_, dim = c(ndpost, n_sample, d))
  for (k in seq_len(ndpost)) {
    if (k %% printevery == 0L) cat(k, "\n")
    for (i in seq_len(n_sample)) {
      y_rep[k,i,] <- stats::rmultinom(n = 1L, size = n_trials[i],
                                      prob = draws_prob[i, ,k])
    }
  }
  if (relative) y_rep <- .normalize_composition(y_rep)
  y_rep
}

#' Posterior predictive distribution under the ZANIM model
#'
#' Generate posterior predictive samples from a ZANIM model given
#' posterior draws of the population-level count probabilities \eqn{\theta_{ij}},
#' and structural zero probabilities, \eqn{\zeta_{ij}}.
#'
#' For each posterior draw \eqn{k = 1,\dots,ndpost} and observation
#' \eqn{i = 1,\dots,n}, a count vector is generated as follows
#'
#' \deqn{Y^{(k)}_{i} \sim \mathrm{Multinomial}(N_i, \boldsymbol{\vartheta}^{(k)}_{i})}
#'
#' where \eqn{N_i} is the number of trials for observation \eqn{i}
#' and \eqn{\boldsymbol{\vartheta}^{k}_{i}} are the posterior draws of the
#' individual-level count probabilities implied by the ZANIM distribution. See
#' below for further details.
#'
#' @param n_trials Integer vector of length \eqn{n} giving the total number of
#' trials for each observation.
#' @param draws_theta A 3D array of posterior draws of the population-level count
#' probabilities with dimensions \eqn{n \times d \times ndpost}, where \eqn{n} is
#' the number of observations, \eqn{d} is the number of categories, and \eqn{ndpost} is the
#' number of posterior samples.
#' @param draws_zeta A 3D array containing posterior draws of the structural
#' zero probabilities \eqn{\zeta_{ij}} with dimensions \eqn{n \times d \times ndpost}.
#' @param relative Logical. If \code{TRUE}, the simulated posterior predictive
#' samples are returned as compositional vectors, i.e.,
#' \eqn{y^{(k)}_{ij}/N_i}.
#' @param printevery Integer specifying how often progress is printed during sampling.
#'
#' @return
#' A 3D array of posterior predictive samples with dimensions
#' \eqn{ndpost \times n \times d}. If \code{relative = TRUE}, the array
#' contains relative abundances instead of counts.
#'
#' @details
#' This function generates posterior predictive samples using the following
#' hierarchical representation of the ZANIM distribution conditional on the
#' draws of the population-level parameters \eqn{\theta_{ij}} and \eqn{\zeta_{ij}}.
#' Specifically, for each observation \eqn{i} and category \eqn{j},
#'
#' \deqn{
#' z_{ij} \mid \zeta_j \overset{\mathrm{ind.}}{\sim}
#' \mathrm{Bernoulli}(1 - \zeta_j), \qquad j = 1,\ldots,d.
#' }
#'
#' Then, conditional on the latent indicators
#' \eqn{\mathbf{z}_i = (z_{i1},\ldots,z_{id})}, the counts satisfies
#'
#' \deqn{
#' \mathbf{Y}_i \mid N_i, \boldsymbol{\theta}, \mathbf{z}_i \sim
#' \begin{cases}
#' \delta_{\mathbf{0}_d}(\cdot), & \text{if } z_{ij}=0 \text{ for all } j, \\
#' \mathrm{Multinomial}_d\!\left(N_i, \vartheta_{i1}, \ldots, \vartheta_{id}\right),
#' & \text{otherwise},
#' \end{cases}
#' }
#'
#' where \eqn{\delta_{\mathbf{0}_d}} denotes the Dirac measure with unit
#' mass at the zero vector and
#'
#' \deqn{
#' \vartheta_{ij} = \frac{z_{ij}\theta_j}{\sum_{k=1}^d z_{ik}\theta_k},
#' \qquad j = 1,\ldots,d.
#' }
#'
#'
#'
#' @examples
#' n <- 5
#' d <- 3
#' ndpost <- 10
#'
#' probs <- array(runif(n * d * ndpost), dim = c(n, d, ndpost))
#' probs <- sweep(probs, c(1, 3), apply(probs, c(1, 3), sum), "/")
#' zetas <- array(runif(n * d * ndpost), dim = c(n, d, ndpost))
#' n_trials <- sample(20:50, n, replace = TRUE)
#' y_rep <- ppd_zanim(n_trials, probs, zetas)
.ppd_zanim <- function(n_trials, draws_prob, draws_zeta, relative, printevery) {

  d <- dim(draws_prob)[2L]
  ndpost <- dim(draws_prob)[3L]
  n_sample <- dim(draws_prob)[1L]
  y_rep <- array(0L, dim = c(ndpost, n_sample, d))
  for (k in seq_len(ndpost)) {
    if (k %% printevery == 0L) cat(k, "\n")
    y_rep[k,,] <- .rzanim_vec(n = n_sample, sizes = n_trials, probs = draws_prob[,,k],
                              zetas = draws_zeta[,,k], d = d)
  }
  if (relative) y_rep <- .normalize_composition(y_rep)
  return(y_rep)
}

# TODO: document
.ppd_zanidm <- function(n_trials, draws_alpha, draws_zeta, relative, printevery) {

  d <- dim(draws_alpha)[2L]
  ndpost <- dim(draws_alpha)[3L]
  n_sample <- dim(draws_alpha)[1L]
  y_rep <- array(0L, dim = c(ndpost, n_sample, d))
  for (k in seq_len(ndpost)) {
    if (k %% printevery == 0L) cat(k, "\n")
    y_rep[k,,] <- .rzanidm_vec(n = n_sample, sizes = n_trials,
                               alphas = draws_alpha[,,k],
                               zetas = draws_zeta[,,k], d = d)
  }
  if (relative) y_rep <- .normalize_composition(y_rep)
  return(y_rep)
}
.ppd_zanim_ln <- function(n_trials, draws_prob, draws_zeta, draws_chol_Sigma_V, Bt,
                          relative, printevery) {

  d <- dim(draws_prob)[2L]
  dm1 <- d - 1L
  ndpost <- dim(draws_prob)[3L]
  n_sample <- dim(draws_prob)[1L]
  y_rep <- array(0L, dim = c(ndpost, n_sample, d))

  for (k in seq_len(ndpost)) {
    if (k %% printevery == 0L) cat(k, "\n")
    for (i in seq_len(n_sample)) {
      v <- stats::rnorm(dm1) %*% draws_chol_Sigma_V[,,k]
      u <- drop(v %*% Bt)
      z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - draws_zeta[i,,k])
      if (all(z == 0L)) next
      p <- z * draws_prob[i,,k] * exp(u)
      y_rep[k,i,] <- stats::rmultinom(n = 1L, size = n_trials[i], prob = p / sum(p))
    }
  }
  if (relative) y_rep <- .normalize_composition(y_rep)
  return(y_rep)
}
.ppd_mln <- function(n_trials, draws_prob, draws_chol_Sigma_V, Bt, relative,
                     printevery) {

  d <- dim(draws_prob)[2L]
  dm1 <- d - 1L
  ndpost <- dim(draws_prob)[3L]
  n_sample <- dim(draws_prob)[1L]
  y_rep <- array(0L, dim = c(ndpost, n_sample, d))

  for (k in seq_len(ndpost)) {
    if (k %% printevery == 0L) cat(k, "\n")
    for (i in seq_len(n_sample)) {
      v <- stats::rnorm(dm1) %*% draws_chol_Sigma_V[,,k]
      u <- drop(v %*% Bt)
      p <- draws_prob[i,,k] * exp(u)
      y_rep[k,i,] <- stats::rmultinom(n = 1L, size = n_trials[i], prob = p / sum(p))
    }
  }
  if (relative) y_rep <- .normalize_composition(y_rep)
  return(y_rep)
}
.ppd_dm <- function(n_trials, draws_alpha, relative, printevery) {

  d <- dim(draws_alpha)[2L]
  ndpost <- dim(draws_alpha)[3L]
  n_sample <- dim(draws_alpha)[1L]
  y_rep <- array(0L, dim = c(ndpost, n_sample, d))
  for (k in seq_len(ndpost)) {
    if (k %% printevery == 0L) cat(k, "\n")
    y_rep[k,,] <- rdm(n = n_sample, sizes = n_trials, alphas = draws_alpha[,, k])
  }
  if (relative) y_rep <- .normalize_composition(y_rep)
  return(y_rep)
}

# Wrapper function for conditional
.ppd_conditional <- function(object, relative, printevery) {
  .ppd_multinomial(n_trials = object$n_trials, draws_prob = object$draws_abundance,
                   relative = relative, printevery = printevery)
}

# Internal functions to compute PDP loading the draws of the parameters by batch
#' @param n_trials vector with the individual-level number of trials.
#' @param output_dir,output_dir_chol_V the directory where the predictions are
#' located, and the posterior draws of the Cholesky decomposition of \eqn{Sigma_V}
#' for models with logistic normal random effect.
#' @param n_pred number of observations
#' @param d dimension
#' @param Bt transpose of contrast matrix used in the identifiable covariance matrix V.
#' @param batch_size size of the bacth to load the predictions
.ppd_multinomial_batch <- function(n_trials, output_dir, n_pred, d, ndpost,
                                   batch_size, relative, printevery) {
  ff_theta <- file.path(output_dir, "theta_ij.bin")

  # Create a vector to load the predictions by batch
  look_head <- seq.int(from = 1L, to = ndpost, by = batch_size)
  y_rep <- array(0L, dim = c(ndpost, n_pred, d))
  # Load by batch and generate the posterior-predictive
  for (l in seq_len(length(look_head))) {
    shift <- look_head[l] - 1L
    # Load \theta_ij
    draws_prob <- .load_bin_batch(fname = ff_theta, n = n_pred, d = d,
                                  k = look_head[l], m = batch_size)
    for (k in seq_len(batch_size)) {
      if ((shift + k) %% printevery == 0L) cat(shift + k, "\n")
      for (i in seq_len(n_pred)) {
        y_rep[shift + k, i, ] <- stats::rmultinom(1L, size = n_trials[i],
                                                  prob = draws_prob[i, , k])
      }
    }
  }
  if (relative) y_rep <- .normalize_composition(y_rep)
  y_rep
}
.ppd_mln_batch <- function(n_trials, output_dir, output_dir_chol_V, n_pred, d,
                           ndpost, Bt, batch_size, relative, printevery) {
  ff_theta <- file.path(output_dir, "theta_ij.bin")
  ff_chol_Sigma_V <- file.path(output_dir_chol_V, "chol_Sigma_V.bin")

  dm1 <- d - 1L
  # Create a vector to load the predictions by batch
  look_head <- seq.int(from = 1L, to = ndpost, by = batch_size)
  y_rep <- array(0L, dim = c(ndpost, n_pred, d))
  # Load the posterior parameters by batch and generate the posterior-predictive
  for (l in seq_len(length(look_head))) {
    shift <- look_head[l] - 1L
    # Load \theta_ij
    draws_prob <- .load_bin_batch(fname = ff_theta, n = n_pred, d = d,
                                  k = look_head[l], m = batch_size)
    # Load draws of chol(Sigma_V)
    draws_chol_Sigma_V <- .load_bin_batch(fname = ff_chol_Sigma_V, n = dm1, d = dm1,
                                          k = look_head[l], m = batch_size)
    # Simulating
    for (k in seq_len(batch_size)) {
      if ((shift + k) %% printevery == 0L) cat(shift + k, "\n")
      for (i in seq_len(n_pred)) {
        v <- stats::rnorm(dm1) %*% draws_chol_Sigma_V[,,k]
        u <- drop(v %*% Bt)
        p <- draws_prob[i,,k] * exp(u)
        y_rep[shift + k, i, ] <- stats::rmultinom(1L, size = n_trials[i], prob = p / sum(p))
      }
    }
  }
  if (relative) y_rep <- .normalize_composition(y_rep)
  y_rep
}
.ppd_zanim_batch <- function(n_trials, output_dir, n_pred, d, ndpost, batch_size,
                             relative, printevery) {
  ff_theta <- file.path(output_dir, "theta_ij.bin")
  ff_zeta <- file.path(output_dir, "zeta_ij.bin")

  # Create a vector to load the predictions by batch
  look_head <- seq.int(from = 1L, to = ndpost, by = batch_size)
  y_rep <- array(0L, dim = c(ndpost, n_pred, d))
  # Load the posterior parameters by batch and generate the posterior-predictive
  for (l in seq_len(length(look_head))) {
    shift <- look_head[l] - 1L
    # Load \theta_ij
    thetas <- .load_bin_batch(fname = ff_theta, n = n_pred, d = d, k = look_head[l],
                              m = batch_size)
    # Load \zeta_{ij}
    zetas <- .load_bin_batch(fname = ff_zeta, n = n_pred, d = d, k = look_head[l],
                             m = batch_size)
    for (k in seq_len(batch_size)) {
      if ((shift + k) %% printevery == 0L) cat(shift + k, "\n")
      y_rep[shift + k,,] <- .rzanim_vec(n = n_pred, sizes = n_trials,
                                        probs = thetas[, , k], zetas = zetas[, , k],
                                        d = d)
    }
  }
  if (relative) y_rep <- .normalize_composition(y_rep)
  y_rep
}
.ppd_zanim_ln_batch <- function(n_trials, output_dir, output_dir_chol_V, n_pred, d,
                                ndpost, Bt, batch_size, relative, printevery) {
  ff_theta <- file.path(output_dir, "theta_ij.bin")
  ff_zeta <- file.path(output_dir, "zeta_ij.bin")
  ff_chol_Sigma_V <- file.path(output_dir_chol_V, "chol_Sigma_V.bin")

  dm1 <- d - 1L
  # Create a vector to load the predictions by batch
  look_head <- seq.int(from = 1L, to = ndpost, by = batch_size)
  y_rep <- array(0L, dim = c(ndpost, n_pred, d))
  # Load the posterior parameters by batch and generate the posterior-predictive
  for (l in seq_len(length(look_head))) {
    shift <- look_head[l] - 1L
    # Load \theta_ij and \zeta_{ij}
    draws_prob <- .load_bin_batch(fname = ff_theta, n = n_pred, d = d,
                                  k = look_head[l], m = batch_size)
    draws_zeta <- .load_bin_batch(fname = ff_zeta, n = n_pred, d = d,
                                  k = look_head[l], m = batch_size)
    # Load draws of chol(Sigma_V)
    draws_chol_Sigma_V <- .load_bin_batch(fname = ff_chol_Sigma_V, n = dm1, d = dm1,
                                          k = look_head[l], m = batch_size)
    # Simulating
    for (k in seq_len(batch_size)) {
      if ((shift + k) %% printevery == 0L) cat(shift + k, "\n")
      for (i in seq_len(n_pred)) {
        v <- stats::rnorm(dm1) %*% draws_chol_Sigma_V[,,k]
        u <- drop(v %*% Bt)
        z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - draws_zeta[i,,k])
        if (all(z == 0L)) next
        p <- z * draws_prob[i,,k] * exp(u)
        y_rep[shift + k, i, ] <- stats::rmultinom(1L, size = n_trials[i], prob = p / sum(p))
      }
    }
  }
  if (relative) y_rep <- .normalize_composition(y_rep)
  y_rep
}



# Dispatch methods for different models
#' @export
ppd <- function(object, relative = TRUE, ...) {
  UseMethod("ppd")
}

#' @export
ppd.MultinomialBART <- function(object, relative = TRUE,
                                in_sample = TRUE,
                                draws_prob = NULL,
                                output_dir = NULL,
                                n_trials = NULL,
                                n_pred = NULL,
                                ndpost = object$ndpost,
                                batch_size = 50L,
                                printevery = 100L, ...) {
  if (in_sample) {
    return(.ppd_multinomial(n_trials = object$n_trials, draws_prob = object$draws_theta,
                            relative = relative, printevery = printevery))
  }
  if (is.null(n_trials))
    stop("{n_trials} must be provided for out-of-sample predictions.")
  if (!is.null(draws_prob)) {
    return(.ppd_multinomial(n_trials = n_trials, draws_prob = draws_prob,
                            relative = relative, printevery = printevery))
  } else {
    # Check if file exist before call the function
    if (!file.exists(file.path(output_dir, "theta_ij.bin")))
      stop("file with the posterior predictions of theta_{ij} does not exist.")
    # Compute PPD loading the prediction by batch
    return(.ppd_multinomial_batch(n_trials = n_trials, output_dir = output_dir,
                                  n_pred = n_pred, d = object$d, ndpost = ndpost,
                                  batch_size = batch_size, relative = relative,
                                  printevery = printevery))
  }

}
#' @export
ppd.MultinomialLNBART <- function(object, relative = TRUE, conditional = FALSE,
                                  in_sample = TRUE, draws_prob = NULL,
                                  draws_chol_Sigma_V = NULL, output_dir = NULL,
                                  n_trials = NULL, n_pred = NULL,
                                  ndpost = object$ndpost, batch_size = 50L,
                                  printevery = 100L, ...) {


  if (in_sample) {
    if (conditional) return(.ppd_conditional(object, relative, printevery))
    return(.ppd_mln(n_trials = object$n_trials, draws_prob = object$draws_theta,
                    draws_chol_Sigma_V = object$draws_chol_Sigma_V, Bt = object$Bt,
                    relative = relative, printevery = printevery))
  }
  if (is.null(n_trials))
    stop("{n_trials} must be provided for out-of-sample predictions.")
  if (!is.null(draws_prob) && !is.null(draws_chol_Sigma_V)) {
    return(.ppd_mln(n_trials = n_trials, draws_prob = draws_prob,
                    draws_chol_Sigma_V = draws_chol_Sigma_V, Bt = object$Bt,
                    relative = relative, printevery = printevery))
  } else {
    if (!file.exists(file.path(output_dir, "theta_ij.bin")))
      stop("file with the posterior predictions of theta_{ij} does not exist.")
    if (!file.exists(file.path(object$forests_dir, "chol_Sigma_V.bin")))
      stop("file with the posterior draws of chol(Sigma_V) does not exist.")
    # Compute PPD loading the prediction by batch
    return(.ppd_mln_batch(n_trials = n_trials, output_dir = output_dir,
                          output_dir_chol_V = object$forests_dir,
                          n_pred = n_pred, d = object$d, Bt = object$Bt,
                          ndpost = ndpost, batch_size = batch_size,
                          relative = relative, printevery = printevery))
  }

}
#' @export
ppd.ZANIMBART <- function(object, relative = TRUE, conditional = FALSE,
                          in_sample = TRUE, draws_prob = NULL, draws_zeta = NULL,
                          output_dir = NULL, n_trials = NULL, n_pred = NULL,
                          ndpost = object$ndpost, batch_size = 50L,
                          printevery = 100L, ...) {

  if (in_sample) {
    if (conditional) return(.ppd_conditional(object, relative, printevery))
    return(.ppd_zanim(n_trials = object$n_trials, draws_prob = object$draws_theta,
                      draws_zeta = object$draws_zeta, relative = relative,
                      printevery = printevery))
  }

  if (is.null(n_trials))
    stop("{n_trials} must be provided for out-of-sample predictions.")
  if (!is.null(draws_prob) && !is.null(draws_zeta)) {
    return(.ppd_zanim(n_trials = n_trials, draws_prob = draws_prob,
                      draws_zeta = draws_zeta, relative = relative,
                      printevery = printevery))
  } else {
    if (!file.exists(file.path(output_dir, "theta_ij.bin")))
      stop("file with the posterior predictions of theta_{ij} does not exist.")
    if (!file.exists(file.path(output_dir, "zeta_ij.bin")))
      stop("file with the posterior draws of theta_{ij} does not exist.")
    # Compute PPD loading the prediction by batch
    return(.ppd_zanim_batch(n_trials = n_trials, output_dir = output_dir,
                            n_pred = n_pred, d = object$d,
                            ndpost = ndpost, batch_size = batch_size,
                            relative = relative, printevery = printevery))
  }


}
#' @export
ppd.ZANIMLNBART <- function(object, relative = TRUE, conditional = FALSE,
                            in_sample = TRUE, draws_prob = NULL, draws_zeta = NULL,
                            draws_chol_Sigma_V = NULL, output_dir = NULL,
                            n_trials = NULL, n_pred = NULL, ndpost = object$ndpost,
                            batch_size = 50L, printevery = 100L, ...) {

  if (in_sample) {
    if (conditional) return(.ppd_conditional(object, relative, printevery))
    return(.ppd_zanim_ln(n_trials = object$n_trials, draws_prob = object$draws_theta,
                         draws_zeta = object$draws_zeta,
                         draws_chol_Sigma_V = object$draws_chol_Sigma_V, Bt = object$Bt,
                         relative = relative, printevery = printevery))
  }

  if (is.null(n_trials))
    stop("{n_trials} must be provided for out-of-sample predictions.")
  if (!is.null(draws_prob) && !is.null(draws_zeta) && !is.null(draws_chol_Sigma_V)) {
    return(.ppd_zanim_ln(n_trials = n_trials, draws_prob = draws_prob,
                         draws_zeta = draws_zeta,
                         draws_chol_Sigma_V = draws_chol_Sigma_V, Bt = object$Bt,
                         relative = relative, printevery = printevery))
  } else {
    if (!file.exists(file.path(output_dir, "theta_ij.bin")))
      stop("file with the posterior predictions of theta_{ij} does not exist.")
    if (!file.exists(file.path(output_dir, "zeta_ij.bin")))
      stop("file with the posterior draws of zeta_{ij} does not exist.")
    if (!file.exists(file.path(object$forests_dir, "chol_Sigma_V.bin")))
      stop("file with the posterior draws of chol(Sigma_V) does not exist.")
    # Compute PPD loading the prediction by batch
    return(.ppd_zanim_ln_batch(n_trials = n_trials, output_dir = output_dir,
                               output_dir_chol_V = object$forests_dir,
                               n_pred = n_pred, d = object$d, Bt = object$Bt,
                               ndpost = ndpost, batch_size = batch_size,
                               relative = relative, printevery = printevery))
  }


}
#' @export
ppd.ZANIMLNRegression <- function(object, relative = TRUE, conditional = FALSE,
                                  printevery = 100L, ...) {
  if (conditional) return(.ppd_conditional(object, relative, printevery))
  .ppd_zanim_ln(n_trials = object$n_trials, draws_prob = object$draws_theta,
                draws_zeta = object$draws_zeta,
                draws_chol_Sigma_V = object$draws_chol_Sigma_V, Bt = object$Bt,
                relative = relative, printevery = printevery)
}
#' @export
ppd.ZANIMRegression <- function(object, relative = TRUE, conditional = FALSE,
                                printevery = 100L, ...) {
  if (conditional) return(.ppd_conditional(object, relative, printevery))
  .ppd_zanim(n_trials = object$n_trials, draws_prob = object$draws_theta,
             draws_zeta = object$draws_zeta, relative = relative,
             printevery = printevery)
}
#' @export
ppd.ZANIDMRegression <- function(object, relative = TRUE, conditional = FALSE,
                                 in_sample = TRUE, draws_alpha = NULL,
                                 draws_zeta = NULL, n_trials = NULL,
                                 ndpost = object$ndpost,
                                 printevery = 100L, ...) {

  if (in_sample) {
    if (conditional) return(.ppd_conditional(object, relative, printevery))
    return(.ppd_zanidm(n_trials = object$n_trials, draws_alpha = object$draws_alpha,
                       draws_zeta = object$draws_zeta, relative = relative,
                       printevery = printevery))
  }
  if (is.null(n_trials))
    stop("{n_trials} must be provided for out-of-sample predictions.")
  if (!is.null(draws_alpha) && !is.null(draws_zeta)) {
    return(.ppd_zanidm(n_trials = object$n_trials, draws_alpha = draws_alpha,
                       draws_zeta = draws_zeta, relative = relative,
                       printevery = printevery))
  }
}
#' @export
ppd.DMRegression <- function(object, relative = TRUE, conditional = FALSE,
                             in_sample = TRUE, draws_alpha = NULL,
                             n_trials = NULL,
                             ndpost = object$ndpost,
                             printevery = 100L, ...) {

  if (in_sample) {
    if (conditional) return(.ppd_conditional(object, relative, printevery))
    return(.ppd_dm(n_trials = object$n_trials, draws_alpha = object$draws_alpha,
                   relative = relative, printevery = printevery))
  }
  if (is.null(n_trials))
    stop("{n_trials} must be provided for out-of-sample predictions.")
  if (!is.null(draws_alpha)) {
    return(.ppd_dm(n_trials = n_trials, draws_alpha = draws_alpha,
                   relative = relative, printevery = printevery))
  }
}





