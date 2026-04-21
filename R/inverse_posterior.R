#' Importance sampling approximate the inverse posterior under ML-BART models
#' @param y observed count vector.
#' @param posterior_fx array with the posterior draws of f_j(x_i*).
.prob_x_ml_bart <- function(y, posterior_fx) {
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
    w <- w / sum(w)
    prob_x[, k] <- w
  }
  rowMeans(prob_x)
}

#' Compute the posterior-predictive distribution of the latent field over the proposal
#' distribution of \eqn{x^\ast \sim q(x^\ast)}.
#' @param object an object of class MultinomialBART
#' @param proposal_parms,n_proposal proposal parameters and number of points to draw
#' from the proposal distribution.
#' @param load Logitical indicating if the posterior-predictive distribution of f(x)
#' should be loaded in memory.
#' @param save
#' @param output_dir where to save the predictions, \eqn{f(x^\ast)} and the proposal \eqn{x^\ast}
#' @param x_proposal matrix with proposal values for x*. Default is NULL, which then
#' generate uniform random values as the proposal distribution of x*.
compute_proposal_fx_ml_bart <- function(object, proposal_parms, n_proposal,
                                        load = TRUE, save = FALSE,
                                        output_dir = tempdir(), x_proposal = NULL) {

  if (is.null(x_proposal)) {
    # Simulate x* ~ q(x*) from uniform proposal
    if (object$p == 1L) {
      a <- proposal_parms$min_x
      b <- proposal_parms$max_x
      # x_proposal <- matrix(stats::runif(n_proposal, min = a, max = b), ncol = 1L)
      x_proposal <- matrix(seq(a, b, length.out = n_proposal), ncol = 1L)
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

#' Two-stage Gibbs sampler for the inverse posterior
#' @param y observed count vector.
#' @param x_ini initial value for the x*.
#' @param mean_prior,S_prior mean and covariance matrix prior use in the elliptical
#' slice sampling.
#' @param forests_dir folder where the posterior distribution of forests are located.
#' @param ntrees number of trees used in the forward model
#' @param ndpost number of posterior draws used in the forward model
#' @param n_rep number times to repeat one iteration of elliptical slice sampling.
#' @details
#' Two-stage Gibbs sampler using the elliptical slice sampling.
#'
.gibbs_ml_bart <- function(Y, mean_prior, S_prior, X_ini = NULL, forests_dir,
                           ntrees, ndpost, n_rep = 2L) {
  # Define new module
  ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
  cpp_obj <- new(ml$InversePosterior, ncol(Y), ntrees, ntrees, "ml_bart",
                 forests_dir)

  # If there is no initial value sample from the prior
  if (is.null(X_ini)) {
    n <- nrow(Y)
    p <- length(mean_prior)
    X_ini <- matrix(nrow = n, ncol = p)
    cS <- chol(S_prior)
    for (i in seq_len(n)) X_ini[i, ] <- stats::rnorm(p) %*% cS + mean_prior
  }

  # Run sampler for each Y
  lapply(seq_len(n), function(i) {
    cat(i, " of ", n, "\n")
    cpp_obj$SamplerMLBARTeSS(Y[i, ], as.matrix(X_ini[i,, drop = FALSE]),
                             as.integer(ndpost), mean_prior, S_prior, n_rep)

  })
}



#' Sample form the inverse posterior using the ML-BART model.
#' @param object an object of class
#' @param Y matrix or vector with the compositional counts.
#' @param method which sampler method to use. Options are is, sir and ess.
.is_mlbart <- function(object, Y, proposal_parms, n_proposal = 5000L,
                       sir = FALSE, n_resampling = floor(n_proposal/2),
                       dir_posterior_fx = NULL, save = FALSE,
                       output_dir = tempdir(), x_proposal = NULL) {

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
    cat("Generaing the proposal for x* and computing the posterior-predictive of f(x*) \n")
    # Generate proposal
    cache_proposal_fx <- compute_proposal_fx_ml_bart(object, proposal_parms, n_proposal,
                                                     load = TRUE, save = save,
                                                     output_dir = output_dir,
                                                     x_proposal = x_proposal)
  }
  # Estimated importanse sampling probabilities
  probs <- apply(Y, 1, function(y) {
    .prob_x_ml_bart(y = y, posterior_fx = cache_proposal_fx$posterior_fx)
  })

  # If SIR then return a list with the resampling x_proposal
  if (sir) {
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

# TODO:
inverse_posterior_mlbart <- function(object, Y, ess_parms,
                                     proposal_parms, n_proposal = 5000L,
                                     sir = FALSE, n_resampling = floor(n_proposal/2),
                                     dir_posterior_fx = NULL, save = FALSE,
                                     output_dir = tempdir(), x_proposal = NULL) {

}


#' Compute the volume (proportional to determinant) of each simplex
#' @param X matrix with p columns
#' @param i matrix with p+1 columns indicating the simplicies
#' Returns the determinants () of the simplices.
.volume_simplex <- function(i, X) {
  # if (!is.matrix(i)) i <- matrix(i, nrow = 1)
  apply(i, 1, \(j) {
    S <- X[j, ]
    X0 <- apply(S[-1, ], 1, `-`, y = S[1, ])
    det(X0)
  })
}
#' Sample from the convex hull of X
#' @param n integer for the number of samples.
#' @param X an n\times p matrix.
#' @description
#' This function was written by the user of Stack Exchange @whuber for the following
#' post
#' https://stats.stackexchange.com/questions/663559/transform-convex-hull-vertex-weights-for-uniform-distribution/663594#663594
#'
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



