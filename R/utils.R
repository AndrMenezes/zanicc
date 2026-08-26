#' Multinomial likelihood function
#'
#' Computes the multinomial likelihood function and the log-predictive distribution
#' for multinomial compound models.
#'
#' @param x A matrix of observed count-compositional data.
#' @param prob A vector or matrix of compositional probabilities.
#' @param log Logical; if `TRUE`, return the log-likelihood, otherwise return the likelihood.
#' @param draws_prob A three-dimensional array of posterior draws of
#' the compositional probabilities.
#' @param printevery Frequency at which to print progress while computing the
#' log-predictive distribution.
#'
#' @return `dmultinomial()` returns a vector of multinomial
#'   (log-)likelihoods. `lpd_multinomial()` returns a matrix of
#'   log-predictive probabilities, with posterior draws in rows and
#'   observations in columns.
#'

#' @rdname multinomial_likelihood
#' @export
dmultinomial <- function(x, prob, log = TRUE) {
  x <- x + 0.5
  # storage.mode(x) <- "integer"
  N <- matrixStats::rowSums2(x)
  N <- if (length(unique(N)) == 1) N[1L] else N
  if (is.matrix(prob)) {
    xp <- x * log(prob)
  } else {
    xp <- t(t(x) * log(prob))
  }
  r <- lgamma(N + 1) + matrixStats::rowSums2(xp - lgamma(x + 1), na.rm = TRUE)
  return( if (log) r else exp(r))
}


#' @rdname multinomial_likelihood
#' @export
lpd_multinomial <- function(x, draws_prob, printevery = 100L) {
  n <- dim(draws_prob)[1L]
  ndpost <- dim(draws_prob)[3L]
  lpl <- matrix(nrow = ndpost, ncol = n)
  for (k in seq_len(ndpost)) {
    if (k %% printevery == 0L) cat(k, "\n")
    lpl[k, ] <- dmultinomial(x = x, prob = draws_prob[, , k], log = TRUE)
  }
  lpl
}

#' Read binary MCMC output
#'
#' @description
#'
#' Binary data are exported during the MCMC sampling. These functions read the
#' binary data into memory and format it as a three-dimensional array.
#'
#' [load_bin_predictions()] reads posterior predictions, while
#' [load_bin_coefficients()] reads posterior coefficient draws. The binary
#' files are expected to contain values stored as 8-byte doubles.
#'
#'
#' @param fname Character string giving the name and path of the binary file.
#' @param n Integer giving the number of observations.
#' @param p Integer giving the number of model parameters (coefficients).
#' @param d Integer giving the number of categories or columns of the
#' response vector \eqn{Y}.
#' @param m Integer giving the number of MCMC iterations to read.
#'
#' @return A numeric array containing the values read from the binary file.
#' For [load_bin_predictions()], the dimensions are \code{c(n, d, m)}.
#' For [load_bin_coefficients()], the dimensions are \code{c(p, d, m)}.
#' The first dimension indexes observations or model parameters, the second
#' indexes categories, and the third indexes MCMC iterations.
#'
#'

#' @rdname load_bin
#' @export
load_bin_predictions <- function(fname, n, d, m) {
  array(.load_bin(fname, n*d*m), dim = c(n, d, m))
}

#' @rdname load_bin
#' @export
load_bin_coefficients <- function(fname, p, d, m) {
  array(.load_bin(fname, p*d*m), dim = c(p, d, m))
}

.load_bin <- function(fname, len) {
  con <- file(fname, "rb")
  on.exit(close(con))
  readBin(con, what = "double", n = len)
}
.load_bin_batch <- function(fname, n, d, k, m, arr = TRUE) {
  con <- file(fname, "rb")
  on.exit(close(con))
  # 8 bytes per double
  offset <- (k - 1) * n * d * 8
  # re-position the connections and read the binary data
  seek(con, where = offset, origin = "start", rw = "read")
  data <- readBin(con, what = "double", n = n * d * m)
  if (arr) array(data, dim = c(n, d, m)) else data
}


#' Summarise posterior draws
#'
#' Summarises posterior draws of sample-specific parameters by their mean,
#' median, and credible interval.
#'
#' @param x A matrix or three-dimensional array of posterior draws.
#' @param prob The probability excluded from the credible interval.
#'
#' @return A data frame containing posterior summaries for each sample,
#' optionally by category if `x` is given as a three-dimensional array.
#'
#' @rdname summarise_draws
#' @export
summarise_draws <- function(x, prob = 0.05) {
  n <- nrow(x)
  data.frame(id = seq_len(n), mean = rowMeans(x), median = apply(x, 1L, median),
             ci_lower = apply(x, 1, quantile, prob / 2),
             ci_upper = apply(x, 1, quantile, 1 - prob / 2))
}
#' @rdname summarise_draws
#' @export
summarise_draws_3d <- function(x, prob = 0.05) {
  d <- dim(x)[2L]
  n <- dim(x)[1L]
  l <- vector(mode = "list", length = d)
  for (j in seq_len(d)) l[[j]] <- cbind(summarise_draws(x[,j,]), category = j)
  do.call(rbind, l)
}

# Aux functions to plot posterior predictive against one covariate
.plot_fit_curve <- function(data) {
  ggplot2::ggplot(data, aes(x = x, y = theta)) +
    ggplot2::geom_line() +
    ggplot2::geom_line(aes(y = mean), col = "dodgerblue",  linewidth = 0.8) +
    ggplot2::geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                         fill = "dodgerblue", alpha = 0.3)
}
# Similar as above but faceted by category
.plot_fit_curve_3d <- function(data) {
  .plot_fit_curve(data) + ggplot2::facet_wrap(~category)
}

#' Normalise count-compositional matrix
#'
#' @description
#' Transform a count-compositional matrix into a empirical compositional matrix
#' by dividing the category-specific counts of each sample by their respective
#' total counts.
#'
#' @param x A matrix of multivariate count-compositional data.
#' Rows correspond to observations and columns correspond to categories.
#' @return Matrix of empirical composition on the continuous simplex.
.normalize_composition <- function(x) {
  x <- sweep(x = x, MARGIN = c(1, 2), STATS = apply(x, c(1, 2), sum), FUN = "/")
  # Rare case when n_trials = 0
  x[is.na(x)] <- 0.0
  x
}

# Ledermann bound
.ledermann <- function(q)  floor(q + 0.5 * (1 - sqrt(8L * q + 1L)))

# pmf of beta-binomial
.dbetabinomial <- function(x, n, a, b, log = TRUE) {
  out <- lchoose(n, x) + lbeta(x + a, n - x + b) - lbeta(a, b)
  if (log) return(out) else return(exp(out))
}

# log-sum-exp
.log_sum_exp <- function(x) {
  ma <- max(x)
  ma + log(sum(exp(x - ma)))
}

# Get the sets \mathcal{S}_j
.get_set_S <- function(d, j) {
  indexes <- seq_len(d)[-j]
  all_sets <- vector(mode = "list", length = d - 2L)
  counter <- 1L
  for (k in seq_len(d - 2L)) {
    all_sets[[counter]] <- utils::combn(x = indexes, m = k)
    counter <- counter + 1L
  }
  all_sets
}

# Get the sets \mathcal{R}_{j, h}
.get_set_R <- function(d, j, h) {
  indexes <- seq_len(d)[-c(j, h)]
  if (length(indexes) == 1) return(list(matrix(indexes, nrow = 1, ncol = 1)))
  all_sets <- vector(mode = "list", length = length(indexes))
  counter <- 1L
  for (k in seq_along(indexes)) {
    all_sets[[counter]] <- utils::combn(x = indexes, m = k)
    counter <- counter + 1L
  }
  all_sets
}

#' Create a rectangle uniform grid inside region of the observed data
#' @param X matrix with the observed data. It should have at least two columns.
#' @param step_size double with the step size to create an exhaustive grid
#' @param scale_factors vector with double to re-scale the values `X[j]` from 0 to `scale_factor[j]`.
#' @description
#' First, create an exhaustive grid using `seq(0.0, scale_factors[j], by = step_size)`,
#' then filter each point in this grid checking if at least one observed value of `X`
#' is inside the grid.
#'
create_rectangle_grid <- function(X, step_size = 1.0,
                                  scale_factors = rep(20, ncol(X))) {

  p <- ncol(X)
  if (p < 2L) stop("Create uniform grid for p>1 covariates")

  n <- nrow(X)
  # Get the range of the X's
  min_ <- apply(X, 2, min)
  max_ <- apply(X, 2, max)
  range_ <- max_ - min_

  # Re-scale each column to vary between (0, scale_factor_j)
  X_scale <- matrix(nrow = n, ncol = p)
  for (j in seq_len(p)) {
    X_scale[, j] <- scale_factors[j] * (X[, j] - min_[j]) / range_[j]
  }
  # Create an exhaustive grid (TODO: this is inefficient, need to think a better way)
  X_grid <- expand.grid(lapply(seq_len(p), function(j) seq(0.0, scale_factors[j],
                                                           by = step_size)))
  X_grid <- as.matrix(X_grid)
  n_grid <- nrow(X_grid)
  # Loop through the exhaustive grid and keep only the points that belongs to observed data +- step_size
  keep <- logical(length = n_grid)
  for (i in seq_len(n_grid)) {
    cond <- abs(sweep(X_scale, 2, X_grid[i, ], "-")) < step_size
    keep[i] <- any(rowSums(cond) == p)
  }
  if (sum(keep) == 0) stop("Did not find any points inside the observed data.")
  cat("Keep" , 100*mean(keep), "% of the ", n_grid, "points in the uniform exhaustive grid.")
  # Filter the data and scale the grid into the original scale of X
  X_grid <- X_grid[keep, ]
  for (j in seq_len(p)) {
    X_grid[, j] <- X_grid[, j] * range_[j] / scale_factors[j] + min_[j]
  }
  # Return the grid
  X_grid
}


#' Compute the posterior distribution of individual-level probabilities under the ZANIM and ZANIM-LN models
#' @param thetas array with \eqn{(n \times d \times r)} dimension for the posterior distribution of \eqn{\theta_{ij}^{(r)}}.
#' @param zetas array with \eqn{(n \times d \times r)} dimension for the posterior distribution of \eqn{\zeta_{ij}^{(r)}}.
#' @param chol_Sigma_V posterior draws of Cholesky decomposition of.
#' @param Bt Matrix for the contrast relate to the sum-to-zero constraint.
#' @param verbose logical to keep track of the posterior draws.
#' @param printevery integer to print the posterior draws.
#' TODO: These two functions aren't precise because they are not condition on Y* to
#' generate the latent structural zero z_{ij}, though we use the posterior draws.
#'
compute_vartheta_zanim <- function(thetas, zetas, verbose = FALSE,
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
compute_vartheta_zanimln <- function(thetas, zetas, chol_Sigma_V, Bt,
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




