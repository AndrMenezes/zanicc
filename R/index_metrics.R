#' @name cc_indices
#'
#' @title Summary indices for multivariate count-compositional data
#'
#' @description
#' A collection of indices for summarising multivariate count-compositional data.
#' These functions quantify different aspects of the multivariate
#' count-compositional distribution, including zero-inflation,
#' dispersion, variability, and compositional diversity.
#' These indices provide complementary summaries of count-compositional data and
#' are useful for exploratory analyses and posterior predictive checks.
#'
#' @param Y A count-compositional matrix with samples in rows and categories in
#' columns. For the shannon_entropy, the matrix should be compositional, with
#' each row sums to one.
#' @param x A vector of counts for a single category.
#' @param N A vector containing the total counts associated with `x`.
#' This argument is required only for the `zi_binomial()` function.
#' @param standardise Logical. If `TRUE`, return the standardised version of the
#' binomial zero-inflation index. Defaults to `FALSE`.
#'
#' @details
#' Most functions operate on a count-compositional matrix with samples in rows
#' and categories in columns.
#' The exceptions are the functions `zi_poisson()`,  `zi_neg_bin()`, and `zi_binomial()`, which
#' instead operate on a single count vector corresponding to one category.
#' The function `shannon_entropy()` expects compositional data (each row sums to
#' one). If a count-compositional matrix is supplied, the rows are normalised
#' before computing the average normalised Shannon entropy.
#'
#' ## Zero-inflation indices
#'
#' * `zi_poisson()`: zero-inflation index relative to the Poisson distribution.
#' * `zi_neg_bin()`: zero-inflation index relative to the negative binomial distribution.
#' See Blasco-Moreno et al. (2019) for details.
#' * `zi_binomial()`: zero-inflation index relative to the binomial distribution.
#' See Kim et al. (2018) for details.
#' * `zi_multinomial()`: multivariate zero-inflation index for count-compositional
#' data relative to the multinomial distribution. See Menezes et al. (2026a) for details.
#'
#' ## Dispersion and variability indices
#'
#' * `gdi()`: generalised dispersion index. See Kokonendji and Puig (2018) for details.
#' * `mdi()`: multiple marginal dispersion Index. See Kokonendji and Puig (2018) for details.
#' * `mcv()`: multivariate coefficient of variation. See Albert and Zhang (2010) for details
#'
#' ## Diversity indices
#'
#' * `shannon_entropy()`: average normalised Shannon entropy.
#'
#' @return
#' A single numeric value summarising one aspect of the count-compositional data.
#'
#' @references
#'
#' Albert, A. and Zhang, L. (2010), A novel definition of the multivariate coefficient of variation,
#' \emph{Biometrical Journal}, \strong{52(5)}, 667--675.
#'
#' Blasco-Moreno, A., Pérez-Casany, M., Puig, P., Morante, M. and Castells, E. (2019), What does a zero
#' mean? Understanding false, random and structural zeros in ecology, \emph{Methods in Ecology and Evolution}
#' \strong{10(7)}, 949--959.
#'
#' Kim, H., Weiß, C. and Möller, T. (2018), Testing for an excessive number of
#' zeros in time series of bounded counts. \emph{Statistical Methods & Applications}, \strong{27}, 689--714.
#'
#' Kokonendji, C. C. and Puig, P. (2018), Fisher dispersion index for multivariate count distributions: A review
#' and a new proposal, Journal of Multivariate Analysis \strong{165}, 180--193.
#'
#' Menezes, A. F. B., Parnell, A. C. and Murphy, K. (2026a), Bayesian nonparametric models for zero-inflated
#' count-compositional data using ensembles of regression trees. \emph{arXiv preprint}, \strong{arXiv:2601.08067} <https://arxiv.org/abs/2601.08067v2>


#' @rdname cc_indices
#' @export
zi_multinomial <- function(Y) {
  N <- rowSums(Y)
  p <- colSums(Y) / sum(N)
  q <- outer(N, p, function(Ni, pj) (1 - pj)^Ni)
  p0 <- sum(Y == 0)
  p0_teo <- sum(q)
  index <- (p0 - p0_teo) / length(Y)
  index
}

#' @rdname cc_indices
#' @export
gdi <- function(Y) {
  m <- colMeans(Y)
  cv <- stats::cov(Y)
  drop((crossprod(sqrt(m), cv) %*% sqrt(m)) / crossprod(m))
}

#' @rdname cc_indices
#' @export
mdi <- function(Y) {
  m <- colMeans(Y)
  v <- diag(stats::cov(Y))
  di <- v / m
  drop(sum(m^2 * di) / crossprod(m))
}

#' @rdname cc_indices
#' @export
mcv <- function(Y) {
  m <- colMeans(Y)
  v <- stats::cov(Y)
  drop(sqrt((crossprod(m, v) %*% m) / sum(m^2)))
}

#' @rdname cc_indices
#' @export
shannon_entropy <- function(Y) {
  # N <- rowSums(Y)
  # if (!all(N == 1.0)) {
  #   warning(
  #     "The rows of `Y` do not sum to one and are therefore not compositional vectors.\n",
  #     "Rows will be normalized before computing the Shannon entropy."
  #   )
  #   Y <- .normalise_composition(Y)
  # }
  n <- nrow(Y)
  log_d <- log(ncol(Y))
  terms <- numeric(n)
  for (i in seq_len(n)) terms[i] <- -sum(Y[i, ] * log(Y[i, ]), na.rm = TRUE) / log_d
  mean(terms)
}

#' @rdname cc_indices
#' @export
zi_neg_bin <- function(x) {
  p0 <- mean(x == 0)
  if (p0 == 0.0) {
    return(0.0)
  }
  s2 <- stats::var(x)
  m <- mean(x)
  1.0 + (s2 - m) * log(p0) / (m^2 * (log(s2) - log(m)))
}

#' @rdname cc_indices
#' @export
zi_poisson <- function(x) {
  p0 <- mean(x == 0)
  if (p0 == 0.0) {
    return(0.0)
  }
  1.0 + log(p0) / mean(x)
}

#' @rdname cc_indices
#' @export
zi_binomial <- function(x, N, standardise = FALSE) {
  sum_N <- sum(N)
  n <- nrow(x)
  p0 <- mean(x == 0)
  p_hat <- sum(x) / sum_N
  p0_teo <- mean((1 - p_hat)^N)
  index <- p0 - p0_teo
  if (standardise) {
    var_p <- 1 / n^2 * sum((1 - p_hat)^N * (1 - (1 - p_hat)^N))
    # var_p <- p_hat * (1 - p_hat) / sum_N * ( mean(N * (1 - p_hat)^(N - 1) ) )^2
    index <- index / sqrt(var_p)
  }
  index
}

#' @name recovery_metrics
#'
#' @title Parameter recovery metrics for count-compositional models
#'
#' @description
#' A collection of metrics to assess the performance of count-compositional
#' models in recovering their parameters from simulated data sets. See below for
#' further details.
#'
#' @param true_values A matrix containing the true values of the model parameter
#' being evaluated. Rows correspond to observations and columns
#' correspond to categories.
#' @param estimates A matrix containing posterior point estimates (posterior
#' means or medians) of the corresponding parameter. Must have the same
#' dimensions as `true_values`.
#' @param estimates_lo A matrix containing lower posterior credible interval
#' intervals for the parameter estimates. Must have the same dimensions as
#' `true_values`.
#' @param estimates_up A matrix containing upper posterior credible interval
#' intervals for the parameter estimates. Must have the same dimensions as
#' `true_values`.
#' @param ep A small positive constant used in `compute_kl_simplex()` to avoid
#' undefined values when the true probability is positive but the estimated
#' probability is zero. Defaults to `1.0`.
#'
#' @details
#' The functions are designed for evaluating parameter recovery in
#' count-compositional models. The appropriate metric depends on the type of
#' parameter being assessed.
#'
#' ## Divergence-based metrics
#'
#' * `compute_kl_simplex()`: Kullback-Leibler divergence between probability
#'   vectors defined on the simplex.
#' * `compute_kl_prob()`: Kullback-Leibler divergence for scalar probability parameters.
#' * `compute_js()`: Jensen-Shannon divergence between probability vectors.
#'
#' ## Distance-based metrics
#'
#' * `compute_hellinger()`: Hellinger distance between true values and posterior
#'   estimates.
#' * `compute_frob()`: Frobenius norm between true values and posterior
#'   estimates.
#' * `compute_abs_diff()`: Mean absolute difference between true values and
#'   posterior estimates.
#'
#' ## Posterior uncertainty
#'
#' * `compute_coverage()`: Compute the empirical coverage given the credible interval
#' of the parameters.
#'
#' In the simulation studies conducted in Menezes et al. (2026a), we assessed
#' and compared different models with respect to their ability to estimate the
#' following parameters:
#'
#' \describe{
#'   \item{population-level count probabilities, \eqn{\theta_{ij}} }{
#'    It provides the information underlying the observed compositional counts.
#'    The vector \eqn{\bm{\theta}_{i} = (\theta_{i1}, \ldots, \theta_{id}) \in \mathbb{S}^d}
#'    lie in the continuous simplex space
#'   \eqn{\mathbb{S}^d=\{\bm{\theta}_i\in\mathbb{R}^d; \theta_{ij} > 0, \sum_{j=1}^d \theta_{ij}=1\}}.
#'
#'   For these parameters, we use the Kullback-Leibler divergence for
#'   parameters on the simplex, averaged over the observations.
#'   This is implemented in the function `compute_kl_simplex()`.
#'
#'   }
#'
#'   \item{population-level structural zeros probabilities, \eqn{\zeta_{ij}} }{
#'   It provides the information on the probability a given observation \eqn{i} of
#'   category \eqn{j} is structural zero.
#'   Each \eqn{\zeta_{ij} \in [0, 1]}.
#'
#'   For these parameters, we use the Kullback-Leibler divergence averaged over
#'   the observations, implemented in the function `compute_kl_prob()`.
#'
#'   }
#'
#'   \item{individual-level structural zero probabilities, \eqn{\vartheta_{ij}}}{
#'   It describe within- and between-subject heterogeneity, while \eqn{\bm{\theta}_i}
#'   characterises the counts at a global level.
#'   The vector \eqn{\bm{\vartheta}_{i} = (\vartheta_{i1}, \ldots, \vartheta_{id}) \in \mathbb{S}^d}
#'   also lie in the continuous simplex space
#'   \eqn{\mathbb{S}^d=\{\bm{\vartheta}_i\in\mathbb{R}^d; \vartheta_{ij} \geq 0, \sum_{j=1}^d \vartheta_{ij}=1\}},
#'
#'   However, note that \eqn{\vartheta_{ij}} can have spikes at zero.
#'   Because of this, we use the Jensen-Shannon divergence
#'   averaged over the observations for these parameters, implemented in the function `compute_js`.
#'   }
#' }
#'
#' @references
#' Menezes, A. F. B., Parnell, A. C. and Murphy, K. (2026a), Bayesian nonparametric models for zero-inflated
#' count-compositional data using ensembles of regression trees. \emph{arXiv preprint}, \strong{arXiv:2601.08067} <https://arxiv.org/abs/2601.08067v2>
#'
#' @return
#' A numeric value or vector containing the recovery metric. The returned value
#' measures the discrepancy between the true parameter values and their
#' posterior estimates; smaller values indicate better recovery, while coverage
#' values closer to the nominal credible level indicate better calibration.
#'


#' @rdname recovery_metrics
#' @export
compute_frob <- function(true_values, estimates) {
  sqrt(sum((estimates - true_values)^2))
}
#' @rdname recovery_metrics
#' @export
compute_abs_diff <- function(true_values, estimates) {
  mean(abs(estimates - true_values))
}
#' @rdname recovery_metrics
#' @export
compute_coverage <- function(true_values, estimates_lo, estimates_up) {
  mean((true_values >= estimates_lo) & (true_values <= estimates_up))
}
#' @rdname recovery_metrics
#' @export
compute_kl_simplex <- function(true_values, estimates, ep = 1.0) {
  # Critical case: theta >0 and draws == 0
  idx <- which((true_values > 0.0) & estimates == 0.0)
  estimates[idx] <- ep
  log_ratio <- log(true_values / estimates)
  # continuity as limit: lim x -> 0 of x log x = 0:
  log_ratio[log_ratio == -Inf] <- 0.0
  # log(0/0) = 0:
  log_ratio[is.na(log_ratio)] <- 0.0
  kl_terms <- true_values * log_ratio
  mean(rowSums(kl_terms))
}
#' @rdname recovery_metrics
#' @export
compute_kl_prob <- function(true_values, estimates) {
  n <- nrow(true_values)
  d <- ncol(true_values)
  kl <- numeric(d)
  for (j in seq_len(d)) {
    true_curr <- true_values[, j]
    est_curr <- estimates[, j]
    kl_terms <- true_curr * log(true_curr / est_curr)
    lr_1p <- log1p(-true_curr) - log1p(-est_curr)
    lr_1p[lr_1p == -Inf] <- 0.0
    lr_1p[is.na(lr_1p)] <- 0.0
    kl_terms <- kl_terms + (1 - true_curr) * lr_1p
    kl[j] <- mean(kl_terms)
  }
  kl
}
#' @rdname recovery_metrics
#' @export
compute_js <- function(true_values, estimates) {
  t1 <- estimates * log(2.0 * estimates / (estimates + true_values))
  t1[is.na(t1)] <- 0.0
  t2 <- true_values * log(2.0 * true_values / (estimates + true_values))
  t2[is.na(t2)] <- 0.0
  mean(rowSums(t1 + t2))
}

#' @rdname recovery_metrics
#' @export
compute_hellinger <- function(true_values, estimates) {
  1.0 / sqrt(2) * mean(rowSums((sqrt(true_values) - sqrt(estimates))^2))
}

# d <- 4
# n <- 10
# true_values <- matrix(rexp(n*d), ncol = d, nrow = n)
# true_values[2, 1] <- 0.0
# true_values[1, 1] <- 0.0
# true_values <- sweep(true_values, 1, rowSums(true_values), "/")
# estimates <- matrix(rexp(n*d), ncol = d, nrow = n)
# estimates[1, 1] <- 0.0
# estimates <- sweep(estimates, 1, rowSums(estimates), "/")


#' @name posterior_chain_metrics
#'
#' @title Posterior chain metrics for count-compositional models
#'
#' @description
#' Metrics for assessing the convergence and stability of posterior samples from
#' count-compositional models.
#'
#' @param reference_values A matrix containing the reference values of the
#' parameter. These correspond to the true parameter values in simulation
#' studies or reference estimates in real-data analyses.
#' Rows correspond to observations and columns correspond to categories.
#' @param draws A three-dimensional array of posterior draws with dimensions
#' \eqn{n \times d \times M}, where \eqn{M} is the number of posterior
#' samples.
#' @param ep A small positive constant used in
#' [compute_kl_simplex_chain()] to avoid undefined logarithms when a
#' reference probability is positive but the corresponding posterior draw is
#' zero. Defaults to `1.0`.
#'
#' @details
#'
#' Unlike the functions in `\link{recovery_metrics}`, which
#' evaluate posterior point estimates (e.g., posterior means or medians), these
#' functions compute the discrepancy between each posterior draw and the
#' corresponding reference values.
#' They are primarily intended for monitoring the convergence of the MCMC
#' algorithms of the count-compositional models and evaluating the mixing of the
#' posterior chains.
#'
#' The currently functions implemented are:
#'
#' * `compute_frob_chain()`: Frobenius norm between each posterior draw and the
#' reference values.
#' * `compute_kl_simplex_chain()`: Kullback--Leibler divergence for parameters
#' defined on the simplex.
#' * `compute_kl_prob_chain()`: Bernoulli Kullback--Leibler divergence for
#' scalar probability parameters.
#'
#' @return
#' `compute_frob_chain()` and `compute_kl_simplex_chain()` return a numeric
#' vector of length equal to the number of posterior draws, where each element
#' contains the corresponding metric for each posterior sample.
#'
#' `compute_kl_prob_chain()` returns a matrix whose rows correspond to posterior
#' draws and whose columns correspond to categories.
#'
#' @rdname posterior_chain_metrics
#' @export
compute_frob_chain <- function(reference_values, draws) {
  ndpost <- dim(draws)[3]
  diffs <- (array(reference_values, dim = c(dim(reference_values), ndpost)) - draws)^2
  sqrt(apply(diffs, 3, sum))
}
#' @rdname posterior_chain_metrics
#' @export
compute_kl_simplex_chain <- function(reference_values, draws, ep = 1.0) {
  d <- dim(draws)[2]
  ndpost <- dim(draws)[3]
  # Fixing critical case: true_values > 0 and draws == 0
  for (k in seq_len(ndpost)) {
    for (j in seq_len(d)) {
      idx <- which((reference_values[, j] > 0) & (draws[, j, k] == 0))
      draws[idx, j, k] <- ep
    }
  }
  # Compute the ratio
  log_ratio <- log(array(reference_values, dim = c(dim(reference_values), ndpost)) / draws)
  # 0 log(x) = 0, justify by the continuity limit
  log_ratio[log_ratio == -Inf] <- 0.0
  # log(0/0) = 0
  log_ratio[is.na(log_ratio)] <- 0
  kl_terms <- array(reference_values, dim = c(dim(reference_values), ndpost)) * log_ratio
  colMeans(apply(kl_terms, 3, rowSums))
}
#' @rdname posterior_chain_metrics
#' @export
compute_kl_prob_chain <- function(reference_values, draws) {
  d <- ncol(reference_values)
  n <- nrow(reference_values)
  ndpost <- dim(draws)[3]
  kl <- matrix(nrow = ndpost, ncol = d)
  for (j in seq_len(d)) {
    # Broadcast in order to compute the KL for each draw of \zeta
    true_curr <- matrix(reference_values[, j], nrow = n, ncol = ndpost)
    draws_curr <- draws[, j, ]
    kl_terms <- true_curr * log(true_curr / draws_curr)
    kl_terms <- kl_terms + (1 - true_curr) * (log1p(-true_curr) - log1p(-draws_curr))
    kl[, j] <- colMeans(kl_terms)
  }
  kl
}

# Check if x is in the interval.
# @param interval matrix
# @param x vector
.is_inside <- function(interval, x) {
  p <- length(x)
  isin <- logical(p)
  for (j in seq_len(p)) {
    isin[j] <- x[j] >= interval[j, 1] && x[j] <= interval[j, 2]
  }
  if (all(isin)) return(1L)
  else return(0L)
}

# Compute the mode using kernel density estimates
.get_mode <- function(X, ...) {
  apply(X, 2, function(x) {
    dd <- stats::density(x, ...)
    dd$x[which.max(dd$y)]
  })
}

# Compute metrics for p = 1
.get_metrics_1p <- function(x, x_draws, probs) {

  mu <- mean(x_draws)
  md <- stats::median(x_draws)
  dd <- stats::density(x_draws)
  mo <- dd$x[which.max(dd$y)]

  # HPD coverages
  coverages <- numeric(length = length(probs))
  coda_obj <- coda::as.mcmc(x_draws)
  for (k in seq_len(length(probs))) {
    hdi <- coda::HPDinterval(coda_obj, prob = probs[k])
    coverages[k] <- x >= hdi[1, 1] && x <= hdi[1, 2]
  }
  names(coverages) <- paste0("coverage_", 100*probs)

  c(mae = abs(x - md),
    msep = (x - mu)^2,
    dmode = (x - mo)^2,
    crps = scoringRules::crps_sample(y = x, dat = t(x_draws)),
    coverages)
}

# Compute metrics for p > 1
.get_metrics_p <- function(x, x_draws, probs, joint_coverage, method_kde,
                           binned_kde, pilot_kde) {
  x <- unname(x)
  p <- length(x)
  mu <- colMeans(x_draws)
  md <- apply(x_draws, 2, stats::median)
  # Marginal HPD coverages
  marginal_coverages <- numeric(length = length(probs)*p)
  coda_obj <- coda::as.mcmc(x_draws)
  for (k in seq_len(length(probs))) {
    hdi <- coda::HPDinterval(coda_obj, prob = probs[k])
    for (j in seq_len(p)) {
      marginal_coverages[(k - 1)*p + j] <- x[j] >= hdi[j, 1] && x[j] <= hdi[j, 2]
    }
  }
  names(marginal_coverages) <- unlist(lapply(paste0("coverage_", 100*probs),
                                             function(u) paste0(u, "_x", 1:p)))
  # CRPS for each covariate
  crps <- sapply(seq_len(p), function(j) {
    scoringRules::crps_sample(y = x[j], dat = t(x_draws[,j]))
  })
  names(crps) <- paste0("crps_", 1:p)
  # Energy score
  es <- scoringRules::es_sample(y = x, dat = t(x_draws))
  # Joint coverage and `dmode` statistic
  dmode <- coverage <- NULL
  if (joint_coverage) {
    if (method_kde == "kde") {
      H <- tryCatch(ks::Hpi(x = x_draws, binned = binned_kde, pilot = pilot_kde),
                    error = function(e) NULL)
      test <- tryCatch(expr = chol(H), error = function(e) NULL)
      # If binned=TRUE doesn't work, then fit KDE without binned
      if (is.null(test)) H <- ks::Hpi(x = x_draws, binned = FALSE, pilot = "dscalar")
      den_kde <- ks::kde(x = x_draws, H = H, binned = binned_kde)
      # Evaluate density at posterior draws
      dens_draws <- stats::predict(den_kde, x = x_draws)
      # Evaluate density at observed values of x
      dens_x <- stats::predict(den_kde, x = x)
    } else if (method_kde == "vine") {
      den_vine <- rvinecopulib::vine(data = x_draws)
      dens_draws <- rvinecopulib::dvine(x = x_draws, vine = den_vine)
      dens_x <- rvinecopulib::dvine(x = x, vine = den_vine)
    }
    # Compute cutoff for the HPD using the method of Hyndman (1996)
    cutoffs <- stats::quantile(dens_draws, prob = 1.0 - probs)
    coverage <- as.integer(dens_x >= cutoffs)
    names(coverage) <- paste0("coverage_", 100*probs)
    # mode
    index <- which.max(dens_draws)
    mo <- x_draws[index, ]
    dmode <- sum((x - mo)^2)
  }
  # Output
  c(mae = sum(abs(x - md)), msep = sum((x - mu)^2),
    dmode = dmode, es = es, coverage, marginal_coverages, crps)
}

#' Prediction metrics
#' @param x A matrix of observed values, with rows corresponding to observations
#' and columns to variables.
#' @param draws An array of posterior draws. The first two dimensions contain
#' posterior draws and variables, respectively, and the third dimension indexes
#' observations.
#' @param probs Numeric vector with the probability level of the marginal and joint
#' HPD interval and region, respectively.
#' @param joint_coverage Logical. Whether to compute the joint coverage when the
#' dimension of `x` is greater than one. This implements the density quantile method
#' of Hyndman (1996) using a kernel density estimate of the multivariate density function.
#' @param parallel Logical. Whether to compute the metrics in parallel for each sample.
#' Useful if `joint_coverage=TRUE` and the number of samples are large.
#' It uses the function `mclapply` of the `parallel` package, so it only works on Unix-based systems.
#' @param ncores Number of cores to compute the metrics in parallel.
#' @param verbose Logical. Whether to print the progress by each sample.
#' @param method_kde Character indicating which kernel density estimate to use when
#' computing the joint coverage. Options are the `kde` and `vine`.
#' The `kde` option calls the function [ks::kde()] for the kernel density estimate, and
#' it suffers from the curse of dimensionality, while the option `vine` calls
#' the function [rvinecopulib::vine()] which uses vine copula to compute multivariate
#' kernel density estimates for high-dimensional cases.
#' @param method_kde Character indicating which kernel density estimate to use when
#' computing the joint coverage.
#' Options are `kde` and `vine`. The `kde` option calls [ks::kde()] for the kernel
#' density estimate, while vine` calls [rvinecopulib::vine()] for
#' high-dimensional cases.
#' @param binned_kde,pilot_kde Further arguments passed to [ks::kde()].
#' @param binned_kde,pilot_kde Further arguments passed to [ks::kde()].
#'
#' @return A named vector containing the mean prediction metrics across observations.
#' The metrics are mean absolute error (`mae`),
#' mean squared error based on posterior means (`msep`),
#' mean squared error based on posterior modes (`dmode`),
#' energy score (`es`) for `p > 1`,
#' marginal continuous ranked probability score (`crps_x{j}`) for each dimension of `x`,
#' and marginal empirical coverage of the highest posterior density intervals for
#' different specified probability level `probs`. (`coverage_{prob}_x{j}`).
#' If `joint_coverage=TRUE` and `p > 1`, joint HPD region coverage is additionally
#' returned as `coverage_{prob}`.
#'
#' @references
#' Hyndman, R. J. (1996), Computing and graphing highest density regions, \emph{The American Statistician}
#' \strong{50(2)}, 120--126.
#'
#' @examples
#' set.seed(54)
#' ndpost <- 1000L
#' p <- 3L
#' n <- 100L
#' Sigma <- tcrossprod(stats::rnorm(p)) + diag(stats::rgamma(n = p, shape = 2, rate = 0.5))
#' Sigma_inv <- solve(Sigma)
#' lower_chol_Sigma <- t(chol(Sigma))
#' mu <- stats::rnorm(p, mean = 1.0)
#
# # Data
#' x <- matrix(nrow = n, ncol = p)
#' for (i in seq_len(n)) x[i, ] <- drop(lower_chol_Sigma %*% stats::rnorm(p) + mu)
#
# # Prior
#' mu_0 <- rep(0.0, p)
#' Sigma_0 <- diag(x = 1, nrow = p, ncol = p)
#' Sigma_0_inv <- solve(Sigma_0)
# # Parameters of the posterior-predictive distribution
#' xbar <- colMeans(x)
#' Sigma_post <- solve(Sigma_0_inv + n * Sigma_inv)
#' mu_post <- Sigma_post %*% (Sigma_0_inv %*% mu_0 + n * Sigma_inv %*% xbar)
#' Sigma_pp <- Sigma_post + Sigma
#
# # Simulate from the posterior predictive distribution
#' x_draws <- array(dim = c(ndpost, p, n))
#' lower_chol_Sigma_pp <- t(chol(Sigma_pp))
#' for (k in seq_len(ndpost)) {
#'  for (i in seq_len(n)) {
#'     x_draws[k,,i] <- drop(lower_chol_Sigma_pp %*% stats::rnorm(p) + mu_post)
#'   }
#' }
#
#' res1 <- compute_prediction_metrics(x = x, draws = x_draws, parallel = FALSE,
#'                                    verbose = FALSE, joint_coverage = FALSE)
#' res1
#' res2 <- compute_prediction_metrics(x = x, draws = x_draws, parallel = FALSE,
#'                                    verbose = TRUE, joint_coverage = TRUE,
#'                                    method_kde = "kdevine")
#' res3 <- compute_prediction_metrics(x = x, draws = x_draws, parallel = TRUE,
#'                                    verbose = TRUE, joint_coverage = TRUE,
#'                                    ncores = 4L, method_kde = "kde")
#' cbind(res2, res3)
#'
#' @export
compute_prediction_metrics <- function(x, draws,
                                       probs = c(0.95, 0.50),
                                       joint_coverage = TRUE,
                                       parallel = FALSE, ncores = 10L,
                                       verbose = TRUE,
                                       method_kde = c("kde", "vine"),
                                       binned_kde = TRUE, pilot_kde = "dscalar") {

  method_kde <- match.arg(method_kde)
  n <- nrow(x)
  p <- ncol(x)
  stopifnot(n == dim(draws)[3L])
  stopifnot(p == dim(draws)[2L])


  if (p > 6 && method_kde == "kde") {
    warning("Using {method_kde='vine'} as kde::ks does not support p > 5")
    method_kde <- "vine"
  }

  if (p == 1L) {
    if (parallel) {
      totalcores <- parallel::detectCores()
      if (ncores - 1L >= totalcores) ncores <- totalcores - 1L
      by_draws <- parallel::mclapply(seq_len(n), function(i) {
        if (verbose) cat(i, "of", n, "\n")
        .get_metrics_1p(x = x[i, ], x_draws = draws[,1L,i], probs = probs)
      }, mc.cores = ncores)
    } else {
      by_draws <- lapply(seq_len(n), function(i) {
        if (verbose) cat(i, "of", n, "\n")
        .get_metrics_1p(x = x[i, ], x_draws = draws[,1L,i], probs = probs)
      })
    }
  } else {
    if (parallel) {
      totalcores <- parallel::detectCores()
      if (ncores - 1L >= totalcores) ncores <- totalcores - 1L
      by_draws <- parallel::mclapply(seq_len(n), function(i) {
        if (verbose) cat(i, "of", n, "\n")
        .get_metrics_p(x = x[i, , drop = TRUE], x_draws = draws[,,i], probs = probs,
                       joint_coverage = joint_coverage, method_kde = method_kde,
                       binned_kde = binned_kde, pilot_kde = pilot_kde)
      }, mc.cores = ncores)
    } else {
      by_draws <- lapply(seq_len(n), function(i) {
        if (verbose) cat(i, "of", n, "\n")
        .get_metrics_p(x = x[i, , drop = TRUE], x_draws = draws[,,i], probs = probs,
                       joint_coverage = joint_coverage, method_kde = method_kde,
                       binned_kde = binned_kde, pilot_kde = pilot_kde)
      })
    }
  }
  rowMeans(do.call(cbind, by_draws))
}

#' Classification metrics
#'
#' Computes common binary classification performance metrics from observed
#' (`truth`) and estimated (`estimated`) class labels.
#'
#' @param truth A vector of true binary class labels, coded as `0` and `1`.
#' @param estimated A vector of estimated binary class labels, coded as `0` and `1`.
#'
#' @return A named vector containing `precision`, `recall`, `specificity`,
#' Matthews correlation coefficient (`mcc`), and F1 score (`f1`).
#'
#' @export
compute_classification_metrics <- function(truth, estimated) {
  select <- which(estimated == 1)
  not_selected <- which(estimated == 0)
  included <- which(truth == 1)
  excluded <- which(truth == 0)
  tp <- sum(select %in% included)
  tn <- sum(not_selected %in% excluded)
  fp <- sum(select %in% excluded)
  fn <- sum(not_selected %in% included)

  rcl <- tp / (tp + fn) # sensitivity
  prc <- tp / (tp + fp)
  spec <- tn / (fp + tn)
  mcc <- (tp * tn - fp * fn) / (sqrt(tp + fp) * sqrt(tp + fn) * sqrt(tn + fp) * sqrt(tn + fn))
  if (is.na(mcc)) mcc <- 0.0
  # f1 <- 2 * tp / (2 * tp + fn + fp)
  f1 <- 2 * prc * rcl / (prc + rcl)
  c(precision = prc, recall = rcl, specificity = spec, mcc = mcc, f1 = f1)
}
