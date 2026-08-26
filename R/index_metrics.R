#' @name count_composition_indices
#'
#' @title Summary indices for multivariate count-compositional data
#'
#' @description
#' A collection of indices for summarising multivariate compositional count data.
#' These functions quantify different aspects of the multivariate
#' count-compositional distribution, including zero-inflation,
#' dispersion, variability, and compositional diversity.
#' These indices provide complementary summaries of count-compositional data and
#' are useful for exploratory analyses, and posterior predictive checks.
#'
#' @param Y A count-compositional matrix with samples in rows and categories in
#' columns.
#' @param x A vector of counts for a single category.
#' @param N A vector containing the total counts associated with `x`.
#' This arguments is specific for the `zi_binomial()` function.
#' @param standardise Logical. If `TRUE`, return the standardized version of the
#' binomial zero-inflation index.
#'
#' @details
#' Most functions operate on a count-compositional matrix with samples in rows
#' and categories in columns.
#' The exceptions are the functions `zi_poisson()`,  `zi_neg_bin()`, and `zi_binomial()`
#' instead operate on a single count vector corresponding to one category.
#' The function `shannon_entropy()` expects compositional data (each row sums to
#' one). If a count-compositional matrix is supplied, the rows are normalized
#' before computing the average normalized Shannon entropy.
#'
#' ## Zero-inflation indices
#'
#' * `zi_poisson()`: zero-inflation index relative to the Poisson distribution.
#' * `zi_neg_bin()`: zero-inflation index relative to the negative binomial distribution.
#' See Blasco-Moreno et al. (2019) for details.
#' * `zi_binomial()`: zero-inflation index relative to the binomial distribution.
#' See Kim et al. (2018) for details.
#' * `zi_multinomial()`: multivariate zero-inflation index for count-compositional
#' data relative to the multinomial distribution. See Menezes et al. (2026) for details.
#'
#' ## Dispersion and variability indices
#'
#' * `gdi()`: generalised dispersion index. See Kokonendji and Puig (2018) for details.
#' * `mdi()`: multiple marginal dispersion Index. See Kokonendji and Puig (2018) for details.
#' * `mcv()`: multivariate coefficient of variation. See Albert and Zhang (2010) for details
#'
#' ## Diversity indices
#'
#' * `shannon_entropy()`: average normalized Shannon entropy.
#'
#' @return
#' A single numeric value summarizing one aspect of the count-compositional data.
#'
#' @references
#'
#' Albert, A. and Zhang, L. (2010), A novel definition of the multivariate coefficient of variation,
#' \emph{Biometrical Journal}, \strong{52(5)}, 667--675.
#'
#' Blasco-Moreno, A., P{\'e}rez-Casany, M., Puig, P., Morante, M. and Castells, E. (2019), What does a zero
#' mean? Understanding false, random and structural zeros in ecology, \emph{Methods in Ecology and Evolution}
#' \strong{10(7)}, 949--959.
#'
#' Kim, H., Wei{\ss}, C. and M{\"o}ller, T. (2018), Testing for an excessive number of
#' zeros in time series of bounded counts. \emph{Statistical Methods \& Applications}, \strong{27}, 689--714.
#'
#' Kokonendji, C. C. and Puig, P. (2018), Fisher dispersion index for multivariate count distributions: A review
#' and a new proposal, Journal of Multivariate Analysis \strong{165}, 180--193.
#'
#' Menezes, A. F. B., Parnell, A. C. and Murphy, K. (2026), Bayesian nonparametric models for zero-inflated
#' count-compositional data using ensembles of regression trees. <https://arxiv.org/abs/2601.08067>
#'

#' @rdname count_composition_indices
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

#' @rdname count_composition_indices
#' @export
gdi <- function(Y) {
  m <- colMeans(Y)
  cv <- cov(Y)
  drop((crossprod(sqrt(m), cv) %*% sqrt(m)) / crossprod(m))
}

#' @rdname count_composition_indices
#' @export
mdi <- function(Y) {
  m <- colMeans(Y)
  v <- diag(cov(Y))
  di <- v / m
  drop(sum(m^2 * di) / crossprod(m))
}

#' @rdname count_composition_indices
#' @export
mcv <- function(Y) {
  m <- colMeans(Y)
  v <- cov(Y)
  drop( sqrt( (crossprod(m, v) %*% m) / sum(m^2) ))
}

#' @rdname count_composition_indices
#' @export
shannon_entropy <- function(Y) {
  N <- rowSums(Y)
  if (!all(N == 1.0)) {
    warning(
      "The rows of `Y` do not sum to one and are therefore not compositional vectors.\n",
      "Rows will be normalized before computing the Shannon entropy.")
    Y <- .normalize_composition(Y)
  }
  n <- nrow(Y)
  log_d <- log(ncol(Y))
  terms <- numeric(n)
  for (i in seq_len(n)) terms[i] <- -sum(Y[i, ] * log(Y[i, ]), na.rm = TRUE) / log_d
  mean(terms)
}

#' @rdname count_composition_indices
#' @export
zi_neg_bin <- function(x) {
  p0 <- mean(x == 0)
  if (p0 == 0.0) return(0.0)
  s2 <- var(x)
  m <- mean(x)
  1.0 + (s2 - m) * log(p0) / (m^2 * (log(s2) - log(m)))
}

#' @rdname count_composition_indices
#' @export
zi_poisson <- function(x) {
  p0 <- mean(x == 0)
  if (p0 == 0.0) return(0.0)
  1.0 + log(p0) / mean(x)
}

#' @rdname count_composition_indices
#' @export
zi_binomial <- function(x, N, standardise = FALSE) {
  sum_N <- sum(N)
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
#' probability is zero.
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
#' * `compute_coverage()`: Compute the empirical coverage given the crebible inteval
#' of the parameters.
#'
#' In the simulation studies conducted in Menezes et al. (2025), we assessed the
#' and comparing different models with respect their ability to estimate the
#' following parameters:
#'
#' \describe{
#'   \item{population-level count probabilities, \eqn{\theta_{ij}} }{
#'    It provides the information underlying the observed compositional counts.
#'    The vector \eqn{\pmb{\theta}_{i} = (\theta_{i1}, \ldots, \theta_{id}) \in \mathbb{S}^d}
#'    lie in the continuous simplex space
#'   \eqn{\mathbb{S}^d=\{\bm{\theta}\in\mathbb{R}^d; \theta_{ij} > 0, \sum_{j=1}^d \theta_{ij}=1\}}.
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
#'   Each \eqn{\zeta_{ij} \in (0, 1)}.
#'
#'   For these parameters, we use the Kullback-Leibler divergence averaged over
#'   the observations, implemented in the function `compute_kl_prob()`.
#'
#'   }
#'
#'   \item{individual-level structural zero probabilities, \eqn{\vartheta_{ij}}}{
#'   It describe within- and between-subject heterogeneity, while \eqn{\bm{\theta}_i}
#'   characterises the counts at a global level.
#'   The vector \eqn{\pmb{\vartheta}_{i} = (\vartheta_{i1}, \ldots, \vartheta_{id}) \in \mathbb{S}^d}
#'   also lie in the continuous simplex space
#'   \eqn{\mathbb{S}^d=\{\bm{\vartheta}\in\mathbb{R}^d; \vartheta_{ij} \geq 0, \sum_{j=1}^d \vartheta_{ij}=1\}},
#'
#'   However, note that \eqn{\vartheta_{ij}} can be have spikes at zero.
#'   Because of this, for these parameters we use the Jensen-Shannon divergence
#'   averaged over the observations, implemented in the function `compute_js`.
#'   }
#' }
#'
#' @references
#' Menezes, A. F. B., Parnell, A. C. and Murphy, K. (2026), Bayesian nonparametric models for zero-inflated
#' count-compositional data using ensembles of regression trees. <https://arxiv.org/abs/2601.08067>
#'
#' @return
#' A numeric value or vector containing the recovery metric. The returned value
#' measures the discrepancy between the true parameter values and their
#' posterior estimates; smaller values indicate better recovery, while coverage
#' values closer to the nominal credible level indicate better calibration.
#'
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
compute_kl_simplex <- function(true_values, estimates) {
  ep = 1.0
  # Critical case: theta >0 and draws == 0
  idx <- which((true_values > 0.0) & estimates == 0.0)
  estimates[idx] <- ep
  log_ratio <- log(true_values / estimates)
  # continuity as limit: lim x -> 0 of x log x = 0:
  log_ratio[log_ratio == -Inf] <- 0.0
  # log(0/0) = 0:
  log_ratio[is.na(log_ratio)] <- 0.0
  kl_terms <- true_values*log_ratio
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
    kl_terms <- kl_terms  + (1 - true_curr) * lr_1p
    kl[j] <- mean(kl_terms)
  }
  kl
}
#' @rdname recovery_metrics
#' @export
compute_js <- function(true_values, estimates) {
  t1 <- estimates*log(2.0*estimates / (estimates + true_values))
  t1[is.na(t1)] <- 0.0
  t2 <- true_values*log(2.0*true_values / (estimates + true_values))
  t2[is.na(t2)] <- 0.0
  mean(rowSums(t1 + t2))
}

#' @rdname recovery_metrics
#' @export
compute_hellinger <- function(true_values, estimates) {
  1.0 / sqrt(2)*mean(rowSums((sqrt(true_values) - sqrt(estimates))^2))
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
#' `compute_kl_simplex_chain()` to avoid undefined logarithms when a
#' reference probability is positive but the corresponding posterior draw is
#' zero.
#'
#' @details
#'
#' Unlike the functions in `\link{recovery_metrics}`, which
#' evaluate posterior point estimates (e.g., posterior means or medians), these
#' functions compute the discrepancy between each posterior draw and the
#' corresponding reference values.
#' They are primarily intended for monitoring the convergence of MCMC
#' algorithms of the count-compositional models and evaluating the mixing of
#' posterior chain.
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
compute_kl_simplex_chain <-  function(reference_values, draws, ep = 1.0) {
  d <- dim(draws)[2]
  ndpost <- dim(draws)[3]
  # Fixing critical case: true_values > 0 and draws == 0
  for (k in seq_len(ndpost)) {
    for (j in seq_len(d)) {
      idx <- which((reference_values[, j] > 0) & (draws[,j,k] == 0))
      draws[idx,j,k] <- ep
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
    kl_terms <- kl_terms  + (1 - true_curr) * (log1p(-true_curr) - log1p(-draws_curr))
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
  for (j in seq_len(p))
    isin[j] <- x[j] >= interval[j, 1]  && x[j] <= interval[j, 2]
  if (all(isin)) return(1L) else return(0L)
}

# Compute the mode using kernel density estimates
.get_mode <- function(X) {
  apply(X, 2, function(x) {
    dd <- density(x)
    dd$x[which.max(dd$y)]
  })
}

#' Prediction metrics
#' @param x A matrix of observed values, with rows corresponding to observations
#' and columns to variables.
#' @param draws An array of posterior draws. The first two dimensions contain
#' posterior draws and variables, respectively, and the third dimension indexes
#' observations.
#' @return A named vector containing the mean prediction metrics across
#' observations.
#' The metrics are mean absolute error (`mae`), mean squared error based on
#' posterior means (`msep`), squared error based on posterior modes (`dmode`),
#' continuous ranked probability score (`crps`), and 95\% and 50\% empirical
#' coverage of the highest posterior interval (`coverage_95` and `coverage_50`).
#' @export
compute_prediction_metrics <- function(x, draws) {
  n <- nrow(x)
  stopifnot(n == dim(draws)[3L])
  l <- lapply(seq_len(n), function(i) {
    post <- as.matrix(draws[,,i])
    mu <- colMeans(post)
    md <- apply(post, 2, median)
    mo <- .get_mode(post)
    c(mae = sum(abs(x[i, ] - md)),
      msep = sum((x[i, ] - mu)^2),
      dmode = sum((x[i, ] - mo)^2),
      crps = scoringRules::crps_sample(y = x[i, ], dat = t(post)),
      coverage_95 = .is_inside(coda::HPDinterval(coda::as.mcmc(post), prob = 0.95), x[i, ]),
      coverage_50 = .is_inside(coda::HPDinterval(coda::as.mcmc(post), prob = 0.50), x[i, ])
    )
  })
  rowMeans(do.call(cbind, l))
}


#' Classification metrics
#'
#' Computes common binary classification performance metrics from observed
#' (`truth`) and estimated (`estimated`) class labels.
#'
#' @param truth A vector of true binary class labels, coded as `0` and `1`.
#' @param estimated A vector of estimated binary class labels, coded as `0` and `1`.
#'
#' @return A named vector containing sensitivity (`sens`), specificity (`spec`),
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
  sensitivity <- tp / (fn + tp)
  specificity <- tn / (fp + tn)
  mcc <- (tp * tn - fp * fn)/(sqrt(tp + fp) * sqrt(tp + fn) * sqrt(tn + fp) * sqrt(tn + fn))
  f1 <- 2 * tp / (2 * tp + fn + fp)
  c(sens = sensitivity, spec = specificity, mcc = mcc, f1 = f1)
}

