#' Univariate zero-inflated index
#' @param x vector of counts.
#' @export
zi_neg_bin <- function(x) {
  p0 <- mean(x == 0)
  if (p0 == 0.0) return(0.0)
  s2 <- var(x)
  m <- mean(x)
  1.0 + (s2 - m) * log(p0) / (m^2 * (log(s2) - log(m)))
}
#' @export
zi_poisson <- function(x) {
  p0 <- mean(x == 0)
  if (p0 == 0.0) return(0.0)
  1.0 + log(p0) / mean(x)
}
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
#' Zero-inflated index for multinomial distribution
#' @param Y An n by d count-compositional matrix.
#' @description It average over the ZI binomial index of each marginal component.
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

#' Generalised dispersion index (GDI) for multivariate counts
#' @export
gdi <- function(Y) {
  m <- colMeans(Y)
  cv <- cov(Y)
  drop((crossprod(sqrt(m), cv) %*% sqrt(m)) / crossprod(m))
}
#' @export
mdi <- function(Y) {
  m <- colMeans(Y)
  v <- diag(cov(Y))
  di <- v / m
  drop(sum(m^2 * di) / crossprod(m))
}
#' @export
mcv <- function(Y) {
  m <- colMeans(Y)
  v <- cov(Y)
  drop( sqrt( (crossprod(m, v) %*% m) / sum(m^2) ))
}
#' Compute the average Shannon entropy for a compositional data
#' @export
shannon_entropy <- function(Y) {
  n <- nrow(Y)
  log_d <- log(ncol(Y))
  terms <- numeric(n)
  for (i in seq_len(n)) terms[i] <- -sum(Y[i, ] * log(Y[i, ]), na.rm = TRUE)/log_d
  mean(terms)
}

#' Compute Frobenius norm and KL divergence for each draw of the parameter against the
#' true (or reference) values.
#' @param true_values matrix n by d with the true values of the parameter.
#' @param draws draws with dimension n x d x ndpost, with the draws of the parameter.
#' @description The functions above give the FROB norm and KL divergence for each
#' draw of the parameter.
#' @export
compute_frob_chain <- function(true_values, draws) {
  ndpost <- dim(draws)[3]
  diffs <- (array(true_values, dim = c(dim(true_values), ndpost)) - draws)^2
  sqrt(apply(diffs, 3, sum))
}
#' @export
compute_kl_simplex_chain <-  function(true_values, draws, ep = 1.0) {
  d <- dim(draws)[2]
  ndpost <- dim(draws)[3]
  # Fixing critical case: true_values > 0 and draws == 0
  for (k in seq_len(ndpost)) {
    for (j in seq_len(d)) {
      idx <- which((true_values[, j] > 0) & (draws[,j,k] == 0))
      draws[idx,j,k] <- ep
    }
  }
  # Compute the ratio
  log_ratio <- log(array(true_values, dim = c(dim(true_values), ndpost)) / draws)
  # 0 log(x) = 0, justify by the continuity limit
  log_ratio[log_ratio == -Inf] <- 0.0
  # log(0/0) = 0
  log_ratio[is.na(log_ratio)] <- 0
  kl_terms <- array(true_values, dim = c(dim(true_values), ndpost)) * log_ratio
  colMeans(apply(kl_terms, 3, rowSums))
}
#' @export
compute_kl_prob_chain <- function(true_values, draws) {
  d <- ncol(true_values)
  n <- nrow(true_values)
  ndpost <- dim(draws)[3]
  kl <- matrix(nrow = ndpost, ncol = d)
  for (j in seq_len(d)) {
    # Broadcast in order to compute the KL for each draw of \zeta
    true_curr <- matrix(true_values[, j], nrow = n, ncol = ndpost)
    draws_curr <- draws[, j, ]
    kl_terms <- true_curr * log(true_curr / draws_curr)
    kl_terms <- kl_terms  + (1 - true_curr) * (log1p(-true_curr) - log1p(-draws_curr))
    kl[, j] <- colMeans(kl_terms)
  }
  kl
}

#' Compute Frobenius norm, absolute norm, coverage and KL divergences using the posterior
#' mean as an estimator of the given parameter against the true (or reference)
#' values.
#' @param true_values matrix n by d with the true values of the parameter.
#' @param estimates,estimates_lo,estimates_up posterior estimates (mean or median)
#' and lower and upper estimates of the parameters. All must have the same dimension
#' as the true_values, that is, n x d.
#' @param ep small amount to add in the compute_kl_simplex function to treat cases where
#' `true_values > 0` and `estimates = 0`.
#' @description The functions above give the FROB norm and KL divergence using the
#' posterior mean as estimator of the parameters. For the coverage the (1-p)\%
#' credible interval is used.
#' @export
compute_frob <- function(true_values, estimates) {
  sqrt(sum((estimates - true_values)^2))
}
#' @export
compute_abs_diff <- function(true_values, estimates) {
  mean(abs(estimates - true_values))
}
#' @export
compute_coverage <- function(true_values, estimates_lo, estimates_up) {
  mean((true_values >= estimates_lo) & (true_values <= estimates_up))
}
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
  kl_terms <- true_values*log_ratio
  mean(rowSums(kl_terms))
}
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
# true_values <- matrix(c(1.0, 0.2, 0.5, 0.7), ncol = 2)
# estimates <- matrix(c(0.9, 0.225, 0.55, 0.6), ncol = 2)

#' Compute metrics of classification for variable selection.
#' @param truth vector of 0 and 1 with true labels
#' @param estimated vector of estimated labels.
#' @details
#' Minor modifications of the code from the function `ZIDM::select_perf`.
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
