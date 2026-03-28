#' Vectorise function to compute the log-likelihood from multinomial
#' Thanks to Keefe.
.dmultinomial <- function(x, prob, log = TRUE) {
  x <- x + 0.5
  storage.mode(x) <- "integer"
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

#' Vectorise function to compute log-likelihood of Dirichlet-multinomial distribution.
#' @description
#' Y and a0 should have same dimension, n\times d, that is, we have subject-specific
#' predictions.
.ddirichletmultinomial <- function(Y, a0) {
  s_a0 <- rowSums(a0)
  sum_row_y <- rowSums(Y)
  t0 <- lgamma(s_a0) - rowSums(lgamma(a0))
  t1 <- rowSums(lgamma(Y + a0)) - lgamma(s_a0 + sum_row_y)
  # Constant that depends only the data
  t2 <- lgamma(sum_row_y + 1L) - rowSums(lgamma(Y + 1L))
  return(t0 + t1 + t2)
}

#' Log-predictive distribution for multinomial compound models
.lpd_multinomial <- function(Y, draws_prob, printevery = 100L) {
  n <- dim(draws_prob)[1L]
  ndpost <- dim(draws_prob)[3L]
  lpl <- matrix(nrow = ndpost, ncol = n)
  for (k in seq_len(ndpost)) {
    if (k %% printevery == 0L) cat(k, "\n")
    lpl[k, ] <- .dmultinomial(x = Y, prob = draws_prob[, , k])
  }
  lpl
}

#' Read binary data related to the draws of \eqn{\theta} and \eqn{\zeta}.
#' @description Binary data is exported during the MCMC of ZANIM-BART. This function
#' read it in memory and format as an array. The `.load_bin` read all first \eqn{m},
#' while the `.load_bin_batch` read from the \eqn{k}-th to the \eqn{m}-th iteration.
#' @param fname file name along with the path.
#' @param n number of sample size.
#' @param d number of categories/columns of the Y vector.
#' @param m number of iterations to read, i.e., the batch size.
#' @param k first iteration to read.
#'
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

#' @export
load_bin_predictions <- function(fname, n, d, m) {
  array(.load_bin(fname, n*d*m), dim = c(n, d, m))
}
#' @export
load_bin_coefficients <- function(fname, p, d, m) {
  array(.load_bin(fname, p*d*m), dim = c(p, d, m))
}


#' Generic functions for summarise the posterior draws of parameters
#' Rows are parameters and columns are draws
#' @export
summarise_draws <- function(x, prob = 0.05) {
  n <- nrow(x)
  data.frame(id = seq_len(n), mean = rowMeans(x), median = apply(x, 1L, median),
             ci_lower = apply(x, 1, quantile, prob / 2),
             ci_upper = apply(x, 1, quantile, 1 - prob / 2))
}
#' @export
summarise_draws_3d <- function(x, prob = 0.05) {
  d <- dim(x)[2L]
  n <- dim(x)[1L]
  l <- vector(mode = "list", length = d)
  for (j in seq_len(d)) l[[j]] <- cbind(summarise_draws(x[,j,]), category = j)
  do.call(rbind, l)
}

#' Aux functions to plot posterior predictive against one covariate
.plot_fit_curve <- function(data) {
  ggplot(data, aes(x = x, y = theta)) +
    geom_line() +
    geom_line(aes(y = mean), col = "dodgerblue",  linewidth = 0.8) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3)
}
#' Similar as above but faceted by category
.plot_fit_curve_3d <- function(data) {
  .plot_fit_curve(data) +
    facet_wrap(~category)
}

#' Normalise count-compositional matrix into compositional vectors at simples
.normalize_composition <- function(x) {
  x <- sweep(x = x, MARGIN = c(1, 2), STATS = apply(x, c(1, 2), sum), FUN = "/")
  # Rare case when n_trials = 0
  x[is.na(x)] <- 0.0
  x
}

#' Ledermann bound
#' Thanks to Keefe.
.ledermann <- function(q)  floor(q + 0.5 * (1 - sqrt(8L * q + 1L)))

#' pmf of beta-binomial
.dbetabinomial <- function(x, n, a, b, log = TRUE) {
  out <- lchoose(n, x) + lbeta(x + a, n - x + b) - lbeta(a, b)
  if (log) return(out) else return(exp(out))
}

#' log-sum-exp
.log_sum_exp <- function(x) {
  ma <- max(x)
  ma + log(sum(exp(x - ma)))
}

#' Get the sets \mathcal{S}_j in the defintion
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
