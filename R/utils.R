.rzanim <- function(size, prob, zeta, d) {
  z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - zeta)
  is_zero <- z == 0L
  if (all(is_zero)) {
    x <- rep(0L, d)
  } else if (sum(is_zero) == d - 1L) {
    x <- rep(0L, d)
    x[!is_zero] <- size
  } else {
    # prob_z <- prob
    # prob_z[is_zero] <- 0.0
    # prob_z <- prob_z / (1.0 - sum(prob[is_zero]))
    prob_z <- z*prob / sum(z*prob)
    x <- stats::rmultinom(n = 1L, size = size, prob = prob_z)[, 1L]
  }
  list(x, z)
}

#' @export
rzanim_vec <- function(n, sizes, probs, zetas, d = ncol(zetas)) {
  bu <- sapply(X = seq_len(n), FUN = function(i) {
    .rzanim(size = sizes[i], prob = probs[i, ], zeta = zetas[i, ], d = d)[[1L]]
  }, simplify = FALSE)
  do.call(rbind, bu)
}
#' @export
ldzanim <- function(x, prob, zeta) log_pmf_zanim(x = x, prob = prob, zeta = zeta)


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
#' @export
.load_bin <- function(fname, n, d, m) {
  con <- file(fname, "rb")
  on.exit(close(con))
  array(readBin(con, what = "double", n = n * d * m),
        dim = c(n, d, m))
}
#' @export
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

#' Common function for summarise the draws
#' Rows are parameters and columns are draws
#' @export
.summarise_draws <- function(x, prob = 0.05) {
  n <- nrow(x)
  data.frame(id = seq_len(n),
             mean = rowMeans(x),
             median = apply(x, 1L, median),
             ci_lower = apply(x, 1, quantile, prob / 2),
             ci_upper = apply(x, 1, quantile, 1 - prob / 2))
}
#' @export
.summarise_draws_3d <- function(x, prob = 0.05) {
  d <- dim(x)[2L]
  n <- dim(x)[1L]
  l <- vector(mode = "list", length = d)
  for (j in seq_len(d)) {
    l[[j]] <- cbind(.summarise_draws(x[,j,]), category = j)
  }
  do.call(rbind, l)
}

#' Compute metrics of classification for variable selection.
#' @param truth vector of 0 and 1 with true labels
#' @param estimated vector of estimated labels.
#' @details
#' Minor modifications of the code from the function `ZIDM::select_perf`.
#'
#' @export
compute_class <- function(truth, estimated) {
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

#' Compute Frobenius metric based on the draws and the true values
#' @export
compute_frob <- function(true_values, draws) {
  ndpost <- dim(draws)[3]
  diffs <- (array(true_values, dim = c(dim(true_values), ndpost)) - draws)^2
  sqrt(apply(diffs, 3, sum))
}

#' Compute KL divergence for probability defined in the simplex
#' @export
compute_kl_simplex <- function(true_values, draws) {
  ndpost <- dim(draws)[3]
  log_ratio <- log(array(true_values, dim = c(dim(true_values), ndpost)) / draws)
  # 0 log(x) = 0 (weird convention?)
  log_ratio[is.infinite(log_ratio)] <- 0.0
  log_ratio[is.na(log_ratio)] <- 0.0
  kl_terms <- array(true_values, dim = c(dim(true_values), ndpost)) * log_ratio
  colMeans(apply(kl_terms, 3, rowSums))
}

#' Compute KL divergence for discrete probabilities
#' @export
compute_kl_prob <- function(true_values, draws) {
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

#' Compute zero-inflated index
#' @export
zi_b <- function(x) {
  p0 <- mean(x == 0)
  if (p0 == 0) return(0)
  s2 <- var(x)
  m <- mean(x)
  1.0 + (s2 - m) * log(p0) / (m^2 * (log(s2) - log(m)))
}
#' @export
zi_p <- function(x) {
  p0 <- mean(x == 0)
  if (p0 == 0) return(0)
  1 + log(p0) / mean(x)
}


.plot_fit_curve <- function(data) {
  ggplot(data, aes(x = x, y = theta)) +
    geom_line() +
    geom_line(aes(y = mean), col = "dodgerblue",  linewidth = 0.8) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3)
}

.plot_fit_curve_3d <- function(data) {
  .plot_fit_curve(data) +
    facet_wrap(~category)
}

#' Ledermann bound, thanks to Keefe.
.ledermann <- function(q)  floor(q + 0.5 * (1 - sqrt(8L * q + 1L)))

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
