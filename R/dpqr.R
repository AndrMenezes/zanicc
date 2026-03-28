#' Zero-&-N-Inflated Multinomial Distribution
#' @name zanim
#' @aliases zanim rzanim dzanim_marginal moments_zanim
#'
#' @description Random number generation and probability mass function for the
#' Zero-&-N-Inflated Multinomial distribution. It also contains the
#' marginal probability mass function and their expected value.
#'
#' @param x vector. observed vector of compositional counts. It should have the
#' same length as `prob` and `zeta`.
#' @param n integer. The number of samples to generate.
#' @param size integer. The trial parameter of the multinomial distribution.
#' @param prob numeric vector. The probability vector for the multinomial distribution.
#' @param zeta numeric vector. The probability of structural zero for each category.
#' @param j,h integers. Index corresponding to the component for computing the
#' marginal PMF of \eqn{Y_j} and for computing the covariance between \eqn{Y_j} and \eqn{Y_h}.
#' @param log logical; if \code{TRUE}, probabilities p are given as log(p).
#'
#' @return
#' The `rzanim` function returns a matrix of integer counts with `n` rows and
#' `d` columns, where `d` is the number of categories.

#' @rdname zanim
#' @export
rzanim <- function(n, size, prob, zeta) {
  d <- length(prob)
  x <- matrix(data = 0L, nrow = n, ncol = d)
  if ((length(size) != 1L) & (length(size) != n))
    stop("The length of number of trials {size} should be one or the same as {n}.")
  if (length(size) == 1L) size <- rep(size, n)
  omzeta <- 1 - zeta
  for (i in seq_len(n)) {
    z <- stats::rbinom(n = d, size = 1L, prob = omzeta)
    is_zero <- z == 0L
    if (all(is_zero)) x[i, ] <- rep(0L, d)
    else if (sum(is_zero) == d - 1L) {
      x[i, ] <- rep(0L, d)
      x[i, !is_zero] <- size
    } else {
      prob_z <- z * prob / sum(z * prob)
      x[i, ] <- stats::rmultinom(n = 1L, size = size[i], prob = prob_z)[, 1L]
    }
  }
  x
}

#' @rdname zanim
#' @export
dzanim <- function(x, prob, zeta, log = TRUE) {
  out <- log_pmf_zanim(x = x, prob = prob, zeta = zeta)
  if (log) return(out) else return(exp(out))
}

#' @rdname zanim
#' @export
dzanim_marginal <- function(x, size, prob, zeta, j = 1L, log = FALSE) {
  d <- length(prob)
  out <- rep(-Inf, 2^(d - 1))
  log_zeta <- log(zeta)
  log_1mzeta <- log1p(-zeta)
  if (x == 0L) {
    out[1L] <- log_zeta[j]
    out[2L] <- size * log1p(-prob[j]) + sum(log_1mzeta)
  } else if (x == size) {
    out[1L] <- log_1mzeta[j] + sum(log_zeta[-j])
    out[2L] <- size * sum(log(prob)) + sum(log_1mzeta)
  } else {
    out[2L] <- sum(log_1mzeta) + stats::dbinom(
      x = x, size = size,
      prob = prob[j], log = TRUE
    )
  }
  # Compute the contribution of reduced Binomials
  idx <- 1L
  S_sets <- .get_set_S(d = d, j = j)
  for (k in seq_len(length(S_sets))) {
    S_subset <- S_sets[[k]]
    for (h in seq_len(ncol(S_subset))) {
      to_zero <- S_subset[, h]
      eta_S <- sum(log_1mzeta[-to_zero]) + sum(log_zeta[to_zero])
      out[idx + 2L] <- eta_S + stats::dbinom(
        x = x, size = size,
        prob = prob[j] / (1 - sum(prob[to_zero])),
        log = TRUE
      )
      idx <- idx + 1L
    }
  }
  out <- .log_sum_exp(out)
  if (log) return(out) else return(exp(out))
}

#' @rdname zanim
#' @export
moments_zanim <- function(size, prob, zeta, j) {
  d <- length(prob)
  # Compute the weights for N-inflated and non-inflation
  eta_j_N <- (1 - zeta[j]) * prod(zeta[-j])
  eta_d <- prod(1 - zeta)
  # Mean for N-inflated and Binomial components
  out_mean <- eta_j_N + eta_d * prob[j]
  out_var <- eta_j_N * size^2 + eta_d * (size * prob[j] * (1 - prob[j]) + size^2 * prob[j]^2)
  # Add the mean contribution of the "reduced" Binomials
  S_sets <- .get_set_S(d = d, j = j)
  for (k in seq_len(length(S_sets))) {
    S_subset <- S_sets[[k]]
    for (h in seq_len(ncol(S_subset))) {
      to_zero <- S_subset[, h]
      eta_S <- prod((1 - zeta[-to_zero])) * prod(zeta[to_zero])
      p_S <- prob[j] / (1 - sum(prob[to_zero]))
      out_mean <- out_mean + eta_S * p_S
      out_var <- out_var + eta_S * (size * p_S * (1 - p_S) + size^2 * p_S^2)
    }
  }
  list(mean = size * out_mean, var = out_var - size^2 * out_mean^2)
}

#' @rdname zanim
#' @export
covariance_zanim <- function(size, prob, zeta, j, h) {
  d <- length(prob)
  # Get the means
  mean_j <- moments_zanim(size = size, prob = prob, zeta = zeta, j = j)$mean
  mean_h <- moments_zanim(size = size, prob = prob, zeta = zeta, j = h)$mean
  # Compute the multinomial covariance
  eta_d <- prod(1 - zeta)
  c_jh <- eta_d * prob[j] * prob[h]
  R_sets <- .get_set_R(d = d, j = j, h = h)
  for (k in seq_len(length(R_sets))) {
    R_subset <- R_sets[[k]]
    for (i in seq_len(ncol(R_subset))) {
      to_zero <- R_subset[, i]
      eta_R <- prod((1 - zeta[-to_zero])) * prod(zeta[to_zero])
      p_R_j <- prob[j] / (1 - sum(prob[to_zero]))
      p_R_h <- prob[h] / (1 - sum(prob[to_zero]))
      c_jh <- c_jh + eta_R * p_R_j * p_R_h
    }
  }
  size * (size - 1) * c_jh - mean_j * mean_h
}

#' Zero-&-N-Inflated Dirichlet-Multinomial Distribution
#' @name zanidm
#' @aliases zanidm rzanidm dzanidm
#'
#'
#' @description Random number generation and probability mass function for the
#' Zero-&-N-Inflated Dirichlet-Multinomial distribution. It also contains the
#' marginal probability mass function and their expected value.
#'
#' @param x vector. observed vector of compositional counts. It should have
#' same length as `alpha` and `zeta`.
#' @param n integer. The number of samples to generate.
#' @param size integer. The trial parameter of multinomial distribution.
#' @param alpha numeric vector. Positive real numbers.
#' @param zeta numeric vector. Excess of zero parameter.
#' @param j,h integers. Index corresponding to the component for computing the
#' marginal PMF of \eqn{Y_j} and for computing the covariance between \eqn{Y_j}
#' and \eqn{Y_h}.
#' @param log logical; if \code{TRUE}, probabilities p are given as log(p).

#' @rdname zanidm
#' @export
rzanidm <- function(n, size, alpha, zeta) {
  d <- length(alpha)
  x <- matrix(data = 0L, nrow = n, ncol = d)
  if ((length(size) != 1L) & (length(size) != n))
    stop("The length of number of trials {size} should be one or the same as {n}.")
  if (length(size) == 1L) size <- rep(size, n)
  omzeta <- 1 - zeta
  for (i in seq_len(n)) {
    ld <- stats::rgamma(n = d, shape = alpha, rate = 1)
    zs <- stats::rbinom(n = d, size = 1, prob = omzeta)
    ld <- ld * zs
    if (all(ld == 0)) x[i, ] <- 0L
    else {
      x[i, ] <- stats::rmultinom(n = 1L, size = size[i], prob = ld / sum(ld))
    }
  }
  x
}

#' @rdname zanidm
#' @export
dzanidm <- function(x, alpha, zeta, log = TRUE) {
  out <- log_pmf_zanidm(x = x, alpha = alpha, zeta = zeta)
  if (log) return(out) else return(exp(out))
}


#' @rdname zanidm
#' @export
dzanidm_marginal <- function(x, size, alpha, zeta, j = 1L, log = FALSE) {
  d <- length(alpha)
  S_sets <- .get_set_S(d = d, j = j)
  a_ <- alpha[j]
  b_ <- sum(alpha[-j])
  out <- rep(-Inf, 2^(d - 1))
  log_zeta <- log(zeta)
  log_1mzeta <- log1p(-zeta)
  if (x == 0L) {
    out[1L] <- log_zeta[j]
  } else if (x == size) {
    out[1L] <- log_1mzeta[j] + sum(log_zeta[-j])
  }
  out[2L] <- sum(log_1mzeta) + .dbetabinomial(x = x, n = size, a = a_, b = b_,
                                              log = TRUE)
  # Compute the contribution of reduced Binomials
  idx <- 1L
  S_sets <- .get_set_S(d = d, j = j)
  for (k in seq_len(length(S_sets))) {
    S_subset <- S_sets[[k]]
    for (h in seq_len(ncol(S_subset))) {
      to_zero <- S_subset[, h]
      b_S <- sum(alpha[-c(j, to_zero)])
      eta_S <- sum(log_1mzeta[-to_zero]) + sum(log_zeta[to_zero])
      out[idx + 2L] <- eta_S + .dbetabinomial(x = x, n = size, a = a_, b = b_S,
                                              log = TRUE)
      idx <- idx + 1L
    }
  }
  out <- .log_sum_exp(out)
  if (log) return(out) else return(exp(out))
}

#' @rdname zanidm
#' @export
moments_zanidm <- function(size, alpha, zeta, j) {
  alpha_j <- alpha[j]
  d <- length(alpha)
  eta_j_N <- (1 - zeta[j]) * prod(zeta[-j])
  eta_d <- prod(1 - zeta)
  out_mean <- eta_j_N + eta_d * alpha_j / sum(alpha)
  alpha_0_j <- sum(alpha[-j])
  out_var <- eta_j_N * size^2 +
    eta_d * (size * alpha_j * (size * (1 + alpha_j) + alpha_0_j)
             / ((alpha_j + alpha_0_j) * (1 + alpha_j + alpha_0_j)))
  S_sets <- .get_set_S(d = d, j = j)
  for (k in seq_len(length(S_sets))) {
    S_subset <- S_sets[[k]]
    for (h in seq_len(ncol(S_subset))) {
      to_zero <- S_subset[, h]
      eta_S <- prod((1 - zeta[-to_zero])) * prod(zeta[to_zero])
      alpha_0_j <- sum(alpha[-c(to_zero, j)])
      out_mean <- out_mean + eta_S * alpha_j / (alpha_j + alpha_0_j)
      out_var <- out_var +
        eta_S * (size * alpha_j * (size * (1 + alpha_j) + alpha_0_j)
                 / ((alpha_j + alpha_0_j) * (1 + alpha_j + alpha_0_j)))
    }
  }
  list(mean = size * out_mean,
       var = out_var - size^2 * out_mean^2)
}

#' @rdname zanidm
#' @export
covariance_zanidm <- function(size, alpha, zeta, j, h) {
  d <- length(alpha)
  # Get the means
  mean_j <- moments_zanidm(size = size, alpha = alpha, zeta = zeta, j = j)$mean
  mean_h <- moments_zanidm(size = size, alpha = alpha, zeta = zeta, j = h)$mean
  # Compute the multinomial covariance
  eta_d <- prod(1 - zeta)
  a0 <- sum(alpha)
  a_jh <- alpha[j]*alpha[h]
  c_jh <- eta_d * a_jh / a0^2 * (size - (size + a0) / (1 + a0))
  R_sets <- .get_set_R(d = d, j = j, h = h)
  for (k in seq_len(length(R_sets))) {
    R_subset <- R_sets[[k]]
    for (i in seq_len(ncol(R_subset))) {
      to_zero <- R_subset[, i]
      eta_R <- prod((1 - zeta[-to_zero])) * prod(zeta[to_zero])
      a0 <- sum(alpha[-to_zero])
      c_jh <- c_jh + eta_R * a_jh / a0^2 * (size - (size + a0) / (1 + a0))
    }
  }
  size * c_jh - mean_j * mean_h
}

#' Dirichlet-Multinomial Distribution
#' @name dm
#' @aliases dm rdm ddm
#'
#'
#' @description Random number generation and probability mass function for the
#' Dirichlet-Multinomial distribution.
#' @param x matrix. observed vector of compositional counts. It should have
#' same length as `alpha`.
#' @param n integer. The number of samples to generate.
#' @param sizes vector. The number of trials.
#' @param alphas matrix. The concentration parameter Positive real numbers.
#' @param log logical; if \code{TRUE}, probabilities p are given as log(p).
#'
#' @rdname dm
#' @export
rdm <- function(n, sizes, alphas) {
  if (is.vector(alphas)) d <- length(alphas) else d <- ncol(alphas)
  if (n > 1L) {
    if (length(sizes) == 1L) sizes <- rep(sizes, n)
    if (is.vector(alphas)) {
      alphas <- matrix(alphas, nrow = n, ncol = d, byrow = TRUE)
    }
  }
  x <- matrix(data = 0L, nrow = n, ncol = d)
  for (i in seq_len(n)) {
    a <- stats::rgamma(n = d, shape = alphas[i, ], rate = 1.0)
    p <- a / sum(a)
    x[i, ] <- stats::rmultinom(n = 1L, size = sizes[i], prob = p)[, 1L]
  }
  x
}
#' @rdname dm
#' @export
ddm <- function(x, alphas, log = TRUE) {
  s_alphas <- rowSums(alphas)
  sum_row_x <- rowSums(x)
  t0 <- lgamma(s_alphas) - rowSums(lgamma(alphas))
  t1 <- rowSums(lgamma(x + alphas)) - lgamma(s_alphas + sum_row_x)
  # Constant that depends only the data
  t2 <- lgamma(sum_row_x + 1L) - rowSums(lgamma(x + 1L))
  out <- t0 + t1 + t2
  if (log) return(out) else return(exp(out))
}


#' Generate one sample from ZANIM and ZANIDM distribution.
#' Returns the counts and indicator of zero-inflation.
.rzanim <- function(size, prob, zeta, d) {
  z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - zeta)
  is_zero <- z == 0L
  if (all(is_zero)) {
    x <- rep(0L, d)
  } else if (sum(is_zero) == d - 1L) {
    x <- rep(0L, d)
    x[!is_zero] <- size
  } else {
    prob_z <- z*prob / sum(z*prob)
    x <- stats::rmultinom(n = 1L, size = size, prob = prob_z)[, 1L]
  }
  list(x, z)
}
.rzanidm <- function(size, alpha, zeta, d) {
  z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - zeta)
  is_zero <- z == 0L
  if (all(is_zero)) {
    x <- rep(0L, d)
  } else if (sum(is_zero) == d - 1L) {
    x <- rep(0L, d)
    x[!is_zero] <- size
  } else {
    g <- stats::rgamma(n = d, shape = alpha, rate = 1.0)
    prob_z <- z*g / sum(z*g)
    x <- stats::rmultinom(n = 1L, size = size, prob = prob_z)[, 1L]
  }
  list(x, z)
}

#' Vectorise version that assumed `sizes`, `probs`/`alphas` and `zetas` have same dimension
#' and are subject-specific.
.rzanim_vec <- function(n, sizes, probs, zetas, d = ncol(zetas)) {
  do.call(rbind, sapply(X = seq_len(n), FUN = function(i) {
    .rzanim(size = sizes[i], prob = probs[i, ], zeta = zetas[i, ], d = d)[[1L]]
  }, simplify = FALSE))
}
.rzanidm_vec <- function(n, sizes, alphas, zetas, d = ncol(zetas)) {
  do.call(rbind, sapply(X = seq_len(n), FUN = function(i) {
    .rzanidm(size = sizes[i], alpha = alphas[i, ], zeta = zetas[i, ], d = d)[[1L]]
  }, simplify = FALSE))
}


