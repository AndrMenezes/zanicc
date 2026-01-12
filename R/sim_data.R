#' @export
sim_data_zanim_bspline_curve <- function(n, d, n_trials, dof_bs_theta = 6L,
                                         dof_bs_zeta = 4L,
                                         link_zeta = c("logit", "probit")) {

  link_zeta <- match.arg(link_zeta)
  link_foo <- switch(link_zeta, "logit" = stats::plogis, "probit" = stats::pnorm)

  # Linear predictor for \theta
  X <- as.matrix(seq(-1.0, 1.0, length.out = n)) # runif(n = n, min = -1, max = 1)
  X1_bs <- splines::bs(X, dof_bs_theta)
  betas_theta <- matrix(stats::rnorm(d * dof_bs_theta), dof_bs_theta, d)
  betas_theta[1L, ] <- betas_theta[1L, ] - seq(from = 4, to = 0, length.out = d)
  eta_theta <- X1_bs %*% betas_theta

  # Linear predictor for \zeta
  X2_bs <- splines::bs(X, dof_bs_zeta)
  betas_zeta <- matrix(stats::rnorm(d * dof_bs_zeta, -1), dof_bs_zeta, d)
  betas_zeta[1L, ] <- betas_zeta[1L, ] - seq(from = 4, to = 0, length.out = d)
  eta_zeta <- X2_bs %*% betas_zeta

  # Generate data
  Y <- Z <- vartheta_truth <- theta_truth <- zeta_truth <- matrix(0L, nrow = n, ncol = d)
  for (i in seq_len(n)) {
    l <- exp(eta_theta[i, ])
    theta_truth[i, ] <- l / sum(l)
    zeta_truth[i, ] <- link_foo(eta_zeta[i, ])
    tmp <- .rzanim(size = n_trials, prob = theta_truth[i, ],
                   zeta = zeta_truth[i, ], d = d)
    Y[i, ] <- tmp[[1L]]
    Z[i, ] <- tmp[[2L]]
    vartheta_truth[i, ] <- l * Z[i, ] / sum(l * Z[i, ])
  }
  data_sim <- data.frame(
    id = rep(seq_len(n), each = d),
    category = rep(seq_len(d), times = n),
    x = rep(X[, 1L], each = d),
    theta = c(t(theta_truth)),
    zeta = c(t(zeta_truth)),
    total = c(t(Y)),
    z = c(t(Z)),
    prop = c(apply(Y, 1L, function(z) z / sum(z))))

  list(df = data_sim, Y = Y, X = X, Z = Z, theta = theta_truth, zeta = zeta_truth,
       abundance = vartheta_truth)
}

#' @export
sim_data_zanim_ln_bspline_curve <- function(n, d = 3L, n_trials, dof_bs_theta = 6L,
                                            dof_bs_zeta = 4L, link_zeta = c("logit", "probit"),
                                            covariance = c("toeplitz", "exponential", "fam"),
                                            rho = 0.8, s2 = 1.0, lg = 0.4, q_factors = 4L) {

  covariance <- match.arg(covariance)
  link_zeta <- match.arg(link_zeta)
  if (length(n_trials) == 1) n_trials <- rep(n_trials, n)
  link_foo <- switch(link_zeta, "logit" = stats::plogis, "probit" = stats::pnorm)
  # Generate covariance matrix
  B <- contr.sum(d)
  B <- qr.Q(qr(B))
  if (covariance == "exponential") {
    x <- stats::runif(d - 1)
    Sigma_V <- s2*exp(-as.matrix(dist(x))/lg)
  } else if (covariance == "toeplitz") {
    Sigma_V <- matrix(0, d - 1, d - 1)
    Sigma_V <- rho^abs(row(Sigma_V) - col(Sigma_V))
  } else if (covariance == "fam") {
    Gamma <- matrix(data = stats::runif((d - 1) * q_factors, 0, 1), nrow = d - 1,
                    ncol = q_factors)
    Psi <- diag(x = stats::runif((d - 1) * (d - 1), 0.1, 1), nrow = d - 1,
                ncol = d - 1)
    Sigma_V <- tcrossprod(Gamma) + Psi
  }
  # Generate random effects u_{ij}
  true_Sigma_U <- B %*% tcrossprod(Sigma_V, B)
  chol_Sigma_U <- chol(true_Sigma_U)
  U <- matrix(nrow = n, ncol = d)
  for (i in seq_len(n)) {
    z <- stats::rnorm(n = d)
    U[i, ] <- drop(z %*% chol_Sigma_U)
  }
  # Linear predictor for \theta
  X <- as.matrix(seq(-1.0, 1.0, length.out = n))
  X_bs_theta <- splines::bs(X, dof_bs_theta)
  betas_theta <- matrix(stats::rnorm(d * dof_bs_theta), dof_bs_theta, d)
  betas_theta[1L, ] <- betas_theta[1L, ] - seq(from = 4, to = 0, length.out = d)
  etas_theta <- X_bs_theta %*% betas_theta
  # Linear predictor for \zeta
  X_bs_zeta <- splines::bs(X, dof_bs_zeta)
  betas_zeta <- matrix(stats::rnorm(d * dof_bs_zeta, -1), dof_bs_zeta, d)
  betas_zeta[1L, ] <- betas_zeta[1L, ] - seq(from = 4, to = 0, length.out = d)
  etas_zeta <- X_bs_zeta %*% betas_zeta
  # Generate data
  Y <- Z <- true_abundance <- true_thetas <- true_zetas <- matrix(0L, nrow = n, ncol = d)
  for (i in seq_len(n)) {
    ee_u <- exp(etas_theta[i, ] + U[i, ])
    true_thetas[i, ] <- ee_u / sum(ee_u)
    true_zetas[i, ] <- link_foo(etas_zeta[i, ])
    tmp <- .rzanim(size = n_trials[i], prob = true_thetas[i, ],
                   zeta = true_zetas[i, ], d = d)
    Y[i, ] <- tmp[[1L]]
    Z[i, ] <- tmp[[2L]]
    true_abundance[i, ] <- ee_u * Z[i, ] / sum(ee_u * Z[i, ])
  }
  data_sim <- data.frame(
    id = rep(seq_len(n), each = d),
    category = rep(seq_len(d), times = n),
    x = rep(X[, 1L], each = d),
    theta = c(t(true_thetas)),
    zeta = c(t(true_zetas)),
    total = c(t(Y)),
    z = c(t(Z)))
  list(df = data_sim, Y = Y, X = X, Z = Z, theta = true_thetas, zeta = true_zetas,
       abundance = true_abundance, Sigma_U = true_Sigma_U, U = U)
}

#' @export
sim_data_zanim_friedman <- function(n, n_trials, p_theta = 10L, p_zeta = 10L,
                                    link = stats::plogis) {
  if (length(n_trials) != n) n_trials <- rep(n_trials[1L], n)

  # Friedman for the prob of categories
  x_theta <- matrix(stats::runif(n * p_theta), nrow = n, ncol = p_theta,
                    byrow = TRUE)
  f1 <- sin(pi * x_theta[, 1L] * x_theta[, 2L]) + (x_theta[, 3L] - 0.5)^3
  f2 <- -1 + 2*x_theta[, 1L] * x_theta[, 2L] + x_theta[, 3L]
  f3 <- 0.5 * (x_theta[, 1L] + x_theta[, 2L]) + x_theta[, 3L]
  f <- cbind(f1, f2, f3)

  # Friedman for the zero inflation part
  # Assume now that zeta_{ij} = \zeta_i for all j = 1,2,3
  x_zeta <- matrix(stats::runif(n * p_zeta), nrow = n, ncol = p_zeta, byrow = TRUE)
  eta <- sin(pi * x_zeta[, 1] * x_zeta[, 2]) + (x_zeta[, 3] - 1)
  zeta_truth <- link(eta)

  # Generate data
  Y <- Z <- vartheta_truth <- theta_truth <- matrix(0L, nrow = n, ncol = 3L)
  for (i in seq_len(n)) {
    l <- exp(f[i, ])
    theta_truth[i, ] <- l / sum(l)
    tmp <- .rzanim(size = n_trials[1L], prob = theta_truth[i, ],
                   zeta = rep(zeta_truth[i], 3L), d = 3L)
    Y[i, ] <- tmp[[1L]]
    Z[i, ] <- tmp[[2L]]
    vartheta_truth[i, ] <- l * Z[i, ] / sum(l * Z[i, ])
  }
  d <- 3L
  data_sim <- data.frame(
    id = rep(seq_len(n), each = d),
    category = rep(seq_len(d), times = n),
    theta = c(t(theta_truth)),
    zeta = rep(zeta_truth, each = d),
    total = c(t(Y)),
    z = c(t(Z)),
    prop = c(apply(Y, 1L, function(u) u / sum(u))))
  list(df = data_sim, Y = Y, X_theta = x_theta, X_zeta = x_zeta,
       theta = theta_truth, zeta = zeta_truth, abundance = vartheta_truth)
}


#' @export
sim_data_multinomial_bspline_curve <- function(n, d, n_trials, dof_bs = 6L) {
  # Linear predictor
  X <- as.matrix(seq(-1.0, 1.0, length.out = n))
  X1_bs <- splines::bs(X, dof_bs)
  betas_theta <- matrix(stats::rnorm(d * dof_bs), dof_bs, d)
  betas_theta[1L, ] <- betas_theta[1L, ] - seq(from = 4, to = 0, length.out = d)
  eta_theta <- X1_bs %*% betas_theta
  # Generate data
  Y <- theta_truth <- matrix(0L, nrow = n, ncol = d)
  for (i in seq_len(n)) {
    theta_truth[i, ] <- exp(eta_theta[i, ]) / sum(exp(eta_theta[i, ]))
    Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials, prob = theta_truth[i, ])
  }
  data_sim <- data.frame(
    id = rep(seq_len(n), each = d),
    category = rep(seq_len(d), times = n),
    x = rep(X[, 1L], each = d),
    theta = c(t(theta_truth)),
    total = c(t(Y)),
    prop = c(apply(Y, 1L, function(z) z / sum(z))))
  list(df = data_sim, Y = Y, X = X, theta = theta_truth)
}

#' @export
sim_data_trinomial_friedman <- function(n, n_trials, p = 10L) {
  x <- matrix(stats::runif(n * p), nrow = n)
  # f1 <- sin(pi * x[, 1] * x[,2]) + (x[,3] - 0.5)^3
  # f2 <- -1 + 2*x[, 2L] * x[, 4L] + exp(x[, 5L])
  # f3 <- 0.5 * (x[, 3L] + x[, 4L]) + x[, 5L]
  f1 <- sin(pi * x[, 1] * x[,2]) + (x[,3] - 0.5)^3
  f2 <- -1 + 2*x[, 1L] * x[, 2L] + exp(x[, 3L])
  f3 <- 0.5 * (x[, 1L] + x[, 2L]) + x[, 3L]
  f <- cbind(f1, f2, f3)
  dim(f)
  # Generate data
  Y <- theta_truth <- matrix(0L, nrow = n, ncol = 3L)
  for (i in seq_len(n)) {
    theta_truth[i, ] <- exp(f[i, ]) / sum(exp(f[i, ]))
    Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials,
                               prob = theta_truth[i, ])
  }
  data_sim <- data.frame(
    id = rep(seq_len(n), each = 3L),
    category = rep(seq_len(3L), times = n),
    theta = c(t(theta_truth)),
    total = c(t(Y)),
    prop = c(apply(Y, 1L, function(z) z / sum(z))))
  list(df = data_sim, Y = Y, X = x)
}

#' @export
sim_data_binary_bspline_curve <- function(n, dof_bs = 4L, link = stats::plogis) {
  x <- seq(-1.0, 1.0, length.out = n)
  X_bs <- splines::bs(as.matrix(x), dof_bs)
  betas <- matrix(stats::rnorm(dof_bs, mean = -1.0), dof_bs)
  eta <- X_bs %*% betas
  theta <- link(eta)
  y <- stats::rbinom(n = n, size = 1, prob = theta)
  data.frame(id = seq_len(n), y = y, x = x, theta = theta)
}

#' @export
sim_data_binary_friedman <- function(n, p = 10L, link = stats::plogis) {
  X <- matrix(stats::runif(n * p), nrow = n, ncol = p, byrow = TRUE)
  eta <- sin(pi * X[, 1] * X[, 2]) + (X[, 3] - 0.5)
  theta <- link(eta)
  y <- stats::rbinom(n = n, size = 1L, prob = theta)
  list(y = y, X = X, theta = theta)
}
