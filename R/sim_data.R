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
    ee_u1 <- exp(etas_theta[i, ])
    true_thetas[i, ] <- ee_u1 / sum(ee_u1)
    ee_u2 <- exp(etas_theta[i, ] + U[i, ])
    prob <- ee_u2/sum(ee_u2)
    true_zetas[i, ] <- link_foo(etas_zeta[i, ])
    tmp <- .rzanim(size = n_trials[i], prob = prob,
                   zeta = true_zetas[i, ], d = d)
    Y[i, ] <- tmp[[1L]]
    Z[i, ] <- tmp[[2L]]
    true_abundance[i, ] <- ee_u2 * Z[i, ] / sum(ee_u2 * Z[i, ])
  }
  data_sim <- data.frame(
    id = rep(seq_len(n), each = d),
    category = rep(seq_len(d), times = n),
    x = rep(X[, 1L], each = d),
    theta = c(t(true_thetas)),
    zeta = c(t(true_zetas)),
    abundance = c(t(true_abundance)),
    total = c(t(Y)),
    z = c(t(Z)))
  list(df = data_sim, Y = Y, X = X, Z = Z,
       theta = true_thetas, zeta = true_zetas,
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

  if (length(n_trials) == 1L) n_trials <- rep(n_trials, n)
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
    Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials[i], prob = theta_truth[i, ])
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

#' @export
sim_data_multinomial_2d <- function(n_grid = 20, n_sample = n_grid^2, d, n_trials,
                                    region = c("square", "convexhull"),
                                    xmax = 2.0, X_aux) {

  # Covariates
  region <- match.arg(region)
  if (region == "convexhull") {
    X <- suppressWarnings(rconvexhull(n = n_sample, X = X_aux))
  } else {
    x1 <- seq(-xmax, xmax, length.out = n_grid)
    x2 <- seq(-xmax, xmax, length.out = n_grid)
    X <- expand.grid(x1 = x1, x2 = x2)
    X <- as.matrix(X)
  }
  n <- nrow(X)

  if (length(n_trials) == 1L) n_trials <- rep(n_trials, n)

  # Linear predictor
  eta_theta <- matrix(nrow = n, ncol = d)
  # Intercept
  a0 <- stats::runif(n = d, -2.3, -1.0)
  for (j in seq_len(d)) {
    parms <- stats::runif(4, 1, 4)
    eta_theta[, j] <- a0[j] + 1/(1 + exp(-parms[1] * X[, 1L] - parms[2] * X[, 2L]))
    eta_theta[, j] <- eta_theta[, j] + 1/(1 + exp(-parms[3]*X[, 1L] - parms[4] * X[, 2L]))
  }
  eta_theta <- exp(eta_theta)
  # Generate data
  Y <- true_theta <- matrix(0L, nrow = n, ncol = d)
  for (i in seq_len(n)) {
    true_theta[i, ] <- eta_theta[i, ] / sum(eta_theta[i, ])
    Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials[i], prob = true_theta[i, ])
  }
  data_sim <- data.frame(
    id = rep(seq_len(n), each = d),
    category = rep(seq_len(d), times = n),
    x1 = rep(X[, 1L], each = d),
    x2 = rep(X[, 2L], each = d),
    theta = c(t(true_theta)),
    total = c(t(Y)),
    prop = c(apply(Y, 1L, function(z) z / sum(z))))
  list(df = data_sim, Y = Y, X = X, theta = true_theta)
}

#' @export
sim_data_zanicc_2d <- function(n_grid = 20, n_sample = n_grid^2, d, n_trials,
                               region = c("square", "convexhull"),
                               xmax = 2.0, X_aux = NULL, random_effects = TRUE,
                               structural_zero = TRUE,
                               q_factors = .ledermann(d)) {

  # Covariates
  region <- match.arg(region)
  if (region == "convexhull") {
    if (is.null(X_aux)) {
      data(pollen_data, package = "zanicc")
      X_aux <- pollen_data$X[, c("gdd5", "mtco")]
      X_aux <- scale(X_aux)
    }
    X <- suppressWarnings(rconvexhull(n = n_sample, X = X_aux))
  } else {
    x1 <- seq(-xmax, xmax, length.out = n_grid)
    x2 <- seq(-xmax, xmax, length.out = n_grid)
    X <- expand.grid(x1 = x1, x2 = x2)
    X <- as.matrix(X)
  }
  n <- nrow(X)

  if (length(n_trials) == 1L) n_trials <- rep(n_trials, n)

  # Structural zero
  if (structural_zero) {
    eta_zeta <- matrix(nrow = n, ncol = d)
    b0 <- stats::runif(n = d, -1.0, -0.5)
    for (j in seq_len(d)) {
      scale_z <- stats::runif(1, 1.0, 4.0)
      eta_zeta[, j] <- b0[j] + 1.0/(1.0 + exp(-scale_z * X[, 1L] * X[, 2L]))
    }
    # Population-level structural zero probability
    true_zetas <- stats::pnorm(eta_zeta)
  }
  # Random effects
  U <- matrix(0, n_sample, d)
  if (random_effects) {
    Gamma <- matrix(stats::runif(d * q_factors, 0, 1), d, q_factors)
    Psi <- diag(seq(0.32, 0.35, length.out = d))
    Sigma_U <- tcrossprod(Gamma) + Psi
    chol_Sigma_U <- chol(Sigma_U)
    for (i in seq_len(n_sample)) U[i, ] <- drop(stats::rnorm(d) %*% chol_Sigma_U)
  }
  # Population-level count probability
  eta_theta <- matrix(nrow = n, ncol = d)
  a0 <- stats::runif(n = d, -2.3, -1.0)
  for (j in seq_len(d)) {
    parms_c <- stats::runif(4, 1.0, 4.0)
    eta_theta[, j] <- a0[j] + 1.0/(1.0 + exp(-parms_c[1] * X[, 1L] - parms_c[2] * X[, 2L]))
    eta_theta[, j] <- eta_theta[, j] + 1.0/(1.0 + exp(-parms_c[3]*X[, 1L] + parms_c[4] * X[, 2L]))
  }
  eta_theta <- exp(eta_theta)

  # Indicator for structural zero
  z <- rep(1L, d)

  # Generate data
  Y <- Z <- true_thetas <- true_varthetas <- matrix(nrow = n, ncol = d)
  for (i in seq_len(n)) {

    # Structural zeros
    if (structural_zero) {
      z <- stats::rbinom(d, 1L, prob = 1.0 - true_zetas[i, ])
      # Avoid all zeros
      while (all(z == 0L)) {
        z <- stats::rbinom(d, 1L, prob = 1.0 - true_zetas[i, ])
      }
    }
    Z[i, ] <- z
    is_zero <- z == 0L

    eU <- exp(U[i, ])
    true_thetas[i, ] <- eta_theta[i, ] / sum(eta_theta[i, ])
    true_varthetas[i, ] <- z * eta_theta[i, ] * eU / sum(z * eta_theta[i, ] * eU)
    if (sum(is_zero) == d - 1L) {
      Y[i, ] <- rep(0L, d)
      Y[i, !is_zero] <- n_trials[i]
    } else {
      Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials[i],
                                 prob = true_varthetas[i, ])
    }

  }
  data_sim <- data.frame(
    id = rep(seq_len(n), each = d),
    category = rep(seq_len(d), times = n),
    x1 = rep(X[, 1L], each = d),
    x2 = rep(X[, 2L], each = d),
    theta = c(t(true_thetas)),
    zeta = if (structural_zero) c(t(true_zetas)) else NULL ,
    vartheta = c(t(true_varthetas)),
    total = c(t(Y)),
    prop = c(apply(Y, 1L, function(y) y / sum(y))))

  list(df = data_sim,
       Y = Y, X = X, Z = if (structural_zero) Z else NULL,
       true_thetas = true_thetas,
       true_zetas = if (structural_zero) true_zetas else NULL,
       true_varthetas = true_varthetas, U = if (random_effects) U else NULL)

}

#' @export
#' @description
#' Simulate data following the scenario 1 of the main paper, which
#' corresponds to d=4 with cosine, sine, and polynomial predictors as defined in
#' main paper.
sim_zanim_ln_s1 <- function(n_sample, random_effects = TRUE, structural_zero = TRUE,
                            seed = 1212) {

  set.seed(seed)

  n_trials <- sample(seq.int(100L, 500L), n_sample, replace = TRUE)
  X <- as.matrix(seq(-1, 1, length.out = n_sample))

  # Linear predictors
  eta_theta <- cbind(5*cos(pi*X), 1.5*sin(2*pi*X), 2*(X^3), -2*(X^2))
  alphas <- exp(eta_theta)

  if (structural_zero) {
    intercept <- rep(1.5, 4)
    eta_zeta <- cbind(exp(-5.0 * X^2), X - 2*(X - 0.5)^2, -2*X + 3 * X^3,
                      3*X - 2 * X^3)
    eta_zeta <- t(t(eta_zeta) - intercept)
    true_zetas <- stats::pnorm(eta_zeta)
  }


  # Random effects
  U <- matrix(0, n_sample, d)
  if (random_effects) {
    q_factors <- 2L
    Gamma <- matrix(stats::runif(d * q_factors, 0, 1), d, q_factors)
    Psi <- diag(seq(0.32, 0.35, length.out = d))
    Sigma_U <- tcrossprod(Gamma) + Psi
    chol_Sigma_U <- chol(Sigma_U)
    for (i in seq_len(n_sample)) U[i, ] <- drop(stats::rnorm(d) %*% chol_Sigma_U)
  }

  # Indicator for structural zero
  z <- rep(1L, d)

  # Keep data
  Y <- Z <- true_thetas <- true_varthetas <- matrix(0, n_sample, d)
  # Sampling
  for (i in seq_len(n_sample)) {

    # Structural zeros
    if (structural_zero) {
      z <- stats::rbinom(d, 1L, prob = 1.0 - true_zetas[i, ])
      # avoid all zeros
      while (all(z == 0L)) {
        z <- stats::rbinom(d, 1L, prob = 1.0 - true_zetas[i, ])
      }
    }
    Z[i, ] <- z
    is_zero <- z == 0L

    # Population-level count probabilities
    true_thetas[i, ] <- alphas[i, ] / sum(alphas[i, ])

    # Individual-level count probabilities
    vartheta <- z * alphas[i, ] * exp(U[i, ])
    true_varthetas[i, ] <- vartheta / sum(vartheta)

    # Simulate the counts
    if (sum(is_zero) == d - 1L) {
      Y[i, ] <- 0
      Y[i, !is_zero] <- n_trials[i]
    } else {
      Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials[i],
                                 prob = true_varthetas[i, ])
    }
  }
  list(Y = Y, X = X, Z = if (structural_zero) Z else NULL,
       true_thetas = true_thetas,
       true_zetas = if (structural_zero) true_zetas else NULL,
       true_varthetas = true_varthetas, U = if (random_effects) U else NULL)
}
