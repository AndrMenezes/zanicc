test_that("check dart prior on zanim", {
  rm(list = ls())
  devtools::load_all()
  path_res <- "./tests/testthat/dart_prior"
  # Simulate data
  n <- 500L
  n_trials <- sample(1e3:2e3, size = n, replace = TRUE)
  d <- 3L
  p_theta <- p_zeta <- 10L

  set.seed(666)

  rho <- 0.4
  Sigma <- matrix(0, p_theta, p_theta)
  Sigma <- rho^abs(row(Sigma) - col(Sigma))
  chol_Sigma <- chol(Sigma)


  # Friedman for the prob of categories
  # x_theta <- matrix(nrow = n, ncol = p_theta)
  # for (i in seq_len(n)) {
  #   x_theta[i, ] <- 4+drop(stats::rnorm(n = p_theta) %*% chol_Sigma)
  # }
  x_theta <- matrix(stats::runif(n * p_theta), nrow = n, ncol = p_theta,
                    byrow = TRUE)
  f1 <- sin(pi * x_theta[, 1L] * x_theta[, 2L]) + (x_theta[, 3L] - 0.5)^3
  f2 <- -1 + 2*x_theta[, 4L] * x_theta[, 5L] + exp(x_theta[, 6L])
  f3 <- 0.5 * (x_theta[, 7L] + x_theta[, 8L]) + sqrt(x_theta[, 9L] * x_theta[, 9L])
  f <- cbind(f1, f2, f3)

  # Friedman for the zero inflation part
  # Assume now that zeta_{ij} = \zeta_i for all j = 1,2,3
  x_zeta <- matrix(stats::runif(n * p_zeta), nrow = n, ncol = p_zeta)
  # x_zeta <- matrix(nrow = n, ncol = p_zeta)
  # for (i in seq_len(n)) {
  #   x_zeta[i, ] <- 4+drop(stats::rnorm(n = p_zeta) %*% chol_Sigma)
  # }
  eta_1 <- -1 + sin(pi * x_zeta[, 1] * x_zeta[, 2]) + (x_zeta[, 3] - 1)^3
  eta_2 <- -2.2 + cos(pi * x_zeta[, 4] * x_zeta[, 5]) + sqrt(x_zeta[, 6])
  eta_3 <- 0.5 + log(x_zeta[, 7] * x_zeta[, 8]) + x_zeta[, 9]^3
  zeta_truth <- cbind(eta_1, eta_2, eta_3)
  zeta_truth <- pnorm(zeta_truth)


  plot(x_zeta[, 6L], zeta_truth[, 2L])
  plot(x_zeta[, 6L], rowMeans(zeta_zidm[, 2L, ]))
  plot(x_zeta[, 6L], rowMeans(mod_zanim$draws_zeta[, 2L, ]))



  # Generate data
  Y <- Z <- theta_truth <- matrix(0L, nrow = n, ncol = 3L)
  for (i in seq_len(n)) {
    theta_truth[i, ] <- exp(f[i, ]) / sum(exp(f[i, ]))
    tmp <- zanimixreg:::.rzanim(size = 800L, prob = theta_truth[i, ],
                                zeta = zeta_truth[i, ], d = 3L)
    Y[i, ] <- tmp[[1L]]
    Z[i, ] <- tmp[[2L]]
  }
  colMeans(1-Z)
  colMeans(Y==0)

  # Fit model
  NDPOST <- 1000L
  NSKIP <- 1000L
  NTREES <- 100L

  X <- cbind(x_theta, x_zeta)
  X <- scale(X)
  # Fiting ZANIM-BART
  mod_zanim <- ZANIMBART$new(Y = Y, X_theta = X, X_zeta = X,
                             link_zeta = "probit", shared_trees = FALSE)
  mod_zanim$SetupMCMC(v0_theta = 3.5 / sqrt(2), v0_zeta = 2.0, ntrees_theta = NTREES,
                      ntrees_zeta = NTREES, ndpost = NDPOST, nskip = NSKIP,
                      update_sigma_theta = TRUE, keep_draws = TRUE,
                      sparse = c(TRUE, TRUE), alpha_random = c(FALSE, FALSE),
                      alpha_sparse = c(1.0, 1.0))
  mod_zanim$RunMCMC()
  # mod_zanim$cpp_obj$alpha_sparse_mult
  # mod_zanim$cpp_obj$alpha_sparse_zi
  # mod_zanim$cpp_obj$splitprobs_mult


  X_wint <- cbind(1, x_zeta)[, 1L, drop = FALSE]
  mod_zanim_reg <- ZANIMLinearRegression$new(Y = Y, X_wint, X_wint)
  mod_zanim_reg$SetupMCMC()
  mod_zanim_reg$RunMCMC()


  mod_zanidm_reg <- ZANIDMLinearRegression$new(Y = Y, X_wint, X_wint)
  mod_zanidm_reg$SetupMCMC()
  mod_zanidm_reg$RunMCMC()

  mean(compute_frob(true_values = zeta_truth, draws = mod_zanim$draws_zeta))
  mean(compute_frob(true_values = zeta_truth, draws = mod_zanim_reg$draws_zeta))
  mean(compute_frob(true_values = zeta_truth, draws = mod_zanidm_reg$draws_zeta))
  # mean(compute_frob(true_values = zeta_truth, draws = zeta_zidm))

  colMeans(compute_kl_prob(true_values = zeta_truth, draws = mod_zanim$draws_zeta))
  colMeans(compute_kl_prob(true_values = zeta_truth, draws = mod_zanim_reg$draws_zeta))
  colMeans(compute_kl_prob(true_values = zeta_truth, draws = mod_zanidm_reg$draws_zeta))
  # colMeans(compute_kl_prob(true_values = zeta_truth, draws = zeta_zidm))

  # Check
  vc_theta_dart <- mod_zanim$GetVarCount(parameter = "theta")
  vc_zeta_dart <- mod_zanim$GetVarCount(parameter = "zeta")

  mean_vc_theta_dart <- apply(vc_theta_dart, c(1, 2), mean)
  prob_vc_theta_dart <- apply(vc_theta_dart > 0, c(1, 2), mean)

  mean_vc_zeta_dart <- apply(vc_zeta_dart, c(1, 2), mean)
  prob_vc_zeta_dart <- apply(vc_zeta_dart > 0, c(1, 2), mean)

  data_dart_theta <- data.frame(prob = c(prob_vc_theta_dart),
                                covariate = rep(1:ncol(X), times = 3),
                                category = rep(1:3, each = ncol(X)))
  p_vc_theta <- ggplot(data_dart_theta, aes(x = covariate, y = prob)) +
    facet_wrap(~category, ncol = 1) +
    geom_point() +
    geom_point(data = dplyr::filter(data_dart_theta, covariate %in% c(1:3),
                                    category == 1L),
               col = "red") +
    geom_point(data = dplyr::filter(data_dart_theta, covariate %in% c(4:6),
                                    category == 2L),
               col = "red") +
    geom_point(data = dplyr::filter(data_dart_theta, covariate %in% c(7:9),
                                    category == 3L),
               col = "red") +
    scale_x_continuous(breaks = scales::pretty_breaks(8), limits = c(1, ncol(X))) +
    scale_y_continuous(breaks = scales::pretty_breaks(6), limits = c(0, 1)) +
    labs(x = "Covariate k", y = "Prob[k in model]")
  save_plot(filename = file.path(path_res, "prob_vc_theta3.png"), plot = p_vc_theta,
            bg = "white", base_height = 7.0)


  data_dart_zeta <- data.frame(prob = c(prob_vc_zeta_dart),
                               covariate = rep(1:ncol(X), times = 3),
                               category = rep(1:3, each = ncol(X)))
  p_vc_zeta <- ggplot(data_dart_zeta, aes(x = covariate, y = prob)) +
    facet_wrap(~category, ncol = 1) +
    geom_point() +
    geom_point(data = dplyr::filter(data_dart_zeta, covariate %in% c((p_theta+1):(p_theta+3)),
                                    category == 1L),
               col = "red") +
    geom_point(data = dplyr::filter(data_dart_zeta, covariate %in% c((p_theta+4):(p_theta+6)),
                                    category == 2L),
               col = "red") +
    geom_point(data = dplyr::filter(data_dart_zeta, covariate %in% c((p_theta+7):(p_theta+9)),
                                    category == 3L),
               col = "red") +
    scale_x_continuous(breaks = scales::pretty_breaks(6)) +
    scale_y_continuous(breaks = scales::pretty_breaks(6), limits = c(0, 1)) +
    labs(x = "Covariate k", y = "Prob[k in model]")
  save_plot(filename = file.path(path_res, "prob_vc_zeta3.png"), plot = p_vc_zeta,
            bg = "white", base_height = 7.0)

})


test_that("check dart prior with linear structure", {

  rm(list = ls())
  library(zanimixreg)
  set.seed(1212)
  n_sample <- 100L
  n_trials <- sample(1e3:2e3, size = n_sample, replace = TRUE)
  d <- 20L
  d_r <- 6L
  p <- 40L
  p_r <- 6L
  theta0 <- c(0.1, 0.01, 0.005)[3L]
  tau <- (1 - theta0) / theta0

  rho <- 0.4
  Sigma <- matrix(0, p, p)
  Sigma <- rho^abs(row(Sigma) - col(Sigma))
  # diag(Sigma) <- diag(Sigma) + sqrt(.Machine$double.eps)
  chol_Sigma <- chol(Sigma)
  X <- matrix(nrow = n_sample, ncol = p)
  for (i in seq_len(n_sample)) {
    X[i, ] <- drop(stats::rnorm(n = p) %*% chol_Sigma)
  }

  # X <- matrix(stats::rnorm(n_sample * p), ncol = p)

  eta_alpha <- eta_zeta <- matrix(nrow = n_sample, ncol = d)
  true_coef_counts <- matrix(data = 0.0, nrow = p, ncol = d)
  true_coef_zi <- matrix(data = 0.0, nrow = p, ncol = d)

  fc <- 2.0
  fz <- 2.0
  for (j in seq_len(d_r)) {
    chosen_counts <- sample.int(n = p, size = p_r)
    chosen_zi <- sample.int(n = p, size = p_r)
    true_coef_counts[chosen_counts, j] <- seq(0.6 * fc, 0.9 * fc, length = p_r) * c(1, -1)
    true_coef_zi[chosen_zi, j] <- seq(0.6 * fz, 0.9 * fz, length = p_r) * c(1, -1)
  }
  intercept_counts <- stats::runif(n = d, -2.3, -1.0)
  intercept_zi <- stats::runif(n = d, -1.0, -0.5)
  for (j in seq_len(d)) {
    eta_alpha[, j] <- intercept_counts[j] + drop(X %*% true_coef_counts[, j])
    eta_zeta[, j] <- intercept_zi[j] + drop(X %*% true_coef_zi[, j])
  }
  dm <- TRUE
  alphas <- exp(eta_alpha)
  true_zetas <- stats::pnorm(eta_zeta)
  apply(true_zetas, 2, quantile)
  apply(alphas, 2, quantile)

  Y <- Z <- true_thetas <- true_varthetas <- matrix(nrow = n_sample, ncol = d)
  for (i in seq_len(n_sample)) {
    z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
    is_zero <- z == 0L
    # Hack to avoid all zeros (it happen very rarely)
    while (all(is_zero)) {
      z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
      is_zero <- z == 0L
    }
    p_ij <- alphas[i, ] / sum(alphas[i, ])
    if (dm) {
      # a <- alphas[i, ] * tau
      g <- stats::rgamma(n = d, shape = p_ij * tau, rate = 1.0)
    } else {
      g <- p_ij
    }
    true_thetas[i, ] <- g / sum(g)
    true_varthetas[i, ] <- z * g / sum(z * g)
    if (sum(is_zero) == d - 1L) {
      Y[i, ] <- rep(0L, d)
      Y[i, !is_zero] <- n_trials[i]
    } else {
      Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials[i],
                                 prob = true_varthetas[i, ])
    }
    Z[i, ] <- z
  }
  colMeans(Y == 0)
  colMeans(1 - Z)

  # Fit model
  NDPOST <- 5000L
  NSKIP <- 10000L
  NTREES <- 100L

  # X <- scale(X)
  # Fiting ZANIM-BART
  mod_zanim <- ZANIMBART$new(Y = Y, X_theta = X, X_zeta = X,
                             link = "probit", shared_trees = FALSE)
  mod_zanim$SetupMCMC(v0_theta = 3.5 / sqrt(2), v0_zeta = 2.0, ntrees_theta = NTREES,
                      ntrees_zeta = NTREES, ndpost = NDPOST, nskip = NSKIP,
                      update_sigma_theta = TRUE,
                      sparse = c(TRUE, TRUE), alpha_random = c(FALSE, FALSE),
                      alpha_sparse = c(1.0, 1.0))
  mod_zanim$RunMCMC()
  mod_zanim$cpp_obj$alpha_sparse_mult
  mod_zanim$cpp_obj$alpha_sparse_zi
  mod_zanim$cpp_obj$splitprobs_mult

  mppi_theta_zanim <- apply(mod_zanim$cpp_obj$varcount_mcmc_theta > 0, c(1, 2), mean)
  mppi_zeta_zanim <- apply(mod_zanim$cpp_obj$varcount_mcmc_zeta > 0, c(1, 2), mean)

  x11()
  par(mfrow = c(5, 4))
  for (j in seq_len(d)) {
    ids <- which(true_coef_counts[, j] != 0)
    plot(mppi_theta_zanim[, j], main = paste0("ZANIM ", j), ylim = c(0, 1))
    points(ids, mppi_theta_zanim[ids, j], col = "red", pch = 19)
    abline(h = 0.5, col = "red")
  }

  x11()
  par(mfrow = c(5, 4))
  for (j in seq_len(d)) {
    ids <- which(true_coef_zi[, j] != 0)
    plot(mppi_zeta_zanim[, j], main = paste0("ZANIM ", j), ylim = c(0, 1))
    points(ids, mppi_zeta_zanim[ids, j], col = "red", pch = 19)
    abline(h = 0.5, col = "red")
  }

  ###
  mod_zanimln <- ZANIMLogNormalBART$new(Y = Y, X_theta = X, X_zeta = X)
  mod_zanimln$SetupMCMC(v0_theta = 3.5 / sqrt(2), k_zeta = 2.0,
                        ntrees_theta = NTREES,
                        ntrees_zeta = NTREES, ndpost = NDPOST, nskip = NSKIP,
                        update_sigma_theta = TRUE, covariance_type = 2L,
                        q_factors = 10L,
                        sparse = c(TRUE, TRUE), alpha_random = c(FALSE, FALSE),
                        alpha_sparse = c(1.0, 1.0))
  mod_zanimln$RunMCMC()

  mppi_theta_zanimln <- apply(mod_zanimln$cpp_obj$varcount_mcmc_theta > 0, c(1, 2), mean)
  mppi_zeta_zanimln <- apply(mod_zanimln$cpp_obj$varcount_mcmc_zeta > 0, c(1, 2), mean)

  x11()
  par(mfrow = c(5, 4))
  for (j in seq_len(d)) {
    ids <- which(true_coef_counts[, j] != 0)
    plot(mppi_theta_zanimln[, j], main = paste0("ZANIM-LN ", j), ylim = c(0, 1))
    points(ids, mppi_theta_zanimln[ids, j], col = "red", pch = 19)
    abline(h = 0.5, col = "red")
  }

  x11()
  par(mfrow = c(5, 4))
  for (j in seq_len(d)) {
    ids <- which(true_coef_zi[, j] != 0)
    plot(mppi_zeta_zanimln[, j], main = paste0("ZANIM-LN ", j), ylim = c(0, 1))
    points(ids, mppi_zeta_zanimln[ids, j], col = "red", pch = 19)
    abline(h = 0.5, col = "red")
  }

  ##########
  # Fit ZIDM-reg
  mod_zidm <- ZIDM::ZIDMbvs_R(Z = Y, X = X, X_theta = X,
                              iterations = 20000L, thin = 10L)
  str(mod_zidm)
  # Perform burn-in
  to_rmv <- seq_len(1000L)
  beta_alpha_zidm <- mod_zidm[["beta_gamma"]][, , -to_rmv]
  beta_zeta_zidm <- mod_zidm[["beta_theta"]][, , -to_rmv]

  # Compute the linear predictions and map to the parameter space ----------------
  ndpost_zidm <- dim(beta_alpha_zidm)[3L]
  alpha_zidm <- theta_zidm <- zeta_zidm <- array(dim = c(n_sample, d, ndpost_zidm))
  X_wint <- cbind(1, X)
  for (t in seq_len(ndpost_zidm)) {
    for (j in seq_len(d)) {
      alpha_zidm[,j,t] <- exp(X_wint %*% as.matrix(beta_alpha_zidm[j,,t]))
      zeta_zidm[,j,t] <- stats::plogis(X_wint %*% as.matrix(beta_zeta_zidm[j,,t]),
                                       lower.tail = FALSE)
    }
  }
  psis <- mod_zidm$cc[, ,-to_rmv]
  tn <- apply(psis, c(1, 3), sum)
  vartheta_zidm <- sweep(x = psis, MARGIN = c(1, 3), STATS = tn, FUN = "/")

  # Marginal posterior-probability of inclusion
  mppi_theta_zidm <- t(apply(mod_zidm$varphi[ ,-1, -to_rmv ], c(1, 2), mean))
  mppi_zeta_zidm <- t(apply(mod_zidm$zeta[ ,-1, -to_rmv ], c(1, 2), mean))

  x11()
  par(mfrow = c(5, 4))
  for (j in seq_len(d)) {
    ids <- which(true_coef_counts[, j] != 0)
    plot(mppi_theta_zidm[, j], main = paste0("ZIDM ", j), ylim = c(0, 1))
    points(ids, mppi_theta_zidm[ids, j], col = "red", pch = 19)
    abline(h = 0.5, col = "red")
  }

  x11()
  par(mfrow = c(5, 4))
  for (j in seq_len(d)) {
    ids <- which(true_coef_zi[, j] != 0)
    plot(mppi_zeta_zidm[, j], main = paste0("ZIDM ", j), ylim = c(0, 1))
    points(ids, mppi_zeta_zidm[ids, j], col = "red", pch = 19)
    abline(h = 0.5, col = "red")
  }

})






