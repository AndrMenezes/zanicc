test_that("zanim linear reg", {
  library(ggplot2)
  rm(list = ls())
  devtools::load_all()
  n_sample <- 100L
  n_trials <- 400L
  d <- 3L
  p <- 2L
  set.seed(6669)
  X <- cbind(1, matrix(stats::rnorm(n_sample * p, sd = 0.5), ncol = p))
  eta_alpha <- eta_zeta <- matrix(nrow = n_sample, ncol = d)
  true_coef_counts <- rbind(c(0.1, 0.2, 0.3),
                            c(2.0, -3.0, 1.0)
                            , c(4.0, -4.0, 3.0)
  )
  true_coef_zi <- rbind(c(-2.0, -1.5, -1.0),
                        c(-1.0, 2.0, 1.5)
                        , c(0.5, -1.5, 2.0)
  )
  rownames(true_coef_counts) <- rownames(true_coef_zi) <- c("intercept", "x1", "x2")
  for (j in seq_len(d)) {
    eta_alpha[, j] <- true_coef_counts[1, j] + X[, 2L] * true_coef_counts[2, j] + X[, 3L] * true_coef_counts[3, j]
    eta_zeta[, j] <- true_coef_zi[1, j] + X[, 2L] * true_coef_zi[2, j] + X[, 3L] * true_coef_zi[3, j]
  }
  par(mfrow = c(1, 3))
  for (j in seq_len(d)) plot(X[, 2L], eta_alpha[, j])
  # par(mfrow = c(1, 3))
  # for (j in seq_len(d)) plot(X[, 3L], eta_zeta[, j])

  alphas <- exp(eta_alpha)
  true_zetas <- stats::pnorm(eta_zeta)
  apply(true_zetas, 2, quantile)
  apply(alphas, 2, quantile)
  a0 <- 0.001 # 0.025, 0.005
  tau <- (1 - a0) / a0

  Y <- Z <- true_thetas <- true_varthetas <- matrix(nrow = n_sample, ncol = d)
  for (i in seq_len(n_sample)) {
    z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
    is_zero <- z == 0L
    # Hack to avoid all zeros (it happen very rarely)
    # while (all(is_zero)) {
    #   z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
    #   is_zero <- z == 0L
    # }
    p_ij <- alphas[i, ] / sum(alphas[i, ])
    g <- stats::rgamma(n = d, shape = tau * p_ij, rate = 1.0)
    # g <- alphas[i, ]
    true_thetas[i, ] <- g / sum(g)
    true_varthetas[i, ] <-  z * g / sum(z * g)
    if (all(is_zero)) {Y[i, ] <- 0L; true_varthetas[i, ] <- 0.0}
    if (sum(is_zero) == d - 1L) {
      Y[i, ] <- rep(0L, d)
      Y[i, !is_zero] <- n_trials
    } else {
      Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials, prob = true_varthetas[i, ])
    }
    Z[i, ] <- z
  }
  cbind(all=colMeans(Y==0), structural=colMeans(Z==0), sampling=colMeans(Y==0) - colMeans(Z==0))
  # Fit ZANIM-linear reg

  # devtools::load_all()
  mod_zanim_reg <- ZANIMRegression$new(Y = Y, X_theta = X, X_zeta = X)
  mod_zanim_reg$SetupMCMC(sd_prior_beta_theta = rep(1, p + 1))
  mod_zanim_reg$RunMCMC()
  mod_zanim_reg$PosterioMeanCoef(parameter = "theta")
  true_coef_counts
  mod_zanim_reg$PosterioMeanCoef(parameter = "zeta")
  true_coef_zi
  par(mfrow = c(1, 3))
  plot(mod_zanim_reg$draws_betas_theta[1L, 1L, ], type = "l")
  plot(mod_zanim_reg$draws_betas_theta[1L, 2L, ], type = "l")
  plot(mod_zanim_reg$draws_betas_theta[1L, 3L, ], type = "l")

  plot(mod_zanim_reg$draws_betas_theta[3L, 1L, ], type = "l")
  plot(mod_zanim_reg$draws_betas_theta[3L, 2L, ], type = "l")
  plot(mod_zanim_reg$draws_betas_theta[3L, 3L, ], type = "l")

  coda::effectiveSize(coda::as.mcmc(compute_frob_chain(
    true_values = true_varthetas, draws = mod_zanim_reg$draws_abundance)))
  coda::effectiveSize(coda::as.mcmc(compute_frob_chain(
    true_values = true_thetas, draws = mod_zanim_reg$draws_theta)))

  # devtools::load_all()
  mod_zanim_ln_reg <- ZANIMLNRegression$new(Y = Y, X_theta = X, X_zeta = X)
  mod_zanim_ln_reg$SetupMCMC(sd_prior_beta_theta = rep(1, p + 1),
                             covariance_type = "wishart", ndpost = 5000, nskip = 30000)
  mod_zanim_ln_reg$RunMCMC()
  mod_zanim_ln_reg$PosterioMeanCoef(parameter = "theta")
  true_coef_counts
  mod_zanim_ln_reg$PosterioMeanCoef(parameter = "zeta")
  true_coef_zi

  par(mfrow = c(1, 3))
  plot(mod_zanim_ln_reg$draws_betas_theta[1L, 1L, ], type = "l")
  plot(mod_zanim_ln_reg$draws_betas_theta[1L, 2L, ], type = "l")
  plot(mod_zanim_ln_reg$draws_betas_theta[1L, 3L, ], type = "l")

  plot(mod_zanim_ln_reg$draws_betas_theta[2L, 1L, ], type = "l")
  plot(mod_zanim_ln_reg$draws_betas_theta[2L, 2L, ], type = "l")
  plot(mod_zanim_ln_reg$draws_betas_theta[2L, 3L, ], type = "l")

  coda::effectiveSize(coda::as.mcmc(compute_frob_chain(
    true_values = true_varthetas, draws = mod_zanim_ln_reg$draws_abundance)))
  coda::effectiveSize(coda::as.mcmc(compute_frob_chain(
    true_values = true_thetas, draws = mod_zanim_ln_reg$draws_theta)))


  c(zanim_reg = compute_frob(true_values = true_varthetas,
                             estimates = apply(mod_zanim_reg$draws_abundance, c(1, 2), mean)),
    zanim_ln_reg = compute_frob(true_values = true_varthetas,
                                estimates = apply(mod_zanim_ln_reg$draws_abundance, c(1, 2), mean))
  )
  c(zanim_reg = compute_frob(true_values = true_thetas,
                             estimates = apply(mod_zanim_reg$draws_theta, c(1, 2), mean)),
    zanim_ln_reg = compute_frob(true_values = true_thetas,
                                estimates = apply(mod_zanim_ln_reg$draws_theta, c(1, 2), mean))
  )
  c(
    zanim_reg = coda::effectiveSize(coda::as.mcmc(compute_frob_chain(
      true_values = true_varthetas, draws = mod_zanim_reg$draws_abundance))),
    zanim_ln_reg = coda::effectiveSize(coda::as.mcmc(compute_frob_chain(
      true_values = true_varthetas, draws = mod_zanim_reg$draws_abundance)))
  )
  c(
    zanim_reg = coda::effectiveSize(coda::as.mcmc(compute_kl_simplex_chain(
      true_values = true_thetas, draws = mod_zanim_reg$draws_theta))),
    zanim_ln_reg = coda::effectiveSize(coda::as.mcmc(compute_kl_simplex_chain(
      true_values = true_thetas, draws = mod_zanim_reg$draws_theta)))
  )


})



