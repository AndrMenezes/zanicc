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
  true_coef_zi <- rbind(c(-3.0, -3.5, -2.0),
                        c(-1.0, 2.0, 1.5)
                        , c(0.5, -1.5, 2.0)
                        )
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
  theta0 <- 0.1 # 0.025, 0.005
  tau <- (1 - theta0) / theta0

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
    g <- stats::rgamma(n = d, shape = tau * p_ij, rate = 1.0)
    # g <- alphas[i, ]
    true_thetas[i, ] <- g / sum(g)
    true_varthetas[i, ] <-  z * g / sum(z * g)
    if (sum(is_zero) == d - 1L) {
      Y[i, ] <- rep(0L, d)
      Y[i, !is_zero] <- n_trials
    } else {
      Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials, prob = true_varthetas[i, ])
    }
    Z[i, ] <- z
  }

  # Fit ZANIM-linear reg

  # devtools::load_all()
  mod_zanim_lr <- ZANIMLinearRegression$new(Y = Y, X_theta = X, X_zeta = X)
  mod_zanim_lr$SetupMCMC(sd_prior_beta_theta = rep(10, p + 1))
  mod_zanim_lr$RunMCMC()

  mod_zanim_lr$PosterioMeanCoef("theta")
  true_coef_counts
  mod_zanim_lr$PosterioMeanCoef("zeta")
  true_coef_zi

  # devtools::load_all()
  mod_zanidm_lr <- ZANIDMLinearRegression$new(Y = Y, X_alpha = X, X_zeta = X)
  mod_zanidm_lr$SetupMCMC(sd_prior_beta_alpha = rep(10, p + 1))
  mod_zanidm_lr$RunMCMC()

  mod_zanidm_lr$PosterioMeanCoef("alpha")
  true_coef_counts
  mod_zanidm_lr$PosterioMeanCoef("zeta")
  true_coef_zi

  # Use K. code
  mod_zidm <- ZIDM::ZIDMbvs_R(Z = Y, X = X[, -1L], X_theta = X[, -1L],
                              iterations = 40000L, thin = 10L)
  # Perform burn-in
  to_rmv <- seq_len(1000L)
  beta_alpha_zidm <- mod_zidm[["beta_gamma"]][, , -to_rmv]
  beta_zeta_zidm <- mod_zidm[["beta_theta"]][, , -to_rmv]

  # Compute the linear predictions and map to the parameter space ----------------
  ndpost_zidm <- dim(beta_alpha_zidm)[3L]
  alpha_zidm <- theta_zidm <- zeta_zidm <- array(dim = c(n_sample, d, ndpost_zidm))
  X_wint <- X # cbind(1, X)
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
  t(apply(beta_alpha_zidm, c(1, 2), mean))
  true_coef_counts

  apply(beta_alpha_zidm, c(1, 2), function(x) coda::effectiveSize(coda::as.mcmc(x)))
  apply(mod_zanidm_lr$draws_betas_alpha, c(1, 2), function(x) coda::effectiveSize(coda::as.mcmc(x)))


  mod_dm = ZIDM::DMbvs_R(Z = Y, X = X[, -1L], iterations = 40000L, thin = 10L)
  beta_alpha_m <- mod_dm$beta_gamma[, , -to_rmv]
  t(apply(beta_alpha_m, c(1, 2), mean))
  true_coef_counts

  compute_frob <- function(true_values, draws) {
    ndpost <- dim(draws)[3]
    diffs <- (array(true_values, dim = c(dim(true_values), ndpost)) - draws)^2
    sqrt(apply(diffs, 3, sum))
  }
  frobs_zanim <- compute_frob(true_varthetas, mod_zanim_lr$draws_abundance)
  frobs_zanidm <- compute_frob(true_varthetas, mod_zanidm_lr$draws_abundance)
  frobs_zidm <- compute_frob(true_varthetas, vartheta_zidm)
  yrange <- range(c(frobs_zanidm, frobs_zanim, frobs_zidm))
  par(mfrow = c(1, 3))
  plot(frobs_zanim, type = "l", main = "ZANIM", ylim = yrange)
  plot(frobs_zanidm, type = "l", main = "ZANIDM", ylim = yrange)
  plot(frobs_zidm, type = "l", main = "ZIDM", ylim = yrange)
  graphics.off()


  par(mfrow  = c(3, 3))
  for (j in seq_len(d)) {
    plot(true_varthetas[, j], rowMeans(mod_zanim_lr$draws_abundance[, j, ]),
         main = paste0("ZANIM ", j), ylab = "posterior mean", xlab = "truth")
    abline(0, 1, col = 2)
    plot(true_varthetas[, j], rowMeans(mod_zanidm_lr$draws_abundance[, j, ]),
         main = paste0("ZANIDM ", j), ylab = "posterior mean", xlab = "truth")
    abline(0, 1, col = 2)
    plot(true_varthetas[, j], rowMeans(vartheta_zidm[, j, ]),
         main = paste0("ZIDM ", j), ylab = "posterior mean", xlab = "truth")
    abline(0, 1, col = 2)
  }

  ll_zanim <- mod_zanim_lr$LogPredictiveLikelihood()
  ll_zanidm <- mod_zanidm_lr$LogPredictiveLikelihood()

  par(mfrow = c(1, 2))
  loglik_zanim <- rowSums(ll_zanim)
  loglik_zanidm <- rowSums(ll_zanidm)
  yrange <- range(c(loglik_zanim, loglik_zanidm))
  plot(rowSums(ll_zanim), type = "l", ylim = yrange)
  plot(rowSums(ll_zanidm), type = "l", ylim = yrange)

  dim(ll_zanim)
  dim(ll_zanidm)

  as.data.frame(loo::loo_compare(list(zanim = loo::loo(ll_zanim),
                                      zanidm = loo::loo(ll_zanidm))))

  yrep <- mod_zanim_lr$GetPosteriorPredictive()
  bayesplot::ppc_ecdf_overlay(y = Y[, 1L], yrep = yrep[, , 1L])
  dim(yrep)

})



