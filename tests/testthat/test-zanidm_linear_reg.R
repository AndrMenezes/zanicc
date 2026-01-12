test_that("zanidm linear reg is working", {

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
  b_a <- cbind(c(0.1, 0.2, 0.3),
               c(2.0, -3.0, 1.0)
               , c(4.0, -4.0, 3.0)
               )
  b_z <- cbind(c(-3.0, -3.5, -2.0),
               c(-1.0, 2.0, 1.5)
               , c(0.5, -1.5, 2.0)
               )
  for (j in seq_len(d)) {
    eta_alpha[, j] <- b_a[j, 1] + X[, 2L] * b_a[j, 2] + X[, 3L] * b_a[j, 3]
    eta_zeta[, j] <- b_z[j, 1] + X[, 2L] * b_z[j, 2] + X[, 3L] * b_z[j, 3]
  }

  par(mfrow = c(1, 3))
  for (j in seq_len(d)) plot(X[, 2L], eta_alpha[, j])
  # par(mfrow = c(1, 3))
  # for (j in seq_len(d)) plot(X[, 3L], eta_zeta[, j])

  alphas <- exp(eta_alpha)
  true_zetas <- pnorm(eta_zeta)
  apply(true_zetas, 2, quantile)
  apply(alphas, 2, quantile)
  a0 <- 0.05
  tau <- 1#(1 - a0) / a0

  Y <- Z <- true_thetas <- true_varthetas <- matrix(nrow = n_sample, ncol = d)

  for (i in seq_len(n_sample)) {
    z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
    is_zero <- z == 0L
    # Hack to avoid all zeros (it happen very rarely)
    while (all(is_zero)) {
      z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
      is_zero <- z == 0L
    }
    g <- stats::rgamma(n = d, shape = tau * alphas[i, ], rate = 1.0)
    # g <- alphas[i, ]
    true_thetas[i, ] <- g / sum(g)
    true_varthetas[i, ] <- z * g / sum(z * g)
    if (sum(is_zero) == d - 1L) {
      Y[i, ] <- rep(0L, d)
      Y[i, !is_zero] <- n_trials
    } else {
      Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials, prob = true_varthetas[i, ])
    }
    Z[i, ] <- z
  }

  par(mfrow = c(3, 3))
  for (j in seq_len(d)) plot(X[, 2L], eta_alpha[, j])
  for (j in seq_len(d)) plot(X[, 2L], true_thetas[, j])
  for (j in seq_len(d)) plot(X[, 2L], true_varthetas[, j])

  X_alpha <- X_zeta <- X
  sd_prior_beta_alpha = rep(10, ncol(X))
  sd_prior_beta_zeta = diag(10, ncol(X))
  ndpost = 5000L; nskip = 5000L; nthin = 1L

  # devtools::load_all()
  ml <- Rcpp::Module(module = "zanidm_linear_reg", PACKAGE = "zanimixreg")
  obj <- new(ml$ZANIDMLinearReg, Y, X, X)
  obj$SetMCMC(sd_prior_beta_alpha, sd_prior_beta_zeta, ndpost, nskip, nthin)
  obj$RunMCMC()

  dim(obj$draws_abundance)

  mod_zanim <- zanim_linear_reg(Y = Y, X_theta = X, X_zeta = X)
  t(apply(mod_zanim$mod$draws_betas_theta, c(1, 2), mean))
  b_a

  t(apply(obj$draws_betas_alpha, c(1, 2), mean))
  b_a


  #########################
  #
  mod_zidm <- ZIDM::ZIDMbvs_R(Z = Y, X = X[, -1L], X_theta = X[, -1L],
                              iterations = 40000L, thin = 10L)
  str(mod_zidm)
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


  # Compare
  par(mfrow = c(3, 2))
  for (j in seq_len(d)) {
    plot(true_varthetas[, j], rowMeans(obj$draws_abundance[,j,]), main = j)
    abline(0, 1, col = "red")
    plot(true_varthetas[, j], rowMeans(vartheta_zidm[,j,]), main = j)
    abline(0, 1, col = "red")
  }

  apply(beta_alpha_zidm, c(1, 2), mean)
  t(apply(obj$draws_betas_alpha, c(1, 2), mean))
  b_a


  apply(beta_zeta_zidm, c(1, 2), mean)
  t(apply(obj$draws_betas_zeta, c(1, 2), mean))
  b_z

  dim(beta_alpha_zidm)


  ## Fit iid
  mod_zanidm <- zanidist::ZANIDM$new(Y = Y)
  mod_zanidm$run_mcmc(update_alpha = "da_ptn")
  dim(mod_zanidm$abundance)
  par(mfrow = c(2, 2))
  for (j in seq_len(d)) {
    plot(x = true_varthetas[, j], y = rowMeans(mod_zanidm$abundance[,j,]),
         xlab = "true", ylab = "estimate", main = "ZANIDM")
    abline(0, 1, col = "red")
  }



  compute_frob <- function(true_values, draws) {
    ndpost <- dim(draws)[3]
    diffs <- (array(true_values, dim = c(dim(true_values), ndpost)) - draws)^2
    sqrt(apply(diffs, 3, sum))
  }

  frobs_zidm <- compute_frob(true_values = true_varthetas, draws = vartheta_zidm)
  frobs_zanidm <- compute_frob(true_values = true_varthetas,
                               draws = obj$draws_abundance)
  frobs_zanidm_iid <- compute_frob(true_values = true_varthetas,
                                   draws = mod_zanidm$abundance)
  mean(frobs_zidm)
  mean(frobs_zanidm)
  mean(frobs_zanidm_iid)

  ylim <- range(c(frobs_zidm, frobs_zanidm, frobs_zanidm_iid))
  par(mfrow = c(1, 3))
  plot(frobs_zidm, type = "l", ylim = ylim, main = "ZIDM-reg")
  plot(frobs_zanidm, type = "l", ylim = ylim, main = "ZANIDM-reg")
  plot(frobs_zanidm_iid, type = "l", ylim = ylim, main = "ZANIDM (iid)")




})
