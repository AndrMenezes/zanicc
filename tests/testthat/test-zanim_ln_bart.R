test_that("test ZANIM-LN-BART with different covariance structures", {
  library(ggplot2)
  # Simulate data
  rm(list = ls())
  devtools::load_all()
  n_sample <- 100L
  n_trials <- 200L
  d <- 3L
  set.seed(6669)
  # Simulate a covariance matrix
  list_sim <- sim_data_zanim_ln_bspline_curve(n = n_sample, d = d, n_trials = n_trials,
                                              link_zeta = "probit", rho = 0.8,
                                              covariance = "toeplitz")
  Y <- list_sim$Y
  X <- list_sim$X
  true_Sigma_U <- list_sim$Sigma_U
  true_thetas <- list_sim$theta
  any(rowSums(Y) == 0)

  data_sim <- list_sim$df
  ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = theta), linewidth = 0.4) +
    facet_wrap(~category, scales = "free_y")

  apply(true_thetas, 2, quantile)

  # Fitting models
  NDPOST <- 500L
  NSKIP <- 500L
  NTREES_THETA <- NTREES_ZETA <- 20L
  # devtools::load_all()
  # ini <- proc.time()
  m_diag <- ZANIMLogNormalBART$new(Y = Y, X_theta = X, X_zeta = X)
  m_diag$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                   ndpost = NDPOST, nskip = NSKIP, covariance_type = 0L,
                   update_sigma_theta = TRUE, keep_draws = TRUE)
  m_diag$RunMCMC()
  # yrep <- m_diag$GetPosteriorPredictive()
  # ll <- m_diag$LogPredictiveLikelihood(Y = Y)

  # devtools::load_all()
  m_full <- ZANIMLogNormalBART$new(Y = Y, X_theta = X, X_zeta = X)
  m_full$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                   ndpost = NDPOST, nskip = NSKIP, covariance_type = 1L,
                   update_sigma_theta = TRUE, keep_draws = TRUE)
  m_full$RunMCMC()

  # devtools::load_all()
  m_fm <- ZANIMLogNormalBART$new(Y = Y, X_theta = X, X_zeta = X)
  m_fm$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                 ndpost = NDPOST, nskip = NSKIP, covariance_type = 2L,
                 q_factors = 2L, update_sigma_theta = TRUE, keep_draws = TRUE)
  m_fm$RunMCMC()

  true_Sigma_U
  apply(m_diag$cpp_obj$draws_Sigma_U, c(1, 2), mean)
  apply(m_full$cpp_obj$draws_Sigma_U, c(1, 2), mean)
  apply(m_fm$cpp_obj$draws_Sigma_U, c(1, 2), mean)

  cov2cor(apply(m_full$cpp_obj$draws_Sigma_U, c(1, 2), mean))
  cov2cor(apply(m_diag$cpp_obj$draws_Sigma_U, c(1, 2), mean))
  cov2cor(apply(m_fm$cpp_obj$draws_Sigma_U, c(1, 2), mean))
  cov2cor(true_Sigma_U)

  dim(m_diag$draws_theta)
  par(mfrow = c(2, 3), mar = c(4, 4, 2, 1))
  for (j in seq_len(d)) {
    # plot(true_thetas[, j], rowMeans(m_diag$draws_theta[, j, ]), xlab = "true", ylab = "estimate",
    #      main = paste0("j = ", j))
    # abline(0, 1, col = "red")
    plot(true_thetas[, j], rowMeans(m_full$cpp_obj$draws_theta[, j, ]), xlab = "true", ylab = "estimate",
         main = paste0("j = ", j))
    abline(0, 1, col = "red")
    plot(true_thetas[, j], rowMeans(m_fm$cpp_obj$draws_theta[, j, ]), xlab = "true", ylab = "estimate",
         main = paste0("j = ", j))
    abline(0, 1, col = "red")
  }

  # Trace plot of \phi_i
  par(mfrow = c(3, 3), mar = c(4, 4, 2, 1))
  for (i in sample.int(n_sample, 9)) {
    x_diag <- m_diag$cpp_obj$draws_phi[i, ]
    x_full <- m_full$cpp_obj$draws_phi[i, ]
    ylim <- range(c(x_diag, x_full))
    ess_diag <- coda::effectiveSize(coda::mcmc(x_diag))
    ess_full <- coda::effectiveSize(coda::mcmc(x_full))
    plot(x_full, type = "l", ylab = expression(phi[i]), ylim = ylim,
         main = paste0("ESS diag: ", round(ess_diag, 2), " ESS full: ", round(ess_full, 2)))
    lines(x_diag, col = "red")
  }

  ess_diag <- apply(m_diag$draws_phi, 1, function(x) coda::effectiveSize(coda::as.mcmc(x)))
  ess_full <- apply(m_full$draws_phi, 1, function(x) coda::effectiveSize(coda::as.mcmc(x)))
  Y[which(ess_full == 0), ]
  Y[which(ess_diag == 0), ]
  ess_full <- ess_full[-which(ess_full == 0)]
  ess_diag <- ess_diag[-which(ess_diag == 0)]
  cbind(diag = quantile(ess_diag), full = quantile(ess_full))



  dim(true_thetas)
  dim(m_diag$draws_theta)

  compute_kl_iter <- function(true_thetas, draws_theta) {
    ndpost <- dim(draws_theta)[3]
    log_ratio <- log(array(true_thetas, dim = c(dim(true_thetas), ndpost)) / draws_theta)
    kl_terms <- array(true_thetas, dim = c(dim(true_thetas), ndpost)) * log_ratio
    colMeans(apply(kl_terms, 3, rowSums))
  }
  kl_diag <- compute_kl_iter(true_thetas = true_thetas,
                              draws_theta = m_diag$cpp_obj$draws_theta)
  kl_full <- compute_kl_iter(true_thetas = true_thetas,
                              draws_theta = m_full$cpp_obj$draws_theta)
  kl_fm <- compute_kl_iter(true_thetas = true_thetas,
                           draws_theta = m_fm$cpp_obj$draws_theta)

  ess_kl_diag <- coda::effectiveSize(coda::as.mcmc(kl_diag))
  ess_kl_fm <- coda::effectiveSize(coda::as.mcmc(kl_fm))
  ess_kl_full <-  coda::effectiveSize(coda::as.mcmc(kl_full))

  graphics.off()
  plot(kl_full, type = "l", ylab = "KL", xlab = "Iteration",
       main = paste0("ESS fm: ", round(ess_kl_fm, 2), " and ESS full: ", round(ess_kl_full, 2)))
  # lines(kl_diag, type = "l", col = "red")
  lines(kl_fm, type = "l", col = "red")

  mean(kl_full)
  mean(kl_diag)

  data_zeta <- .summarise_draws_3d(x = stats::pnorm(m_fm$cpp_obj$draws_zeta))
  data_zeta$x <- rep(c(X), times = 3L)
  head(data_zeta)

  p_zeta <- ggplot(data = data_sim) +
    geom_line(data = data_sim, mapping = aes(x = x, y = zeta), linewidth = 0.4) +
    geom_line(data = data_zeta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = data_zeta, aes(x = x, ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3) +
    facet_wrap(~category, scales = "free_y")
  p_zeta

})

test_that("test out of sample predictions in ZANIM-LN-BART", {
  # Simulate data
  rm(list = ls())
  devtools::load_all()
  n_sample <- 800L
  n_trials <- 200L
  d <- 3L
  set.seed(6669)
  # Simulate a covariance matrix
  list_sim <- sim_data_zanim_ln_bspline_curve(n = n_sample, d = d, n_trials = n_trials,
                                              link_zeta = "probit", rho = 0.8,
                                              covariance = "toeplitz")

  # Simulate from ZANIM
  # list_sim <- sim_data_zanim_bspline_curve(n = n_sample, d = d, n_trials = n_trials,
  #                                          link_zeta = "probit")

  Y <- list_sim$Y
  X <- list_sim$X
  true_thetas <- list_sim$theta

  # Split data
  n_train <- 500L
  Y_train <- Y[1:n_train, ]
  X_train <- X[1:n_train, , drop = FALSE]
  Y_test <- Y_train#Y[-(1:n_train), ]
  X_test <- X_train#X[-(1:n_train), , drop = FALSE]
  true_thetas_train <- true_thetas[1:n_train, ]
  true_thetas_test <- true_thetas[-(1:n_train),]

  true_Sigma_U <- list_sim$Sigma_U
  true_U <- list_sim$U
  any(rowSums(Y) == 0)

  # Fitting model
  NDPOST <- 1000L
  NSKIP <- 1000L
  NTREES_THETA <- NTREES_ZETA <- 10L
  devtools::load_all()
  # ini <- proc.time()
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/zanim_ln", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  mod <- ZANIMLogNormalBART$new(Y = Y_train, X_theta = X_train, X_zeta = X_train)
  mod$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                ndpost = NDPOST, nskip = NSKIP, covariance_type = 1L,
                update_sigma_theta = TRUE, keep_draws = TRUE, path = path_res)
  mod$RunMCMC()
  list.files(mod$path)

  # Import the cholesky of V
  # con <- file(file.path(path_res, "chol_Sigma_V.bin"), "rb")
  # chol_sigma_V <- array(readBin(con, what = "double", n = (d - 1) * (d - 1) * NDPOST),
  #                       dim = c(d-1, d-1, NDPOST))
  # close(con)

  # X_test <- X[1:4, , drop = FALSE]
  # n_samples <- NDPOST
  mod$cpp_obj$ComputePredictProb(X_test, NDPOST, mod$ntrees_theta,
                                                    path_res, path_res,
                                                    as.integer(TRUE))
  draws_theta_test <- .load_bin(fname = file.path(path_res, "theta_ij.bin"),
                                n = nrow(X_test), d = mod$d, m = NDPOST)

  # dim(draws_lambdas_test)
  # U <- matrix(nrow = nrow(Y_test), ncol = d)
  # # chol_Sigma_U <- chol(true_Sigma_U)
  # draws_theta_test <- array(dim = dim(draws_lambdas_test))
  # B <- qr.Q(qr(stats::contr.sum(d)))
  # M <- 500L
  # for (k in seq_len(NDPOST)) {
  #   # Add the random effect
  #   lambda_ij <- draws_lambdas_test[,,k]
  #   # el <- exp(lambda_ij)
  #   # theta_ij <- sweep(el, MARGIN = 1, STATS = rowSums(el), FUN = "/")
  #
  #   ## Repeat this process m times (monte carlo integration??)
  #   theta_ij <- matrix(0.0, nrow = nrow(Y_test), ncol = d)
  #   for (h in seq_len(M)) {
  #     # Generate from random effect
  #     for (i in seq_len(nrow(Y_test))) {
  #       V = drop(stats::rnorm(n = d-1) %*% chol_sigma_V[,,k])
  #       U[i, ] <- drop(B %*% V)
  #       #drop(stats::rnorm(n = d) %*% chol_sigma_V[,,k])
  #     }
  #     # Add the random effect and normalise
  #     el <- exp(lambda_ij + U)
  #     theta_ij <- theta_ij + sweep(el, MARGIN = 1, STATS = rowSums(el), FUN = "/")
  #   }
  #
  #   ###
  #   draws_theta_test[,,k] <- theta_ij / M
  # }


  par(mfrow = c(2, 3))
  for (j in seq_len(d)) {
    plot(true_thetas_train[,j], rowMeans(mod$draws_theta[,j,]), main = j,
         xlab = "true", ylab = "posterior mean")
    abline(0, 1, col = "red")
  }
  for (j in seq_len(d)) {
    plot(true_thetas_test[,j], rowMeans(draws_theta_test[,j,]), main = j,
         xlab = "true", ylab = "posterior mean")
    abline(0, 1, col = "red")
  }



  ## Check the posterior-predictive

  mod$cpp_obj$ComputePredictProbZero(X_test, NDPOST, mod$ntrees_zeta, path_res,
                                     path_res, as.integer(TRUE))
  draws_zeta_test <- .load_bin(fname = file.path(path_res, "zeta_ij.bin"),
                               n = nrow(X_test), d = mod$d, m = NDPOST)
  # all.equal(draws_zeta_test, mod$draws_zeta)

  # Generate the PPC
  y_ppc_in <- mod$GetPosteriorPredictive()

  y_ppc_out <- array(0L, dim = c(NDPOST, nrow(Y_test), d))
  for (k in seq_len(NDPOST)) {
    if (k %% 100 == 0) cat(k, "\n")
    y_ppc_out[k,,] <- rzanim_vec(n = nrow(Y_test), sizes = rowSums(Y_test),
                                 probs = draws_theta_test[,,k],
                                 zetas = draws_zeta_test[,,k], d = d)
  }
  # all.equal(y_ppc_in, y_ppc_out)

  apply(y_ppc_in, 3, mean)
  apply(y_ppc_out, 3, mean)


  list_plots <- list()
  for (j in seq_len(d)) {
    list_plots[[j]] <- cowplot::plot_grid(
      bayesplot::ppc_ecdf_overlay(y = Y_train[, j], yrep = y_ppc_in[,,j]) + ggtitle("In sample"),
      bayesplot::ppc_ecdf_overlay(y = Y_test[, j], yrep = y_ppc_out[,,j]) + ggtitle("Out sample")
    )
  }
  x11();cowplot::plot_grid(plotlist = list_plots, ncol = 1)

  list_plots <- list()
  for (j in seq_len(d)) {
    list_plots[[j]] <- cowplot::plot_grid(
      bayesplot::ppc_stat(y = Y_train[, j], yrep = y_ppc_in[,,j], stat = "sd") + ggtitle("In sample"),
      bayesplot::ppc_stat(y = Y_test[, j], yrep = y_ppc_out[,,j], stat = "sd") + ggtitle("Out sample")
    )
  }
  x11();cowplot::plot_grid(plotlist = list_plots, ncol = 1)

  list_plots

})
