test_that("test out of sample predictions in ZANIM-LN-BART", {
  rm(list = ls())
  devtools::load_all()

  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/zanim_ln", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate data
  d <- 4L
  n_sample <- 600L

  set.seed(1212)
  n_trials <- sample(seq.int(100L, 500L), n_sample, replace = TRUE)
  X <- as.matrix(seq(-1, 1, length.out = n_sample))
  eta_theta <- cbind(5*cos(pi*X), 1.5*sin(2*pi*X), 2*(X^3), -2*(X^2))
  eta_zeta <- matrix(nrow = n_sample, ncol = d)
  intercept <- rep(1.5, d) # seq(1.5, 2.5, length.out = d)
  eta_zeta <- cbind(exp(-5.0 * X^2), X - 2*(X - 0.5)^2, -2*X + 3 * X^3, 3*X - 2 * X^3)
  eta_zeta <- t(t(eta_zeta) - intercept)
  true_zetas <- stats::pnorm(eta_zeta)
  alphas <- exp(eta_theta)
  # Generate covariance matrix
  q_factors <- 2L
  Gamma <- matrix(data = stats::runif(d * q_factors, 0, 1), nrow = d, ncol = q_factors)
  Psi <- diag(x = seq(0.32, 0.35, length.out = d), nrow = d, ncol = d)
  true_Sigma_U <- tcrossprod(Gamma) + Psi
  chol_Sigma_U <- chol(true_Sigma_U)
  U <- matrix(nrow = n_sample, ncol = d)
  for (i in seq_len(n_sample)) {
    z <- stats::rnorm(n = d)
    U[i, ] <- drop(z %*% chol_Sigma_U)
  }
  Y <- Z <- true_thetas <- true_varthetas <- matrix(nrow = n_sample, ncol = d)
  for (i in seq_len(n_sample)) {
    z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
    is_zero <- z == 0L
    while (all(is_zero)) {
      z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
      is_zero <- z == 0L
    }
    p_ij <- alphas[i, ] / sum(alphas[i, ])
    true_thetas[i, ] <- p_ij
    true_varthetas[i, ] <- z * alphas[i, ] * exp(U[i, ]) / sum(z * alphas[i, ] * exp(U[i, ]))
    if (sum(is_zero) == d - 1L) {
      Y[i, ] <- rep(0L, d)
      Y[i, !is_zero] <- n_trials[i]
    } else {
      Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials[i], prob = true_varthetas[i, ])
    }
    Z[i, ] <- z
  }

  # Split data
  idx_train <- sample(1:n_sample, size = 400L, replace = FALSE)
  Y_train <- Y[idx_train, ]
  X_train <- X[idx_train, , drop = FALSE]
  Y_test <- Y[-idx_train, ]
  X_test <- X[-idx_train, , drop = FALSE]
  n_trials_train <- rowSums(Y_train)
  n_trials_test <- rowSums(Y_test)
  theta_train <- true_thetas[idx_train, ]
  theta_test <- true_thetas[-idx_train, ]
  Y_train_rel <- sweep(Y_train, 1, n_trials_train, "/")
  Y_test_rel <- sweep(Y_test, 1, n_trials_test, "/")

  # Fitting model

  # devtools::load_all()
  zanim_ln_bart <- zanicc(Y = Y_train, X_count = X_train, X_zi = X_train,
                          model = "zanim_bart", ntrees_theta = 20, ntrees_zeta = 20,
                          ndpost = 1000, nskip = 500, save_trees = TRUE,
                          forests_dir = path_res)

  # Predict function
  pred_theta <- predict(object = zanim_ln_bart, newdata = X_train, type = "theta",
                        load = TRUE, output_dir = path_res)
  expect_equal(pred_theta, zanim_ln_bart$draws_theta)
  pred_zeta <- predict(object = zanim_ln_bart, newdata = X_train, type = "zeta",
                       load = TRUE, output_dir = path_res)
  expect_equal(pred_zeta, zanim_ln_bart$draws_zeta)

  draws_chol_Sigma_V <- load_bin_predictions(file.path(path_res, "chol_Sigma_V.bin"),
                                  n = d-1, d = d-1, m = 1000)
  expect_equal(draws_chol_Sigma_V, zanim_ln_bart$draws_chol_Sigma_V)


  x11(); par(mfrow = c(2,2))
  for (j in seq_len(d))  plot(theta_train[, j], rowMeans(pred_theta[,j,]))

  # PPD function
  ppd1 <- ppd(object = zanim_ln_bart, in_sample = TRUE)
  ppd2 <- ppd(object = zanim_ln_bart, in_sample = FALSE, draws_prob = pred_theta,
              draws_zeta = pred_zeta, draws_chol_Sigma_V = draws_chol_Sigma_V,
              n_trials = n_trials_train)
  ppd3 <- ppd(object = zanim_ln_bart, in_sample = FALSE, n_trials = n_trials_train,
              output_dir = path_res, n_pred = nrow(Y_train))
  cowplot::plot_grid(
    bayesplot::ppc_ecdf_overlay(y = Y_train_rel[, 1L], yrep = ppd1[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_train_rel[, 1L], yrep = ppd2[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_train_rel[, 1L], yrep = ppd3[,,1L])
  )
  t_dat <- shannon_entropy(Y_train_rel)
  t_ppd <- apply(ppd2, 1, shannon_entropy)
  hist(t_ppd); abline(v = t_dat)

  t_dat <- gdi(Y_train_rel)
  t_ppd <- apply(ppd1, 1, gdi)
  hist(t_ppd); abline(v = t_dat)

  t_dat <- mean(Y_train_rel == 0)
  t_ppd <- apply(ppd1, 1, function(x) mean(x == 0))
  hist(t_ppd); abline(v = t_dat)

  t_dat <- sum(cor(Y_train_rel))
  t_ppd <- apply(ppd1, 1, function(x) sum(cor(x)))
  hist(t_ppd); abline(v = t_dat)

  # Out of sample
  unlink(file.path(path_res, "theta_ij.bin"))
  unlink(file.path(path_res, "zeta_ij.bin"))
  pred2_theta <- predict(object = zanim_ln_bart, newdata = X_test, type = "theta",
                         load = TRUE, output_dir = path_res)
  rowSums(pred2_theta[,,1])
  pred2_zeta <- predict(object = zanim_ln_bart, newdata = X_test, type = "zeta",
                        load = TRUE, output_dir = path_res)
  x11()
  par(mfrow = c(2,2))
  for (j in seq_len(d))  plot(theta_test[, j], rowMeans(pred2_theta[,j,]))
  ppd2 <- ppd(object = zanim_ln_bart, in_sample = FALSE, draws_prob = pred2_theta,
              draws_chol_Sigma_V = draws_chol_Sigma_V, draws_zeta = pred2_zeta,
              n_trials = n_trials_test)
  ppd3 <- ppd(object = zanim_ln_bart, in_sample = FALSE, n_trials = n_trials_test,
              output_dir = path_res, n_pred = nrow(Y_test))
  cowplot::plot_grid(
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 1L], yrep = ppd2[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 1L], yrep = ppd3[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 2L], yrep = ppd2[,,2L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 2L], yrep = ppd3[,,2L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 3L], yrep = ppd2[,,3L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 3L], yrep = ppd3[,,3L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 4L], yrep = ppd2[,,4L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 4L], yrep = ppd3[,,4L]), ncol = 4
  )

  t_dat_test <- shannon_entropy(Y_test_rel)
  t_ppd <- apply(ppd2, 1, shannon_entropy)
  hist(t_ppd); abline(v = t_dat_test)

  t_dat_test <- gdi(Y_test_rel)
  t_ppd <- apply(ppd2, 1, gdi)
  hist(t_ppd); abline(v = t_dat_test)

  # Counts
  ppd2 <- ppd(object = zanim_ln_bart, in_sample = FALSE, relative = FALSE,
              draws_prob = pred2_theta, draws_zeta = pred2_zeta,
              draws_chol_Sigma_V = draws_chol_Sigma_V, n_trials = n_trials_test)

  t_dat_test <- zi_multinomial(Y_test)
  t_ppd <- apply(ppd2, 1, zi_multinomial)
  hist(t_ppd); abline(v = t_dat_test)

})


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
  NDPOST <- 1000L
  NSKIP <- 5000L
  NTREES_THETA <- NTREES_ZETA <- 20L
  # devtools::load_all()
  m_diag <- zanicc(Y = Y, X_count = X, X_zi = X, model = "zanim_ln_bart",
                   ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                   ndpost = NDPOST, nskip = NSKIP, keep_draws = TRUE,
                   covariance_type = "diag")
  m_wishart <- zanicc(Y = Y, X_count = X, X_zi = X, model = "zanim_ln_bart",
                   ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                   ndpost = NDPOST, nskip = NSKIP, keep_draws = TRUE,
                   covariance_type = "wishart")
  m_fa <- zanicc(Y = Y, X_count = X, X_zi = X, model = "zanim_ln_bart",
                   ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                   ndpost = NDPOST, nskip = NSKIP, keep_draws = TRUE,
                   covariance_type = "fa")

  true_Sigma_U
  apply(m_diag$draws_chol_Sigma_V, c(1, 2), function(x) mean( crossprod(x) ))
  apply(m_full$draws_chol_Sigma_V, c(1, 2), mean)
  apply(m_fm$draws_chol_Sigma_V, c(1, 2), mean)

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

  mod <- ZANIMLNBART$new(Y = Y_train, X_theta = X_train, X_zeta = X_train)
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
  draws_theta_test <- load_bin_predictions(fname = file.path(path_res, "theta_ij.bin"),
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
  draws_zeta_test <- load_bin_predictions(fname = file.path(path_res, "zeta_ij.bin"),
                               n = nrow(X_test), d = mod$d, m = NDPOST)
  # all.equal(draws_zeta_test, mod$draws_zeta)

  # Generate the PPC
  y_ppc_in <- mod$GetPosteriorPredictive()

  y_ppc_out <- array(0L, dim = c(NDPOST, nrow(Y_test), d))
  for (k in seq_len(NDPOST)) {
    if (k %% 100 == 0) cat(k, "\n")
    y_ppc_out[k,,] <- .rzanim_vec(n = nrow(Y_test), sizes = rowSums(Y_test),
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
