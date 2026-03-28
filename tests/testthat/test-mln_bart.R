test_that("multinomial logistic normal BART", {
  devtools::load_all()
  rm(list = ls());gc()
  d <- 4L
  n_sample <- 400L
  set.seed(1212)
  n_trials <- sample(seq.int(100L, 500L), n_sample, replace = TRUE)
  X <- as.matrix(seq(-1, 1, length.out = n_sample))
  eta_theta <- cbind(5*cos(pi*X), 1.5*sin(2*pi*X), 2*(X^3), -2*(X^2))
  eta_zeta <- matrix(nrow = n_sample, ncol = d)
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
  cov(U)
  true_Sigma_U
  Y <- true_thetas <- true_varthetas <- matrix(nrow = n_sample, ncol = d)
  for (i in seq_len(n_sample)) {
    p_ij <- alphas[i, ] / sum(alphas[i, ])
    true_thetas[i, ] <- p_ij
    true_varthetas[i, ] <- alphas[i, ] * exp(U[i, ]) / sum(alphas[i, ] * exp(U[i, ]))
    Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials[i], prob = true_varthetas[i, ])
  }
  devtools::load_all()
  self <- zanicc(Y = Y, X_count = X, model = "mult_ln_bart", ndpost = 1000,
                 nskip = 1000, ntrees_theta = 50L)
  y_fitted <- self$GetPosteriorPredictive(conditional_rf = TRUE)
  y_marginal <- self$GetPosteriorPredictive(conditional_rf = FALSE)

  Y_rel <- sweep(Y, 1, rowSums(Y), FUN = "/")
  cowplot::plot_grid(
    bayesplot::ppc_ecdf_overlay(y = Y_rel[, 1L], yrep = y_fitted[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_rel[, 1L], yrep = y_marginal[,,1L])
  )
  cowplot::plot_grid(
    bayesplot::ppc_ecdf_overlay(y = Y_rel[, 2L], yrep = y_fitted[,,2L]),
    bayesplot::ppc_ecdf_overlay(y = Y_rel[, 2L], yrep = y_marginal[,,2L])
  )

  ll_cond <- self$LogPredictiveLikelihood(Y = Y, conditional_rf = TRUE)
  ll_marg <- self$LogPredictiveLikelihood(Y = Y, conditional_rf = FALSE)

  loo::loo(ll_cond)
  loo::loo(ll_marg)

  MC <- 50L
  ndpost <- 1000L
  U <- matrix(nrow = n, ncol = self$d)
  # Compute the log-likelihood
  lpl <- matrix(nrow = ndpost, ncol = n)
  for (k in seq_len(ndpost)) {
    cat(k, "\n")
    vals <- sapply(seq_len(MC), function(m) {
      # Generate random effects
      Z <- matrix(data = stats::rnorm(self$d - 1), nrow = n, ncol = self$d - 1)
      U <- Z %*% (self$draws_chol_Sigma_V[,,k] %*% self$Bt)
      # Compute the probabilities
      probs <- self$draws_theta[, ,k] * exp(U)
      probs <- sweep(probs, 1, rowSums(probs), "/")
      .dmultinomial(x = Y, prob = probs)
    }, simplify = "array")
    # log-sum-exp
    maxlog <- apply(vals, 1, max)
    lpl[k, ] <- maxlog + log(rowMeans(exp(vals - maxlog)))
  }
  mean(rowSums(lpl))
  mean(rowSums(ll_cond))
  loo::loo(ll_cond)
  loo::loo(lpl)

  MC <- 50L
  lpl2 <- matrix(nrow = ndpost, ncol = n)
  for (k in seq_len(ndpost)) {
    cat(k, "\n")
    for (i in seq_len(self$n)) {
      # Monte Carlo approximation
      sapply(X = seq_len(MC), function(m) {

      })

      lik_vals <- numeric(M)
      for (m in seq_len(M)) {
        v <- stats::rnorm(self$d - 1) %*% self$draws_chol_Sigma_V[,,k]
        u <- drop(v %*% self$Bt)
        p <- self$draws_theta[i,,k] * exp(u)
        lik_vals[m] <- stats::dmultinom(x = Y[i, ],prob = p / sum(p), log = TRUE)
      }
      # v <- stats::rnorm(self$d - 1) %*% self$draws_chol_Sigma_V[,,k]
      # u <- drop(v %*% self$Bt)
      # p <- self$draws_theta[i,,k] * exp(u)
      # lpl2[k,i] <- stats::dmultinom(x = Y[i, ], prob = p / sum(p), log = TRUE)
      maxlog <- max(lik_vals)
      lpl2[k,i] <- maxlog + log(mean(exp(lik_vals - maxlog)))
    }
  }
  mean(rowSums(lpl))
  mean(rowSums(lpl2)) #[1] -6816.427
  mean(rowSums(ll_cond))
})



test_that("predict and ppd functions", {
  devtools::load_all()
  rm(list = ls());gc()

  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/mln_bart", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate data
  d <- 4L
  n_sample <- 400L
  set.seed(1212)
  n_trials <- sample(seq.int(100L, 500L), n_sample, replace = TRUE)
  X <- as.matrix(seq(-1, 1, length.out = n_sample))
  eta_theta <- cbind(5*cos(pi*X), 1.5*sin(2*pi*X), 2*(X^3), -2*(X^2))
  eta_zeta <- matrix(nrow = n_sample, ncol = d)
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
  cov(U)
  true_Sigma_U
  Y <- true_thetas <- true_varthetas <- matrix(nrow = n_sample, ncol = d)
  for (i in seq_len(n_sample)) {
    p_ij <- alphas[i, ] / sum(alphas[i, ])
    true_thetas[i, ] <- p_ij
    true_varthetas[i, ] <- alphas[i, ] * exp(U[i, ]) / sum(alphas[i, ] * exp(U[i, ]))
    Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials[i], prob = true_varthetas[i, ])
  }
  colMeans(Y==0)
  zi_multinomial(Y)

  # Split data
  idx_train <- sample(1:n_sample, size = 300L, replace = FALSE)
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
  # devtools::load_all()
  mln_bart <- zanicc(Y = Y_train, X_count = X_train, model = "mult_ln_bart",
                     ndpost = 1000, nskip = 5000, ntrees_theta = 20L,
                     save_trees = TRUE, covariance_type = "wishart",
                     forests_dir = path_res)

  # Predict function
  pred <- predict(object = mln_bart, newdata = X_train, load = TRUE,
                  output_dir = path_res)
  expect_equal(pred, mln_bart$draws_theta)
  # file.remove(file.path(path_res, "theta_ij.bin"))
  # pred3 <- predict(object = mln_bart, newdata = head(X_train), load = TRUE,
  #                 output_dir = path_res)
  # expect_equal(pred[1:6,,], mln_bart$draws_theta[1:6,,])
  # rowSums(pred[1:6,,1])
  # rowSums(pred3[1:6,,1])

  x11()
  par(mfrow = c(2,2))
  for (j in seq_len(d))  plot(theta_train[, j], rowMeans(pred[,j,]))

  draws_chol_Sigma_V <- load_bin_predictions(file.path(path_res, "chol_Sigma_V.bin"), d - 1,
                                  d - 1, 1000)
  expect_equal(draws_chol_Sigma_V, mln_bart$draws_chol_Sigma_V)

  # PPD function
  ppd1 <- ppd(object = mln_bart, in_sample = TRUE)
  ppd2 <- ppd(object = mln_bart, in_sample = FALSE, draws_prob = pred,
              draws_chol_Sigma_V = draws_chol_Sigma_V, n_trials = n_trials_train)
  ppd3 <- ppd(object = mln_bart, in_sample = FALSE, n_trials = n_trials_train,
              output_dir = path_res, n_pred = nrow(Y_train))
  cowplot::plot_grid(
    bayesplot::ppc_ecdf_overlay(y = Y_train_rel[, 1L], yrep = ppd1[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_train_rel[, 1L], yrep = ppd2[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_train_rel[, 1L], yrep = ppd3[,,1L])
  )
  t_dat <- shannon_entropy(Y_train_rel)
  t_ppd <- apply(ppd1, 1, shannon_entropy)
  hist(t_ppd); abline(v = t_dat)

  t_dat <- gdi(Y_train_rel)
  t_ppd <- apply(ppd1, 1, gdi)
  hist(t_ppd); abline(v = t_dat)

  # Out of sample
  unlink(file.path(path_res, "theta_ij.bin"))
  pred2 <- predict.MultinomialLNBART(object = mln_bart, newdata = X_test, load = TRUE,
                                     output_dir = path_res)
  x11()
  par(mfrow = c(2,2))
  for (j in seq_len(d))  plot(theta_test[, j], rowMeans(pred2[,j,]))
  ppd2 <- ppd(object = mln_bart, in_sample = FALSE, draws_prob = pred2,
              draws_chol_Sigma_V = draws_chol_Sigma_V, n_trials = n_trials_test)
  ppd3 <- ppd(object = mln_bart, in_sample = FALSE, n_trials = n_trials_test,
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
  ppd2 <- ppd(object = mln_bart, relative = FALSE, in_sample = FALSE,
              draws_prob = pred2, draws_chol_Sigma_V = draws_chol_Sigma_V,
              n_trials = n_trials_test)

  t_dat_test <- zi_multinomial(Y_test)
  t_ppd <- apply(ppd2, 1, zi_multinomial)
  hist(t_ppd); abline(v = t_dat_test)

})



