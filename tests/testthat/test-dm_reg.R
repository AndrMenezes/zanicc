test_that("DM regression model", {
  rm(list = ls()); gc()
  devtools::load_all()
  # set.seed(12)
  d <- 4L
  p <- 2L
  n_sample <- 500L
  n_trials <- sample(seq.int(1000, 5000), size = n_sample, replace = TRUE)
  X <- cbind(1, matrix(stats::rnorm(n_sample * p), ncol = p))
  X <- X[, 1L:2L, drop = FALSE]
  p <- ncol(X)
  true_betas <- matrix(stats::rnorm(d * p), ncol = p , nrow = d)
  true_alphas <- true_thetas <- true_varthetas <- matrix(nrow = n_sample, ncol = d)
  for (j in seq_len(d)) true_alphas[, j] <- exp(X %*% true_betas[j, ])
  range(true_alphas)
  true_thetas <- sweep(true_alphas, 1, rowSums(true_alphas), "/")

  Y <- matrix(nrow = n_sample, ncol = d)
  for (i in seq_len(n_sample)) {
    g <- stats::rgamma(n = d, shape = true_alphas[i, ], rate = 1.0)
    true_varthetas[i, ] <- g / sum(g)
    Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials[i],
                               prob = true_varthetas[i, ])[, 1L]
  }
  Y_rel <- sweep(Y, 1, n_trials, "/")
  head(Y)
  plot(Y[, 1L]~X[, 2L])
  # devtools::load_all()
  mod <- DMRegression$new(Y, X)
  mod$SetupMCMC(ndpost = 2000L, nskip = 8000L)
  mod$RunMCMC()
  mod$elapsed_time
  betas_mean <- t(mod$PosterioMeanCoef()) #apply(mod$draws_betas, c(1,2), mean)#
  cbind(truth = true_betas[, 1L], mean = betas_mean[, 1L])
  cbind(truth = true_betas[, 2L], mean = betas_mean[, 2L])

  par(mfrow = c(2, 2))
  plot(mod$draws_betas[1,1,], type = "l")
  plot(mod$draws_betas[1,2,], type = "l")
  plot(mod$draws_betas[1,3,], type = "l")
  plot(mod$draws_betas[1,4,], type = "l")

  plot(mod$draws_betas[2,1,], type = "l")
  plot(mod$draws_betas[2,2,], type = "l")
  plot(mod$draws_betas[2,3,], type = "l")
  plot(mod$draws_betas[2,4,], type = "l")


  yppc <- mod$GetPosteriorPredictive(conditional = FALSE, relative = FALSE)
  bayesplot::ppc_ecdf_overlay(y = Y[, 1L], yrep = yppc[,,1L])

  yppc2 <- mod$GetPosteriorPredictive(conditional = FALSE, relative = TRUE)

  bayesplot::ppc_ecdf_overlay(y = Y_rel[, 2L], yrep = yppc2[,,2L])

  bayesplot::ppc_ecdf_overlay(y = Y_rel[, 1L], yrep = yppc2[,,1L])

  ll <- mod$LogPredictiveLikelihood(Y = Y)
  dim(ll)
  loo::loo(ll)

  par(mfrow = c(2, 2))
  for (j in seq_len(d)) {
    plot(true_varthetas[, j], rowMeans(mod$draws_abundance[,j,]))
  }
  for (j in seq_len(d)) {
    plot(true_alphas[, j], rowMeans(mod$draws_alpha[,j,]))
  }





  ml <- Rcpp::Module(module = "dm_linear_reg", PACKAGE = "zanicc")
  mod <- new(ml$DMLinearReg, Y, X)
  # S <- matrix(c(0.004698629, -0.001036410, -0.001036410,  0.002189974), ncol=2)
  S <- array(0, dim = c(p, p, d))
  for (j in seq_len(d)) diag(S[,,j]) <- 1.0
  # chol(S)
  mod$SetMCMC(S, 10000L, 50000L, 1L)
  mod$RunMCMC()

  betas_mean <- t(apply(mod$draws_betas, c(1, 2), mean))
  cbind(truth = true_betas[, 1L], mean = betas_mean[, 1L])
  cbind(truth = true_betas[, 2L], mean = betas_mean[, 2L])

  # Trace plot
  par(mfrow = c(2, 2))
  plot(mod$draws_betas[1,1,], type = "l")
  plot(mod$draws_betas[1,2,], type = "l")
  plot(mod$draws_betas[1,3,], type = "l")
  plot(mod$draws_betas[1,4,], type = "l")

  par(mfrow = c(2, 2))
  plot(mod$draws_betas[2,1,], type = "l")
  plot(mod$draws_betas[2,2,], type = "l")
  plot(mod$draws_betas[2,3,], type = "l")
  plot(mod$draws_betas[2,4,], type = "l")

  for (j in seq_len(d)) {
    plot(true_varthetas[, j], rowMeans(mod$draws_abundance[,j,]))
  }
  for (j in seq_len(d)) {
    plot(true_alphas[, j], rowMeans(mod$draws_alphas[,j,]))
  }

  # ### Run again with different covariance matrix
  #
  # S_mean <- simplify2array(list(cov(t(mod$draws_betas[,1,])),
  #                               cov(t(mod$draws_betas[,2,])),
  #                               cov(t(mod$draws_betas[,3,])),
  #                               cov(t(mod$draws_betas[,4,]))))
  # # S_opt <- S_mean * 2.5^2 / p
  # ml <- Rcpp::Module(module = "dm_linear_reg", PACKAGE = "zanicc")
  # mod2 <- new(ml$DMLinearReg, Y, X)
  # mod2$SetMCMC(S_mean, 5000L, 5000L, 1L)
  # mod2$RunMCMC()
  # betas_mean <- t(apply(mod2$draws_betas, c(1, 2), mean))
  # cbind(truth = true_betas[, 1L], mean = betas_mean[, 1L])
  # cbind(truth = true_betas[, 2L], mean = betas_mean[, 2L])
  #
  # # Trace plot
  # par(mfrow = c(2, 2))
  # plot(mod2$draws_betas[1,1,], type = "l")
  # plot(mod2$draws_betas[1,2,], type = "l")
  # plot(mod2$draws_betas[1,3,], type = "l")
  # plot(mod2$draws_betas[1,4,], type = "l")
  #
  # par(mfrow = c(2, 2))
  # plot(mod2$draws_betas[2,1,], type = "l")
  # plot(mod2$draws_betas[2,2,], type = "l")
  # plot(mod2$draws_betas[2,3,], type = "l")
  # plot(mod2$draws_betas[2,4,], type = "l")
  # for (j in seq_len(d)) {
  #   plot(true_varthetas[, j], rowMeans(mod2$draws_abundance[,j,]))
  # }

})

test_that("test in and out sample predictions of DM-regression ", {
  rm(list = ls()); gc()
  devtools::load_all()

  # Path to save
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/dm_reg", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate data
  set.seed(12)
  d <- 4L
  p <- 2L
  n_sample <- 500L
  n_trials <- sample(seq.int(1000, 5000), size = n_sample, replace = TRUE)
  X <- cbind(1, matrix(stats::rnorm(n_sample * p), ncol = p))
  X <- X[, 1L:2L, drop = FALSE]
  p <- ncol(X)
  true_betas <- matrix(stats::rnorm(d * p), ncol = p , nrow = d)
  true_alphas <- true_thetas <- true_varthetas <- matrix(nrow = n_sample, ncol = d)
  for (j in seq_len(d)) true_alphas[, j] <- exp(X %*% true_betas[j, ])
  true_thetas <- sweep(true_alphas, 1, rowSums(true_alphas), "/")
  Y <- matrix(nrow = n_sample, ncol = d)
  for (i in seq_len(n_sample)) {
    g <- stats::rgamma(n = d, shape = true_alphas[i, ], rate = 1.0)
    true_varthetas[i, ] <- g / sum(g)
    Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials[i],
                               prob = true_varthetas[i, ])[, 1L]
  }
  Y_rel <- sweep(Y, 1, n_trials, "/")
  NDPOST <- 5000L
  NSKIP <- 5000L
  # devtools::load_all()
  mod <- zanicc(Y, X, model = "dm_reg", ndpost = NDPOST, nskip = NSKIP,
                save_draws = TRUE, dir_draws = path_res)

  # Save model
  save_model(mod, model_dir = path_res)
  rm(mod); gc()
  # Load model
  mod <- load_model(model_dir = path_res)

  # Load draws
  draws_betas_loaded <- load_bin_coefficients(file.path(path_res, "draws_betas.bin"),
                                              p = p,d = d,m = NDPOST)
  expect_equal(draws_betas_loaded, mod$draws_betas)
  # Compute linear predictions
  pred_alpha <- predict(mod, newdata = X, type = "alpha")
  #expect_equal(pred_alpha, mod$draws_alpha)
  pred_theta <- predict(mod, newdata = X, type = "theta")
  #expect_equal(pred_theta, mod$draws_theta)
  # PPD
  yrep1 <- ppd(object = mod, conditional = TRUE)
  yrep2 <- ppd(object = mod, in_sample = FALSE, draws_alpha = pred_alpha,
               n_trials = rowSums(Y))
  cowplot::plot_grid(
    bayesplot::ppc_ecdf_overlay(y = Y_rel[, 1L], yrep = yrep1[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_rel[, 1L], yrep = yrep2[,,1L])
  )

})





