library(ggplot2)
library(cowplot)
devtools::load_all()
# Theme
theme_set(theme_cowplot() + background_grid())

test_that("bspline", {
  rm(list = ls())

  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/probit", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate some data
  n_sample <- 1000L
  set.seed(1212)
  data_sim <- sim_data_binary_bspline_curve(n = n_sample, link = stats::pnorm)
  y <- data_sim$y
  X <- as.matrix(data_sim$x)
  p <- ncol(X)
  head(data_sim)
  # ggplot(data = data_sim, aes(x = x, y = theta)) + geom_line()

  out <- probit_bart(y = y, X = X, path = path_res, ntrees = 100L,
                     ndpost = 2000L, nskip = 1000L)
  dim(out$mod$draws)
  rowMeans(out$mod$varcount_mcmc)

  out$mod$avg_leaves / (out$ndpost)
  out$mod$avg_depth / (out$ndpost)

  # Check the posterior point estimates
  df_draws <- .summarise_draws(x = out$draws)
  df_draws <- dplyr::left_join(df_draws, data_sim, by = "id")
  .plot_fit_curve(df_draws)
  expect_lt(abs(mean(df_draws$mean - df_draws$theta)), 1e-1)

  # Check the predict function
  pred <- predict.probit_bart(object = out, X = head(X), ndpost = 10L)
  expect_equal(stats::pnorm(pred), out$draws[1:6, 1:10])

  # Delete files and folders
  unlink(x = path_res, recursive = TRUE)

})

test_that("friedman", {
  rm(list = ls())

  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/probit", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate some data, only the first 3 is important!
  n_sample <- 500L
  p <- 10L
  set.seed(1212)
  tmp <- sim_data_binary_friedman(n = n_sample, p = p, link = stats::pnorm)
  y <- tmp$y
  table(y)
  X <- tmp$X
  theta_truth <- tmp$theta
  quantile(theta_truth)

  # BART
  p_bart <- probit_bart(y = y, X = X, path = tempdir(), ntrees = 100L,
                        ndpost = 2000L, nskip = 1000L, sparse = FALSE)

  # DART fixed concentration parameter at 1.0
  p_dart_1 <- probit_bart(y = y, X = X, path = tempdir(), ntrees = 100L,
                          ndpost = 5000L, nskip = 2000L, sparse = TRUE,
                          alpha_sparse = 1.0, alpha_random = FALSE)
  p_dart_1$mod$alpha_sparse

  cbind(bart = rowMeans(p_bart$mod$varcount_mcmc),
        dart_1 = rowMeans(p_dart_1$mod$varcount_mcmc))

  cbind(bart = rowMeans(p_bart$mod$varcount_mcmc > 0),
        dart_1 = rowMeans(p_dart_1$mod$varcount_mcmc > 0))

  cbind(bart = p_bart$mod$splitprobs,
        dart = p_dart_1$mod$splitprobs)

  mean_prob <- rowMeans(p_bart$draws)
  yhat <- 1L * (mean_prob > 0.5)
  table(y, yhat)
  mean(y == yhat)
  # Remove files
  unlink(x = path_res, recursive = TRUE)
})


