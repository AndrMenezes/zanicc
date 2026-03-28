library(ggplot2)
library(cowplot)
devtools::load_all()
# Theme
theme_set(theme_cowplot() + background_grid() + theme(legend.position = "top"))

test_that("bspline", {
  rm(list = ls())

  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/sharedlogit", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate some data
  n_sample <- 1000L
  set.seed(1212)
  data_sim <- sim_data_binary_bspline_curve(n = n_sample)
  y <- data_sim$y
  X <- as.matrix(data_sim$x)
  p <- ncol(X)
  head(data_sim)
  # ggplot(data = data_sim, aes(x = x, y = theta)) + geom_line()

  out <- shared_logit_bart(y = y, X = X, path = path_res)
  dim(out$mod$draws)
  rowMeans(out$mod$varcount_mcmc)

  # Check the posterior point estimates
  df_draws <- .summarise_draws(x = out$draws)
  df_draws <- dplyr::left_join(df_draws, data_sim, by = "id")
  .plot_fit_curve(df_draws)
  expect_lt(abs(mean(df_draws$mean - df_draws$theta)), 1e-1)

  # Check the predict function
  pred <- predict.shared_logit_bart(object = out, X = head(X), ndpost = 10L)
  expect_equal(pred, out$draws[1:6, 1:10])

  # Delete files and folders
  unlink(x = path_res, recursive = TRUE)
})

test_that("friedman", {
  rm(list = ls())

  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/sharedlogit", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate some data
  n_sample <- 500L
  p <- 10L
  set.seed(1212)
  tmp <- sim_data_binary_friedman(n = n_sample, p = p)
  y <- tmp$y
  table(y)
  X <- tmp$X
  theta_truth <- tmp$theta
  quantile(theta_truth)

  # BART
  slogit_bart <- shared_logit_bart(y = y, X = X, path = path_res, ntrees = 100L,
                                   ndpost = 5000L, nskip = 2000L, sparse = FALSE)

  # DART fixed concentration parameter at 1.0
  slogit_dart_1 <- shared_logit_bart(y = y, X = X, path = path_res, ntrees = 100L,
                                     ndpost = 5000L, nskip = 2000L, sparse = TRUE,
                                     alpha_sparse = 1.0, alpha_random = TRUE)
  slogit_dart_1$mod$alpha_sparse
  cbind(bart = rowMeans(slogit_bart$mod$varcount_mcmc),
        dart_1 = rowMeans(slogit_dart_1$mod$varcount_mcmc))

  cbind(bart = rowMeans(slogit_bart$mod$varcount_mcmc > 0),
        dart_1 = rowMeans(slogit_dart_1$mod$varcount_mcmc > 0))

  cbind(bart = slogit_bart$mod$splitprobs,
        dart = slogit_dart_1$mod$splitprobs)

  mean_prob <- rowMeans(slogit_dart_1$draws)
  yhat <- 1L * (mean_prob > 0.5)
  table(y, yhat)
  mean(y == yhat)

  #
  prob_bart <- rowMeans(slogit_bart$mod$varcount_mcmc > 0)
  prob_dart <- rowMeans(slogit_dart_1$mod$varcount_mcmc > 0)
  data_dart <- data.frame(prob = c(prob_dart), covariate = 1:ncol(X),
                          split_prior = "dirichlet")
  data_bart <- data.frame(prob = c(prob_bart), covariate = 1:ncol(X),
                          split_prior = "uniform")
  data_vc <- rbind(data_bart, data_dart)
  p_vc <- ggplot(data_vc, aes(x = covariate, y = prob, col = split_prior)) +
    geom_point() +
    scale_x_continuous(breaks = scales::pretty_breaks(6)) +
    scale_y_continuous(breaks = scales::pretty_breaks(6), limits = c(0, 1)) +
    labs(x = "Covariate k", y = "Prob[k in model]", col = "") +
    ggtitle("Probability of chosen covariate k for different priors on the split prob.")
  save_plot(filename = file.path(path_res, "prob_vc.png"), plot = p_vc,
            bg = "white", base_height = 7.0)

  # Remove files
  unlink(x = path_res, recursive = TRUE)
})
