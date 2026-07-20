library(ggplot2)
library(cowplot)
devtools::load_all()
# Theme
theme_set(theme_cowplot() + background_grid() + theme(legend.position = "top"))



test_that("bspline", {
  rm(list = ls())

  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  forests_dir <- file.path("./tests/testthat/sharedlogit", time_id, "forests")
  if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

  # Wrapper function
  shared_logit_bart <- function(y, X, v0 = 3.5 / sqrt(2), ntrees = 20L,
                                ndpost = 1000L, nskip = 1000L, printevery = 10L,
                                numcut = 100L, power = 2.0, base = 0.95,
                                proposals_prob = c(0.25, 0.25, 0.50),
                                splitprobs = rep(1 / ncol(X), ncol(X)), sparse = FALSE,
                                sparse_parms = c(ncol(X), 0.5, 1.0), alpha_sparse = 1.0,
                                alpha_random = FALSE, xinfo = matrix(),
                                forests_dir = tempdir()) {

    # Call the C++ class in R
    ml <- Rcpp::Module(module = "shared_logit_bart", PACKAGE = "zanicc")
    obj <- new(ml$SharedLogitBART, y, X)
    # Set up MCMC
    obj$SetMCMC(v0, ntrees, ndpost, nskip, printevery, numcut, power, base,
                proposals_prob, splitprobs, as.integer(sparse), sparse_parms,
                alpha_sparse, as.integer(alpha_random), xinfo, forests_dir)
    # Run MCMC
    obj$RunMCMC()
    # Save information
    lt <- list(call = match.call(), mod = obj, forests_dir = forests_dir, ntrees = ntrees,
               ndpost = ndpost, draws = obj$draws, n = length(y))
    class(lt) <- "shared_logit_bart"
    lt
  }

  # Simulate some data
  n_sample <- 1000L
  set.seed(1212)
  data_sim <- zanicc::sim_data_binary_bspline_curve(n = n_sample)
  y <- data_sim$y
  X <- as.matrix(data_sim$x)
  p <- ncol(X)
  head(data_sim)
  # ggplot(data = data_sim, aes(x = x, y = theta)) + geom_line()

  out <- shared_logit_bart(y = y, X = X, forests_dir = path_res)
  dim(out$mod$draws)
  rowMeans(out$mod$varcount_mcmc)

  # Check the posterior point estimates
  df_draws <- zanicc::summarise_draws(x = out$draws)
  df_draws <- dplyr::left_join(df_draws, data_sim, by = "id")
  zanicc:::.plot_fit_curve(df_draws)
  expect_lt(abs(mean(df_draws$mean - df_draws$theta)), 1e-1)

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
