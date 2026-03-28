rm(list = ls())
library(ggplot2)
library(cowplot)
devtools::load_all()
# Theme
theme_set(theme_cowplot() + background_grid() + theme(legend.position = "top"))

test_that("bspline multinomial", {
  rm(list = ls())
  devtools::load_all()

  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/multinomial", time_id, "draws")
  path_res2 <- file.path("./tests/testthat/multinomial_shared", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)
  if (!dir.exists(path_res2)) dir.create(path_res2, recursive = TRUE)

  # Simulate some data
  n_trials <- 500L
  n_sample <- 1000L
  d <- 4L
  set.seed(1212)
  sim_tmp <- sim_data_multinomial_bspline_curve(d = 4, n = n_sample,
                                                n_trials = n_trials)
  Y <- sim_tmp$Y
  X <- sim_tmp$X
  p <- ncol(X)
  data_sim <- sim_tmp$df
  # head(data_sim)
  # ggplot(data = data_sim, aes(x = x, y = theta)) +
  #   facet_wrap(~category) +
  #   geom_line()

  # Non-shared trees
  mod_nonshared <- MultinomialBART$new(Y = Y, X = X, shared_trees = FALSE)
  mod_nonshared$SetupMCMC(v0 = 3.5 / sqrt(2), ntrees = 200L, ndpost = 1000L,
                          nskip = 500L, update_sigma = TRUE, path = path_res,
                          save_trees = TRUE)
  mod_nonshared$RunMCMC()
  mod_nonshared$accept_rate
  mod_nonshared$avg_leaves
  hist(mod_nonshared$cpp_obj$sigma_mcmc)

  # Shared trees
  mod_shared <- MultinomialBART$new(Y = Y, X = X, shared_trees = TRUE)
  mod_shared$SetupMCMC(v0 = 3.5 / sqrt(2), ntrees = 200L, ndpost = 1000L,
                       nskip = 500L, update_sigma = TRUE, path = path_res2)
  mod_shared$RunMCMC()
  mod_shared$accept_rate
  mod_shared$avg_leaves
  hist(mod_shared$cpp_obj$sigma_mcmc)

  ll_nonshared <- mod_nonshared$LogPredictiveLikelihood()

  # Non-shared tree
  df_draws_ns <- .summarise_draws_3d(x = mod_nonshared$cpp_obj$draws)
  df_draws_ns <- dplyr::left_join(df_draws_ns, data_sim, by = c("id", "category"))
  p_ns <- .plot_fit_curve_3d(df_draws_ns)

  # Shared tree
  df_draws_s <- .summarise_draws_3d(x = mod_shared$cpp_obj$draws)
  df_draws_s <- dplyr::left_join(df_draws_s, data_sim, by = c("id", "category"))
  p_s <- .plot_fit_curve_3d(df_draws_s)

  plot_grid(p_ns + ggtitle("Non-shared"), p_s + ggtitle("Shared"))

  # Compute log-likelihood
  ypp_ns <- mod_nonshared$GetPosteriorPredictive()
  ypp_s <- mod_shared$GetPosteriorPredictive()

  # Compute the posterior predictive distribution
  lpl_ns <- mod_nonshared$LogPredictiveLikelihood()
  lpl_s <- mod_shared$LogPredictiveLikelihood()

  loo_ns <- loo::loo(lpl_ns)
  loo_s <- loo::loo(lpl_s)
  as.data.frame(loo::loo_compare(list(non_shared = loo_ns, shared = loo_s)))

  dim(ypp_s)
  bayesplot::ppc_ecdf_overlay(y = Y[, 1L], yrep = t(ypp_s[, 1L, ]))
  bayesplot::ppc_ecdf_overlay(y = Y[, 1L], yrep = t(ypp_ns[, 1L, ]))

})

test_that("test predict and ppd functions for multinomialBART class", {

  rm(list = ls())
  devtools::load_all()

  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/multinomial", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate some data
  n_trials <- 500L
  n_sample <- 1000L
  d <- 4L
  set.seed(1212)
  sim_tmp <- sim_data_multinomial_bspline_curve(d = 4, n = n_sample,
                                                n_trials = n_trials)

  idx_train <- sample(1:n_sample, size = 800L)
  Y_train <- sim_tmp$Y[idx_train, ]
  X_train <- sim_tmp$X[idx_train, , drop = FALSE]
  Y_test <- sim_tmp$Y[-idx_train, ]
  X_test <- sim_tmp$X[-idx_train, , drop = FALSE]
  theta_test <- sim_tmp$theta[-idx_train, , drop = FALSE]
  data_sim_train <- sim_tmp$df[idx_train, ]
  data_sim_test <- sim_tmp$df[-idx_train, ]
  Y_train_rel <- sweep(Y_train, 1, rowSums(Y_train), "/")
  Y_test_rel <- sweep(Y_test, 1, rowSums(Y_test), "/")

  ml_bart <- zanicc(Y = Y_train, X_count = X_train, model = "mult_bart",
                    ntrees_theta = 20L, ndpost = 1000L, nskip = 3000L,
                    path = path_res, keep_draws = TRUE, save_trees = TRUE)

  pred1 <- predict.MultinomialBART(ml_bart, newdata = X_train, load = TRUE,
                                   output_dir = path_res)
  expect_equal(pred1, ml_bart$draws_theta)

  ppd1 <- .ppd_multinomial_batch(n_trials = rowSums(Y_train), output_dir = path_res,
                                 n_pred = nrow(Y_train), d = d, ndpost = 1000,
                                 batch_size = 50L, relative = TRUE,
                                 printevery = 100)
  ppd2 <- ppd(object = ml_bart, in_sample = FALSE, output_dir = path_res,
              n_trials = rowSums(Y_train), n_pred = nrow(Y_train), batch_size = 50L)
  ppd3 <- ppd(object = ml_bart, in_sample = TRUE)
  cowplot::plot_grid(
    bayesplot::ppc_ecdf_overlay(y = Y_train_rel[, 1L], yrep = ppd1[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_train_rel[, 2L], yrep = ppd2[,,2L]),
    bayesplot::ppc_ecdf_overlay(y = Y_train_rel[, 2L], yrep = ppd3[,,2L])
  )

  unlink(x = file.path(path_res, "theta_ij.bin"))
  # Out of sample
  pred2 <- predict.MultinomialBART(ml_bart, newdata = X_test, load = TRUE,
                                   output_dir = path_res)
  par(mfrow = c(2,2))
  for (j in seq_len(d))  plot(theta_test[, j], rowMeans(pred2[,j,]))

  res1 <- ppd(object = ml_bart, in_sample = FALSE, output_dir = path_res,
              n_trials = rowSums(Y_test), n_pred = nrow(Y_test), batch_size = 50L)
  res2 <- ppd(object = ml_bart, in_sample = FALSE, draws_prob = pred2,
              n_trials = rowSums(Y_test), n_pred = nrow(Y_test))

  cowplot::plot_grid(
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 1L], yrep = res1[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 1L], yrep = res2[,,1L])
  )
  # zi_multinomial(Y_train)
  # apply(Y_train, 2, zi_binomial, N = rowSums(Y_train))

})


test_that("friedman trinomial", {
  rm(list = ls())
  devtools::load_all()
  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/multinomial", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate some data
  n_trials <- 100L
  n_sample <- 500L
  set.seed(1212)
  sim_tmp <- sim_data_trinomial_friedman(n = n_sample, n_trials = n_trials, p = 20L)
  Y <- sim_tmp$Y
  X <- sim_tmp$X
  p <- ncol(X)
  data_sim <- sim_tmp$df

  mbart <- MultinomialBART$new(Y = Y, X = X)
  mbart$SetupMCMC(ntrees = 100L, ndpost = 5000, nskip = 2000,
                  update_sigma = TRUE, sparse = FALSE, alpha_random = FALSE,
                  keep_draws = TRUE)
  mbart$RunMCMC()

  mean_vc <- apply(mbart$cpp_obj$varcount_mcmc, c(1, 2), mean)
  prob_vc <- apply(mbart$cpp_obj$varcount_mcmc > 0, c(1, 2), mean)

  mdart <- MultinomialBART$new(Y = Y, X = X)
  mdart$SetupMCMC(ntrees = 100L, ndpost = 5000, nskip = 2000,
                  update_sigma = TRUE, sparse = TRUE, alpha_random = TRUE,
                  keep_draws = TRUE)
  mdart$RunMCMC()

  mdart$cpp_obj$alpha_sparse
  mbart$cpp_obj$alpha_sparse

  mean_vc_d <- apply(mdart$cpp_obj$varcount_mcmc, c(1, 2), mean)
  prob_vc_d <- apply(mdart$cpp_obj$varcount_mcmc > 0, c(1, 2), mean)

  mean_vc; mean_vc_d
  prob_vc; prob_vc_d

  data_dart <- data.frame(prob = c(prob_vc_d), covariate = rep(1:ncol(X), times = 3),
                          category = rep(1:3, each = ncol(X)),
                          split_prior = "dirichlet")
  data_bart <- data.frame(prob = c(prob_vc), covariate = rep(1:ncol(X), times = 3),
                          category = rep(1:3, each = ncol(X)),
                          split_prior = "uniform")
  data_vc <- rbind(data_bart, data_dart)
  p_vc <- ggplot(data_vc, aes(x = covariate, y = prob, col = split_prior)) +
    facet_wrap(~category, ncol = 1) +
    geom_point() +
    scale_x_continuous(breaks = scales::pretty_breaks(6)) +
    scale_y_continuous(breaks = scales::pretty_breaks(6), limits = c(0, 1)) +
    labs(x = "Covariate k", y = "Prob[k in model]", col = "") +
    ggtitle("Probability of chosen covariate k for different priors on the split prob.")
  save_plot(filename = file.path(path_res, "prob_vc.png"), plot = p_vc,
            bg = "white", base_height = 7.0)

  mbart$mod$splitprobs
  mdart$mod$splitprobs

  ll_dart <- lpd_multinomial(Y = Y, draws_vartheta = mdart$draws_theta)
  loo::loo(ll_dart)


  # Delete files and folders
  unlink(x = path_res, recursive = TRUE)
})
