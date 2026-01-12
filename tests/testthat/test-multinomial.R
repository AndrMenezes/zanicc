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
                          sparse = TRUE, alpha_random = TRUE)
  mod_nonshared$RunMCMC()
  mod_nonshared$accept_rate
  mod_nonshared$avg_leaves
  hist(mod_nonshared$cpp_obj$sigma_mcmc)

  # Shared trees
  mod_shared <- MultinomialBART$new(Y = Y, X = X, shared_trees = TRUE)
  mod_shared$SetupMCMC(v0 = 3.5 / sqrt(2), ntrees = 50L, ndpost = 1000L,
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

  # cowplot::plot_grid(p1, p2)

  # en <- df_draws |>
  #   dplyr::group_by(id) |>
  #   dplyr::summarise(entropy = sum(theta * log(mean))) |>
  #   dplyr::pull(entropy)
  # en_s <- df_draws_s |>
  #   dplyr::group_by(id) |>
  #   dplyr::summarise(entropy = sum(theta * log(mean))) |>
  #   dplyr::pull(entropy)
  # -mean(en)
  # -mean(en_s)


#
#   out_1 <- multinomial_bart(Y = Y, X = X, path = path_res, ntrees = 200L,
#                            update_sigma = FALSE)
#   out_2 <- multinomial_bart(Y = Y, X = X, path = path_res, ntrees = 200L,
#                            update_sigma = TRUE)
#   hist(out_2$mod$sigma_mcmc)
#   abline(v = 3.5/sqrt(200), col = "red")
#   quantile(out_2$mod$sigma_mcmc)
#
#   # Check the posterior point estimates
#   df_draws <- .summarise_draws_3d(x = out_1$draws)
#   df_draws <- dplyr::left_join(df_draws, data_sim, by = c("id", "category"))
#   .plot_fit_curve_3d(df_draws)
#   expect_lt(mean(df_draws$mean - df_draws$theta), 1e-10)
#
#   # Check the predict function
#   pred <- predict.multinomial_bart(object = out, X = head(X), ndpost = 10L)
#   expect_equal(pred, out$draws[1:6, , 1:10])
#
#   # Delete files and folders
#   unlink(x = path_res, recursive = TRUE)


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

  mbart <- multinomial_bart(Y = Y, X = X, path = path_res, ntrees = 100L,
                            ndpost = 5000L, nskip = 2000L, printevery = 100L)
  mean_vc <- apply(mbart$mod$varcount_mcmc, c(1, 2), mean)
  prob_vc <- apply(mbart$mod$varcount_mcmc > 0, c(1, 2), mean)

  mdart <- multinomial_bart(Y = Y, X = X, path = path_res, ntrees = 100L,
                            ndpost = 5000L, nskip = 2000L, sparse = TRUE,
                            alpha_sparse = 1.0, alpha_random = TRUE)
  mdart$mod$alpha_sparse
  mbart$mod$alpha_sparse

  mean_vc_d <- apply(mdart$mod$varcount_mcmc, c(1, 2), mean)
  prob_vc_d <- apply(mdart$mod$varcount_mcmc > 0, c(1, 2), mean)

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

  # Delete files and folders
  unlink(x = path_res, recursive = TRUE)
})
