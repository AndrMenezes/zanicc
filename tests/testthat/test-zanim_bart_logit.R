library(ggplot2)
library(cowplot)
devtools::load_all()
# Theme
theme_set(theme_cowplot() + background_grid() + theme(legend.position = "top"))

test_that("bspline", {

  rm(list = ls())
  devtools::load_all()
  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/zanim_logit", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate data
  set.seed(1212)
  n_trials <- 100L
  n_sample <- 400L
  d <- 4L
  tmp <- sim_data_zanim_bspline_curve(n = n_sample, d = d, n_trials = n_trials)
  X <- tmp$X
  Y <- tmp$Y
  data_sim <- tmp$df
  colMeans(Y == 0)

  mod <- ZANIMBART$new(Y = Y, X_theta = X, X_zeta = X, link = "logit")
  mod$SetupMCMC(v0_theta = 1.5 / sqrt(2), ntrees_theta = 10L, ntrees_zeta = 10L,
                ndpost = 2000L, nskip = 2000L, printevery = 50,
                update_sigma_theta = TRUE, path = path_res)
  mod$RunMCMC()
  mod$elapsed_time

  hist(mod$cpp_obj$sigma_mult_mcmc)

  # Average number of leaves
  mod$avg_leaves_theta
  mod$avg_leaves_zeta

  # Acceptance rate
  mod$accept_rate_theta
  mod$accept_rate_zeta

  expect_type(mod$GetVarCount(parameter = "theta"), "double")
  expect_type(mod$GetVarCount(parameter = "zeta"), "double")

  X_pred <- matrix(seq(min(X), max(X), length.out = 100L))
  mod$ComputePredictions(X = X_pred, parameter = "theta")
  mod$ComputePredictions(X = X_pred, parameter = "zeta")
  expect_equal(mod$n_samples_pred, mod$ndpost)

  d_theta <- mod$LoadPredictions(parameter = "theta", n_samples = 2000L)
  d_zeta <- mod$LoadPredictions(parameter = "zeta", n_samples = 2000L)
  # expect_equal(dim(d_zeta), c(nrow(X), 4L, 500L))
  # expect_equal(dim(d_theta), c(nrow(X), 4L, 500L))
  range(d_theta)
  range(data_sim$theta)
  range(d_zeta)
  range(data_sim$zeta)

  data_theta <- .summarise_draws_3d(x = d_theta)
  data_theta$x <- rep(c(X_pred), times = 4L)
  head(data_theta)
  range(data_theta$mean)
  range(data_sim$theta)

  # Plotting
  p_theta <- ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x1, y = theta), linewidth = 0.4) +
    facet_wrap(~category, scales = "free_y") +
    geom_line(data = data_theta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = data_theta, aes(x = x, ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3)
  p_theta
  save_plot(filename = file.path(path_res, "theta.png"), plot = p_theta,
            bg = "white", base_height = 7.0)

  data_zeta <- .summarise_draws_3d(x = d_zeta)
  data_zeta$x <- rep(c(X_pred), times = 4L)
  head(data_zeta)

  p_zeta <- ggplot(data = data_sim) +
    geom_line(data = data_sim, mapping = aes(x = x1, y = zeta), linewidth = 0.4) +
    geom_line(data = data_zeta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = data_zeta, aes(x = x, ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3) +
    facet_wrap(~category, scales = "free_y")
  p_zeta
  save_plot(filename = file.path(path_res, "zeta.png"), plot = p_zeta,
            bg = "white", base_height = 7.0)

  ppc <- mod$GetPosteriorPredictive(n_trials = mod$n_trials, batch_size = 100)

  ppc_appended <- do.call(rbind, lapply(seq(d), function(i) ppc[,i,1:600]))
  g_vec <- rep(seq_len(d), each = nrow(Y))
  p <- bayesplot::ppc_ecdf_overlay_grouped(y = c(Y), yrep = t(ppc_appended),
                                           group = g_vec)
  save_plot(filename = file.path(path_res, "ppc_bspline.png"), plot = p,
            bg = "white", base_height = 7.0)

  # Compute the log-predictive density
  # ntrees_theta <- 10L
  # ntrees_zeta <- 10L
  # mod$cpp_obj$LogPredictiveDensity(Y[1L, ], X[1L, ], 10, ntrees_theta, ntrees_zeta,
  #                                  path_res)

  unlink(x = path_res, recursive = TRUE)
})

test_that("friedman", {

  rm(list = ls())
  devtools::load_all()
  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X") #"2025-Jul-01-15:44:23"#
  path_res <- file.path("./tests/testthat/zanim_logit", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate data
  set.seed(1212)
  n_trials <- 500L
  n_sample <- 500L
  tmp <- sim_data_zanim_friedman(n = n_sample, n_trials = n_trials, p_theta = 20L,
                                 p_zeta = 20L)
  X_theta <- tmp$X_theta
  X_zeta <- tmp$X_zeta
  Y <- tmp$Y
  data_sim <- tmp$df
  colMeans(Y == 0)

  mod <- ZANIMBART$new(Y = Y, X_theta = X_theta, X_zeta = X_zeta, link = "logit")
  mod$SetupMCMC(v0_theta = 1.5 / sqrt(2), ntrees_theta = 100L, ntrees_zeta = 100L,
                ndpost = 5000L, nskip = 2000L, printevery = 50, path = path_res,
                sparse = c(FALSE, FALSE))
  mod$RunMCMC()

  vc_theta_bart <- mod$GetVarCount(parameter = "theta")
  vc_theta_bart_2 <- mod$cpp_obj$GetVarCount(mod$ndpost, mod$ntrees_theta, "theta",
                                             mod$path)
  expect_equal(vc_theta_bart, vc_theta_bart_2)

  vc_zeta_bart <- mod$GetVarCount(parameter = "zeta")
  vc_zeta_bart_2 <- mod$cpp_obj$GetVarCount(mod$ndpost, mod$ntrees_zeta, "zeta",
                                            mod$path)
  expect_equal(vc_zeta_bart, vc_zeta_bart_2)

  mean_vc_theta_bart <- apply(vc_theta_bart, c(1, 2), mean)
  prob_vc_theta_bart <- apply(vc_theta_bart > 0, c(1, 2), mean)

  mean_vc_zeta_bart <- apply(vc_zeta_bart, c(1, 2), mean)
  prob_vc_zeta_bart <- apply(vc_zeta_bart > 0, c(1, 2), mean)

  # DART prior on both parameters
  path_res2 <- file.path("./tests/testthat/zanim_sharedlogit", time_id, "sparse",
                        "draws")
  if (!dir.exists(path_res2)) dir.create(path_res2, recursive = TRUE)
  mod2 <- ZANIMBART$new(Y = Y, X_theta = X_theta, X_zeta = X_zeta, link = "logit")
  mod2$SetupMCMC(v0_theta = 1.5 / sqrt(2), ntrees_theta = 100L, ntrees_zeta = 100L,
                ndpost = 5000L, nskip = 2000L, printevery = 50, path = path_res2,
                sparse = c(TRUE, TRUE) ,alpha_random = c(TRUE, TRUE),
                alpha_sparse = c(1.0, 1.0))
  mod2$RunMCMC()

  mod2$cpp_obj$alpha_sparse_zi
  mod2$cpp_obj$alpha_sparse_mult

  vc_theta_dart <- mod2$GetVarCount(parameter = "theta")
  vc_theta_dart_2 <- mod2$cpp_obj$GetVarCount(mod2$ndpost, mod2$ntrees_theta, "theta",
                                              mod2$path)
  expect_equal(vc_theta_dart, vc_theta_dart_2)

  vc_zeta_dart <- mod2$GetVarCount(parameter = "zeta")
  vc_zeta_dart_2 <- mod2$cpp_obj$GetVarCount(mod2$ndpost, mod2$ntrees_zeta, "zeta",
                                             mod2$path)
  expect_equal(vc_zeta_dart, vc_zeta_dart_2)

  mean_vc_theta_dart <- apply(vc_theta_dart, c(1, 2), mean)
  prob_vc_theta_dart <- apply(vc_theta_dart > 0, c(1, 2), mean)

  mean_vc_zeta_dart <- apply(vc_zeta_dart, c(1, 2), mean)
  prob_vc_zeta_dart <- apply(vc_zeta_dart > 0, c(1, 2), mean)

  data_dart <- data.frame(prob = c(prob_vc_theta_dart),
                          covariate = rep(1:ncol(X_theta), times = 3),
                          category = rep(1:3, each = ncol(X_theta)),
                          split_prior = "dirichlet")
  data_bart <- data.frame(prob = c(prob_vc_theta_bart),
                          covariate = rep(1:ncol(X_theta), times = 3),
                          category = rep(1:3, each = ncol(X_theta)),
                          split_prior = "uniform")
  data_vc_theta <- rbind(data_bart, data_dart)
  p_vc_theta <- ggplot(data_vc_theta, aes(x = covariate, y = prob,
                                          col = split_prior, shape = split_prior)) +
    facet_wrap(~category, ncol = 1) +
    geom_point() +
    scale_x_continuous(breaks = scales::pretty_breaks(8), limits = c(1, ncol(X_theta))) +
    scale_y_continuous(breaks = scales::pretty_breaks(6), limits = c(0, 1)) +
    labs(x = "Covariate k", y = "Prob[k in model]")
  save_plot(filename = file.path(path_res, "prob_vc_theta.png"), plot = p_vc_theta,
            bg = "white", base_height = 7.0)

  par(mfrow = c(1, 2))
  matplot(prob_vc_theta_dart, ylim = c(0, 1))
  matplot(prob_vc_theta_bart, ylim = c(0, 1))

  #
  data_dart <- data.frame(prob = c(prob_vc_zeta_dart),
                          covariate = rep(1:ncol(X_zeta), times = 3),
                          category = rep(1:3, each = ncol(X_zeta)),
                          split_prior = "dirichlet")
  data_bart <- data.frame(prob = c(prob_vc_zeta_bart),
                          covariate = rep(1:ncol(X_zeta), times = 3),
                          category = rep(1:3, each = ncol(X_zeta)),
                          split_prior = "uniform")
  data_vc_zeta <- rbind(data_bart, data_dart)
  p_vc_zeta <- ggplot(data_vc_zeta, aes(x = covariate, y = prob, col = split_prior)) +
    facet_wrap(~category, ncol = 1) +
    geom_point() +
    scale_x_continuous(breaks = scales::pretty_breaks(6)) +
    scale_y_continuous(breaks = scales::pretty_breaks(6), limits = c(0, 1)) +
    labs(x = "Covariate k", y = "Prob[k in model]")
  save_plot(filename = file.path(path_res, "prob_vc_zeta.png"), plot = p_vc_zeta,
            bg = "white", base_height = 7.0)

  # Workflow for the log-likelihood
  mod$ComputePredictions(X = X_theta, n_samples = 2000L, path_out = path_res,
                         parameter = "theta")
  mod$ComputePredictions(X = X_zeta, n_samples = 2000L, path_out = path_res,
                         parameter = "zeta")
  ll_bart <- mod$LogPredictiveLikelihood(Y = Y, n_samples = 2000L, path = path_res,
                                         n_pred = nrow(Y), ncores = 20L,
                                         logfile = "log.txt")
  # For DART
  mod2$ComputePredictions(X = X_theta, n_samples = 2000L, path_out = path_res2,
                         parameter = "theta")
  mod2$ComputePredictions(X = X_zeta, n_samples = 2000L, path_out = path_res2,
                         parameter = "zeta")
  ll_dart <- mod2$LogPredictiveLikelihood(Y = Y, n_samples = 2000L, path = path_res2,
                                          n_pred = nrow(Y), ncores = 20L,
                                          logfile = "log.txt")
  dim(ll_dart)

  ess <- function(x) coda::effectiveSize(coda::as.mcmc(x))
  apply(ll_bart, 2, ess)
  apply(ll_dart, 2, ess)

  loo_bart <- loo::loo(ll_bart)
  waic_bart <- loo::waic(ll_bart)
  loo_dart <- loo::loo(ll_dart)
  waic_dart <- loo::waic(ll_dart)
  loo::loo_compare(list(bart = loo_bart, dart = loo_dart))


  unlink(x = path_res, recursive = TRUE)

})
