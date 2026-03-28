library(ggplot2)
library(cowplot)
theme_set(theme_cowplot() + background_grid() + theme(legend.position = "top"))
test_that("zanim-bart with shared trees", {

  rm(list = ls())
  devtools::load_all()
  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/zanim_shared_probit", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate data
  set.seed(1212)
  n_trials <- 100L
  n_sample <- 400L
  d <- 4L
  tmp <- sim_data_zanim_bspline_curve(n = n_sample, d = d, n_trials = n_trials,
                                      link_zeta = "probit")
  X <- tmp$X
  Y <- tmp$Y
  data_sim <- tmp$df
  colMeans(Y == 0)

  #
  mod <- ZANIMBART$new(Y = Y, X_theta = X, X_zeta = X, link_zeta = "probit",
                       shared_trees = FALSE)
  mod$SetupMCMC(v0_theta = 1.5 / sqrt(2), ntrees_theta = 10L, ntrees_zeta = 10L,
                ndpost = 2000L, nskip = 2000L, printevery = 50,
                update_sigma_theta = TRUE, path = path_res,
                sparse = c(TRUE, TRUE), alpha_random = c(TRUE, TRUE),
                alpha_sparse = c(1.0, 1.0)
                )
  mod$RunMCMC()
  mod$elapsed_time

  hist(mod$cpp_obj$sigma_mult_mcmc)

  # Average number of leaves
  mod$avg_leaves_theta
  mod$avg_leaves_zeta

  # Acceptance rate
  mod$accept_rate_theta
  mod$accept_rate_zeta


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


})
