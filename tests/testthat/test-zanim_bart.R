library(ggplot2)
library(cowplot)

test_that("bspline probit", {
  rm(list = ls())
  devtools::load_all()
  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/zanim_probit", time_id, "draws")
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
  colMeans(1 - tmp$Z)

  mod <- ZANIMBART$new(Y = Y, X_theta = X, X_zeta = X, link_zeta = "probit",
                       shared_trees = FALSE)
  mod$SetupMCMC(v0_theta = 3.5 / sqrt(2), ntrees_theta = 20L, ntrees_zeta = 20L,
                ndpost = 2000L, nskip = 200L, update_sigma_theta = TRUE,
                path = path_res, keep_draws = TRUE)#tempdir())
  mod$RunMCMC()
  mod$elapsed_time

  hist(mod$cpp_obj$sigma_mult_mcmc)

  # dim(out$draws_phi)
  par(mfrow = c(3, 3), mar = c(4, 4, 2, 1))
  for (i in sample.int(n_sample, 9)) {
    x <- mod$draws_phi[i, ]
    ess <- coda::effectiveSize(coda::mcmc(x))
    plot(x, type = "l", ylab = expression(phi[i]), main = paste0("ESS ", round(ess, 2)))
  }

  posterior_thetas <- mod$draws_theta
  dim(posterior_thetas)
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  for (j in seq_len(d)) {
    plot(tmp$theta[, j], rowMeans(posterior_thetas[, j, ]), xlab = "true", ylab = "estimate")
    abline(0, 1, col = "red")
  }

  dim(tmp$theta)
  ndpost <- dim(posterior_thetas)[3]
  true_thetas <- tmp$theta
  kl_ <- numeric(ndpost)
  for (i in seq_len(ndpost)) {
    if (i %% 100 == 0) cat(i, "\n")
    kl_[i] = mean(rowSums(true_thetas * log(true_thetas / mod$draws_theta[,,i])))
  }
  coda::effectiveSize(coda::as.mcmc(kl_))
  graphics.off()
  plot(kl_, type = "l")

  # Average number of leaves
  mod$avg_leaves_theta
  mod$avg_leaves_zeta

  # Acceptance rate
  mod$accept_rate_theta
  mod$accept_rate_zeta

  mod$ComputePredictions(X = head(X), parameter = "theta")
  mod$ComputePredictions(X = head(X), parameter = "zeta")
  d_theta <- mod$LoadPredictions(parameter = "theta", n_samples = 2000L)
  d_zeta <- mod$LoadPredictions(parameter = "zeta", n_samples = 2000L)

  dim(mod$draws_theta[1:6,,])
  dim(d_theta)
  rowSums(d_theta[,,1L])
  rowSums(mod$draws_theta[1:6,,1L])

  all.equal(d_theta, mod$draws_theta[1:6,,])
  all.equal(d_zeta, mod$draws_zeta[1:6,,])


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
    geom_line(mapping = aes(x = x, y = theta), linewidth = 0.4) +
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
