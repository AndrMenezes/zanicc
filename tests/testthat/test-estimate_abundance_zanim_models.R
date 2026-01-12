
test_that("test abundance estimates of zanim and zanim-ln models", {
  rm(list = ls())
  devtools::load_all()
  # Path
  # time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  # path_res <- file.path("./tests/testthat/zanim_probit", time_id, "draws")
  # if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate data
  set.seed(1212)
  n_trials <- 100L
  n_sample <- 400L
  d <- 4L
  # tmp <- sim_data_zanim_bspline_curve(n = n_sample, d = d, n_trials = n_trials,
  #                                     link_zeta = "probit")
  tmp <- sim_data_zanim_ln_bspline_curve(n = n_sample, d = d, n_trials = n_trials,
                                         link_zeta = "probit", covariance = "exp",
                                         lg = 0.4)
  X <- tmp$X
  Y <- tmp$Y
  Z <- tmp$Z
  true_thetas <- tmp$theta
  true_abundance <- tmp$abundance

  # Fitting ZANIM-BART
  mod_zanim <- ZANIMBART$new(Y = Y, X_theta = X, X_zeta = X, link_zeta = "probit",
                       shared_trees = FALSE)
  mod_zanim$SetupMCMC(v0_theta = 1.5 / sqrt(2), ntrees_theta = 20L, ntrees_zeta = 20L,
                      ndpost = 2000L, nskip = 200L, update_sigma_theta = TRUE,
                      path = tempdir(), keep_draws = TRUE)
  mod_zanim$RunMCMC()

  any(mod_zanim$draws_abundance == 0)
  dim(mod_zanim$draws_abundance)
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  for (j in seq_len(d)) {
    plot(true_abundance[, j], rowMeans(mod_zanim$draws_abundance[, j, ]),
         xlab = "true", ylab = "estimate")
    abline(0, 1, col = "red")
  }

  # Fitting ZANIM-BART
  mod_zanim_ln <- ZANIMLogNormalBART$new(Y = Y, X_theta = X, X_zeta = X)
  mod_zanim_ln$SetupMCMC(v0_theta = 1.5 / sqrt(2), ntrees_theta = 20L,
                         ntrees_zeta = 20L, ndpost = 2000L, nskip = 200L,
                         update_sigma_theta = TRUE, path = tempdir(),
                         covariance_type = 1L,
                         keep_draws = TRUE)
  mod_zanim_ln$RunMCMC()

  expect_true(any(mod_zanim_ln$draws_abundance == 0))
  dim(mod_zanim_ln$draws_abundance)

  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  for (j in seq_len(d)) {
    plot(true_abundance[, j], rowMeans(mod_zanim_ln$draws_abundance[, j, ]),
         xlab = "true", ylab = "estimate")
    abline(0, 1, col = "red")
  }




})
