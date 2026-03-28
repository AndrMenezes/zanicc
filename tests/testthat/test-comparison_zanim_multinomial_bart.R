rm(list = ls())
library(ggplot2)
library(cowplot)
devtools::load_all()
# Theme
theme_set(theme_cowplot() + background_grid() + theme(legend.position = "top"))

test_that("friedman", {
  rm(list = ls())
  devtools::load_all()

  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X") #"2025-Jul-01-15:44:23"#
  path_res_1 <- file.path("./tests/testthat/zanim_logit", time_id, "draws")
  if (!dir.exists(path_res_1)) dir.create(path_res_1, recursive = TRUE)
  path_res_2 <- file.path("./tests/testthat/multinomial", time_id, "draws")
  if (!dir.exists(path_res_2)) dir.create(path_res_2, recursive = TRUE)

  # Simulate data
  set.seed(1212)
  n_trials <- 500L
  n_sample <- 500L
  tmp <- sim_data_zanim_friedman(n = n_sample, n_trials = n_trials,
                                 p_theta = 20L, p_zeta = 20L)
  X_theta <- tmp$X_theta
  X_zeta <- tmp$X_zeta
  X <- cbind(X_theta, X_zeta)
  Y <- tmp$Y
  data_sim <- tmp$df
  colMeans(Y == 0)

  NDPOST <- 1000L
  NSKIP <- 1000L
  NTREES <- 100L

  # Fiting ZANIM-BART
  mod_zanim <- ZANIMBART$new(Y = Y, X_theta = X, X_zeta = X,
                             link = "probit", shared_trees = FALSE)
  mod_zanim$SetupMCMC(v0_theta = 1.5 / sqrt(2), ntrees_theta = NTREES,
                      ntrees_zeta = NTREES, ndpost = NDPOST, nskip = NSKIP,
                      printevery = 50, path = path_res_1, update_sigma_theta = TRUE,
                      sparse = c(TRUE, TRUE), alpha_random = c(TRUE, TRUE),
                      alpha_sparse = c(1.0, 1.0))
  mod_zanim$RunMCMC()
  mod_zanim$cpp_obj$alpha_sparse_mult
  mod_zanim$cpp_obj$alpha_sparse_zi
  mod_zanim$cpp_obj$splitprobs_mult

  # Computing the in-predictions
  mod_zanim$ComputePredictions(X = X, n_samples = NDPOST, path_out = path_res_1,
                         parameter = "theta")
  mod_zanim$ComputePredictions(X = X, n_samples = NDPOST, path_out = path_res_1,
                         parameter = "zeta")

  # Compute log-likelihood
  ll_zanim <- mod_zanim$LogPredictiveLikelihood(Y = Y, n_samples = NDPOST,
                                          path = path_res_1, n_pred = nrow(Y),
                                          ncores = 15L, logfile = "log.txt")

  # Checking convergence of latent variable \phi_i
  par(mfrow = c(3, 3), mar = c(4, 4, 2, 1))
  for (i in sample.int(n_sample, 9)) {
    x <- mod_zanim$cpp_obj$draws_phi[i, ]
    ess <- coda::effectiveSize(coda::mcmc(x))
    plot(x, type = "l", ylab = expression(phi[i]), main = paste0("ESS ", round(ess, 2)))
  }

  posterior_thetas <- mod_zanim$LoadPredictions(parameter = "theta")
  par(mfrow = c(1, 3), mar = c(4, 4, 2, 1))
  for (j in seq_len(3)) {
    plot(tmp$theta[, j], rowMeans(posterior_thetas[, j, ]), xlab = "true", ylab = "estimate")
    abline(0, 1, col = "red")
  }
  posterior_zetas <- mod_zanim$LoadPredictions(parameter = "zeta")
  par(mfrow = c(1, 3), mar = c(4, 4, 2, 1))
  for (j in seq_len(3)) {
    plot(tmp$zeta, rowMeans(posterior_zetas[, j, ]), xlab = "true", ylab = "estimate")
    abline(0, 1, col = "red")
  }

  # Fitting Multinomial-BART
  mod_mult <- MultinomialBART$new(Y = Y, X = X, shared_trees = FALSE)
  mod_mult$SetupMCMC(v0 = 1.5 / sqrt(2), ntrees = NTREES, ndpost = NDPOST,
                     nskip = NSKIP, update_sigma = FALSE, path = path_res_2,
                     sparse = TRUE, alpha_random = TRUE, alpha_sparse = 1.0)
  mod_mult$RunMCMC()

  par(mfrow = c(3, 3), mar = c(4, 4, 2, 1))
  for (i in sample.int(n_sample, 9)) {
    x <- mod_mult$cpp_obj$draws_phi[i, ]
    ess <- coda::effectiveSize(coda::mcmc(x))
    plot(x, type = "l", ylab = expression(phi[i]), main = paste0("ESS ", round(ess, 2)))
  }

  dim(mod_mult$cpp_obj$draws)
  par(mfrow = c(1, 3), mar = c(4, 4, 2, 1))
  for (j in seq_len(3)) {
    plot(tmp$theta[, j], rowMeans(mod_mult$cpp_obj$draws[, j, ]), xlab = "true", ylab = "estimate")
    abline(0, 1, col = "red")
  }




  ll_mult <- mod_mult$LogPredictiveLikelihood()
  dim(ll_mult)
  loo_zanim <- loo::loo(ll_zanim)
  loo_mult <- loo::loo(ll_mult)
  as.data.frame(loo::loo_compare(list(zanim = loo_zanim, multinomial = loo_mult)))

  # Check
  vc_theta_dart <- mod_zanim$GetVarCount(parameter = "theta")
  vc_zeta_dart <- mod_zanim$GetVarCount(parameter = "zeta")

  mean_vc_theta_dart <- apply(vc_theta_dart, c(1, 2), mean)
  prob_vc_theta_dart <- apply(vc_theta_dart > 0, c(1, 2), mean)

  mean_vc_zeta_dart <- apply(vc_zeta_dart, c(1, 2), mean)
  prob_vc_zeta_dart <- apply(vc_zeta_dart > 0, c(1, 2), mean)

  data_dart_theta <- data.frame(prob = c(prob_vc_theta_dart),
                          covariate = rep(1:ncol(X), times = 3),
                          category = rep(1:3, each = ncol(X)),
                          split_prior = "dirichlet")
  p_vc_theta <- ggplot(data_dart_theta, aes(x = covariate, y = prob)) +
    facet_wrap(~category, ncol = 1) +
    geom_point() +
    scale_x_continuous(breaks = scales::pretty_breaks(8), limits = c(1, ncol(X))) +
    scale_y_continuous(breaks = scales::pretty_breaks(6), limits = c(0, 1)) +
    labs(x = "Covariate k", y = "Prob[k in model]")
  save_plot(filename = file.path(path_res_1, "prob_vc_theta.png"), plot = p_vc_theta,
            bg = "white", base_height = 7.0)

  data_dart_zeta <- data.frame(prob = c(prob_vc_zeta_dart),
                                covariate = rep(1:ncol(X), times = 3),
                                category = rep(1:3, each = ncol(X)),
                                split_prior = "dirichlet")
  p_vc_zeta <- ggplot(data_dart_zeta, aes(x = covariate, y = prob)) +
    facet_wrap(~category, ncol = 1) +
    geom_point() +
    scale_x_continuous(breaks = scales::pretty_breaks(6)) +
    scale_y_continuous(breaks = scales::pretty_breaks(6), limits = c(0, 1)) +
    labs(x = "Covariate k", y = "Prob[k in model]")
  save_plot(filename = file.path(path_res_1, "prob_vc_zeta.png"), plot = p_vc_zeta,
            bg = "white", base_height = 7.0)

  unlink(x = path_res, recursive = TRUE)

})
