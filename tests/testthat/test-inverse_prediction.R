library(ggplot2)
library(cowplot)
devtools::load_all()
# Theme
theme_set(theme_cowplot() + background_grid() + theme(legend.position = "top"))

test_that("inverse prediction with bspline", {

  rm(list = ls())
  devtools::load_all()
  # Path
  time_id <- "2025-Jul-22-22:19:32"#format(Sys.time(), "%Y-%b-%d-%X")
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

  mod <- ZANIMBARTLogit$new(Y = Y, X_theta = X, X_zeta = X)
  mod$SetupMCMC(v0_theta = 1.5 / sqrt(2), ntrees_theta = 25L, ntrees_zeta = 25L,
                ndpost = 1000L, nskip = 100L, printevery = 50,
                update_sigma_theta = TRUE, path = path_res)
  # mod$RunMCMC()
  mod$elapsed_time

  ntrees_theta <- 25L
  ntrees_zeta <- 25L
  n_samples <- 1000L
  x_cur <- X[1L, ]
  y_cur <- Y[1L, ]
  # Compute the log-predictive density
  mod$cpp_obj$LogPredictiveDensity(y_cur, x_cur, n_samples, ntrees_theta, ntrees_zeta,
                                   path_res)

  mean_prior <- mean(X)
  sd_prior <- sd(X)
  mod$cpp_obj$InversePosteriorESS(y_cur, x_cur, mean_prior, sd_prior, n_samples,
                                  ntrees_theta, ntrees_zeta, path_res)


  ndpost = 1000L; nskip = 500L; printevery = 1L
  X_prior1 <- rnorm(n = ndpost + nskip, mean = mean_prior, sd = sd_prior)
  X_prior <- seq(mean_prior - 3 * sd_prior, mean_prior + 3 * sd_prior,
                 length.out = ndpost + nskip)
  par(mfrow = c(1, 2))
  hist(X_prior)
  hist(X_prior1)

  # Using the uniform prior
  ini <- proc.time()
  probs <- mod$cpp_obj$GetNormaliseProbsIS(y_cur, matrix(X_prior, ncol = 1),
                                           ndpost + nskip,
                                           n_samples, ntrees_theta, ntrees_zeta,
                                           path_res)
  end <- proc.time() - ini
  hist(log(probs))
  x_is_wo <- sample(X_prior, size = ndpost, replace = FALSE, prob = probs)
  x_is_w <- sample(X_prior, size = ndpost, replace = TRUE, prob = probs)

  x11();par(mfrow = c(1, 2))
  hist(x_is_wo, breaks = 40, xlim = range(c(x_is_wo, x_cur)), main = "Without replace")
  abline(v = x_cur, col = "red")

  hist(x_is_w, breaks = 40, xlim = range(c(x_is_w, x_cur)), main = "With replace")
  abline(v = x_cur, col = "red")


  ini <- proc.time()
  probs1 <- mod$cpp_obj$GetNormaliseProbsIS(y_cur, matrix(X_prior1, ncol = 1),
                                           ndpost + nskip,
                                           n_samples, ntrees_theta, ntrees_zeta,
                                           path_res)
  end <- proc.time() - ini
  hist(log(probs1))
  x_is_wo1 <- sample(X_prior1, size = ndpost, replace = FALSE, prob = probs1)
  x_is_w1 <- sample(X_prior1, size = ndpost, replace = TRUE, prob = probs1)

  x11();par(mfrow = c(1, 2))
  hist(x_is_wo1, breaks = 40, xlim = range(c(x_is_wo1, x_cur)), main = "Without replace")
  abline(v = x_cur, col = "red")

  hist(x_is_w1, breaks = 40, xlim = range(c(x_is_w1, x_cur)), main = "With replace")
  abline(v = x_cur, col = "red")

  out = mod$cpp_obj$SampleInversePosterior(y_cur, x_cur, mean_prior, sd_prior,
                                           ndpost, nskip, printevery, n_samples,
                                           ntrees_theta, ntrees_zeta, path_res)
  dim(out)
  hist(out[1L, ])
  abline(v = x_cur)
  coda::effectiveSize(coda::as.mcmc(out[1, ]))

  # Run in parallel over the samples
  n_test <- 10L
  x_true <- X[sample.int(n = nrow(X), size = 10, replace = FALSE), ]
  list_out <- parallel::mclapply(X = seq_len(n_test), FUN = function(i) {
    mod$cpp_obj$SampleInversePosterior(y_cur, x_true[i], mean_prior, sd_prior,
                                       ndpost, nskip, printevery, n_samples,
                                       ntrees_theta, ntrees_zeta, path_res)
  }, mc.cores = 4L)
  x11()
  par(mfrow = c(3, 3))
  for (i in seq_len(n_test - 1)) {
    xlim = range(list_out[[i]])
    xlim = range(c(xlim, x_true[i]))
    hist(list_out[[i]], xlim = xlim); abline(v = x_true[i], col = "red")
  }

  # Using the IS within the MCMC samples
  # n_samples <- 2001L
  # X_prior <- matrix(rnorm(n_samples, mean(X), sd(X)), ncol = 1L)
  # ntrees_theta <- 25L
  # ntrees_zeta <- 25L
  # mod$cpp_obj$LogPredictiveDensitySeq(Y[1L, ], X_prior, n_samples, ntrees_theta,
  #                                     ntrees_zeta, path_res)

  # Using the ESS within the MCMC samples
  ntrees_theta <- 25L
  ntrees_zeta <- 25L
  x_post = mod$cpp_obj$SampleInversePosteriorSeq(Y[2L, ], X[2L, ], mean(X), sd(X),
                                                 1000L, ntrees_theta, ntrees_zeta,
                                                 path_res)

  unlink(x = path_res, recursive = TRUE)
})
