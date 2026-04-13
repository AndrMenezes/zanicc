test_that("inverse posterior ML-BART univariate X", {


  rm(list = ls())
  devtools::load_all()

  # Path
  time_id <- "2026-Apr-13-12:03:35"#format(Sys.time(), "%Y-%b-%d-%X")
  path_local <- "./tests/testthat/inverse_posterior/ml_bart"
  forests_dir <- file.path(path_local, time_id, "forests")
  if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

  n_sample <- 200L
  d <- 3L
  n_trials <- 500L
  tmp <- sim_data_multinomial_bspline_curve(n = n_sample, d = d, n_trials = n_trials)
  Y <- tmp$Y
  X <- tmp$X

  # Fit forward model
  NDPOST <- 5000L
  # NSKIP <- 5000L
  NTREES <- 200L
  # ml_bart <- zanicc(Y = Y, X_count = X, model = "mult_bart", ntrees_theta = NTREES,
  #                   ndpost = NDPOST, nskip = NSKIP, save_trees = TRUE,
  #                   forests_dir = forests_dir)

  # data_sim <- tmp$df
  # data_theta <- summarise_draws_3d(x = ml_bart$draws_theta)
  # data_theta$x <- rep(X[, 1], times = d)
  # library(ggplot2)
  # ggplot(data_sim, aes(x = x)) +
  #   facet_wrap(~category) +
  #   geom_point(aes(y = prop)) +
  #   geom_hline(data = data.frame(x = rep(x_true, d), prop = y_ast / sum(y_ast),
  #                                category = 1:d),
  #              aes(yintercept = prop), col = "red", linetype = "dashed") +
  #   # geom_point(data = data.frame(x = rep(x_true, d), prop = y_ast / sum(y_ast),
  #   #                              category = 1:d),
  #   #            aes(x = x, y = prop), size = 4, col = "red") +
  #   geom_line(aes(y = theta), col = "black") +
  #   geom_line(data = data_theta, aes(x = x, y = median), col = "dodgerblue") +
  #   geom_ribbon(data = data_theta, aes(x = x, ymin = ci_lower, ymax = ci_upper),
  #               alpha = 0.5, fill = "dodgerblue")

  # Call the class
  devtools::load_all()
  ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
  cpp_obj <- new(ml$InversePosterior, d, NTREES, NTREES, "ml_bart", forests_dir)

  # Sample a case to find its inverse posterior
  id <- sample(n_sample, 1)
  (y_ast <- Y[id, ])
  x_true <- X[id, ]

  x_ini <- runif(1, min(X), max(X))
  res <- cpp_obj$SamplerMLBARTeSS(y_ast, as.matrix(x_ini), NDPOST, 0.0,
                                  as.matrix(1.0), 10)

  dim(res)
  plot(density(res[, 1]) )
  points(x_true, 0.001, col = "blue", pch = 4, cex = 2)
  plot(res[, 1], type = "l")

})

test_that("inverse posterior ML-BART bivariate X", {


  rm(list = ls())
  devtools::load_all()

  # Path
  time_id <- "2026-Apr-13-12:24:25"#format(Sys.time(), "%Y-%b-%d-%X")
  path_local <- "./tests/testthat/inverse_posterior/ml_bart"
  forests_dir <- file.path(path_local, time_id, "forests")
  if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

  n_sample <- 200L
  d <- 3L
  n_trials <- 500L
  tmp <- sim_data_multinomial_2d_covariates(n_grid = 20, d = d, n_trials = n_trials)
  Y <- tmp$Y
  X <- tmp$X
  data_sim <- tmp$df

  ggplot(data = data.frame(x1 = X[, 1], x2 = X[, 2]),
         aes(x = x1, y = x2)) +
    geom_point()

  # ggplot(data = data_sim, aes(x = x1, y = x2, col = theta, fill = theta)) +
  #   facet_wrap(~category) +
  #   geom_raster() +
  #   scale_fill_gradientn(
  #     colours = RColorBrewer::brewer.pal(11, "RdBu")
  #     # limits = c(0, 1)
  #     # , guide = guide_colourbar(barwidth = 10.5, barheight = 1)
  #   )

  # Fit forward model
  NDPOST <- 1000L
  NSKIP <- 1000L
  NTREES <- 200L
  ml_bart <- zanicc(Y = Y, X_count = X, model = "mult_bart", ntrees_theta = NTREES,
                    ndpost = NDPOST, nskip = NSKIP, save_trees = TRUE,
                    forests_dir = forests_dir)
  # data_sim <- tmp$df
  # data_theta <- summarise_draws_3d(x = ml_bart$draws_theta)
  # data_theta <- dplyr::left_join(data_theta, data_sim[, c("id", "category", "theta")],
  #                                by = c("id", "category"))
  # head(data_theta)
  # head(data_sim)
  #
  # library(ggplot2)
  # ggplot(data_theta, aes(x = theta, y = median)) +
  #   facet_wrap(~category) +
  #   geom_point() +
  #   geom_abline(slope = 1, intercept = 0)

  # Call the class
  devtools::load_all()
  ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
  cpp_obj <- new(ml$InversePosterior, d, NTREES, NTREES, "ml_bart", forests_dir)

  id <- sample(n_sample, 1)
  (y_ast <- Y[id, ])
  (x_true <- X[id, ])

  p <- ncol(X)
  S_prior <- cov(X)#diag(1.0, p)
  x_ini <- x_true
  res <- cpp_obj$SamplerMLBARTeSS(y_ast, x_ini, NDPOST, rep(0.0, ncol(X)), S_prior, 10)
  dens <- MASS::kde2d(res[, 1], res[, 2], n = 500)

  par(mfrow = c(1, 3))
  contour(dens$x, dens$y, dens$z, main = "S = I_2")
  abline(v = x_true[1], h = x_true[2])
  plot(density(res[, 1]) )
  points(x_true[1], 0.001, col = "blue", pch = 4, cex = 2)
  plot(density(res[, 2]) )
  points(x_true[2], 0.001, col = "blue", pch = 4, cex = 2)

  res2 <- cpp_obj$SamplerMLBARTeSS(y_ast, x_ini, NDPOST, rep(0.0, ncol(X)), S_prior)
  dens2 <- MASS::kde2d(res2[, 1], res2[, 2], n = 500)
  contour(dens2$x, dens2$y, dens2$z, main = "S = cov(res)")
  abline(v = x_true[1], h = x_true[2])

  dim(res)
  par(mfrow = c(1, 2))

  par(mfrow = c(1, 2))
  plot(res[, 1], type = "l")
  plot(res[, 2], type = "l")

  head(as.data.frame(res))

  # library(ggplot2)
  ggplot(as.data.frame(res), aes(x=V1, y=V2)) +
    geom_density2d() +
    geom_vline(xintercept = x_true[1]) +
    geom_hline(yintercept = x_true[2])
  ggplot(as.data.frame(res2), aes(x=V1, y=V2)) +
    geom_density2d() +
    geom_vline(xintercept = x_true[1]) +
    geom_hline(yintercept = x_true[2])




})
