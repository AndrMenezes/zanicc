test_that("inverse posterior ML-BART univariate X", {


  rm(list = ls())
  devtools::load_all()

  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_local <- "./tests/testthat/inverse_posterior/one_dimension/ml_bart"
  path_res <- file.path(path_local, time_id, "results")
  forests_dir <- file.path(path_res, "forests")
  if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

  set.seed(1212)
  n_sample <- 300L
  d <- 3L
  n_trials <- 500L
  tmp <- sim_data_multinomial_bspline_curve(n = n_sample, d = d, n_trials = n_trials)

  # Split the data
  n_test <- 100L
  id_test <- sample.int(n_sample, n_test)
  Y_test <- tmp$Y[id_test, ]
  X_test <- tmp$X[id_test, , drop = FALSE]
  Y_train <- tmp$Y[id_test, ]
  X_train <- tmp$X[id_test, , drop = FALSE]

  # Fit forward model
  NDPOST <- 5000L
  NSKIP <- 5000L
  NTREES <- 200L

  if (file.exists(file.path(path_res, "mod.rds"))) {
    ml_bart <- load_model(model_dir = path_res)
  } else {
    ml_bart <- zanicc(Y = Y_train, X_count = X_train, model = "mult_bart",
                      ntrees_theta = NTREES, ndpost = NDPOST, nskip = NSKIP,
                      save_trees = TRUE, forests_dir = forests_dir)
    save_model(object = ml_bart, model_dir = path_res)
  }

  # Compute the f(x*) and generate proposal
  N_PROPOSAL <- 2000L
  proposal_parms <- list(min_x = min(X_train), max_x = max(X_train))
  compute_proposal_fx_ml_bart(object = ml_bart, proposal_parms = proposal_parms,
                              n_proposal = N_PROPOSAL, load = FALSE,
                              save = TRUE, output_dir = path_res)
  # IS
  is <- .is_mlbart(object = ml_bart, Y = Y_test, proposal_parms = proposal_parms,
                   dir_posterior_fx = path_res, n_proposal = N_PROPOSAL)
  # SIR
  sir <- .is_mlbart(object = ml_bart, Y = Y_test, proposal_parms = proposal_parms,
                    dir_posterior_fx = path_res, n_proposal = N_PROPOSAL, sir = TRUE)

  length(sir)
  dim(is)

  # eSS
  mean_prior <- 0.0
  S_prior <- diag(1.0, nrow = 1)
  ess <- .gibbs_ml_bart(Y = Y_test, mean_prior = mean_prior, S_prior = S_prior,
                        forests_dir = forests_dir, ntrees = NTREES,
                        ndpost = NDPOST, n_rep = 2L)

  # Visual comparison between the three methods
  x_proposal <- is[, -seq_len(n_test)]
  pdf(file.path(path_res, "inverse_posterior.pdf"), width = 6, height = 3)
  for (i in seq_len(n_test)) {
    x_true <- X_test[i, ]
    cat(i, "\n")
    par(mfrow = c(1, 3), mar = c(3, 3, 1, 1))
    plot(x_proposal, is[, i], type = "S", main = "IS")
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)

    xrange <- range(c(x_true, range(sir[[i]])))
    plot(density(sir[[i]]), type = "S", main = "SIR", xlim = xrange)
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)

    xrange <- range(c(x_true, range(ess[[i]])))
    plot(density(ess[[i]]), type = "S", main = "eSS", xlim = xrange)
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)
  }
  graphics.off()

})

test_that("inverse posterior ML-BART bivariate X", {


  rm(list = ls())
  devtools::load_all()

  # Path
  time_id <-  "2026-Apr-21-16:00:40" #format(Sys.time(), "%Y-%b-%d-%X")
  path_local <- "./tests/testthat/inverse_posterior/ml_bart/two_dimension"
  path_res <- file.path(path_local, time_id, "results")
  forests_dir <- file.path(path_local, time_id, "forests")
  if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

  set.seed(1212)
  d <- 3L
  n_trials <- 500L
  tmp <- sim_data_multinomial_2d(n_grid = 20, d = d, n_trials = n_trials)

  # Split the data
  n_sample <- nrow(tmp$Y)
  n_test <- 100L
  id_test <- sample.int(n_sample, n_test)
  Y_test <- tmp$Y[id_test, ]
  X_test <- tmp$X[id_test, , drop = FALSE]
  Y_train <- tmp$Y[-id_test, ]
  X_train <- tmp$X[-id_test, , drop = FALSE]

  data_sim <- tmp$df[!(tmp$df$id %in% id_test), ]
  data_sim$id <- rep(seq_len(nrow(Y_train)), each = d)

  # ggplot(data = data.frame(x1 = X[, 1], x2 = X[, 2]),
  #        aes(x = x1, y = x2)) +
  #   geom_point()

  # ggplot(data = data_sim, aes(x = x1, y = x2, col = theta, fill = theta)) +
  #   facet_wrap(~category) +
  #   geom_raster() +
  #   scale_fill_gradientn(
  #     colours = RColorBrewer::brewer.pal(11, "RdBu")
  #     # limits = c(0, 1)
  #     # , guide = guide_colourbar(barwidth = 10.5, barheight = 1)
  #   )

  # Fit forward model
  NDPOST <- 5000L
  NSKIP <- 5000L
  NTREES <- 200L

  if (file.exists(file.path(path_res, "mod.rds"))) {
    ml_bart <- load_model(model_dir = path_res)
  } else {
    ml_bart <- zanicc(Y = Y_train, X_count = X_train, model = "mult_bart",
                      ntrees_theta = NTREES, ndpost = NDPOST, nskip = NSKIP,
                      save_trees = TRUE, forests_dir = forests_dir)
    save_model(object = ml_bart, model_dir = path_res)
  }

  # Check the model fit
  data_theta <- summarise_draws_3d(x = ml_bart$draws_theta)
  data_theta <- dplyr::left_join(data_theta, data_sim[, c("id", "category", "theta")],
                                 by = c("id", "category"))
  head(data_theta)
  head(data_sim)

  library(ggplot2)
  ggplot(data_theta, aes(x = theta, y = median)) +
    facet_wrap(~category) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0)

  # Check the rconvexhull function
  x_proposal <- rconvexhull(n = 1000, X = X_train)
  plot(X_train[, 1], X_train[, 2])
  points(x_proposal[, 1], x_proposal[, 2], col = "blue")

  # Compute the f(x*) and generate proposal
  N_PROPOSAL <- 2000L
  proposal_parms <- list(X = X_train)
  compute_proposal_fx_ml_bart(object = ml_bart, proposal_parms = proposal_parms,
                              n_proposal = N_PROPOSAL, load = FALSE,
                              save = TRUE, output_dir = path_res)
  # IS
  is <- .is_mlbart(object = ml_bart, Y = Y_test, proposal_parms = proposal_parms,
                   dir_posterior_fx = path_res, n_proposal = N_PROPOSAL)
  # SIR
  sir <- .is_mlbart(object = ml_bart, Y = Y_test, proposal_parms = proposal_parms,
                    dir_posterior_fx = path_res, n_proposal = N_PROPOSAL, sir = TRUE)

  length(sir)
  dim(is)
  dim(sir[[1]])

  # eSS
  mean_prior <- rep(0.0, 2)
  S_prior <- diag(1.0, nrow = 2)
  ess <- .gibbs_ml_bart(Y = Y_test, mean_prior = mean_prior, S_prior = S_prior,
                        forests_dir = forests_dir, ntrees = NTREES,
                        ndpost = NDPOST, n_rep = 2L)
  # i=1
  # Visual comparison between the three methods
  # x_proposal <- is[, -seq_len(n_test)]
  pdf(file.path(path_res, "inverse_posterior.pdf"), width = 6, height = 3)
  for (i in seq_len(n_test)) {
    x_true <- X_test[i, ]
    cat(i, "\n")
    par(mfrow = c(1, 3), mar = c(3, 3, 1, 1))

    # plot(x_proposal, is[, i], type = "S", main = "IS")
    # points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)

    # Linear interpolation for IS
    is_interp <- interp(x_proposal[, 1], x_proposal[, 2], is[, i], linear = TRUE)

    # Compute the KDE for SIR and eSS
    dens_sir <- MASS::kde2d(sir[[i]][, 1], sir[[i]][, 2])
    dens_ess <- MASS::kde2d(ess[[i]][, 1], ess[[i]][, 2])
    # Common limits
    xrange <- range(c(x_true[1], range(dens_sir$x), range(dens_ess$x), range(is_interp$x)))
    yrange <- range(c(x_true[2], range(dens_sir$y), range(dens_ess$y), range(is_interp$x)))

    # dens_sir <- MASSExtra::kde_2d(ess[[i]][, 1], ess[[i]][, 2], kernel = "opt")
    # dens_ess <- MASSExtra::kde_2d(ess[[i]][, 1], ess[[i]][, 2], kernel = "opt")

    contour(is_interp$x, is_interp$y, is_interp$z, main = "IS", ylim = yrange,
            xlim = xrange)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])

    # SIR
    contour(dens_sir$x, dens_sir$y, dens_sir$z, main = "SIR", ylim = yrange,
            xlim = xrange)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])

    # eSS
    contour(dens_ess$x, dens_ess$y, dens_ess$z, main = "eSS", ylim = yrange,
            xlim = xrange)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])
  }
  graphics.off()

})
