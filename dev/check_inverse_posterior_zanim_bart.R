test_that("ZANIM-BART one-dimension", {

  rm(list = ls())
  devtools::load_all()

  # Path
  time_id <- "2026-Apr-23-11:54:17"#format(Sys.time(), "%Y-%b-%d-%X")
  path_local <- "./tests/testthat/inverse_posterior/zanim_bart/one_dimension"
  path_res <- file.path(path_local, time_id, "results")
  forests_dir <- file.path(path_res, "forests")
  if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

  set.seed(1212)
  n_sample <- 300L
  d <- 3L
  n_trials <- 500L
  tmp <- sim_data_zanim_bspline_curve(n = n_sample, d = d, n_trials = n_trials,
                                      link_zeta = "probit")

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
    zanim_bart <- load_model(model_dir = path_res)
  } else {
    zanim_bart <- zanicc(Y = Y_train, X_count = X_train, X_zi = X_train,
                         model = "zanim_bart", ntrees_theta = NTREES,
                         ntrees_zeta = NTREES, ndpost = NDPOST, nskip = NSKIP,
                         save_trees = TRUE, forests_dir = forests_dir)
    save_model(object = zanim_bart, model_dir = path_res)
  }

  # Compute the f(x*) and generate proposal (do this once)
  N_PROPOSAL <- 2000L
  proposal_parms <- list(min_x = min(X_train), max_x = max(X_train))
  if (!file.exists(file.path(path_res, "theta_ij.bin"))) {
    compute_proposal_fx_zanim_bart(object = zanim_bart, proposal_parms = proposal_parms,
                                   n_proposal = N_PROPOSAL, load = FALSE,
                                   save = TRUE, output_dir = path_res,
                                   conditional = TRUE)
  }

  # posterior_fx_theta <- load_bin_predictions(fname = file.path(path_res, "theta_ij.bin"),
  #                                            n = N_PROPOSAL, d = d, m = NDPOST)
  # posterior_fx_zeta <- load_bin_predictions(fname = file.path(path_res, "zeta_ij.bin"),
  #                                           n = N_PROPOSAL, d = d, m = NDPOST)

  # IS
  is <- .is_zanimbart(object = zanim_bart, Y = Y_test, proposal_parms = proposal_parms,
                      dir_posterior_fx = path_res, n_proposal = N_PROPOSAL)
  is_cond <- .is_zanimbart(object = zanim_bart, Y = Y_test, proposal_parms = proposal_parms,
                           dir_posterior_fx = path_res, n_proposal = N_PROPOSAL,
                           conditional = TRUE)
  # SIR
  sir <- .is_zanimbart(object = zanim_bart, Y = Y_test, proposal_parms = proposal_parms,
                       dir_posterior_fx = path_res, n_proposal = N_PROPOSAL, sir = TRUE)
  sir_cond <- .is_zanimbart(object = zanim_bart, Y = Y_test, proposal_parms = proposal_parms,
                            dir_posterior_fx = path_res, n_proposal = N_PROPOSAL, sir = TRUE,
                            conditional = TRUE)
  # eSS
  mean_prior = rep(0.0, 1); S_prior = diag(1.0, nrow = 1); n_rep = 1
  ess <- .gibbs_sampler(Y = Y_test, mean_prior = mean_prior, S_prior = S_prior,
                        forests_dir = forests_dir, ntrees = NTREES,
                        ndpost = NDPOST, n_rep = 4L, forward_model = "zanim_bart")
  ess_cond <- .gibbs_sampler(Y = Y_test, mean_prior = mean_prior, S_prior = S_prior,
                             forests_dir = forests_dir, ntrees = NTREES,
                             ndpost = NDPOST, n_rep = 4L,
                             forward_model = "zanim_bart", conditional = TRUE)

  x_proposal <- is[, -seq_len(n_test)]
  # par(mfrow = c(1, 2))
  # plot(x_proposal, is[, 1], type = "h", xlim = c(-0.6, -0.4))
  # plot(x_proposal, is_cond[, 1], type = "h", xlim = c(-0.6, -0.4))
  #
  # Visual comparison between the three methods
  pdf(file.path(path_res, "inverse_posterior_new.pdf"), width = 6, height = 3)
  for (i in seq_len(n_test)) {
    x_true <- X_test[i, ]
    cat(i, "\n")
    par(mfrow = c(3, 2), mar = c(3, 3, 1, 1))
    plot(x_proposal, is[, i], type = "h",
         main = paste0("IS, y_i = (", paste0(Y_test[i, ], collapse = ","), ")"))
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)

    plot(x_proposal, is_cond[, i], type = "h", main = "IS (conditional)")
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)

    xrange <- c(-1,1)#range(c(x_true, range(sir[[i]])))
    plot(density(sir[[i]]), main = "SIR", xlim = xrange)
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)
    # xrange <- range(c(x_true, range(sir_cond[[i]])))
    plot(density(sir_cond[[i]]), main = "SIR (conditional)", xlim = xrange)
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)

    xrange <- range(c(1, -1, range(ess_cond[,1,i]), range(ess[,1,i])))
    plot(density(ess[,1,i]), main = "eSS", xlim = xrange)
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)
    plot(density(ess_cond[,1,i]),  main = "eSS (conditional)", xlim = xrange)
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)
  }
  graphics.off()

  # Some checks in the internal functions

  mean_prior = rep(0.0, 1); S_prior = diag(1.0, nrow = 1); n_rep = 1
  Y_test <- Y_train[3,,drop=FALSE]
  n <- nrow(Y_test)
  p <- length(mean_prior)
  X_ini <- matrix(nrow = n, ncol = p)
  cS <- chol(S_prior)
  for (i in seq_len(n)) X_ini[i, ] <- stats::rnorm(p) %*% cS + mean_prior
  #
  # devtools::load_all()
  ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
  cpp_obj <- new(ml$InversePosterior, ncol(Y_test), NTREES, NTREES, "zanim_bart",
                 forests_dir)

  out1=cpp_obj$SamplerZANIMBARTeSS(Y_test, as.matrix(X_ini),
                                   as.integer(NDPOST), mean_prior, S_prior, n_rep, 0L)
  out2=cpp_obj$SamplerZANIMBARTeSS(Y_test, as.matrix(X_ini),
                                   as.integer(NDPOST), mean_prior, S_prior, n_rep, 1L)

  par(mfrow = c(2, 2))
  plot(density(out1[1:NDPOST]), main = "marginal", xlim = c(-1, 1))
  points(x=X_train[1],y=0.001, col = "blue", pch = 19)
  plot(density(out2[1:NDPOST]), main = "conditional", xlim = c(-1, 1))
  points(x=X_train[1],y=0.001, col = "blue", pch = 19)
  plot(density(out1[(NDPOST+1):(NDPOST*2)]), main = "marginal", xlim = c(-1, 1))
  points(x=X_train[2],y=0.001, col = "blue", pch = 19)
  plot(density(out2[(NDPOST+1):(NDPOST*2)]), main = "conditional", xlim = c(-1, 1))
  points(x=X_train[2],y=0.001, col = "blue", pch = 19)

  length(out)
  table(out == 0)


})

test_that("ZANIM-BART two-dimension", {

  rm(list = ls())
  devtools::load_all()

  # Path
  region <- "convexhull" # square
  path_local <- "./tests/testthat/inverse_posterior/zanim_bart/two_dimension"
  path_res <- file.path(path_local, region, "results")
  forests_dir <- file.path(path_res, "forests")
  if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

  set.seed(1212)
  n_sample <- 300L
  d <- 3L
  n_trials <- 500L
  X_aux <- pollen_data$X[, c("gdd5", "mtco")]
  X_aux <- scale(X_aux)
  tmp <- sim_data_zanicc_2d(n_grid = 20, d = d, n_trials = n_trials,
                            region = region, X_aux = X_aux, random_effects = FALSE,
                            structural_zero = TRUE)

  # Split the data
  n_test <- 100L
  id_test <- sample.int(n_sample, n_test)
  Y_test <- tmp$Y[id_test, ]
  X_test <- tmp$X[id_test, , drop = FALSE]
  Y_train <- tmp$Y[-id_test, ]
  X_train <- tmp$X[-id_test, , drop = FALSE]

  data_sim <- tmp$df[!(tmp$df$id %in% id_test), ]
  data_sim$id <- rep(seq_len(nrow(Y_train)), each = d)
  head(data_sim)
  library(ggplot2)
  p1 <- ggplot(data = data_sim, aes(x = x1, y = x2)) +
    facet_wrap(~category) +
    # geom_tile(aes(fill = theta), width = 0.1, height = 0.1) +
    geom_point(aes(col = theta), shape = 4) +
    scale_color_viridis_c(option = "C", limits = c(0, max(data_sim$theta)))
  # scale_color_gradientn(
  #   colours = RColorBrewer::brewer.pal(11, "RdBu")
  #   # ,limits = c(0, 1)
  #   # , guide = guide_colourbar(barwidth = 10.5, barheight = 1)
  # )
  p2 <- ggplot(data = data_sim, aes(x = x1, y = prop)) +
    facet_wrap(~category) +
    geom_point()
  p_grid <- cowplot::plot_grid(p1, p2, ncol = 1)
  p_grid
  cowplot::save_plot(filename = file.path(path_res, "data.png"), plot = p_grid,
                     bg = "white", base_height = 8)

  # Fit forward model
  NDPOST <- 5000L
  NSKIP <- 5000L
  NTREES <- 100L
  if (file.exists(file.path(path_res, "mod.rds"))) {
    zanim_bart <- load_model(model_dir = path_res)
  } else {
    zanim_bart <- zanicc(Y = Y_train, X_count = X_train, X_zi = X_train,
                         model = "zanim_bart", ntrees_theta = NTREES,
                         ntrees_zeta = NTREES, ndpost = NDPOST, nskip = NSKIP,
                         save_trees = TRUE, forests_dir = forests_dir)
    save_model(object = zanim_bart, model_dir = path_res)
  }

  # Compute the f(x*) and generate proposal (do this once)
  N_PROPOSAL <- 2000L
  proposal_parms <- list(X = X_train)
  if (!file.exists(file.path(path_res, "theta_ij.bin"))) {
    compute_proposal_fx_zanimbart(object = zanim_bart, proposal_parms = proposal_parms,
                                  n_proposal = N_PROPOSAL, load = FALSE,
                                  save = TRUE, output_dir = path_res,
                                  conditional = TRUE)
  }

  # posterior_fx_theta <- load_bin_predictions(fname = file.path(path_res, "theta_ij.bin"),
  #                                            n = N_PROPOSAL, d = d, m = NDPOST)
  # posterior_fx_zeta <- load_bin_predictions(fname = file.path(path_res, "zeta_ij.bin"),
  #                                           n = N_PROPOSAL, d = d, m = NDPOST)

  # IS
  is <- .is_zanimbart(object = zanim_bart, Y = Y_test, proposal_parms = proposal_parms,
                      dir_posterior_fx = path_res, n_proposal = N_PROPOSAL)
  is_cond <- .is_zanimbart(object = zanim_bart, Y = Y_test, proposal_parms = proposal_parms,
                           dir_posterior_fx = path_res, n_proposal = N_PROPOSAL,
                           conditional = TRUE)
  # SIR
  sir <- .is_zanimbart(object = zanim_bart, Y = Y_test, proposal_parms = proposal_parms,
                       dir_posterior_fx = path_res, n_proposal = N_PROPOSAL, sir = TRUE)
  sir_cond <- .is_zanimbart(object = zanim_bart, Y = Y_test, proposal_parms = proposal_parms,
                            dir_posterior_fx = path_res, n_proposal = N_PROPOSAL, sir = TRUE,
                            conditional = TRUE)
  # eSS
  mean_prior = rep(0.0, 2); S_prior = diag(1.0, nrow = 2);
  ess <- .gibbs_sampler(Y = Y_test, mean_prior = mean_prior, S_prior = S_prior,
                        forests_dir = forests_dir, ntrees = NTREES,
                        ndpost = NDPOST, n_rep = 4L, forward_model = "zanim_bart")
  ess_cond <- .gibbs_sampler(Y = Y_test, mean_prior = mean_prior, S_prior = S_prior,
                             forests_dir = forests_dir, ntrees = NTREES,
                             ndpost = NDPOST, n_rep = 4L,
                             forward_model = "zanim_bart", conditional = TRUE)

  # Visual comparison between the three methods
  x_proposal <- is[, -seq_len(n_test)]
  pdf(file.path(path_res, "inverse_posterior.pdf"), width = 6, height = 4)
  for (i in seq_len(n_test)) {
    x_true <- X_test[i, ]
    cat(i, "\n")

    # Linear interpolation for IS
    is_interp <- akima::interp(x_proposal[, 1], x_proposal[, 2], is[, i], linear = TRUE)
    is_interp_cond <- akima::interp(x_proposal[, 1], x_proposal[, 2], is_cond[, i], linear = TRUE)

    # Compute the KDE for SIR and eSS
    dens_sir <- MASS::kde2d(sir[[i]][, 1], sir[[i]][, 2])
    dens_sir_cond <- MASS::kde2d(sir_cond[[i]][, 1], sir_cond[[i]][, 2])
    dens_ess <- MASS::kde2d(ess[, 1, i], ess[, 2, i])
    dens_ess_cond <- MASS::kde2d(ess_cond[, 1, i], ess_cond[, 2, i])

    # Common limits
    xrange <- range(c(x_true[1], range(dens_sir$x), range(dens_sir_cond$x),
                      range(dens_ess$x), range(dens_ess_cond$x),
                      range(is_interp_cond$x), range(is_interp$x)))
    yrange <- range(c(x_true[2], range(dens_sir$y), range(dens_sir_cond$y),
                      range(dens_ess$y), range(dens_ess_cond$y),
                      range(is_interp_cond$y), range(is_interp$y)))

    par(mfrow = c(3, 2), mar = c(3, 3, 1, 1))
    # IS
    contour(is_interp$x, is_interp$y, is_interp$z,
            main = paste0("IS, y_i = (", paste0(Y_test[i, ], collapse = ","), ")"),
            ylim = yrange, xlim = xrange)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])

    contour(is_interp_cond$x, is_interp_cond$y, is_interp_cond$z,
            main = "IS (conditional)", ylim = yrange, xlim = xrange)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])

    # SIR
    contour(dens_sir$x, dens_sir$y, dens_sir$z, main = "SIR", ylim = yrange,
            xlim = xrange)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])

    contour(dens_sir_cond$x, dens_sir_cond$y, dens_sir_cond$z,
            main = "SIR (conditional)", ylim = yrange, xlim = xrange)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])

    # eSS
    contour(dens_ess$x, dens_ess$y, dens_ess$z, main = "eSS", ylim = yrange,
            xlim = xrange)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])

    contour(dens_ess_cond$x, dens_ess_cond$y, dens_ess_cond$z,
            main = "eSS (conditional)", ylim = yrange, xlim = xrange)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])

  }
  graphics.off()

  es <- sapply(seq_len(n_test), function(i) {
    c(ess = scoringRules::es_sample(y = X_test[i, ], dat = t(ess[,,i])),
      sir = scoringRules::es_sample(y = X_test[i, ], dat = t(sir[[i]])))
  })
  rowMeans(es)
  #       ess       sir
  # 0.9349305 0.8494944

})
