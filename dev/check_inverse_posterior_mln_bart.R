test_that("MLN-BART one-dimension", {

  rm(list = ls())
  devtools::load_all()

  # Path
  time_id <- "2026-Apr-27-16:21:50"#format(Sys.time(), "%Y-%b-%d-%X")
  path_local <- "./tests/testthat/inverse_posterior/mln_bart/one_dimension"
  path_res <- file.path(path_local, time_id, "results")
  forests_dir <- file.path(path_res, "forests")
  if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

  set.seed(1212)
  d <- 4L
  n_sample <- 400L
  tmp <- sim_zanim_ln_s1(n_sample = n_sample, random_effects = TRUE,
                         structural_zero = FALSE)
  colMeans(tmp$Y == 0)

  # Split the data
  n_test <- 100L
  id_test <- sample.int(n_sample, n_test)
  Y_test <- tmp$Y[id_test, ]
  X_test <- tmp$X[id_test, , drop = FALSE]
  Y_train <- tmp$Y[-id_test, ]
  X_train <- tmp$X[-id_test, , drop = FALSE]

  # Fit forward model
  NDPOST <- 5000L
  NSKIP <- 5000L
  NTREES <- 100L

  if (file.exists(file.path(path_res, "mod.rds"))) {
    mln_bart <- load_model(model_dir = path_res)
  } else {
    mln_bart <- zanicc(Y = Y_train, X_count = X_train,
                       model = "mln_bart", ntrees_theta = NTREES, ndpost = NDPOST,
                       nskip = NSKIP, save_trees = TRUE, forests_dir = forests_dir)
    save_model(object = mln_bart, model_dir = path_res)
  }

  # Compute the f(x*) and generate proposal (do this once)
  N_PROPOSAL <- 2000L
  proposal_parms <- list(min_x = min(X_train), max_x = max(X_train))
  if (!file.exists(file.path(path_res, "theta_ij.bin"))) {
    compute_proposal_fx_mlbart(object = mln_bart, proposal_parms = proposal_parms,
                               n_proposal = N_PROPOSAL, load = FALSE,
                               save = TRUE, output_dir = path_res)
  }

  # IS
  is <- inverse_posterior_mlbart(object = mln_bart, Y = Y_test, method = "is",
                                 proposal_parms = proposal_parms,
                                 n_proposal = N_PROPOSAL,
                                 dir_posterior_fx = path_res)
  # SIR
  sir <- inverse_posterior_mlbart(object = mln_bart, Y = Y_test, method = "is",
                                  proposal_parms = proposal_parms, sir = TRUE,
                                  n_proposal = N_PROPOSAL,
                                  dir_posterior_fx = path_res)


  x_proposal <- is[, -seq_len(n_test)]
  # Visual comparison between the three methods
  pdf(file.path(path_res, "inverse_posterior.pdf"), width = 6, height = 3)
  for (i in seq_len(n_test)) {
    x_true <- X_test[i, ]
    cat(i, "\n")
    par(mfrow = c(1, 2), mar = c(3, 3, 1, 1))
    plot(x_proposal, is[, i], type = "h",
         main = paste0("IS, y_i = (", paste0(Y_test[i, ], collapse = ","), ")"))
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)

    xrange <- c(-1,1)#range(c(x_true, range(sir[[i]])))
    plot(density(sir[[i]]), main = "SIR", xlim = xrange)
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)

  }
  graphics.off()

})

test_that("ZANIM-LN-BART one-dimension", {

  library(ggplot2)
  rm(list = ls())
  devtools::load_all()

  d <- 4L

  # Path
  #time_id <-  #format(Sys.time(), "%Y-%b-%d-%X")
  path_local <- "./tests/testthat/inverse_posterior/zanim_ln_bart/one_dimension"
  path_res <- file.path(path_local, "results", sprintf("d=%i", d))
  forests_dir <- file.path(path_res, "forests")
  if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

  list.files(path_res)

  set.seed(1212)
  n_sample <- 400L
  tmp <- sim_data_zanim_ln_bspline_curve(n = n_sample, d = d, n_trials = 1000,
                                         covariance = "exponential", q_factors = 10L)
  # tmp <- sim_zanim_ln_s1(n_sample = n_sample, random_effects = TRUE,
  #                        structural_zero = TRUE)
  cbind(sampling = colMeans(tmp$Y == 0) - colMeans(1 - tmp$Z),
        structural = colMeans(1 - tmp$Z))

  head(tmp$df)
  ggplot(tmp$df, aes(x = x, y = theta)) +
    facet_wrap(~category) +
    geom_line()
  ggplot(tmp$df, aes(x = x, y = zeta)) +
    facet_wrap(~category) +
    geom_line()

  # Split the data
  n_test <- 100L
  id_test <- sample.int(n_sample, n_test)
  Y_test <- tmp$Y[id_test, ]
  X_test <- tmp$X[id_test, , drop = FALSE]
  Y_train <- tmp$Y[-id_test, ]
  X_train <- tmp$X[-id_test, , drop = FALSE]
  true_thetas <- tmp$theta[-id_test, ]
  true_varthetas <- tmp$abundance[-id_test, ]
  true_zetas <- tmp$zeta[-id_test, ]
  data_sim <- tmp$df[!(tmp$df$id %in% id_test), ]
  data_sim$id <- rep(seq_len(nrow(Y_train)), each = d)

  # Fit forward model
  NDPOST <- 5000L
  NSKIP <- 10000L
  NTREES <- 100L

  if (!file.exists(file.path(path_res, "mod.rds"))) {
    zanim_ln_bart <- zanicc(Y = Y_train, X_count = X_train, X_zi = X_train,
                            model = "zanim_ln_bart", ntrees_theta = NTREES,
                            ntrees_zeta = NTREES, ndpost = NDPOST,
                            nskip = NSKIP, save_trees = TRUE, forests_dir = forests_dir)
    save_model(object = zanim_ln_bart, model_dir = path_res)
  }
  zanim_ln_bart <- load_model(model_dir = path_res)

  # Check if we recover the parameters
  mfrow_op <- c(2,2)#c(4, 5)#c(3, 2)#grDevices::n2mfrow(28)
  pdf(file.path(path_res, "posterior_mean_vs_true.pdf"), width = 8, height = 6)
  par(mfrow = mfrow_op, mar = c(3, 3, 1, 1))
  for (j in seq_len(d)) {
    plot(true_thetas[,j], rowMeans(zanim_ln_bart$draws_theta[,j,]),
         xlab = "true", ylab = "estimate", main = sprintf("theta_{i%d}", j))
    abline(0, 1)
  }
  par(mfrow = mfrow_op, mar = c(3, 3, 1, 1))
  for (j in seq_len(d)) {
    plot(true_varthetas[,j], rowMeans(zanim_ln_bart$draws_abundance[,j,]),
         xlab = "true", ylab = "estimate", main = sprintf("vartheta_{i%d}", j))
    abline(0, 1)
  }
  par(mfrow = mfrow_op, mar = c(3, 3, 1, 1))
  for (j in seq_len(d)) {
    plot(true_zetas[,j], rowMeans(zanim_ln_bart$draws_zeta[,j,]),
         xlab = "true", ylab = "estimate", main = sprintf("zeta_{i%d}", j))
    abline(0, 1)
  }
  graphics.off()

  data_theta <- zanicc::summarise_draws_3d(x = zanim_ln_bart$draws_theta)
  data_zeta <- zanicc::summarise_draws_3d(x = zanim_ln_bart$draws_zeta)
  data_theta$x <- data_zeta$x <- rep(c(X_train), times = d)

  p_theta <- ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = theta, col = "Truth", fill = "Truth"),
              linewidth = 0.8) +
    facet_wrap(~category, labeller = label_parsed) +
    geom_rug(data = dplyr::filter(data_sim, total == 0L),
             mapping = aes(y = NA_real_, x = x)) +
    geom_line(data = data_theta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = data_theta,
                aes(x = x, ymin = ci_lower, ymax = ci_upper), fill = "dodgerblue",
                alpha = 0.3)
  cowplot::save_plot(filename = file.path(path_res, "posterior_theta.png"),
                     plot = p_theta, bg = "white", base_height = 9)
  p_zeta <- ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = zeta, col = "Truth", fill = "Truth"),
              linewidth = 0.8) +
    facet_wrap(~category, labeller = label_parsed) +
    # geom_rug(data = dplyr::filter(data_sim, total == 0L),
    #          mapping = aes(y = NA_real_, x = x)) +
    geom_line(data = data_zeta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = data_zeta,
                aes(x = x, ymin = ci_lower, ymax = ci_upper), fill = "dodgerblue",
                alpha = 0.3)
  cowplot::save_plot(filename = file.path(path_res, "posterior_zeta.png"),
                     plot = p_zeta, bg = "white", base_height = 9)

  # data_zeta_join <- dplyr::left_join(data_zeta, data_sim[, c("id", "category", "zeta")],
  #                               by = c("category", "id"))
  # p_zeta_ <- ggplot(data = data_zeta_join, aes(x = zeta, y = median)) +
  #   facet_wrap(~category) +
  #   geom_point() +
  #   geom_abline(slope = 1, intercept = 0)

  # Generate uniform proposal in the convex-hull
  N_PROPOSAL <- 2000L
  if (file.exists(file.path(path_res, "x_proposal.rds"))) {
    x_proposal <- readRDS(file.path(path_res, "x_proposal.rds"))
  } else {
    x_proposal <- matrix(seq(min(X_train), max(X_train), length.out = N_PROPOSAL),
                         ncol = 1)
    saveRDS(x_proposal, file.path(path_res, "x_proposal.rds"))
  }

  # devtools::load_all()
  sir <- inverse_posterior_zanimlnbart(object = zanim_ln_bart, Y = Y_test,
                                       x_proposal = x_proposal,
                                       dir_posterior_fx = path_res,
                                       method = "sir")
  pdf(file.path(path_res, "density_mi_sir.pdf"), width = 6, height = 3)
  for (i in seq_len(10)) {
    par(mar = c(3, 3, 1, 1))
    plot(density(sir[[i]]), xlim = c(-1, 1))
    points(X_test[i,], y = 0.001, col = "blue", pch = 4, cex = 2)
  }
  graphics.off()
  # eSS
  mean_prior = 0.0; S_prior = diag(1.0, nrow = 1); n_rep = 4L
  ess <- .gibbs_sampler(Y = Y_test, mean_prior = mean_prior, S_prior = S_prior,
                        forests_dir = forests_dir, ntrees = NTREES, ndpost = NDPOST,
                        n_rep = n_rep, forward_model = "zanim_ln_bart")


  x_proposal <- is[, -seq_len(n_test)]
  # Visual comparison between the three methods
  pdf(file.path(path_res, "density_posteriors.pdf"), width = 6, height = 3)
  for (i in seq_len(n_test)) {
    x_true <- X_test[i, ]
    cat(i, "\n")
    par(mfrow = c(1, 2), mar = c(3, 3, 1, 1))
    plot(x_proposal, is[, i], type = "h",
         main = paste0("IS, y_i = (", paste0(Y_test[i, ], collapse = ","), ")"))
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)

    xrange <- c(-1,1)#range(c(x_true, range(sir[[i]])))
    plot(density(sir[[i]]), main = "SIR", xlim = xrange)
    points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)

    # xrange <- range(c(1, -1, range(ess[,1,i]) ))
    # plot(density(ess[,1,i]), main = "eSS", xlim = xrange)
    # points(x_true, 0.00001, col = "blue", pch = 4, cex = 2)
  }
  graphics.off()

  # Some checks in the internal functions

  mean_prior = 0.0; S_prior = diag(1.0, nrow = 1); n_rep = 1
  # Y_test <- Y_train[1:2,,drop=FALSE]
  x_true <- X_test[c(1:3, 93),]
  n <- nrow(Y_test)
  p <- length(mean_prior)
  X_ini <- matrix(nrow = n, ncol = p)
  cS <- chol(S_prior)
  for (i in seq_len(n)) X_ini[i, ] <- stats::rnorm(p) %*% cS + mean_prior
  #
  devtools::load_all()
  ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
  cpp_obj <- new(ml$InversePosterior, ncol(Y_test), NTREES, NTREES, NDPOST, "ml_bart",
                 forests_dir)
  out <- cpp_obj$SamplerZANIMLNBARTeSS(Y_test, as.matrix(X_ini),
                                       as.integer(NDPOST), mean_prior, S_prior,
                                       4, t(zanim_ln_bart$Bt))

  as.matrix(qr.Q(qr(stats::contr.sum(ncol(Y_test)))))

  draws <- array(out, dim = c(NDPOST, 1, 4))
  par(mfrow = c(2, 2))
  plot(density(draws[,,1]))
  points(x_true[1], 0.0001, col = "blue", pch = 4)
  plot(density(draws[,,2]))
  points(x_true[2], 0.0001, col = "blue", pch = 4)
  plot(density(draws[,,3]))
  points(x_true[3], 0.0001, col = "blue", pch = 4)
  plot(density(draws[,,4]))
  points(x_true[4], 0.0001, col = "blue", pch = 4)

})
