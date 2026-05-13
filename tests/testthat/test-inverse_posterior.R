test_that("ZANIM-LN-BART one-dimension", {
  rm(list = ls())
  devtools::load_all()

  d <- 4L
  n_sample <- 1000L

  # Path
  path_local <- "./tests/testthat/inverse_posterior/zanim_ln_bart/one_dimension"
  path_results <- file.path(path_local, sprintf("d=%i", d), "results")
  forests_dir <- file.path(path_results, "forests")
  if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

  if (!file.exists(file.path(path_results, "data.rds"))) {
    # Load pollen data
    data("pollen_data", package = "zanicc")
    n_trials <- rowSums(pollen_data$Y)
    # Use the marginal ZI as upper bound for zetas
    upper_bound_zeta <- apply(pollen_data$Y, 2, zi_binomial, N = n_trials)
    lower_bound_zeta <- colMeans(pollen_data$Y == 0) - upper_bound_zeta
    lower_bound_zeta[lower_bound_zeta == 0] <- 0.001
    stats::qnorm(lower_bound_zeta)
    # Use the centered log-ratio of the empirical mean compositional as intercept
    # for the compositional part
    base_comp <- colMeans(sweep(pollen_data$Y, MARGIN = 1, n_trials, "/"))
    intercept_theta <- log(base_comp / exp(mean(log(base_comp))))
    all.equal(base_comp, exp(intercept_theta)/sum(exp(intercept_theta)))
    set.seed(1212)
    # Sample size and number of categories
    n_trials <- sample(1000L:2000L, size = n_sample, replace = TRUE)
    X_real <- unique(pollen_data$X[, "mtco", drop = FALSE])
    int_theta <- if (d == 28) intercept_theta else stats::runif(d, -2.3, 2.3)
    up <- if (d == 28) upper_bound_zeta else sample(upper_bound_zeta, d)
    list_data <- sim_zanim_ln_gp(n = n_sample, d = d, X_real = X_real,
                                 n_trials = n_trials,
                                 len_scale_theta = 5.0,
                                 upper_bound_zeta = up,
                                 intercept_theta = int_theta)
    saveRDS(object = list_data, file = file.path(path_results, "data.rds"))
  }

  list_data <- readRDS(file = file.path(path_results, "data.rds"))
  # Split the data
  set.seed(1212)
  n_test <- 100L
  id_test <- sample.int(n_sample, n_test)
  Y_test <- list_data$Y[id_test, ]
  X_test <- list_data$X[id_test, , drop = FALSE]
  Y_train <- list_data$Y[-id_test, ]
  X_train <- list_data$X[-id_test, , drop = FALSE]
  true_thetas <- list_data$true_thetas[-id_test, ]
  true_varthetas <- list_data$true_varthetas[-id_test, ]
  true_zetas <- list_data$true_zetas[-id_test, ]
  cbind(structural_zeros = colMeans(1 - list_data$Z),
        sampling_zeros = colMeans(list_data$Y == 0) - colMeans(1 - list_data$Z))

  # Fit forward model
  NDPOST <- 5000L
  NSKIP <- 10000L
  NTREES_THETA <- 100L
  NTREES_ZETA <- 20L

  if (file.exists(file.path(path_results, "mod.rds"))) {
    zanim_ln_bart <- load_model(model_dir = path_results)
  } else {
    zanim_ln_bart <- zanicc(Y = Y_train, X_count = X_train, X_zi = X_train,
                            model = "zanim_ln_bart", ntrees_theta = NTREES_THETA,
                            ntrees_zeta = NTREES_ZETA, ndpost = NDPOST,
                            nskip = NSKIP, save_trees = TRUE, forests_dir = forests_dir,
                            q_factors = .ledermann(d) + 1L)
    save_model(object = zanim_ln_bart, model_dir = path_results)
    zanim_ln_bart <- load_model(model_dir = path_results)

    # Plot true parameters versus estimates
    mfrow_op <- grDevices::n2mfrow(d) #c(3, 2)#
    pdf(file.path(path_results, "diagnostics.pdf"), width = 8, height = 6)
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
    # Plot convergence of vartheta
    par(mfrow = c(1,1), mar = c(3, 3, 1, 1))
    kl <- compute_kl_simplex_chain(true_values = true_varthetas,
                                   draws = zanim_ln_bart$draws_abundance)
    plot(kl, type = "l", main = "KL")

    Y_ppc <- ppd(zanim_ln_bart, relative = FALSE)

    # Plot PPC
    plot_ppc(Y_train, Y_ppc)

    # Plot QQ-plotS
    plot_qqplots(Y_train, Y_ppc, relative = TRUE)

    graphics.off()

    # Compute metrics
    compute_kl_simplex(true_values = true_thetas,
                       estimates = apply(zanim_ln_bart$draws_theta, c(1, 2), mean))
    compute_kl_simplex(true_values = true_varthetas,
                       estimates = apply(zanim_ln_bart$draws_abundance, c(1, 2), mean))
    mean(compute_kl_prob(true_values = true_zetas,
                         estimates = apply(zanim_ln_bart$draws_zeta, c(1, 2), mean)))
    compute_coverage(true_values = true_thetas,
                     apply(zanim_ln_bart$draws_theta, c(1, 2), quantile, probs = 0.025),
                     apply(zanim_ln_bart$draws_theta, c(1, 2), quantile, probs = 0.975))
    compute_coverage(true_values = true_varthetas,
                     apply(zanim_ln_bart$draws_abundance, c(1, 2), quantile, probs = 0.025),
                     apply(zanim_ln_bart$draws_abundance, c(1, 2), quantile, probs = 0.975))

    # Plot parameter against covariates
    data_sim <- list_data$df[!(list_data$df$id %in% id_test), ]
    data_sim$id <- rep(seq_len(nrow(Y_train)), each = d)
    data_theta <- zanicc::summarise_draws_3d(x = zanim_ln_bart$draws_theta)
    data_zeta <- zanicc::summarise_draws_3d(x = zanim_ln_bart$draws_zeta)
    data_theta$x1 <- data_zeta$x1 <- rep(c(X_train), times = d)
    library(ggplot2)
    p_theta <- ggplot(data = data_sim) +
      geom_line(mapping = aes(x = x1, y = theta, col = "Truth", fill = "Truth"),
                linewidth = 0.8) +
      facet_wrap(~category, scales = "free_y") +
      geom_rug(data = dplyr::filter(data_sim, total == 0L),
               mapping = aes(y = NA_real_, x = x1)) +
      geom_line(data = data_theta, mapping = aes(x = x1, y = median),
                col = "dodgerblue") +
      geom_ribbon(data = data_theta,
                  aes(x = x1, ymin = ci_lower, ymax = ci_upper), fill = "dodgerblue",
                  alpha = 0.3)
    cowplot::save_plot(filename = file.path(path_results, "posterior_theta.png"),
                       plot = p_theta, bg = "white", base_height = 9)
    p_zeta <- ggplot(data = data_sim) +
      geom_line(mapping = aes(x = x1, y = zeta, col = "Truth", fill = "Truth"),
                linewidth = 0.8) +
      facet_wrap(~category, labeller = label_parsed) +
      # geom_rug(data = dplyr::filter(data_sim, total == 0L),
      #          mapping = aes(y = NA_real_, x = x)) +
      geom_line(data = data_zeta, mapping = aes(x = x1, y = median),
                col = "dodgerblue") +
      geom_ribbon(data = data_zeta,
                  aes(x = x1, ymin = ci_lower, ymax = ci_upper), fill = "dodgerblue",
                  alpha = 0.3)
    cowplot::save_plot(filename = file.path(path_results, "posterior_zeta.png"),
                       plot = p_zeta, bg = "white", base_height = 9)
  }

  # Generate uniform proposal in the convex-hull
  N_PROPOSAL <- 2000L
  if (file.exists(file.path(path_results, "x_proposal.rds"))) {
    x_proposal <- readRDS(file.path(path_results, "x_proposal.rds"))
  } else {
    x_proposal <- matrix(seq(min(X_train), max(X_train),
                             length.out = N_PROPOSAL), ncol = 1L)
    saveRDS(x_proposal, file.path(path_results, "x_proposal.rds"))
  }

  # Run multiple imputation with SIR
  ff_sir <- file.path(path_results, "sir.rds")
  if (file.exists(ff_sir)) {
    sir <- readRDS(file = ff_sir)
  } else {
    sir <- inverse_posterior_zanimlnbart(object = zanim_ln_bart, Y = Y_test,
                                         x_proposal = x_proposal,
                                         dir_posterior_fx = path_results,
                                         method = "sir")
    # Save results
    saveRDS(object = sir, file = ff_sir)
  }

  # Run eSS
  ff_ess <- file.path(path_results, "ess.rds")
  if (file.exists(ff_ess)) {
    ess <- readRDS(file = ff_ess)
  } else {
    ess <- inverse_posterior_zanimlnbart(object = zanim_ln_bart, Y = Y_test,
                                         x_proposal = x_proposal,
                                         dir_posterior_fx = path_results,
                                         method = "ess",
                                         mean_prior = mean(X_train),
                                         S_prior = diag(1.5*var(X_train[, 1]), nrow = 1),
                                         nburnin = 1L)
    # Save results
    saveRDS(object = ess, file = ff_ess)
  }
  attr(sir, "elapsed_time")
  attr(ess, "elapsed_time")

  sd(X_test)
  sir_metrics <- compute_prediction_metrics(x = X_test, draws = sir)
  ess_metrics <- compute_prediction_metrics(x = X_test, draws = ess)


  # Check eSS
  devtools::load_all()
  ess <- inverse_posterior_zanimlnbart(object = zanim_ln_bart, Y = Y_test[2, , drop = FALSE],
                                       dir_posterior_fx = path_results,
                                       method = "ess", nburnin = 1L,
                                       mean_prior = mean(X_train),
                                       S_prior = diag(1.5*var(X_train[, 1]), nrow = 1))

  ess2 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart, Y = Y_test[2, , drop = FALSE],
                                        dir_posterior_fx = path_results,
                                        method = "ess", nburnin = 1L,
                                        mean_prior = 0.0,
                                        S_prior = diag(1.5*var(X_train[, 1]), nrow = 1))
  plot(density(sir[,1,2]))
  lines(density(ess[,1,1]), col = "red")
  lines(density(ess2[,1,1]), col = "green")
  points(X_test[2, ], 0.001, pch = 19, cex = 2, col = "blue")

})

test_that("ZANIM-LN-BART two-dimension", {

  rm(list = ls())
  devtools::load_all()

  d <- 4L
  region <- "convexhull"

  # Path
  path_local <- "./tests/testthat/inverse_posterior/zanim_ln_bart/two_dimension"
  path_results <- file.path(path_local, region, sprintf("d=%i", d), "results")
  forests_dir <- file.path(path_results, "forests")
  if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

  set.seed(1212)
  n_sample <- 400L
  n_trials <- 500L
  tmp <- sim_data_zanicc_2d(n_grid = 20L, d = d, n_trials = n_trials,
                            region = region)
  # colMeans(tmp$Y==0)
  # colMeans(tmp$Z==0)

  # Split the data
  set.seed(1212)
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
  cowplot::save_plot(filename = file.path(path_results, "data.png"), plot = p_grid,
                     bg = "white", base_height = 8)

  # Fit forward model
  NDPOST <- 5000L
  NSKIP <- 10000L
  NTREES <- 100L

  if (file.exists(file.path(path_results, "mod.rds"))) {
    zanim_ln_bart <- load_model(model_dir = path_results)
  } else {
    zanim_ln_bart <- zanicc(Y = Y_train, X_count = X_train, X_zi = X_train,
                            model = "zanim_ln_bart", ntrees_theta = NTREES,
                            ntrees_zeta = NTREES, ndpost = NDPOST, nskip = NSKIP,
                            save_trees = TRUE, forests_dir = forests_dir, q_factors = 2)
    save_model(object = zanim_ln_bart, model_dir = path_results)
    # Plotting
    true_thetas <- tmp$true_thetas[-id_test, ]
    true_varthetas <- tmp$true_varthetas[-id_test, ]
    true_zetas <- tmp$true_zetas[-id_test, ]
    mfrow_op <- c(2, 2)
    pdf(file.path(path_results, "posterior_mean_vs_true.pdf"), width = 8, height = 6)
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
    plot_ppc(object = zanim_ln_bart, Y = Y_train)
    graphics.off()
    dim(zanim_ln_bart$draws_theta)
    compute_kl_simplex(true_thetas, apply(zanim_ln_bart$draws_theta, c(1, 2), mean))
    compute_frob(true_thetas, apply(zanim_ln_bart$draws_theta, c(1, 2), mean))
    compute_coverage(true_values = true_thetas,
                     apply(zanim_ln_bart$draws_theta, c(1, 2), quantile, probs = 0.025),
                     apply(zanim_ln_bart$draws_theta, c(1, 2), quantile, probs = 0.975))
  }

  # Generate uniform proposal in the convex-hull
  N_PROPOSAL <- 2000L
  if (file.exists(file.path(path_results, "x_proposal.rds"))) {
    x_proposal <- readRDS(file.path(path_results, "x_proposal.rds"))
  } else {
    set.seed(1212)
    x_proposal <- rconvexhull(n = N_PROPOSAL, X = X_train)
    saveRDS(x_proposal, file.path(path_results, "x_proposal.rds"))
  }

  # Inverse posterior
  sir <- inverse_posterior_zanimlnbart(object = zanim_ln_bart, Y = Y_test[1:3, ],
                                       x_proposal = x_proposal,
                                       dir_posterior_fx = path_results,
                                       method = "sir")
  S <- 2*cov(X_train)
  ess <- inverse_posterior_zanimlnbart(object = zanim_ln_bart, Y = Y_test[1, , drop = FALSE],
                                       x_proposal = x_proposal,
                                       dir_posterior_fx = path_results,
                                       method = "ess", nburnin = 1000L, S_prior = S)
  # smooth-ess
  # Get the H-representation of convex-hull
  # Extract matrix and vector for the H-representation of convex-hull
  hull <- geometry::convhulln(X_train, options = "Pp Fa Fx", output.options = "n")
  normals <- hull$normals
  A <- -normals[, -ncol(normals), drop = FALSE]
  b <- -normals[, ncol(normals)]
  m <- colMeans(X_train)#centroid_convexhull(data = X_train, hull = hull)
  plot(hull)
  points(m[1], m[2], col = "blue", pch = 19)
  S <- diag(1, 2)#2*cov(X_train)
  c_ess <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                         Y = Y_test[1:10, , drop = FALSE],
                                         x_proposal = x_proposal,
                                         dir_posterior_fx = path_results,
                                         method = "c_ess", nburnin = 100L,
                                         S_prior = S,
                                         mean_prior = m,
                                         Amat = A, bvec = b, eta = 10000)
  apply(c_ess, c(2, 3), mean)
  attr(sir, "elapsed_time")
  attr(ess, "elapsed_time")
  attr(c_ess, "elapsed_time")

  plot(density(c_ess[, 1,1]))
  plot(density(c_ess[, 2,1]))

  # Get the density of the proposal (uniform)
  dens_prior <- MASS::kde2d(x_proposal[, 1], x_proposal[, 2])
  x1range <- range(x_proposal[, 1])
  x2range <- range(x_proposal[, 2])

  # Plotting only c-eSS
  pdf(file.path(path_results, "inverse_posterior_cess.pdf"), width = 6, height = 3)
  for (i in seq_len(dim(c_ess)[3])) {
    x_true <- X_test[i, ]
    cat(i, "\n")
    dens <- MASS::kde2d(c_ess[, 1, i], c_ess[, 2, i], n = 100)
    # Plotting
    par(mfrow = c(1, 3), mar = c(4, 4, 1, 1))
    # SIR
    contour(dens_prior$x, dens_prior$y, dens_prior$z,
            col = scales::alpha("brown", 0.4), main = "(x1,x2)",
            xlim = range(c(x1range, range(dens$x))),
            ylim = range(c(x2range, range(dens$y)))
            )
    # points(sir[[i]][, 1], sir[[i]][, 2])
    contour(dens$x, dens$y, dens$z, add = TRUE)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])
    # Marginal densities
    plot(density(c_ess[, 1, i]), main = "x1")
    points(x_true[1], 0.00001, col = "blue", pch = 4, cex = 2)
    plot(density(c_ess[, 2, i]), main = "x2")
    points(x_true[2], 0.00001, col = "blue", pch = 4, cex = 2)
  }
  graphics.off()

  # Plotting only SIR
  pdf(file.path(path_results, "inverse_posterior_sir.pdf"), width = 6, height = 3)
  for (i in seq_len(length(sir))) {
    x_true <- X_test[i, ]
    cat(i, "\n")
    dens_sir <- MASS::kde2d(sir[[i]][, 1], sir[[i]][, 2], n = 100)
    # Plotting
    par(mfrow = c(1, 3), mar = c(3, 3, 1, 1))
    # SIR
    contour(dens_prior$x, dens_prior$y, dens_prior$z,
            col = scales::alpha("brown", 0.4), main = "(x1,x2)")
    # points(sir[[i]][, 1], sir[[i]][, 2])
    contour(dens_sir$x, dens_sir$y, dens_sir$z, add = TRUE)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])
    # Marginal densities
    plot(density(sir[[i]][, 1]), main = "x1")
    points(x_true[1], 0.00001, col = "blue", pch = 4, cex = 2)
    plot(density(sir[[i]][, 2]), main = "x2")
    points(x_true[2], 0.00001, col = "blue", pch = 4, cex = 2)
  }
  graphics.off()

  # SIR and eSS
  pdf(file.path(path_results, "inverse_posterior_comparison.pdf"), width = 6, height = 3)
  for (i in seq_len(length(sir))) {
    x_true <- X_test[i, ]
    cat(i, "\n")
    dens_sir <- MASS::kde2d(sir[[i]][, 1], sir[[i]][, 2], n = 100)
    dens_ess <- MASS::kde2d(ess[, 1, i], ess[, 2, i], n = 100)
    dens_cess <- MASS::kde2d(c_ess[, 1, i], c_ess[, 2, i], n = 100)
    # Plotting
    par(mfrow = c(1, 3), mar = c(3, 3, 1, 1))
    # SIR
    contour(dens_prior$x, dens_prior$y, dens_prior$z,
            col = scales::alpha("brown", 0.4), main = "(x1,x2)")
    contour(dens_sir$x, dens_sir$y, dens_sir$z, add = TRUE)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])
    # eSS
    contour(dens_prior$x, dens_prior$y, dens_prior$z,
            col = scales::alpha("brown", 0.4), main = "(x1,x2)",
            xlim = range(c(range(dens_prior$x), range(dens_ess$x))),
            ylim = range(c(range(dens_prior$y), range(dens_ess$y)))
            )
    contour(dens_ess$x, dens_ess$y, dens_ess$z, add = TRUE)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])
    # c-eSS
    contour(dens_prior$x, dens_prior$y, dens_prior$z,
            col = scales::alpha("brown", 0.4), main = "(x1,x2)",
            xlim = range(c(range(dens_prior$x), range(dens_cess$x))),
            ylim = range(c(range(dens_prior$y), range(dens_cess$y)))
            )
    contour(dens_cess$x, dens_cess$y, dens_cess$z, add = TRUE)
    points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
    abline(v = x_true[1], h = x_true[2])
  }
  graphics.off()

  # Check log-likelihood
  devtools::load_all()
  n <- 1000L
  p <- 2L
  X <- matrix(stats::rnorm(n*p), ncol = p)
  hull <- geometry::convhulln(X, options = "Pp Fa Fx", output.options = "n")
  normals <- hull$normals
  A <- -normals[, -ncol(normals), drop = FALSE]
  b <- -normals[, ncol(normals)]
  x = c(-0.1, 0.2)
  ll_linear_constraint <- function(x, eta, A, b) {
    u <- (A %*% x +b )
    # return(u)+ b[i]
    -sum(log1p(exp(-eta * u)))
  }
  ll_linear_constraint(x = c(-0.1, 0.2), eta = 50, A = A, b = b)
  .logIlc(x = c(-0.1, 0.2), mu = c(0,0), A = c(t(A)), b = b, eta = 50)
})


test_that("laplace approximation", {

  rm(list = ls())
  devtools::load_all()
  d <- 4L
  region <- "convexhull"
  # Path
  path_local <- "./tests/testthat/inverse_posterior/zanim_ln_bart/two_dimension"
  path_results <- file.path(path_local, region, sprintf("d=%i", d), "results")
  forests_dir <- file.path(path_results, "forests")
  if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

  # Data
  set.seed(1212)
  n_sample <- 400L
  n_trials <- 500L
  tmp <- sim_data_zanicc_2d(n_grid = 20L, d = d, n_trials = n_trials,
                            region = region)
  # Split the data
  set.seed(1212)
  n_test <- 100L
  id_test <- sample.int(n_sample, n_test)
  Y_test <- tmp$Y[id_test, ]
  X_test <- tmp$X[id_test, , drop = FALSE]
  Y_train <- tmp$Y[-id_test, ]
  X_train <- tmp$X[-id_test, , drop = FALSE]


  zanim_ln_bart <- load_model(model_dir = path_results)
  ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
  cpp_obj <- new(ml$InversePosterior, zanim_ln_bart$d, zanim_ln_bart$ntrees_theta,
                 zanim_ln_bart$ntrees_zeta, zanim_ln_bart$forests_dir)

  # Test the log-likelihood
  i <- 50
  y <- Y_test[i, ]
  x_true <- X_test[i, ]
  chain_index <- 4000L
  ndpost <- zanim_ln_bart$ndpost
  B <- t(zanim_ln_bart$Bt)

  ll <- cpp_obj$LogLikelihoodZANIMLN_2(y, x_true, ndpost, B)
  mean(ll)
  plot(ll)


  k=4
  ll[k]
  cpp_obj$LogLikelihoodZANIMLN(y, x_true, ndpost, k, B)

  x_proposal <- rconvexhull(n = 4000, X = X_train)
  lls <- sapply(seq_len(nrow(x_proposal)), function(i) {
    cpp_obj$LogLikelihoodZANIMLN(y, x_proposal[i, ], ndpost, 1, B)
  })
  cbind(x_proposal, lls)
  maxxx <- cbind(x_proposal, lls)[which.max(lls),]
  terp = interp::interp(x = x_proposal[, 1], y = x_proposal[, 2], z = lls)
  contour(terp$x, terp$y, terp$z)
  points(maxxx[1], maxxx[2], col = "blue", pch = 19, cex = 4)
  points(x_true[1], x_true[2], col = "red", pch = 19, cex = 4)

  # constrained optimisation
  hull <- geometry::convhulln(X_train, options = "Pp Fa Fx", output.options = "n")
  normals <- hull$normals
  A <- -normals[, -ncol(normals), drop = FALSE]
  b <- -normals[, ncol(normals)]
  x_ini <- c(2,0.3)#colMeans(X_train)
  log_like <- function(x, y, ndpost, k, B) -cpp_obj$LogLikelihoodZANIMLN(y, x, ndpost, k, B)
  opt <- ?constrOptim(theta = x_ini, f = log_like, ui = A, ci = -b, y = y, ndpost = ndpost,
                     k = 1, B = B, grad = NULL)
  opt$par
  plot(hull)
  points(opt$par[1], opt$par[2], col = "blue", pch = 19, cex = 4)
  points(x_true[1], x_true[2], col = "red", pch = 19, cex = 4)

  # what about use the posterior mean of the parameters?
  thetas <- load_bin_predictions(fname = file.path(path_results, "theta_ij.bin"),
                                 n = 2000, d = d, m = ndpost)
  zetas <- load_bin_predictions(fname = file.path(path_results, "theta_ij.bin"),
                                 n = 2000, d = d, m = ndpost)
  chol_Sigma_V <- load_bin_coefficients(fname = file.path(forests_dir, "chol_Sigma_V.bin"),
                                        p = d-1, d = d-1, m = ndpost)

  B
  # 1. Get posterior mean of theta, zeta and Sigma_V
  # 2. Compute vartheta
  #   a. conditional on y* to improve
  # 3a. Compute the log-likelihood for each x_proposal, find the maximum and
  # evaluate the minus inverse hessian of multinomial at this value, use this as the
  # laplace approximation for proposal in the is... or this can be seen as the
  # posterior in two-step estimation without uncertainty propagation
  # 3b.



})





