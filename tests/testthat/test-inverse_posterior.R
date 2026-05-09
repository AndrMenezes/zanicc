test_that("ZANIM-LN-BART two-dimension", {

  rm(list = ls())
  devtools::load_all()

  d <- 4L
  region <- "convexhull"

  # Path
  path_local <- "./tests/testthat/inverse_posterior/zanim_ln_bart/two_dimension"
  path_res <- file.path(path_local, region, sprintf("d=%i", d), "results")
  forests_dir <- file.path(path_res, "forests")
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
  cowplot::save_plot(filename = file.path(path_res, "data.png"), plot = p_grid,
                     bg = "white", base_height = 8)

  # Fit forward model
  NDPOST <- 5000L
  NSKIP <- 10000L
  NTREES <- 100L

  if (file.exists(file.path(path_res, "mod.rds"))) {
    zanim_ln_bart <- load_model(model_dir = path_res)
  } else {
    zanim_ln_bart <- zanicc(Y = Y_train, X_count = X_train, X_zi = X_train,
                            model = "zanim_ln_bart", ntrees_theta = NTREES,
                            ntrees_zeta = NTREES, ndpost = NDPOST, nskip = NSKIP,
                            save_trees = TRUE, forests_dir = forests_dir, q_factors = 2)
    save_model(object = zanim_ln_bart, model_dir = path_res)
    # Plotting
    true_thetas <- tmp$true_thetas[-id_test, ]
    true_varthetas <- tmp$true_varthetas[-id_test, ]
    true_zetas <- tmp$true_zetas[-id_test, ]
    mfrow_op <- c(2, 2)
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
  if (file.exists(file.path(path_res, "x_proposal.rds"))) {
    x_proposal <- readRDS(file.path(path_res, "x_proposal.rds"))
  } else {
    set.seed(1212)
    x_proposal <- rconvexhull(n = N_PROPOSAL, X = X_train)
    saveRDS(x_proposal, file.path(path_res, "x_proposal.rds"))
  }

  # Inverse posterior
  sir <- inverse_posterior_zanimlnbart(object = zanim_ln_bart, Y = Y_test[1:3, ],
                                       x_proposal = x_proposal,
                                       dir_posterior_fx = path_res,
                                       method = "sir")
  S <- 2*cov(X_train)
  ess <- inverse_posterior_zanimlnbart(object = zanim_ln_bart, Y = Y_test[1, , drop = FALSE],
                                       x_proposal = x_proposal,
                                       dir_posterior_fx = path_res,
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
                                         dir_posterior_fx = path_res,
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
  pdf(file.path(path_res, "inverse_posterior_cess.pdf"), width = 6, height = 3)
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
  pdf(file.path(path_res, "inverse_posterior_sir.pdf"), width = 6, height = 3)
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
  pdf(file.path(path_res, "inverse_posterior_comparison.pdf"), width = 6, height = 3)
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
  path_res <- file.path(path_local, region, sprintf("d=%i", d), "results")
  forests_dir <- file.path(path_res, "forests")
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


  zanim_ln_bart <- load_model(model_dir = path_res)
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
  thetas <- load_bin_predictions(fname = file.path(path_res, "theta_ij.bin"),
                                 n = 2000, d = d, m = ndpost)
  zetas <- load_bin_predictions(fname = file.path(path_res, "theta_ij.bin"),
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





