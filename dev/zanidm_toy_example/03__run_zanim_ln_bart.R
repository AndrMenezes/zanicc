rm(list = ls())
library(zanicc)
library(ggplot2)

# Path
path_local <- "./dev/zanidm_toy_example"
path_data <- file.path(path_local, "data")
path_results <- file.path(path_local, "results", "zanim_ln_bart")
forests_dir <- file.path(path_results, "forests")
if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

# Import data
list_data <- readRDS(file.path(path_data, "data.rds"))
id_test <- list_data$id_test
n_test <- length(id_test)
Y_test <- list_data$Y[id_test, ]
X_test <- list_data$X[id_test, , drop = FALSE]
Y_train <- list_data$Y[-id_test, ]
X_train <- list_data$X[-id_test, , drop = FALSE]
true_thetas <- list_data$true_thetas[-id_test, ]
true_varthetas <- list_data$true_varthetas[-id_test, ]
true_zetas <- list_data$true_zetas[-id_test, ]
d <- ncol(Y_train)

# Fit forward model
NDPOST <- 5000L
NSKIP <- 10000L
NTREES_THETA <- 100L
NTREES_ZETA <- 100L

if (!file.exists(file.path(path_results, "mod.rds"))) {
zanim_ln_bart <- zanicc(Y = Y_train, X_count = X_train, X_zi = X_train,
                          model = "zanim_ln_bart", ntrees_theta = NTREES_THETA,
                          ntrees_zeta = NTREES_ZETA, ndpost = NDPOST,
                          nskip = NSKIP, save_trees = TRUE,
                          forests_dir = forests_dir)
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
  data_theta$x <- data_zeta$x <- rep(c(X_train), times = d)

  saveRDS(object = data_theta, file = file.path(path_results, "posterior_theta.rds"))
  saveRDS(object = data_zeta, file = file.path(path_results, "posterior_zeta.rds"))

  p_theta <- ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = theta, col = "Truth", fill = "Truth"),
              linewidth = 0.8) +
    facet_wrap(~category, scales = "free_y") +
    geom_rug(data = dplyr::filter(data_sim, total == 0L),
             mapping = aes(y = NA_real_, x = x)) +
    geom_line(data = data_theta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = data_theta,
                aes(x = x, ymin = ci_lower, ymax = ci_upper), fill = "dodgerblue",
                alpha = 0.3)
  cowplot::save_plot(filename = file.path(path_results, "posterior_theta.png"),
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
  cowplot::save_plot(filename = file.path(path_results, "posterior_zeta.png"),
                     plot = p_zeta, bg = "white", base_height = 9)
}

# Generate uniform proposal
N_PROPOSAL <- 2000L
if (!file.exists(file.path(path_results, "x_proposal.rds"))) {
  x_proposal <- matrix(seq(min(X_train), max(X_train),
                           length.out = N_PROPOSAL), ncol = 1L)
  saveRDS(x_proposal, file.path(path_results, "x_proposal.rds"))
}


# Compute the posterior distribution of f^{(c)}_j(x*) and f^{(0)}_j(x*) for x*~\pi(x*) and j=1,...,d
predict(zanim_ln_bart, newdata = x_proposal, load = FALSE, output_dir = path_results,
        type = "theta")
predict(zanim_ln_bart, newdata = x_proposal, load = FALSE, output_dir = path_results,
        type = "zeta")



