rm(list = ls())
devtools::load_all()
library(ggplot2)

d <- 10L
# Path
path_local <- "./dev/zanidm_p2"
path_data <- file.path(path_local, "data")
path_results <- file.path(path_local, "results", d)
forests_dir <- file.path(path_results, "forests")
if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

# Import data
list_data <- readRDS(file.path(path_data, sprintf("data_d=%i.rds", d)))
id_test <- list_data$id_test
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
