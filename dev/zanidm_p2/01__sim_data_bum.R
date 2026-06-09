rm(list = ls())
devtools::load_all()
# library(zanicc)
library(ggplot2)
library(cowplot)

path_local <- "./dev/zanidm_p2"
path_data <- file.path(path_local, "data")

if (!dir.exists(path_data)) dir.create(path_data, recursive = TRUE)

set.seed(1212)

# Dimension, sample size, and number of trials
d <- 10L
n_samples <- 1000L
exp_counts <- 1000L
n_trials <- stats::rpois(n = n_samples, lambda = exp_counts)

# Simulate covariates
data("pollen_data", package = "zanicc")
X_pollen <- unique(pollen_data$X[, c("gdd5", "mtco")])

X <- rconvexhull(n = n_samples, X = X_pollen)
X <- scale(X)
plot(X[, 1], X[, 2])
mu_X <- colMeans(X)
sd_X <- apply(X, 2, sd)
# Functional form for theta = \alpha / sum(\alpha)

fx_theta <- matrix(nrow = n_samples, ncol = d)
a1_scale <- stats::runif(d, .1, 1)#c(4.3, 1.1, 3.2, 4.8)
a2_scale <- stats::runif(d, .1, 1)
b1_optimum <- stats::rnorm(d, mean = mu_X[1])#c(-6, -5, -2, 3)
b2_optimum <- stats::rnorm(d, mean = mu_X[2])
t1_tolerance <- stats::runif(d, .2, 1)#c(.6, .8, 1.0, 1.2)#
t2_tolerance <- stats::runif(d, .2, 1)

for (j in seq_len(d)) {
  fx_theta[, j] <- a1_scale[j] * exp(-(b1_optimum[j] - X[, 1])^2 / t1_tolerance[j]^2) + a2_scale[j] * exp(-(b2_optimum[j] - X[, 2])^2 / t2_tolerance[j]^2)
}

true_thetas <- sweep(fx_theta, 1, rowSums(fx_theta), "/")

ord1 <- order(X[, 1])
ord2 <- order(X[, 2])
par(mfrow = c(2, 2))
for (j in seq_len(d)) {
  plot(X[ord1, 1], fx_theta[ord1, j], ylim = range(fx_theta), type = "l")
}
for (j in seq_len(d)) {
  plot(X[ord2, 2], fx_theta[ord2, j], ylim = range(fx_theta), type = "l")
}

for (j in seq_len(d)) {
  plot(X[ord1, 1], true_thetas[ord1, j], ylim = range(true_thetas),  type = "l")
}

for (j in seq_len(d)) {
  plot(X[ord2, 2], true_thetas[ord2, j], ylim = range(true_thetas), type = "l")
}

# Functional form for zeta
alphas <- runif(d, 0.01, 0.2)#c(0.3, 0.1, 0.4, 0.2)
true_zetas <- matrix(nrow = n_samples, ncol = d)
for (j in seq_len(d)) true_zetas[, j] <- 1 - true_thetas[, j]^alphas[j]

ord1 <- order(X[, 1])
par(mfrow = c(2, 2))
for (j in seq_len(d)) {
  plot(X[ord1, 1], true_zetas[ord1, j], ylim = range(true_zetas, true_thetas), type = "l")
  lines(X[ord1, 1], true_thetas[ord1, j], type = "l", col = "blue")
}
ord2 <- order(X[, 2])
for (j in seq_len(d)) {
  plot(X[ord2, 2], true_zetas[ord2, j], ylim = range(true_zetas, true_thetas), type = "l")
  lines(X[ord2, 2], true_thetas[ord2, j], type = "l", col = "blue")
}

a0 <- 0.1
tau <- (1 - a0) / a0

# Simulate the counts
Y <- Z <- true_varthetas <- matrix(nrow = n_samples, ncol = d)
for (i in seq_len(n_samples)) {
  z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
  while (all(z == 0L))
    z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
  # z=rep(1, d)
  Z[i, ] <- z
  ld <- stats::rgamma(n = d, shape = tau*fx_theta[i,], rate = 1.0)
  ld <- pmax(ld, 1e-10)
  true_varthetas[i, ] <- z * ld / sum(z * ld)
  is_zero <- z == 0L
  if (sum(is_zero) == d - 1L) {
    Y[i, ] <- rep(0L, d)
    Y[i, !is_zero] <- n_trials[i]
  } else {
    Y[i, ] <- drop(stats::rmultinom(n = 1L, size = n_trials[i],
                                    prob = true_varthetas[i, ]))
  }
}

# Check percentage of zeros
tab_zeros <- cbind(all_zeros = colMeans(Y == 0),
                   structural_zeros = colMeans(1 - Z),
                   sampling_zeros = colMeans(Y == 0) - colMeans(1 - Z))
tab_zeros


# Organise in data frame
data_sim <- data.frame(id = rep(seq_len(n_samples), each = d),
                       category = rep(seq_len(d), times = n_samples),
                       x1 = rep(X[, 1], each = d),
                       x2 = rep(X[, 2], each = d),
                       theta = c(t(true_thetas)),
                       vartheta = c(t(true_varthetas)),
                       zeta = c(t(true_zetas)),
                       total = c(t(Y)), z = c(t(Z)),
                       prop = c(apply(Y, 1L, function(z) z/sum(z))))

# Plot of the true parameters
p_theta <- ggplot(data = data_sim, aes(x = x1, y = x2)) +
  facet_wrap(~category) +
  geom_point(aes(col = theta), shape = 4) +
  scale_color_viridis_c(option = "C", limits = c(0, max(data_sim$theta)))
p_zeta <- ggplot(data = data_sim, aes(x = x1, y = x2)) +
  facet_wrap(~category) +
  geom_point(aes(col = zeta), shape = 4) +
  scale_color_viridis_c(option = "C", limits = c(0, max(data_sim$zeta)))
p_vartheta <- ggplot(data = data_sim, aes(x = x1, y = x2)) +
  facet_wrap(~category) +
  geom_point(aes(col = vartheta), shape = 4) +
  scale_color_viridis_c(option = "C", limits = c(0, max(data_sim$vartheta)))

p_theta

# Save plots
cowplot::save_plot(filename = file.path(path_data, sprintf("true_theta_d=%i.png", d)),
                   plot = p_theta, bg = "white", base_height = 8)
cowplot::save_plot(filename = file.path(path_data, sprintf("true_zeta_d=%i.png", d)),
                   plot = p_zeta, bg = "white", base_height = 8)
cowplot::save_plot(filename = file.path(path_data, sprintf("true_vartheta_d=%i.png", d)),
                   plot = p_vartheta, bg = "white", base_height = 8)

# Sample some test observations
n_test <- 100L
id_test <- sample.int(n = n_samples, size = n_test)
list_data <- list(df = data_sim,
                  id_test = id_test,
                  Y = Y, X = X, Z = Z, true_thetas = true_thetas,
                  true_zetas = true_zetas, true_varthetas = true_varthetas,
                  alphas = alphas)

saveRDS(object = list_data, file = file.path(path_data, sprintf("data_d=%i.rds", d)))

