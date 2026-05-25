rm(list = ls())
library(zanicc)
library(ggplot2)
library(cowplot)

path_local <- "./dev/zanidm_toy_example"
path_data <- file.path(path_local, "data")

if (!dir.exists(path_data)) dir.create(path_data, recursive = TRUE)

set.seed(1212)

# Dimension, sample size, and number of trials
d <- 4L
n_samples <- 1000L
exp_counts <- 1000L
n_trials <- stats::rpois(n = n_samples, lambda = exp_counts)

# Climate covariate (say temperature)
data("pollen_data", package = "zanicc")
mu_X <- mean(pollen_data$X[, "mtco"])
sd_X <- sd(pollen_data$X[, "mtco"])
x <- stats::rnorm(n_samples, mean = mu_X, sd = sd_X)

# Functional form for theta = \alpha / sum(\alpha)
fx_theta <- matrix(nrow = n_samples, ncol = d)
a_scale <- c(4.3, 1.1, 3.2, 4.8) #stats::runif(d, 0.1, 5.0)
b_optimum <- c(-6, -5, -2, 3)#stats::rnorm(d, mean = mu_X, sd = sd_X)
t_tolerance <- c(20, 15, 12, 10)#stats::runif(d, 5, 20)
# t_tolerance <- stats::rgamma(d, shape = 9.0, rate = 0.5)

parms_count <- list(a_scale = a_scale, b_optimum = b_optimum,
                    t_tolerance = t_tolerance)

for (j in seq_len(d)) {
  fx_theta[, j] <- a_scale[j] * exp(-(b_optimum[j] - x)^2 / t_tolerance[j]^2)
}
true_thetas <- sweep(fx_theta, 1, rowSums(fx_theta), "/")

ord <- order(x)
par(mfrow = c(2, 2))
for (j in seq_len(d)) {
  plot(x[ord], fx_theta[ord, j], ylim = range(fx_theta), type = "l")
}

for (j in seq_len(d)) {
  plot(x[ord], true_thetas[ord, j], ylim = range(true_thetas), type = "l")
}

# Functional form for zeta
# true_zetas <- matrix(nrow = n_samples, ncol = d)
# b_occ <- c(-5, -30, -9.0, -3)
# tau_occ <- runif(d, 3, 10)
# pi_min <- runif(d, 0.01, 0.20)
# pi_max <- runif(d, 0.40, 0.70)
# 
# parms_zi <- list(pi_min = pi_min, pi_max = pi_max, 
#                  b_optimum = b_occ,
#                  t_tolerance = tau_occ)
# 
# 
# for (j in seq_len(d)) {
#   true_zetas[, j] <- pi_min[j] + (pi_max[j] - pi_min[j]) * exp(-(x - b_occ[j])^2 / tau_occ[j]^2)
# }

alphas <- c(0.3, 0.1, 0.4, 0.2)
true_zetas <- matrix(nrow = n_samples, ncol = d)
for (j in seq_len(d)) {
  true_zetas[, j] <- 1 - true_thetas[, j]^alphas[j]
}

par(mfrow = c(2, 2))
for (j in seq_len(d)) {
  plot(x[ord], true_zetas[ord, j], ylim = range(true_zetas, true_thetas), type = "l")
  lines(x[ord], true_thetas[ord, j], type = "l", col = "blue")
}

a0 <- 0.01
tau <- (1 - a0) / a0

# Simulate the counts
Y <- Z <- true_varthetas <- matrix(nrow = n_samples, ncol = d)
for (i in seq_len(n_samples)) {
  z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
  is_zero <- z == 0L
  # Hack to avoid all zeros (it happen very rarely)
  while (all(is_zero)) {
    z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
    is_zero <- z == 0L
  }
  ld <- stats::rgamma(n = d, shape = tau*fx_theta[i,], rate = 1.0)
  ld <- pmax(ld, 1e-10)
  true_varthetas[i, ] <- z * ld / sum(z * ld)
  if (sum(is_zero) == d - 1L) {
    Y[i, ] <- rep(0L, d)
    Y[i, !is_zero] <- n_trials[i]
  } else {
    Y[i, ] <- drop(stats::rmultinom(n = 1L, size = n_trials[i],
                                    prob = true_varthetas[i, ]))
  }
  Z[i, ] <- z
}

# Check percentage of zeros
tab_zeros <- cbind(all_zeros = colMeans(Y == 0),
                   structural_zeros = colMeans(1 - Z),
                   sampling_zeros = colMeans(Y == 0) - colMeans(1 - Z))
tab_zeros


# Organise in data frame
data_sim <- data.frame(id = rep(seq_len(n_samples), each = d),
                       category = rep(seq_len(d), times = n_samples),
                       x = rep(x, each = d),
                       theta = c(t(true_thetas)),
                       vartheta = c(t(true_varthetas)),
                       zeta = c(t(true_zetas)),
                       total = c(t(Y)), z = c(t(Z)),
                       prop = c(apply(Y, 1L, function(z) z/sum(z))))

# Plot true functional forms and the observed counts
ggplot(data_sim, aes(x = x, y = prop)) + facet_wrap(~category) +
  geom_point() +
  geom_line(aes(y = theta), col = "dodgerblue") +
  geom_line(aes(y = zeta), col = "orangered")

# Sample some test observations
n_test <- 100L
id_test <- sample.int(n = n_samples, size = n_test)
list_data <- list(df = data_sim,
                  id_test = id_test,
                  Y = Y, X = as.matrix(x), Z = Z, true_thetas = true_thetas,
                  true_zetas = true_zetas, true_varthetas = true_varthetas,
                  parms_count = parms_count,
                  # , parms_zi = parms_zi
                  alphas = alphas
                  )
saveRDS(object = list_data, file = file.path(path_data, "data.rds"))





