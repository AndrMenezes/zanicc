rm(list = ls())
library(zanicc)
library(ggplot2)
library(cowplot)

path_local <- "./dev/zanidm_toy_example"
path_data <- file.path(path_local, "data")
path_results <- file.path(path_local, "results")

if (!dir.exists(path_results)) dir.create(path_results, recursive = TRUE)

# Load data
list_data <- readRDS(file.path(path_data, "data.rds"))
data_sim <- list_data$df
X <- list_data$X
Y <- list_data$Y
rangeX <- range(X)
parms_count <- list_data$parms_count
# parms_zi <- list_data$parms_zi
parms_zi <- list_data$alphas
id_test <- list_data$id_test
Y_test <- list_data$Y[id_test,]
X_test <- list_data$X[id_test,, drop = FALSE]
n_test <- nrow(Y_test)
d <- ncol(Y)

# Check percentage of zeros
tab_zeros <- cbind(all_zeros = colMeans(list_data$Y == 0),
                   structural_zeros = colMeans(1 - list_data$Z),
                   sampling_zeros = colMeans(Y == 0) - colMeans(1 - list_data$Z))
tab_zeros

# Plot true functional forms and the observed counts
p_true_ff <- ggplot(data_sim, aes(x = x)) +
  facet_wrap(~category) +
  geom_point(aes(y = prop)) +
  geom_line(aes(y = theta), col = "dodgerblue") +
  geom_line(aes(y = zeta), col = "orangered") +
  labs(x = "x", y = "theta/zeta")
p_true_ff

# True functional form
fx_theta <- function(x, parms) {
  d <- length(parms$a_scale)
  out <- numeric(d)
  for (j in seq_len(d)) {
    out[j] <- parms$a_scale[j] * exp(-(parms$b_optimum[j] - x)^2 / parms$t_tolerance[j]^2)
  }
  out
}
# fx_zeta <- function(x, parms) {
#   d <- length(parms$b_optimum)
#   out <- numeric(d)
#   for (j in seq_len(d)) {
#     out[j] <- parms$pi_min[j] + (parms$pi_max[j] - parms$pi_min[j]) *
#       exp(-(x - parms$b_optimum[j])^2 / parms$t_tolerance[j]^2)
#   }
#   out
# }
fx_zeta <- function(x, thetas, alphas) {
  d <- length(thetas)
  out <- numeric(d)
  for (j in seq_len(d)) {
    out[j] <- 1 - thetas[j]^(alphas[j])
  }
  out
}

# The model is: Y_i ~ ZANIDM[alpha_i, zeta_i], where
# alpha_i = fx_theta(x_i, parms)
# zeta_i = fx_zeta(x_i, parms)
#
# Given a new count y*, we want to compute p(x* | y*) \propto p(y* | x*)p(x*)
# Because we know the true parameters "f", we don't need to integrate out them.
# Therefore, to obtain the above posterior we just need to sample from the target
# density pi(x*) := p(y* | x*)p(x*), where in our case p(y* | x*) is dzanidm, and
# we shall use a uniform prior for p(x*).

# Importance sampling
run_is <- function(y, x_proposal, parms_count, parms_zi) {
  n_proposal <- length(x_proposal)
  log_weights <- numeric(n_proposal)
  for (k in seq_len(n_proposal)) {
    if (k %% 100L == 0L) cat(k, "\n")
    x_cur <- x_proposal[k]
    # Get the parameters
    alpha_cur <- fx_theta(x = x_cur, parms = parms_count)
    # alpha_cur <- pmax(alpha_cur, 1e-10)
    zeta_cur <- fx_zeta(x = x_cur, thetas = alpha_cur / sum(alpha_cur),
                        alphas = parms_zi)
    log_weights[k] <- zanicc:::log_pmf_zanidm(x = y, alpha = alpha_cur,
                                              zeta = zeta_cur)
  }
  probs <- exp(log_weights - max(log_weights))
  probs / sum(probs)
}

# Create an uniform proposal
n_proposal <- 10000L
x_proposal <- seq(rangeX[1], rangeX[2], length.out = n_proposal)

n_resampling <- floor(n_proposal / 2L)
x_posterior <- matrix(nrow = n_resampling, ncol = n_test)

# Run IS for all the test observations
pdf(file = file.path(path_results, "truth_inverse_posterior.pdf"), width = 10, height = 6)
for (i in seq_len(n_test)) {
  y_new <- Y_test[i, ]
  x_true <- X_test[i, ]
  # Importance sampling probabilities
  p_is <- run_is(y = y_new, x_proposal = x_proposal, parms_count = parms_count,
               parms_zi = parms_zi)
  # Resampling
  x_sir <- sample(x_proposal, size = n_proposal/2, replace = TRUE, prob = p_is)
  # Saving
  x_posterior[,i] <- x_sir

  # Plotting
  data_cur <- data.frame(x = rep(x_true, d), prop = y_new / sum(y_new),
                         category = seq_len(d))
  p_example <- p_true_ff +
    geom_hline(data = data_cur,  aes(yintercept = prop), linetype = "dashed",
               col = "red") +
    geom_vline(data = data_cur,  aes(xintercept = x), linetype = "dashed")

  # p_posterior <- ggplot(data = data.frame(x = x_proposal, probs = probs)) +
  #   geom_linerange(aes(x = x,ymin = 0, ymax = probs)) +
  #   annotate(geom = "point", x = x_true, y = min(p_is), col = "blue", size = 4)
  p_posterior <- ggplot(data = data.frame(x = x_sir), aes(x = x)) +
    geom_density() +
    geom_rug() +
    scale_x_continuous(limits = rangeX) +
    annotate(geom = "point", x = x_true, y = min(p_is), col = "blue", size = 4) +
    ggtitle(paste0("y=(",paste0(round(y_new / sum(y_new), digits = 2), collapse = ", "), ")"))

  print(plot_grid(p_example, p_posterior))

}
graphics.off()


saveRDS(object = x_posterior, file = file.path(path_data, "ground_truth_posterior.rds"))



