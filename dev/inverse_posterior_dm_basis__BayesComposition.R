rm(list = ls())
# Install package from the inverse_posterior branch
# remotes::install_github("andrmenezes/zanicc@inverse_posterior")
library(zanicc)
library(BayesComposition)

d <- 4L
n_sample <- 1000L

# Path (Change here)
path_local <- "./tests/testthat/inverse_posterior/zanim_ln_bart/one_dimension"
path_results <- file.path(path_local, sprintf("d=%i", d), "results")


list_data <- readRDS(file = file.path(path_results, "data.rds"))
# Split the data
set.seed(1212)
n_test <- 100L
id_test <- sample.int(n_sample, n_test)
Y_test <- list_data$Y[id_test, ]
X_test <- list_data$X[id_test, , drop = FALSE]
Y_train <- list_data$Y[-id_test, ]
X_train <- list_data$X[-id_test, , drop = FALSE]
true_thetas <- list_data$true_thetas[-id_test, , drop = FALSE]

# Fit forward model
params <- list(
  n_adapt                    = 1000,
  n_mcmc                     = 1000,
  n_thin                     = 10L,
  likelihood                 = "dirichlet-multinomial",
  function_type              = "basis"
  # ## prior on covariance to improve estimation
  # eta                        = 8,
  # additive_correlation       = FALSE,
  # n_chains                   = 4,
  # n_cores                    = 4,
  # n_knots                    = n_knots,
  # X_knots                    = X_knots
  )

# Fit
if (!file.exists(file.path(path_results, "dm_basis.rds"))) {
  dm_gp <- BayesComposition::fit_compositional_data(y = Y_train, X = X_train, params = params,
                                                    progress_directory = path_results,
                                                    progress_file = "progress_file.txt")
  saveRDS(object = dm_gp, file = file.path(path_results, "dm_basis.rds"))
}
dm_gp <- readRDS(file = file.path(path_results, "dm_basis.rds"))
length(dm_gp)

dim(dm_gp[[1]]$beta)
dim(dm_gp[[1]]$mu_beta)

# Extract samples
samples <- BayesComposition::extract_compositional_samples(dm_gp)
dim(samples$beta)
alpha_post <- samples$alpha
theta_post <- sweep(alpha_post, MARGIN = c(1, 2),
                    STATS = apply(alpha_post, MARGIN = c(1, 2), sum),
                    FUN = "/")
theta_post <- aperm(theta_post, perm = c(2, 3, 1))
dim(theta_post)
dim(true_thetas)
kl <- zanicc::compute_frob_chain(true_values = true_thetas,
                                 draws = theta_post)
plot(kl, type = "l")

# plot(samples$beta[,1,1], type = "l")

# Posterior-predictive checks
n_trials <- rowSums(Y_train)
mc <- dim(alpha_post)[1L]
y_ppc <- array(dim = dim(alpha_post))
for (k in seq_len(mc)) {
  if (k %% 10 == 0L) cat(k, "\n")
  for (i in seq_len(nrow(Y_train))) {
    ld <- stats::rgamma(n = d, shape = alpha_post[k, i, ])
    y_ppc[k, i, ] <- drop(stats::rmultinom(n = 1L, size = n_trials[i],
                                           prob = ld / sum(ld)))
  }
}
png(filename = file.path(path_results, "ppc.png"), units = "in", width = 10,
    height = 7, res = 300)
out_ppc <- zanicc::plot_ppc(Y = Y_train, Y_ppc = y_ppc, output = TRUE)
graphics.off()
png(filename = file.path(path_results, "qqplots.png"), units = "in", width = 10,
    height = 7, res = 300)
zanicc::plot_qqplots(Y = Y_train, Y_ppc = y_ppc, relative = TRUE)
graphics.off()



# Inverse posterior
X_ini <- matrix(nrow = n_test, ncol = 1)
for (i in seq_len(n_test)) X_ini[i, ] <- stats::rnorm(1, mean = mean(X_train), sd = sd(X_train))

res <- BayesComposition::predict_compositional_data(
  y_reconstruct = Y_test,
  X_calibrate = X_ini,
  params = params,
  samples = samples,
  progress_directory = paste0(path_results, "/"),
  progress_file = "progress__inverse_posterior.txt")

ip_dm_gp <- array(dim = c(nrow(res$X), 1L, ncol(res$X)))
for (i in seq_len(ncol(res$X))) ip_dm_gp[,1L,i] <- res$X[, i]

compute_prediction_metrics(x = X_test, draws = ip_dm_gp)




