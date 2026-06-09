rm(list = ls())
# library(zanicc)
library(ggplot2)
devtools::load_all()

# Path
path_local <- "./dev/zanidm_toy_example"
path_data <- file.path(path_local, "data")
path_results <- file.path(path_local, "results", "zanim_ln_bart")

# Import data
list_data <- readRDS(file.path(path_data, "data.rds"))
id_test <- list_data$id_test
n_test <- length(id_test)
Y_test <- list_data$Y[id_test, ]
X_test <- list_data$X[id_test, , drop = FALSE]

# Load forward model -----------------------------------------------------------
zanim_ln_bart <- load_model(model_dir = path_results)


# Load uniform proposal --------------------------------------------------------

x_proposal <- readRDS(file.path(path_results, "x_proposal.rds"))
n_proposal <- nrow(x_proposal)


# Load the ground truth --------------------------------------------------------

x_truth_posterior <- readRDS(file = file.path(path_data, "ground_truth_posterior.rds"))


# Test with the inverse posterior ----------------------------------------------
chosen_obs <- c(1L, 2L, 10L, 12) # N-inflation, only one zero, no-zeros and 2-zeros.

i <- 12L #2L #1L
y_new <- Y_test[i, ]
x_true <- X_test[i, ]

B <- t(zanim_ln_bart$Bt)
# ndpost <- zanim_ln_bart$ndpost
ndpost <- 1000L

kernel <- 1L # 0: gaussian, 1: exponential
h <- 0.01 # bandiwith
n_particles <- 10L

# Initialise the class
devtools::load_all()
ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
cpp_obj <- new(ml$InversePosterior, zanim_ln_bart$d, zanim_ln_bart$ntrees_theta,
               zanim_ln_bart$ntrees_zeta, zanim_ln_bart$forests_dir)

# SIR --------------------------------------------------------------------------
ini <- proc.time()
cpp_obj$SIR(Y_test, n_proposal, ndpost, B, path_results)
end <- proc.time() - ini

idx_sir <- cpp_obj$indices_sir + 1L
effsize_sir <- matrix(cpp_obj$ess_sir, nrow = ndpost)
n_samples <- length(chosen_obs)

colMeans(effsize_sir)

# Copy the posterior into an array
x_posterior <- array(dim = c(ndpost, 1, n_samples))
for (i in seq_len(n_samples)) {
   indices <- idx_sir[(1 + ndpost*(i - 1L)):(ndpost*i)]
   x_posterior[,,i] <- x_proposal[indices,]
}
dim(x_posterior)
# Plotting
par(mfrow = c(2, 2))
for (i in chosen_obs) {
   plot(density(x_posterior[,,i]))
   points(X_test[i, ], 0.0001, col = "blue", cex = 2, pch = 19)
}

# ESS --------------------------------------------------------------------------

mu_prior <- mean(list_data$X)
S_prior <- as.matrix(var(list_data$X))
x_ess1 <- cpp_obj$ESSZANIMLNBART(matrix(y_new, nrow = 1), as.matrix(x_true), ndpost, 1L,
                                mu_prior, S_prior, B, 1L)
x_ess10 <- cpp_obj$ESSZANIMLNBART(matrix(y_new, nrow = 1), as.matrix(x_true), ndpost, 10L,
                                mu_prior, S_prior, B, 10L)
x_ess100 <- cpp_obj$ESSZANIMLNBART(matrix(y_new, nrow = 1), as.matrix(x_true), ndpost, 10L,
                                mu_prior, S_prior, B, 100L)
# x_ess500 <- cpp_obj$ESSZANIMLNBART(matrix(y_new, nrow = 1), as.matrix(x_true), ndpost, 10L,
#                                 mu_prior, S_prior, B, 500L)
x_ess1000 <- cpp_obj$ESSZANIMLNBART(matrix(y_new, nrow = 1), as.matrix(x_true), ndpost, 1L,
                                mu_prior, S_prior, B, 1000L)
# x_ess100100 <- cpp_obj$ESSZANIMLNBART(matrix(y_new, nrow = 1), as.matrix(x_true), ndpost, 100L,
#                                 mu_prior, S_prior, B, 100L)

# Compare only ESS versions with different n_particles
d_true <- density(x_truth_posterior[, i])
d_ess1 <- density(x_ess1)
d_ess10 <- density(x_ess10)
d_ess100 <- density(x_ess100)
# d_ess500 <- density(x_ess500)
d_ess1000 <- density(x_ess1000)
# d_ess100100 <- density(x_ess100100)
plot(d_ess1, ylim = range(d_true$y, d_ess10$y, d_ess100$y,  d_ess1000$y)
     , xlim = range(d_true$x, d_ess10$x, d_ess100$x, d_ess1000$x),
     main = "eSS with different values of {n_particles}")
lines(d_ess1, col = "blue")
lines(d_ess10, col = "green")
lines(d_ess100, col = "red")
# lines(d_ess500, col = "green")
lines(d_ess1000, col = "gold")
# lines(d_ess100100, col = "blue")
points(x_true, min(d_true$y), col = "blue", cex = 2, pch = 19)
legend(x = "topright", legend = c(1, 10, 100, 1000), col = c("blue", "green", "red", "gold"),
       lwd = 1)


# ABC-SIR ----------------------------------------------------------------------
idx_abc1 <- cpp_obj$ABCSIRZANIMLNBART(y_new, n_proposal, 1000, B,
                                      path_results, kernel, h, 1)
effsize_abc1 <- 1.0 / cpp_obj$ess_sir

idx_abc10 <- cpp_obj$ABCSIRZANIMLNBART(y_new, n_proposal, 1000, B,
                                       path_results, kernel, h, 10)
effsize_abc10 <- 1.0 / cpp_obj$ess_sir

idx_abc100 <- cpp_obj$ABCSIRZANIMLNBART(y_new, n_proposal, ndpost, B,
                                       path_results, kernel, h, 100)
effsize_abc100 <- 1.0 / cpp_obj$ess_sir

# ESS according number of particles
cbind(np1 = quantile(effsize_abc1),
      np10 = quantile(effsize_abc10)
      # , np100 = quantile(effsize_abc100)
      )

x_abc1 <- x_proposal[idx_abc1 + 1L, ]
x_abc10 <- x_proposal[idx_abc10 + 1L, ]

d_true <- density(x_truth_posterior[, i])
d_abc1 <- density(x_abc1)
d_abc10 <- density(x_abc10)
plot(d_true, ylim = range(d_true$y, d_abc1$y, d_abc10$y)
     , xlim = range(d_true$x, d_abc1$x, d_abc10$x))
lines(d_abc1, col = "red")
lines(d_abc10, col = "blue")
points(x_true, min(d_true$y), col = "blue", cex = 2, pch = 19)



# Population-MC ----------------------------------------------------------------

devtools::load_all()
ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
cpp_obj <- new(ml$InversePosterior, zanim_ln_bart$d, zanim_ln_bart$ntrees_theta,
               zanim_ln_bart$ntrees_zeta, zanim_ln_bart$forests_dir)
# 67 75
i <- 10L #2L #1L
y_new <- Y_test[i, ]
x_true <- X_test[i, ]
scale_prop <- 0.5
ep <- 0.01
prob_level <- 0.01
ndpost <- 100L
n_particles_x <- 1000L
range_prior <- range(list_data$X)
B <- t(zanim_ln_bart$Bt)
cpp_obj$PopulationMC(y_new, ndpost, n_particles_x, B, range_prior, scale_prop,
                     prob_level, ep)

ess_pmc <- cpp_obj$ess_sir
x_pmc <- cpp_obj$x_posterior
# any(is.na(x_pmc))

plot(density(x_pmc), col = "blue")
abline(v = x_true)

pmc <- inverse_posterior_zanimlnbart(object = zanim_ln_bart, Y = Y_test[c(67,75), ],
                                     method = "pmc", n_particles = 1000L,
                                     scale_prop = 0.5,
                                     range_prior = range_prior
                                     , ndpost = 1000)
lapply(pmc, function(x) any(is.na(x)))
plot(density(pmc[[1]]), col = "blue")
plot(density(pmc[[2]]), col = "blue")


# Check ESS
idx_sir1 <- cpp_obj$SIRZANIMLNBART(y_new, n_proposal, ndpost, B,
                                   path_results, 1, 0L)
x_sir1 <- x_proposal[idx_sir1 + 1L, ]
ess_sir1 <- cpp_obj$ess_sir
hist(ess_pmc)
mean(ess_pmc); mean(ess_sir1)
# cbind(ess_pmc, ess_sir1)

plot(density(x_pmc_mat[1, ]), col = "blue", main = "")
lines(density(x_sir1))
abline(v = x_true)


##############
range_prior <- range(list_data$X)
res <- inverse_posterior_zanimlnbart(zanim_ln_bart, Y = Y_test[chosen_obs, ],
                                     dir_posterior_fx = path_results, method = "pmc",
                                     ndpost = 100L,
                                     n_particles = 1000L, range_prior = range_prior,
                                     scale_prop = 0.5)
lapply(res, function(x) attr(x, "ess"))











# Check posterior predictions
theta_posterior <- t(matrix(cpp_obj$theta_posterior, nrow = 4, ncol = n_particles_x))
zeta_posterior <- t(matrix(cpp_obj$zeta_posterior, nrow = 4, ncol = n_particles_x))

# load predictions, theta and zeta given "uniform" proposal
theta_pred <- load_bin_predictions(file.path(path_results, "theta_ij.bin"),
                                   n = n_proposal, d = 4, m = 1)
zeta_pred <- load_bin_predictions(file.path(path_results, "zeta_ij.bin"),
                                  n = n_proposal, d = 4, m = 1)
theta_pred <- theta_pred[,,1]
zeta_pred <- zeta_pred[,,1]


tail(theta_pred)
tail(theta_posterior)

head(zeta_posterior)
head(zeta_pred)



# cpp_obj$ess_sir
x_posterior <- cpp_obj$x_posterior
ess_pmc <- cpp_obj$ess_sir
ess_pmc
effsize_sir1
tail(x_proposal)


x_posterior <- matrix(x_posterior, nrow = n_particles_x, ncol = ndpost)
par(mfrow = c(1, 2))
hist(x_posterior)
abline(v = x_true, col = "blue", lwd = 2)
hist(ess)



idx_sir1 <- cpp_obj$SIRZANIMLNBART(y_new, n_proposal, ndpost, B,
                                   path_results, 1, 0L)
effsize_sir1 <- cpp_obj$ess_sir
x_sir1 <- x_proposal[idx_sir1 + 1L, ]


par(mfrow = c(1, 2))
plot(density(x_posterior), col = "blue")
lines(density(x_sir1))
abline(v = x_true, col = "red", lwd = 2)

plot(density(effsize_sir1))
plot(density(ess), col = "red")


