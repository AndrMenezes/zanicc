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

i <- 10L #2L #1L
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

# ndpost <- 10L
# n_particles_x <- 10L
# n_particles_l <- 1L
# range_prior <- range(list_data$X)
# cpp_obj$PopulationMC(y_new, ndpost, n_particles_x, n_particles_l, B, range_prior)


# ESS --------------------------------------------------------------------------

mu_prior <- mean(list_data$X)
S_prior <- as.matrix(var(list_data$X))
x_ess1 <- cpp_obj$ESSZANIMLNBART(matrix(y_new, nrow = 1), as.matrix(x_true), ndpost, 1L,
                                mu_prior, S_prior, B, 1L)
x_ess10 <- cpp_obj$ESSZANIMLNBART(matrix(y_new, nrow = 1), as.matrix(x_true), ndpost, 1L,
                                mu_prior, S_prior, B, 10L)
x_ess100 <- cpp_obj$ESSZANIMLNBART(matrix(y_new, nrow = 1), as.matrix(x_true), ndpost, 1L,
                                mu_prior, S_prior, B, 100L)
x_ess500 <- cpp_obj$ESSZANIMLNBART(matrix(y_new, nrow = 1), as.matrix(x_true), ndpost, 100L,
                                mu_prior, S_prior, B, 500L)
x_ess1000 <- cpp_obj$ESSZANIMLNBART(matrix(y_new, nrow = 1), as.matrix(x_true), ndpost, 1L,
                                mu_prior, S_prior, B, 1000L)
x_ess100100 <- cpp_obj$ESSZANIMLNBART(matrix(y_new, nrow = 1), as.matrix(x_true), ndpost, 100L,
                                mu_prior, S_prior, B, 100L)

# Compare only ESS versions with different n_particles
d_true <- density(x_truth_posterior[, i])
d_ess1 <- density(x_ess1)
d_ess10 <- density(x_ess10)
d_ess100 <- density(x_ess100)
d_ess500 <- density(x_ess500)
d_ess1000 <- density(x_ess1000)
d_ess100100 <- density(x_ess100100)
plot(d_true, ylim = range(d_true$y, d_ess10$y, d_ess100$y, d_ess500$y,  d_ess1000$y)
     , xlim = range(d_true$x, d_ess10$x, d_ess100$x, d_ess500$x, d_ess1000$x))
# lines(d_ess1, col = "brown")
 # lines(d_ess10, col = "red")
# lines(d_ess100, col = "red")
# lines(d_ess500, col = "green")
lines(d_ess1000, col = "gold")
lines(d_ess100100, col = "blue")
points(x_true, min(d_true$y), col = "blue", cex = 2, pch = 19)


# SIR --------------------------------------------------------------------------
idx_sir1 <- cpp_obj$SIRZANIMLNBART(y_new, n_proposal, ndpost, B,
                                    path_results, 1, 0L)
effsize_sir1 <- 1.0 / cpp_obj$ess_sir

idx_sir10 <- cpp_obj$SIRZANIMLNBART(y_new, n_proposal, ndpost, B,
                                    path_results, 10, 0L)
effsize_sir10 <- 1.0 / cpp_obj$ess_sir

# Using the mixture likelihood
idx_sir1_mix <- cpp_obj$SIRZANIMLNBART(y_new, n_proposal, ndpost, B,
                                    path_results, 1L, 1L)
effsize_sir1_mix <- 1.0 / cpp_obj$ess_sir

idx_sir10_mix <- cpp_obj$SIRZANIMLNBART(y_new, n_proposal, ndpost, B,
                                    path_results, 10, 1L)
effsize_sir10_mix <- 1.0 / cpp_obj$ess_sir


# idx_sir100 <- cpp_obj$SIRZANIMLNBART(y_new, n_proposal, ndpost, B,
#                                      path_results, 100)
# effsize_sir100 <- 1.0 / cpp_obj$ess_sir


x_sir1 <- x_proposal[idx_sir1 + 1L, ]
x_sir10 <- x_proposal[idx_sir10 + 1L, ]
x_sir1_mix <- x_proposal[idx_sir1_mix + 1L, ]
x_sir10_mix <- x_proposal[idx_sir10_mix + 1L, ]


# ESS according number of particles
cbind(
   np1 = quantile(effsize_sir1),
   np10 = quantile(effsize_sir10),
   np1_mix = quantile(effsize_sir1_mix),
   np10_mix = quantile(effsize_sir10_mix)
      # ,np100 = quantile(effsize_sir100)
      )

# Comparison
d_true <- density(x_truth_posterior[, i])
d_sir1 <- density(x_sir1)
d_sir10 <- density(x_sir10)
d_sir1_mix <- density(x_sir1_mix)
d_sir10_mix <- density(x_sir10_mix)
# d_sir100 <- density(x_sir100)
plot(d_true, ylim = range(d_true$y, d_sir10$y)
     , xlim = range(d_true$x, d_sir10$x))
lines(d_sir1, col = "red")
lines(d_sir10, col = "blue")
lines(d_sir1_mix, col = "gold")
lines(d_sir10_mix, col = "darkgreen")
points(x_true, min(d_true$y), col = "blue", cex = 2, pch = 19)

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


i <- 12L #2L #1L
y_new <- Y_test[i, ]
x_true <- X_test[i, ]

ndpost <- 5000L
n_particles_x <- 1000L
n_particles_l <- 2L
range_prior <- range(list_data$X)
cpp_obj$PopulationMC(y_new, ndpost, n_particles_x, n_particles_l, B, range_prior)

# cpp_obj$ess_sir
x_posterior <- cpp_obj$x_posterior

x_posterior <- matrix(x_posterior, nrow = n_particles_x, ncol = ndpost)
par(mfrow = c(1, 2))
hist(x_posterior)
abline(v = x_true)
hist(cpp_obj$ess_sir)
