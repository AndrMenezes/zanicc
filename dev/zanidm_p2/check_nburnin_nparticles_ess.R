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
n_test <- length(id_test)
Y_test <- list_data$Y[id_test, ]
X_test <- list_data$X[id_test, , drop = FALSE]
X_train <- list_data$X[-id_test, , drop = FALSE]
Z_test <- list_data$Z[id_test, , drop = FALSE]
d <- ncol(Y_test)

# Load forward model
zanim_ln_bart <- load_model(model_dir = path_results)

# Priors for the ESS
S_prior <- cov(X_train)
mean_prior <- colMeans(X_train)

i <- 1L
x_true <- X_test[i, ]
y_new <- Y_test[i,, drop = FALSE]

ndpost <- 1000L
parms <- expand.grid(nburnin = 1L, n_particles = c(100L, 1000L))

list_draws <- vector(mode = "list", length = nrow(parms))
# ESS
for (k in seq_len(nrow(parms))) {
  cat(k, "\n")
  ess <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                       Y = y_new, method = "ess",
                                       mean_prior = mean_prior,
                                       S_prior = S_prior,
                                       ndpost = ndpost,
                                       X_ini = matrix(x_true, ncol = 2),
                                       nburnin = parms$nburnin[k],
                                       n_particles = parms$n_particles[k])
  list_draws[[k]] <- ess[,,1L]
}
# saveRDS(object = list_draws, file = file.path(path_results, "draws_y1_prior.rds"))

# Plotting
x_proposal <- readRDS(file = file.path(path_results, "x_proposal.rds"))
x1range <- range(x_proposal[, 1])
x2range <- range(x_proposal[, 2])
dens_prior <- MASS::kde2d(x_proposal[, 1], x_proposal[, 2])

pdf(file.path(path_results, "ess_varying_particles_burnin.pdf"), width = 7, height = 4)
par(mar = c(4, 4, 1, 1), mfrow = c(1, 2))
for (k in seq_len(nrow(parms))) {
  dens_ess <- MASS::kde2d(list_draws[[k]][, 1L], list_draws[[k]][, 2L], n = 100)
  contour(dens_prior$x, dens_prior$y, dens_prior$z,
          col = scales::alpha("brown", 0.4),
          main = sprintf("nburnin=%i and n_particles=%i",
                         parms$nburnin[k], parms$n_particles[k]),
          xlim = range(x1range, list_draws[[k]][, 1]),
          ylim = range(x2range, list_draws[[k]][, 2]) )
  contour(dens_ess$x, dens_ess$y, dens_ess$z, add = TRUE)
  points(x_true[1], x_true[2], col = "blue", pch = 4, cex = 2)
  abline(v = x_true[1], h = x_true[2])
}
graphics.off()
