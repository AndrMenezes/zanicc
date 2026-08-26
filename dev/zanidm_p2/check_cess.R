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
d <- ncol(Y_test)

# Load forward model
zanim_ln_bart <- load_model(model_dir = path_results)

# Priors for the ESS
S1_prior <- cov(X_train)
S2_prior <- 1.5*cov(X_train)
mean_prior <- colMeans(X_train)

i <- 1L
x_true <- X_test[i, ]
y_new <- Y_test[i,, drop = FALSE]

ndpost <- 1000L

# Get A and b for Ax + b >=0
hull <- geometry::convhulln(list_data$X, options = "Pp Fa Fx", output.options = "n")
normals <- hull$normals
A <- -normals[, -ncol(normals), drop = FALSE]
b <- -normals[, ncol(normals)]
cess1 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                      Y = y_new, method = "cess",
                                      mean_prior = mean_prior,
                                      S_prior = S1_prior,
                                      ndpost = ndpost,
                                      X_ini = matrix(x_true, ncol = 2),
                                      nburnin = 1,
                                      Amat = A, bvec = b, eta = 100L,
                                      n_particles = 100L)
cess2 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                      Y = y_new, method = "cess",
                                      mean_prior = mean_prior,
                                      S_prior = S2_prior,
                                      ndpost = ndpost,
                                      X_ini = matrix(x_true, ncol = 2),
                                      nburnin = 1,
                                      Amat = A, bvec = b, eta = 100L,
                                      n_particles = 100L)
# Adapt...
S3_prior <- 2.5^2/2*cov(cess1[,,1])
mean_prior3 <- colMeans(cess1[,,1])
cess3 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                      Y = y_new, method = "cess",
                                      mean_prior = mean_prior3,
                                      S_prior = S3_prior,
                                      ndpost = ndpost,
                                      X_ini = matrix(x_true, ncol = 2),
                                      nburnin = 1,
                                      Amat = A, bvec = b, eta = 100L,
                                      n_particles = 100L)
S4_prior <- 2.5^2/2*cov(cess2[-(1:500),,1])
mean_prior4 <- colMeans(cess2[-(1:500),,1])
cess4 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                      Y = y_new, method = "cess",
                                      mean_prior = mean_prior4,
                                      S_prior = S4_prior,
                                      ndpost = ndpost,
                                      X_ini = matrix(x_true, ncol = 2),
                                      nburnin = 1,
                                      Amat = A, bvec = b, eta = 100L,
                                      n_particles = 100L)

dens_data <- MASS::kde2d(list_data$X[,1], list_data$X[, 2])
dens1 <- MASS::kde2d(cess1[,1, ], cess1[, 2,])
dens2 <- MASS::kde2d(cess2[,1, ], cess2[, 2,])
dens3 <- MASS::kde2d(cess3[,1, ], cess3[, 2,])
dens4 <- MASS::kde2d(cess4[,1, ], cess4[, 2,])

par(mfrow = c(2, 2))

# contour(dens_data$x, dens_data$y, dens_data$z)
contour(dens1$x, dens1$y, dens1$z, col = "brown")
points(cess1[, 1,], cess1[, 2,])
points(x_true[1], x_true[2], pch = 19, col = "blue", cex = 3)
#
# contour(dens_data$x, dens_data$y, dens_data$z)
contour(dens2$x, dens2$y, dens2$z, col = "brown")
points(cess2[, 1,], cess2[, 2,])
points(x_true[1], x_true[2], pch = 19, col = "blue", cex = 3)

# contour(dens_data$x, dens_data$y, dens_data$z)
contour(dens3$x, dens3$y, dens3$z, col = "brown")
points(cess3[, 1,], cess3[, 2,])
points(x_true[1], x_true[2], pch = 19, col = "blue", cex = 3)

# contour(dens_data$x, dens_data$y, dens_data$z)
contour(dens4$x, dens4$y, dens4$z, col = "brown")
points(cess4[, 1,], cess4[, 2,])
points(x_true[1], x_true[2], pch = 19, col = "blue", cex = 3)


