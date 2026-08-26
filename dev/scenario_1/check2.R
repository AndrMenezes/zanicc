rm(list = ls())
library(ggplot2)
devtools::load_all()

args <- commandArgs(trailingOnly = TRUE)
d <- as.integer(args[1L])
replica <- as.integer(args[2L])

d <- 4L
replica <- 1L

# Path
path_local <- sprintf("./dev/scenario_1/d=%i", d)
path_data <- file.path(path_local, "data")
path_results <- file.path(path_local, "results")
forests_dir <- file.path(path_results, "forests")
if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

# Import data
list_data <- readRDS(file.path(path_data, sprintf("data_replica=%i.rds", replica)))
id_test <- list_data$id_test
n_test <- length(id_test)
Y_test <- list_data$Y[id_test, ]
X_test <- list_data$X[id_test, , drop = FALSE]

# Priors
range_prior <- range(list_data$X)
mean_prior <- mean(list_data$X)
S_prior <- var(list_data$X)
sd_prior <- sd(list_data$X)

# Load forward model
zanim_ln_bart <- load_model(model_dir = path_results)
zanim_ln_bart$forests_dir <- forests_dir

###################################################################################
devtools::load_all()
ess1 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                      Y = Y_test[3, ,drop=FALSE], method = "ess",
                                      mean_prior = mean_prior,
                                      S_prior = sd_prior,
                                      nadapt = 0,
                                      nburnin = 1L, n_particles = 100L)

ess2 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                      Y = Y_test[3, ,drop=FALSE], method = "ess",
                                      mean_prior = mean(ess1),
                                      S_prior = sd(ess1),
                                      nadapt = 0,
                                      nburnin = 1L, n_particles = 100L)
ess3 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                      Y = Y_test[3, ,drop=FALSE], method = "ess",
                                      mean_prior = X_test[3, ],
                                      S_prior = 10.0,
                                      nadapt = 0,
                                      nburnin = 1L, n_particles = 100L)
mean(ess3)
plot(density(ess2), col = "blue")
lines(density(ess1))
# lines(density(ess3), col = "red")
abline(v=X_test[3, ])
plot(ess3)

cbind(true = X_test[3, ], mu1 = mean(ess1), mu2 = mean(ess2), sd1 = sd(ess1),
      sd2 = sd(ess2))

sd(ess2); sd(ess1)
mean(ess2); mean(ess1)

ess3 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                      Y = Y_test[1, ,drop=FALSE], method = "ess",
                                      mean_prior = mean_prior,
                                      S_prior = sd_prior,
                                      nadapt = 2500,
                                      nburnin = 1L, n_particles = 100L)

sds <- c(sd(ess1), sd(ess2), sd(ess3))

sd(ess1[1:2500])
mean(ess1)

mean(ess2[1:2500])
sd(ess2[1:2500])

mean(ess)
sd(ess)

####################
ess1 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                      Y = Y_test, method = "ess",
                                      mean_prior = mean_prior,
                                      S_prior = sd_prior,
                                      nadapt = 0,
                                      nburnin = 1L, n_particles = 100L)
ess2 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                      Y = Y_test, method = "ess",
                                      mean_prior = mean_prior,
                                      S_prior = sd_prior,
                                      nadapt = 2000L,
                                      nburnin = 1L, n_particles = 100L)

cess <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                      Y = Y_test, method = "cess",
                                      mean_prior = mean_prior,
                                      S_prior = sd_prior,
                                      lower = range_prior[1L],
                                      upper = range_prior[2L],
                                      eta = 500L,
                                      nburnin = 1L, n_particles = 100L)
dim(ess1)
metrics_ess1 <- compute_prediction_metrics(X_test, ess1)
metrics_ess2 <- compute_prediction_metrics(X_test, ess2)

metrics_cess <- compute_prediction_metrics(X_test, cess)


post <- cbind(
  x_true = X_test[, 1],
  mu_ess1 = apply(ess1, 3, mean), mu_ess2 = apply(ess2, 3, mean),
  sd_ess1 = apply(ess1, 3, sd), sd_ess2 = apply(ess2, 3, sd))
head(post)




