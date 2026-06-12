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

#
devtools::load_all()
ess1 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                     Y = Y_test[1, ,drop=FALSE], method = "ess",
                                     mean_prior = mean_prior,
                                     S_prior = sd_prior,
                                     nadapt = 0,
                                     nburnin = 1L, n_particles = 100L)

ess2 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                     Y = Y_test[1, ,drop=FALSE], method = "ess",
                                     mean_prior = mean(ess1),
                                     S_prior = sd(ess1),
                                     nadapt = 0,
                                     nburnin = 1L, n_particles = 100L)
plot(density(ess1))
lines(density(ess2), col = "blue")
sd(ess2); sd(ess1)

ess3 <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                     Y = Y_test[1, ,drop=FALSE], method = "ess",
                                     mean_prior = mean_prior,
                                     S_prior = sd_prior,
                                     nadapt = 2500,
                                     nburnin = 1L, n_particles = 100L)
sd(ess1[1:2500])
mean(ess1)

mean(ess2[1:2500])
sd(ess2[1:2500])

mean(ess)
sd(ess)
cess <- inverse_posterior_zanimlnbart(object = zanim_ln_bart,
                                      Y = Y_test, method = "cess",
                                      mean_prior = mean_prior,
                                      S_prior = sd_prior,
                                      lower = range_prior[1L],
                                      upper = range_prior[2L],
                                      eta = 500L,
                                      nburnin = 1L, n_particles = 100L)

metrics_ess <- compute_prediction_metrics(X_test, ess)
metrics_cess <- compute_prediction_metrics(X_test, cess)



