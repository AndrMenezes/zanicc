rm(list = ls())
devtools::load_all()

d <- 4L
n_sample <- 1000L

# Path
path_local <- "./tests/testthat/inverse_posterior/zanim_ln_bart/one_dimension"
path_results <- file.path(path_local, sprintf("d=%i", d), "results")
forests_dir <- file.path(path_results, "forests")
if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)


list_data <- readRDS(file = file.path(path_results, "data.rds"))
# Split the data
set.seed(1212)
n_test <- 100L
id_test <- sample.int(n_sample, n_test)
Y_test <- list_data$Y[id_test, ]
X_test <- list_data$X[id_test, , drop = FALSE]

# Load
zanim_ln_bart <- load_model(model_dir = path_results)

####
mean_prior = mean(list_data$X)
S_prior = diag(var(list_data$X[, 1]), nrow = 1)
# Initialise C++ class
devtools::load_all()
ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
cpp_obj <- new(ml$InversePosterior, zanim_ln_bart$d, zanim_ln_bart$ntrees_theta,
               zanim_ln_bart$ntrees_zeta, zanim_ln_bart$forests_dir)

head(Y_test)
nburnin <- 100L
B <- t(zanim_ln_bart$Bt)
ndpost <- zanim_ln_bart$ndpost
i <- 1L
X_ini <- matrix(nrow = 1, ncol = 1)
X_ini[1, ] <- stats::rnorm(n = 1, mean = mean_prior, sd = sd(list_data$X))
ess <- cpp_obj$ESSZANIMLNBART(Y_test[i,, drop = FALSE],
                              X_test[i, ,drop=FALSE],
                              zanim_ln_bart$ndpost, nburnin, mean_prior,
                              S_prior, B, 1)

x_proposal <- readRDS(file.path(path_results, "x_proposal.rds"))
n_proposal <- length(x_proposal)
sir_abc <- cpp_obj$ABCSIRZANIMLNBART(Y_test[i, ], n_proposal, ndpost, B,
                                     path_results, .001)
effsize <- 1.0 / cpp_obj$ess_sir
hist(effsize)



