rm(list = ls())
# Install package from the inverse_posterior branch
# remotes::install_github("andrmenezes/zanicc@inverse_posterior")
library(zanicc)

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


# Create splines basis
# X_train_basis <- cbind(1, X_train, X_train^2, X_train^3)
rangeX <- range(X_train)
df <- 6L
degree <- 3L
knots <- seq(rangeX[1], rangeX[2], length.out = df - degree - 1 + 2)
knots <- knots[1:(df-degree)]
X_train_basis <- splines::bs(X_train, df = df, degree = degree, intercept = FALSE,
                             knots = knots, Boundary.knots = rangeX)



# Using C++
# X_train_basis2 <- BayesComposition::bs_cpp(x = X_train, df = dof, interior_knots = knots,
#                                           degree = degree, intercept = FALSE,
#                                           Boundary_knots = rangeX)
# head(X_train_basis)
# head(X_train_basis2)


# Fit model
if (!file.exists(file.path(path_results, "dm_reg.rds"))) {
  mod <- zanicc(Y = Y_train, X_count = X_train_basis,
                model = "dm_reg", ndpost = 4000L, nskip = 4000L)
  save_model(object = mod, model_dir = path_results, file_name = "dm_reg.rds")
}
mod <- load_model(model_dir = path_results, file_name = "dm_reg.rds")

dim(mod$draws_betas)

Y_ppc <- ppd(mod, relative = FALSE)
plot_qqplots(Y = Y_train, Y_ppc = Y_ppc, relative = TRUE)
plot_ppc(Y = Y_train, Y_ppc = Y_ppc)

# Parameters for the splines
parms_bs <- list(df = df, degree = degree, intercept = FALSE, knots = knots,
                 Boundary.knots = rangeX)


###################################################################################
#' Get the DM concentration parameter
get_alpha <- function(x, betas, parms_bs) {
  # X_mat <- cbind(1, x, x^2, x^3)
  # X_mat <- predict(splines_obj, newx = x)
  X_mat <- splines::bs(x, df = parms_bs$df, degree = parms_bs$degree,
                       intercept = parms_bs$intercept,
                       knots = parms_bs$knots,
                       Boundary.knots = parms_bs$Boundary.knots)
  exp(drop(X_mat %*% betas))
}

#' Update x using ESS
udpate_ess_dm <- function(x, y, betas, sd_prior, mu_prior, parms_bs) {
  # Set log-likelihood threshold
  alpha <- get_alpha(x = x, betas = betas, parms_bs = parms_bs)
  lr <- log(stats::runif(1)) + zanicc:::log_pmf_dm(x = y, size = sum(y), alpha = alpha)
  # Draw angle
  nu <- stats::rnorm(1L, sd = sd_prior)
  angle <- stats::runif(1) * 2*pi;
  angle_max <- angle;
  angle_min <- angle - 2*pi;
  # Draw proposal
  x_proposal <- x * cos(angle) + nu * sin(angle)
  x_tilde <- x_proposal + mu_prior
  alpha_proposal <- get_alpha(x = x_tilde, betas = betas, parms_bs = parms_bs)
  counter <- 0L
  repeat {
    ll <- zanicc:::log_pmf_dm(x = y, size = sum(y), alpha = alpha_proposal)
    if (ll > lr) break
    if (counter > 100) {
      # cat("More than 100 slices, leaving the loop\n", ll, lr)
      break
    }
    if (angle < 0) angle_min <- angle
    else angle_max <- angle
    angle <- angle_min + (angle_max - angle_min) * stats::runif(1L)
    # Draw new proposal
    x_proposal <- x * cos(angle) + nu * sin(angle)
    x_tilde <- x_proposal + mu_prior
    # Compute alpha_proposal
    alpha_proposal <- get_alpha(x = x_tilde, betas = betas, parms_bs = parms_bs)
    counter <- counter + 1L
  }
  return(x_proposal)
}

# Normal prior
sd_prior <- sd(X_train)
mu_prior <- mean(X_train)

# udpate_ess_dm(x = 0.0, y = Y_test[1, ], betas = mod$draws_betas[,,1], sd_prior = sd_prior,
#               mu_prior = mu_prior, parms_bs = parms_bs)

ndpost <- mod$ndpost

# Keep the draws
x_draws <- matrix(nrow = n_test, ncol = ndpost)

# Start MCMC
for (i in seq_len(n_test)) {
  if (i %% 10 == 0) cat(i, "\n")
  y <- Y_test[i, ]
  # Initial value for x
  # x_cur <- X_ini[i, ]
  x_cur <- stats::rnorm(1L, mean = mu_prior, sd = sd_prior)
  for (k in seq_len(ndpost)) {
    # Load current model parameters
    betas <- mod$draws_betas[,,k]
    # Run ESS
    x_cur <- udpate_ess_dm(x = x_cur, y = y, betas = betas, sd_prior = sd_prior,
                           mu_prior = mu_prior, parms_bs = parms_bs)
    x_draws[i, k] <- x_cur + mu_prior
  }
}

plot(x_draws[1, ], type = "l")
plot(density(x_draws[1, ]))
points(X_test[1,], 0.001, col = "blue", pch = 4, cex = 2)

plot(x_draws[2, ], type = "l")
plot(density(x_draws[2, ]))
points(X_test[2,], 0.001, col = "blue", pch = 4, cex = 2)


# Re-arrange the format to use `zanicc` function for compute predictions
ip_dm_splines <- array(dim = c(ndpost, 1L, n_test))
for (i in seq_len(n_test)) ip_dm_splines[,1L,i] <- x_draws[i, ]

compute_prediction_metrics(x = X_test, draws = ip_dm_splines)

# # A tibble: 5 × 6
# method              mae  msep dmode coverage_95 coverage_50
# <chr>             <dbl> <dbl> <dbl>       <dbl>       <dbl>
# 1 sir_zanim_ln_bart 10.4  201.   348.        0.96        0.56
# 2 ess_zanim_ln      11.9  247.   341.        1           0.68
# 3 dm_gp              6.55  86.9  112.        0.97        0.58
# 4 zanim_bart        11.3  276.   332.        0.6         0.23
# 5 ml_bart           11.4  352.   352.        0.16        0.05
#      mae        msep       dmode coverage_95 coverage_50
# 8.706904  119.769922  139.166287    0.970000    0.400000
