rm(list = ls())

devtools::load_all()
# library(zanicc)
library(ggplot2)

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
data_sim <- list_data$df[!(list_data$df$id %in% id_test), ]
data_sim$id <- rep(seq_len(nrow(Y_train)), each = d)

# Create splines basis
# X_train_basis <- cbind(1, X_train, X_train^2, X_train^3)
rangeX <- range(X_train)
df <- 6L
degree <- 3L
knots <- seq(rangeX[1], rangeX[2], length.out = df - degree - 1 + 2)
knots <- knots[1:(df - degree)]
X_train_basis <- splines::bs(X_train, df = df, degree = degree, intercept = FALSE,
                             knots = knots, Boundary.knots = rangeX)

# Fit model
if (!file.exists(file.path(path_results, "zanim_ln_reg.rds"))) {
  mod <- zanicc(Y = Y_train, X_count = X_train_basis, X_zi = X_train_basis,
                model = "zanim_ln_reg", ndpost = 4000L, nskip = 4000L, keep_draws = TRUE)
  save_model(object = mod, model_dir = path_results, file_name = "zanim_ln_reg.rds")
}
mod <- load_model(model_dir = path_results, file_name = "zanim_ln_reg.rds")

# Parameters for the splines
parms_bs <- list(df = df, degree = degree, intercept = FALSE, knots = knots,
                 Boundary.knots = rangeX)



# Utils functions --------------------------------------------------------------

#' Get the ZANIM-LN parameters (this is like a "predict"-type function)
get_parms <- function(x, betas_theta, betas_zeta, parms_bs) {
  X_mat <- splines::bs(x, df = parms_bs$df, degree = parms_bs$degree,
                       intercept = parms_bs$intercept,
                       knots = parms_bs$knots,
                       Boundary.knots = parms_bs$Boundary.knots)
  alpha <- exp(drop(X_mat %*% betas_theta))
  zeta <- stats::pnorm(drop(X_mat %*% betas_zeta))
  list(alpha / sum(alpha), zeta)
}
#' Simulate from ZANIM-LN
rzanimln <- function(n_trial, prob, zeta, chol_Sigma_V, Bt) {
  d <- length(prob)
  dm1 <- d - 1L
  v <- stats::rnorm(dm1) %*% chol_Sigma_V
  u <- drop(v %*% Bt)
  z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - zeta)
  if (all(z == 0L)) return(rep(0L, d))
  p <- z * prob * exp(u)
  drop(stats::rmultinom(n = 1L, size = n_trial, prob = p / sum(p)))
}
#' Summary statistics
s_stats <- function(y) {
  n <- sum(y)
  if (n == 0) return(rep(0.0, length(y)))
  y / n
}
#' Kernel statistics
log_k_gauss <- function(s_obs, s_prop, h) {
  # Euclidean distance between the summary statistics
  u <- sqrt(sum((s_obs - s_prop)^2))
  # log-Gaussian kernel
  return(-0.5 / h^2 * u^2)
  # stats::dnorm(u, mean = 0.0, sd = h, log = TRUE)
}


# ABC-IS -----------------------------------------------------------------------

# Uniform prior, proposal = prior
N_proposal <- 2000L
x_proposal <- seq(rangeX[1], rangeX[2], length.out = N_proposal)

run_abc_is <- function(y_obs, x_proposal, betas_theta, betas_zeta,
                       chol_Sigma_V, Bt, parms_bs, h = 0.01) {
  n_proposal <- length(x_proposal)
  s_obs <- s_stats(y_obs)
  n_trial <- sum(y_obs)
  log_w <- lapply(seq_len(n_proposal), function(i) {
    parm <- get_parms(x = x_proposal[i], betas_theta = betas_theta,
                      betas_zeta = betas_zeta, parms_bs = parms_bs)
    y_prop <- rzanimln(n_trial = n_trial, prob = parm[[1]], zeta = parm[[2]],
                       chol_Sigma_V = chol_Sigma_V, Bt = Bt)
    # Compute summary statistics
    s_prop <- s_stats(y_prop)
    # Return weights
    lk <- log_k_gauss(s_obs = s_obs, s_prop = s_prop, h = h)
    if (is.na(lk)) cat(y_prop, s_prop, parm[[2]], "\n")
    lk
  })
  unlist(log_w)
}


i <- 1L
x_true <- X_test[i,]
y_obs <- Y_test[i, ]

# Run for a given posterior draw
log_weights <- run_abc_is(y_obs = y_obs, x_proposal = x_proposal,
                          betas_theta = mod$draws_betas_theta[,,1],
                          betas_zeta = mod$draws_betas_zeta[,,1],
                          chol_Sigma_V = mod$draws_chol_Sigma_V[,,1],
                          Bt = mod$Bt, parms_bs = parms_bs, h = 0.01)
weights <- exp(log_weights - max(log_weights))
weights <- weights / sum(weights)
plot(x_proposal, weights, type = "h")

ndpost <- mod$ndpost

# Run ABC-SIR
indices_sir <- integer(length = ndpost)
for (k in seq_len(ndpost)) {
  cat(k, "\n")
  log_weights <- run_abc_is(y_obs = y_obs, x_proposal = x_proposal,
                            betas_theta = mod$draws_betas_theta[,,k],
                            betas_zeta = mod$draws_betas_zeta[,,k],
                            chol_Sigma_V = mod$draws_chol_Sigma_V[,,k],
                            Bt = mod$Bt, parms_bs = parms_bs, h = 0.01)
  weights <- exp(log_weights - max(log_weights))
  weights <- weights / sum(weights)
  # Resampling
  indices_sir[k] <- sample.int(N_proposal, size = 1, prob = weights)
}

x_sir <- x_proposal[indices_sir]
plot(density(x_sir))
hist(x_sir)
abline(v = x_true)
