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



# Using C++
# X_train_basis2 <- BayesComposition::bs_cpp(x = X_train, df = dof, interior_knots = knots,
#                                           degree = degree, intercept = FALSE,
#                                           Boundary_knots = rangeX)
# head(X_train_basis)
# head(X_train_basis2)


# Fit model
if (!file.exists(file.path(path_results, "zanim_ln_reg.rds"))) {
  mod <- zanicc(Y = Y_train, X_count = X_train_basis, X_zi = X_train_basis,
                model = "zanim_ln_reg", ndpost = 4000L, nskip = 4000L, keep_draws = TRUE)
  save_model(object = mod, model_dir = path_results, file_name = "zanim_ln_reg.rds")
}
mod <- load_model(model_dir = path_results, file_name = "zanim_ln_reg.rds")

# dim(mod$draws_betas_theta)
#
# Y_ppc <- ppd(mod, relative = FALSE)
# plot_qqplots(Y = Y_train, Y_ppc = Y_ppc, relative = TRUE)
# plot_ppc(Y = Y_train, Y_ppc = Y_ppc)
#
#
# data_theta <- summarise_draws_3d(x = mod$draws_theta)
# data_zeta <- summarise_draws_3d(x = mod$draws_zeta)
# data_theta$x1 <- data_zeta$x1 <- rep(c(X_train), times = d)
#
# p_theta <- ggplot(data = data_sim) +
#   geom_line(mapping = aes(x = x1, y = theta, col = "Truth", fill = "Truth"),
#             linewidth = 0.8) +
#   facet_wrap(~category, scales = "free_y") +
#   geom_rug(data = dplyr::filter(data_sim, total == 0L),
#            mapping = aes(y = NA_real_, x = x1)) +
#   geom_line(data = data_theta, mapping = aes(x = x1, y = median),
#             col = "dodgerblue") +
#   geom_ribbon(data = data_theta,
#               aes(x = x1, ymin = ci_lower, ymax = ci_upper), fill = "dodgerblue",
#               alpha = 0.3)
# cowplot::save_plot(filename = file.path(path_results, "posterior_theta__zanim_ln_reg.png"),
#                    plot = p_theta, bg = "white", base_height = 9)
# p_zeta <- ggplot(data = data_sim) +
#   geom_line(mapping = aes(x = x1, y = zeta, col = "Truth", fill = "Truth"),
#             linewidth = 0.8) +
#   facet_wrap(~category, labeller = label_parsed) +
#   # geom_rug(data = dplyr::filter(data_sim, total == 0L),
#   #          mapping = aes(y = NA_real_, x = x)) +
#   geom_line(data = data_zeta, mapping = aes(x = x1, y = median),
#             col = "dodgerblue") +
#   geom_ribbon(data = data_zeta,
#               aes(x = x1, ymin = ci_lower, ymax = ci_upper), fill = "dodgerblue",
#               alpha = 0.3)
# cowplot::save_plot(filename = file.path(path_results, "posterior_zeta__zanim_ln_reg.png"),
#                    plot = p_zeta, bg = "white", base_height = 9)

# Parameters for the splines
parms_bs <- list(df = df, degree = degree, intercept = FALSE, knots = knots,
                 Boundary.knots = rangeX)


###################################################################################
#' Get the DM concentration parameter
get_parms <- function(x, betas_alpha, betas_zeta, parms_bs) {
  X_mat <- splines::bs(x, df = parms_bs$df, degree = parms_bs$degree,
                       intercept = parms_bs$intercept,
                       knots = parms_bs$knots,
                       Boundary.knots = parms_bs$Boundary.knots)
  alpha <- exp(drop(X_mat %*% betas_alpha))
  zeta <- stats::pnorm(drop(X_mat %*% betas_zeta))
  list(alpha / sum(alpha), zeta)
}

# Monte Carlo approximation
log_pmf_zanim_ln_mc <- function(x, prob, zeta, chol_Sigma_V, B, mc = 100) {
  ll <- lapply(seq_len(mc), function(i) {
    log_pmf_zanim_ln_conditional(x = x, prob = prob, zeta = zeta,
                                 chol_Sigma_V = chol_Sigma_V, B = B)
  })
  ll <- unlist(ll)
  ll <- zanicc:::.log_sum_exp(ll) - log(mc)
  ll
}

log_pmf_zanim_ln_mc2 <- function(x, prob, zeta, chol_Sigma_V, Bt, mc = 100) {
  dm1 <- length(x) - 1L
  ll <- lapply(seq_len(mc), function(i) {
    v <- stats::rnorm(dm1) %*% chol_Sigma_V #draws_chol_Sigma_V[,,k]
    u <- drop(v %*% Bt)
    p <- prob * exp(u)
    log_pmf_zanim(x = x, prob = p / sum(p), zeta = zeta)
  })
  ll <- zanicc:::.log_sum_exp(unlist(ll)) - log(mc)
  if (is.na(ll)) return(-1000)
  ll
}

#' Update x using ESS
udpate_ess <- function(x, y, betas_alpha, betas_zeta, chol_Sigma_V, sd_prior, mu_prior,
                       parms_bs, B) {
  Bt <- t(B)
  # Set log-likelihood threshold
  tmp <- get_parms(x = x, betas_alpha = betas_alpha, betas_zeta = betas_zeta,
                   parms_bs = parms_bs)
  ll <- log_pmf_zanim_ln_mc2(x = y, prob = tmp[[1]], zeta = tmp[[2]],
                             chol_Sigma_V = chol_Sigma_V, Bt = Bt, mc = 100L)
  # log_pmf_zanim_ln_mc(x = y, prob = tmp[[1]], zeta = tmp[[2]],
  #                     chol_Sigma_V = c(chol_Sigma_V), B = B, mc = 100L)
  # ll <- log_pmf_zanim_ln_conditional(x = y, prob = tmp[[1]], zeta = tmp[[2]],
  #                                    chol_Sigma_V = c(chol_Sigma_V), B = B)
  lr <- log(stats::runif(1)) + ll
  # cat(lr, x, ll, "\n")
  # Draw angle
  nu <- stats::rnorm(1L, sd = sd_prior)
  angle <- stats::runif(1) * 2*pi
  angle_max <- angle
  angle_min <- angle - 2*pi
  # Draw proposal
  x_proposal <- x * cos(angle) + nu * sin(angle)
  x_tilde <- x_proposal + mu_prior
  tmp <- get_parms(x = x_tilde, betas_alpha = betas_alpha, betas_zeta = betas_zeta,
                   parms_bs = parms_bs)
  counter <- 0L
  repeat {
    ll <- log_pmf_zanim_ln_mc2(x = y, prob = tmp[[1]], zeta = tmp[[2]],
                               chol_Sigma_V = chol_Sigma_V, Bt = Bt, mc = 10L)
    # cat(lr, x, ll, "\n")
    # ll <- log_pmf_zanim_ln_conditional(x = y, prob = tmp[[1]], zeta = tmp[[2]],
    #                                    chol_Sigma_V = c(chol_Sigma_V), B = B)
    # ll <- log_pmf_zanim_ln_mc(x = y, prob = tmp[[1]], zeta = tmp[[2]],
    #                           chol_Sigma_V = c(chol_Sigma_V), B = B, mc = 100L)
    if (ll > lr) break
    if (abs(ll - lr) < 1e-5) break
    # if (counter > 100) {
    #   # cat("More than 100 slices, leaving the loop\n", ll, lr)
    #   break
    # }
    if (angle < 0) angle_min <- angle
    else angle_max <- angle
    angle <- angle_min + (angle_max - angle_min) * stats::runif(1L)
    # Draw new proposal
    x_proposal <- x * cos(angle) + nu * sin(angle)
    x_tilde <- x_proposal + mu_prior
    # Compute alpha_proposal Compute BART predictions for the new proposal
    tmp <- get_parms(x = x_tilde, betas_alpha = betas_alpha,
                     betas_zeta = betas_zeta, parms_bs = parms_bs)
    counter <- counter + 1L
  }
  return(x_proposal)
}

# Normal prior
sd_prior <- sd(X_train)
mu_prior <- mean(X_train)

# Contrast matrix
Bt <- mod$Bt
B <- t(Bt)

# Test
udpate_ess(x = 0.0, y = Y_test[1, ], betas_alpha = mod$draws_betas_theta[,,1],
           betas_zeta = mod$draws_betas_zeta[,,1],
           chol_Sigma_V = mod$draws_chol_Sigma_V[,,1],
           sd_prior = sd_prior,
           mu_prior = mu_prior, parms_bs = parms_bs, B = B)



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
    betas_alpha <- mod$draws_betas_theta[,,k]
    betas_zeta <- mod$draws_betas_zeta[,,k]
    chol_Sigma_V <- mod$draws_chol_Sigma_V[,,k]
    # Run ESS
    x_cur <- udpate_ess(x = x_cur, y = y, betas_alpha = betas_alpha,
                        betas_zeta = betas_zeta, chol_Sigma_V = chol_Sigma_V,
                        sd_prior = sd_prior,
                        mu_prior = mu_prior, parms_bs = parms_bs, B = B)
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
ip_zanim_ln_splines <- array(dim = c(ndpost, 1L, n_test))
for (i in seq_len(n_test)) ip_zanim_ln_splines[,1L,i] <- x_draws[i, ]

compute_prediction_metrics(x = X_test, draws = ip_zanim_ln_splines)

saveRDS(object = ip_zanim_ln_splines,
        file = file.path(path_results, "ip_zanim_ln_splines_marginal_mc2.rds"))

####################################################################################

# Load the marginal and conditional
ip_zanim_ln_splines_cond <- readRDS(file = file.path(path_results,
                                                     "ip_zanim_ln_splines_conditional.rds"))
ip_zanim_ln_splines_marg <- readRDS(file = file.path(path_results,
                                                     "ip_zanim_ln_splines_marginal_mc.rds"))
ip_zanim_ln_splines_marg2 <- readRDS(file = file.path(path_results,
                                                      "ip_zanim_ln_splines_marginal_mc2.rds"))

marginal <- compute_prediction_metrics(x = X_test, draws = ip_zanim_ln_splines_marg)
marginal2 <- compute_prediction_metrics(x = X_test, draws = ip_zanim_ln_splines_marg2)
conditional <- compute_prediction_metrics(x = X_test, draws = ip_zanim_ln_splines_cond)
rbind(marginal, marginal2, conditional)
