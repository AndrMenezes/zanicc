rm(list = ls())
library(zanicc)
library(ggplot2)
library(cowplot)

# Plot theme
theme_set(
  theme_cowplot(font_size = 16) +
    background_grid() +
    theme(text = element_text(size = 16), legend.position = "top")
)

# Simulate data ----------------------------------------------------------------

n_sample <- 400L
n_trials <- sample(seq.int(100L, 500L), n_sample, replace = TRUE)
set.seed(1212)
d <- 4L
dof_bs_theta <- 6L
X <- as.matrix(seq(-1, 1, length.out = n_sample))
X1_bs <- splines::bs(X, dof_bs_theta)
betas_theta <- matrix(stats::rnorm(d * dof_bs_theta), dof_bs_theta, d)
betas_theta[1L, ] <- betas_theta[1L, ] - seq(from = 4, to = 0, length.out = d)
eta_theta <- X1_bs %*% betas_theta
eta_zeta <- matrix(nrow = n_sample, ncol = d)
intercept <- c(0.5, 1.0, 1.5, 2.0)
for (j in seq_len(d)) {
  eta_zeta[, j] <- sin(2 * pi * X[, 1L]) + X[, 1L]^2 - intercept[j]
}
true_zetas <- pnorm(eta_zeta)
alphas <- exp(eta_theta)
Y <- Z <- true_thetas <- true_varthetas <- matrix(nrow = n_sample, ncol = d)
for (i in seq_len(n_sample)) {
  z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
  is_zero <- z == 0L
  p_ij <- alphas[i, ] / sum(alphas[i, ])
  true_thetas[i, ] <- p_ij
  true_varthetas[i, ] <- z * p_ij / sum(z * p_ij)
  if (all(is_zero)) Y[i, ] <- rep(0L, d)
  else if (sum(is_zero) == d - 1L) {
    Y[i, ] <- rep(0L, d)
    Y[i, !is_zero] <- n_trials[i]
  } else {
    Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials[i],
                               prob = true_varthetas[i, ])
  }
  Z[i, ] <- z
}
data_sim <- data.frame(id = rep(seq_len(n_sample), each = d),
                       category = rep(seq_len(d), times = n_sample),
                       x = rep(X[, 1L], each = d),
                       theta = c(t(true_thetas)),
                       zeta = c(t(true_zetas)),
                       total = c(t(Y)), z = c(t(Z)),
                       prop = c(apply(Y, 1L, function(z) z/sum(z))))
data_sim$category_lab <- paste0("j == ", data_sim$category)
ggplot(data_sim, aes(x = x, y = theta)) + facet_wrap(~category) + geom_line()


# Fitting the models -----------------------------------------------------------


NTREES_THETA <- 200L
NTREES_ZETA <- 200L
NDPOST <- 6000L
NSKIP <- 6000L

# ZANIM-BART
mod_zanim <- ZANIMBART$new(Y = Y, X_theta = X, X_zeta = X)
mod_zanim$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                    ndpost = NDPOST, nskip = NSKIP, update_sigma_theta = TRUE,
                    keep_draws = TRUE)
mod_zanim$RunMCMC()

# Multinomial logistic BART
mod_mult <- MultinomialBART$new(Y = Y, X = X)
mod_mult$SetupMCMC(ntrees = NTREES_THETA,
                   ndpost = NDPOST, nskip = NSKIP, update_sigma = TRUE)
mod_mult$RunMCMC()

# ZANIDM-regression
Xwint <- cbind(1, X)
mod_zanidm <- ZANIDMLinearRegression$new(Y = Y, X_alpha = Xwint, X_zeta = Xwint)
mod_zanidm$SetupMCMC(ndpost = NDPOST, nskip = NSKIP)
mod_zanidm$RunMCMC()

# Elapsed time
mod_zanim$elapsed_time[3L]
mod_mult$elapsed_time[3L]
mod_zanidm$elapsed_time[3L]



# Plot compositional probabilities ---------------------------------------------

COLORS <- colorspace::qualitative_hcl(3, palette = "Dark 3")


data_theta_zanim_bart <- zanicc::.summarise_draws_3d(x = mod_zanim$draws_theta)
data_theta_mult_bart <- zanicc::.summarise_draws_3d(x = mod_mult$draws_theta)
data_theta_zanidm <- zanicc::.summarise_draws_3d(x = mod_zanidm$draws_theta)
data_theta_mult_bart$x <- data_theta_zanim_bart$x <- data_theta_zanidm$x <- rep(c(X), times = 4L)

data_theta <- dplyr::bind_rows(
  dplyr::mutate(data_theta_zanim_bart, model = "ZANIM-BART"),
  dplyr::mutate(data_theta_mult_bart, model = "multinomial-BART"),
  dplyr::mutate(data_theta_zanidm, model = "ZANIDM-reg"))

data_theta$category_lab <- paste0("j == ", data_theta$category)
p_theta <- ggplot(data = data_sim) +
  geom_line(mapping = aes(x = x, y = theta, col = "Truth", fill = "Truth"), linewidth = 0.8) +
  facet_wrap(~category_lab, labeller = label_parsed) +
  geom_rug(data = dplyr::filter(data_sim, total == 0L),
           mapping = aes(y = NA_real_, x = x)) +
  geom_line(data = data_theta, mapping = aes(x = x, y = median, col = model)) +
  geom_ribbon(data = data_theta,
              aes(x = x, ymin = ci_lower, ymax = ci_upper, fill = model),
              alpha = 0.3) +
  labs(y = latex2exp::TeX(r'(Count probabilities, $\theta_{ij}$)'),
       x = expression(x[i]), col = "", fill = "") +
  scale_color_manual(breaks = c("Truth", "multinomial-BART", "ZANIDM-reg", "ZANIM-BART"),
                     values = c("Truth" = "black",
                                "multinomial-BART" = COLORS[1L],
                                "ZANIDM-reg" = COLORS[2L],
                                "ZANIM-BART" = COLORS[3L])) +
  scale_fill_manual(breaks = c("Truth", "multinomial-BART", "ZANIDM-reg", "ZANIM-BART"),
                    values = c("Truth" = "black",
                               "multinomial-BART" = COLORS[1L],
                               "ZANIDM-reg" = COLORS[2L],
                               "ZANIM-BART" = COLORS[3L]))
p_theta
save_plot(filename = "count_prob__sim1.pdf", plot = p_theta,
          base_aspect_ratio = 1.537, base_height = 8)


# Plot structural zero probabilities -------------------------------------------

data_zeta_zanim_bart <- zanicc::.summarise_draws_3d(x = mod_zanim$draws_zeta)
data_zeta_zanidm <- zanicc::.summarise_draws_3d(x = mod_zanidm$draws_zeta)
data_zeta_zanidm$x <- data_zeta_zanim_bart$x <- rep(c(X), times = 4L)
data_zeta_zanim_bart$category_lab <- data_zeta_zanidm$category_lab <- paste0("j == ", data_zeta_zanim_bart$category)

data_zeta <- dplyr::bind_rows(
  dplyr::mutate(data_zeta_zanim_bart, model = "ZANIM-BART"),
  dplyr::mutate(data_zeta_zanidm, model = "ZANIDM-reg"))

p_zeta <- ggplot(data = data_sim) +
  geom_line(mapping = aes(x = x, y = zeta, col = "Truth", fill = "Truth"), linewidth = 0.8) +
  facet_wrap(~category_lab, labeller = label_parsed) +
  geom_rug(data = dplyr::filter(data_sim, total == 0L),
           mapping = aes(y = NA_real_, x = x)) +
  geom_line(data = data_zeta, mapping = aes(x = x, y = median, col = model)) +
  geom_ribbon(data = data_zeta, aes(x = x, ymin = ci_lower, ymax = ci_upper,
                                    fill = model), alpha = 0.3) +
  labs(y = latex2exp::TeX(r'(Zero-inflation probabilities, $\zeta_{ij}$)'),
       x = expression(x[i]), col = "", fill = "") +
  scale_color_manual(breaks = c("Truth", "ZANIDM-reg", "ZANIM-BART"),
                     values = c("Truth" = "black",
                                "ZANIDM-reg" = COLORS[2L],
                                "ZANIM-BART" = COLORS[3L])) +
  scale_fill_manual(breaks = c("Truth", "ZANIDM-reg", "ZANIM-BART"),
                    values = c("Truth" = "black",
                               "ZANIDM-reg" = COLORS[2L],
                               "ZANIM-BART" = COLORS[3L]))
p_zeta
save_plot(filename = "zero_inflation_prob__sim1.pdf",
          plot = p_zeta, base_aspect_ratio = 1.537,
          base_height = 8)
