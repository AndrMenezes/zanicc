test_that("comparison of models", {
  rm(list = ls()); gc()
  devtools::load_all()

  # library(fido)
  # library(zanicc)
  library(ggplot2)
  library(cowplot)

  # Plot theme
  theme_set(
    theme_cowplot(font_size = 16) +
      background_grid() +
      theme(text = element_text(size = 16), legend.position = "top")
  )

  set.seed(1212)
  # Simulate data ----------------------------------------------------------------
  a0 <- 0.01
  tau <- (1 - a0) / a0
  n_sample <- 400L
  n_trials <- sample(seq.int(100L, 500L), n_sample, replace = TRUE)
  d <- 4L
  dof_bs_theta <- 6L
  X <- as.matrix(seq(-1, 1, length.out = n_sample))
  X1_bs <- splines::bs(X, dof_bs_theta)
  betas_theta <- matrix(stats::rnorm(d * dof_bs_theta), dof_bs_theta, d)
  betas_theta[1L, ] <- betas_theta[1L, ] + seq(from = 1, to = 4, length.out = d)
  eta_theta <- X1_bs %*% betas_theta
  eta_zeta <- matrix(nrow = n_sample, ncol = d)
  intercept <- seq(0.5, 4.0, length.out = d)#c(0.5, 1.0, 1.5, 2.0)
  for (j in seq_len(d)) {
    eta_zeta[, j] <- sin(2 * pi * X[, 1L]) + X[, 1L]^2 - intercept[j]
  }
  true_zetas <- stats::pnorm(eta_zeta)
  true_alphas <- exp(eta_theta)
  Y <- Z <- true_thetas <- true_thetas_gamma <- true_varthetas <- matrix(nrow = n_sample, ncol = d)
  for (i in seq_len(n_sample)) {
    z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
    is_zero <- z == 0L
    p_ij <- true_alphas[i, ] / sum(true_alphas[i, ])
    g <- stats::rgamma(n = d, shape = tau * p_ij, rate = 1.0) # tau  *
    # g <- p_ij
    # g <- pmax(g, 1e-10)
    true_thetas[i, ] <- p_ij
    true_thetas_gamma[i, ] <- g / sum(g)
    true_varthetas[i, ] <- z * g / sum(z * g)
    if(any(is.na(true_varthetas[i, ]))) break
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
  colMeans(1 - Z)
  colMeans(Y == 0)
  any(is.na(true_varthetas))
  sum(is.na(true_varthetas))
  data_sim <- data.frame(id = rep(seq_len(n_sample), each = d),
                         category = rep(seq_len(d), times = n_sample),
                         x = rep(X[, 1L], each = d),
                         theta = c(t(true_thetas)),
                         theta_gamma = c(t(true_thetas_gamma)),
                         zeta = c(t(true_zetas)),
                         total = c(t(Y)), z = c(t(Z)),
                         prop = c(apply(Y, 1L, function(z) z/sum(z))))
  data_sim$category_lab <- paste0("j == ", data_sim$category)
  ggplot(data_sim, aes(x = x, y = theta)) + facet_wrap(~category) + geom_line()
  ggplot(data_sim, aes(x = x, y = theta_gamma)) + facet_wrap(~category) + geom_line()
  ggplot(data_sim, aes(x = x, y = zeta)) + facet_wrap(~category) + geom_line()


  # MCMC settings
  NDPOST <- 2000L
  NSKIP <- 2000L
  NTREES_THETA <- 20L
  NTREES_ZETA <- 20L
  Q_FACTORS <- .ledermann(q = d - 1)

  # Fitting the models

  # multinomial-BART
  mod_mult_bart <- MultinomialBART$new(Y = Y, X = X)
  mod_mult_bart$SetupMCMC(ntrees = NTREES_THETA, ndpost = NDPOST, nskip = NSKIP,
                          update_sigma = TRUE)
  mod_mult_bart$RunMCMC()
  # saveRDS(object = mod_mult_bart, file = file.path(path_res, "mult_bart.rds"))
  # mod_mult_bart <- readRDS(file = file.path(path_res, "mult_bart.rds"))

  # multinomial-LN-BART
  mod_mult_ln_bart <- MultinomialLNBART$new(Y = Y, X = X)
  mod_mult_ln_bart$SetupMCMC(ntrees = NTREES_THETA, ndpost = NDPOST, nskip = NSKIP,
                             update_sigma = TRUE, covariance_type = "wishart")
  mod_mult_ln_bart$RunMCMC()
  # saveRDS(object = mod_mult_bart, file = file.path(path_res, "mult_bart.rds"))
  # mod_mult_bart <- readRDS(file = file.path(path_res, "mult_bart.rds"))

  # ZANIM-BART
  mod_zanim_bart <- ZANIMBART$new(Y = Y, X_theta = X, X_zeta = X)
  mod_zanim_bart$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                           ndpost = NDPOST, nskip = NSKIP, update_sigma_theta = TRUE,
                           keep_draws = TRUE)
  mod_zanim_bart$RunMCMC()
  # saveRDS(object = mod_zanim_bart, file = file.path(path_res, "zanim_bart.rds"))
  # mod_zanim_bart <- readRDS(file = file.path(path_res, "zanim_bart.rds"))

  # ZANIM-LN-BART
  mod_zanim_ln_bart <- ZANIMLNBART$new(Y = Y, X_theta = X, X_zeta = X)
  mod_zanim_ln_bart$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                              ndpost = NDPOST, nskip = 2*NSKIP,
                              shape_lsphis = 2.0, a1_gs = 1.5, a2_gs = 2.8,
                              q_factors = Q_FACTORS, update_sigma_theta = TRUE,
                              keep_draws = TRUE, covariance_type = "wishart")
  mod_zanim_ln_bart$RunMCMC()
  # saveRDS(object = mod_zanim_ln_bart, file = file.path(path_res, "zanim_ln_bart.rds"))
  # mod_zanim_ln_bart <- readRDS(file = file.path(path_res, "zanim_ln_bart.rds"))

  # ZANIDM-reg
  Xwint <- cbind(1, X)
  mod_zanidm_reg <- ZANIDMRegression$new(Y = Y, X_alpha = Xwint, X_zeta = Xwint)
  mod_zanidm_reg$SetupMCMC(ndpost = NDPOST, nskip = 2*NSKIP,
                           sd_prior_beta_zeta = diag(1.0, ncol(Xwint)))
  mod_zanidm_reg$RunMCMC()
  # saveRDS(object = mod_zanidm_reg, file = file.path(path_res, "zanidm_reg.rds"))
  # mod_zanidm_reg <- readRDS(file = file.path(path_res, "zanidm_reg.rds"))


  X_t <- t(X)
  Y_t <- t(Y)
  # Specify Priors
  Gamma <- function(X) SE(X, sigma = 1) # Create partial function
  Theta <- function(X) matrix(0, d - 1, ncol(X_t))
  upsilon <- d - 1 + 3
  Xi <- matrix(.4, d - 1, d - 1)
  diag(Xi) <- 1

  # Now fit the model
  ini <- proc.time()
  fit <- fido::basset(Y_t, X_t, upsilon, Theta, Gamma, Xi, n_samples = NDPOST)
  draws_theta_fido <- to_proportions(fit)$Eta
  end <- proc.time() - ini

  # Swap to match my pattern
  draws_theta_fido <- aperm(draws_theta_fido, c(2, 1, 3))


  model_specs <- list(
    mult_bart = list(obj = "mod_mult_bart", theta = "draws_theta", vartheta = "draws_theta"),
    mult_ln_bart = list(obj = "mod_mult_ln_bart", theta = "draws_theta", vartheta = "draws_vartheta"),
    zanim_bart = list(obj = "mod_zanim_bart", theta = "draws_theta", vartheta = "draws_abundance"),
    zanim_ln_bart = list(obj = "mod_zanim_ln_bart", theta = "draws_theta", vartheta = "draws_abundance"),
    zanidm_reg = list(obj = "mod_zanidm_reg", theta = "draws_theta", vartheta = "draws_abundance"),
    mult_gp = list(obj = "mult_gp_silverman", theta = NULL, vartheta = NULL)
  )



  compute_block <- function(true_values, param_name, kl_fun) {

    sapply(names(model_specs), function(name) {

      spec <- model_specs[[name]]

      # special case for fido
      if (name == "mult_gp") {
        draws <- draws_theta_fido
      } else {
        model_obj <- get(spec$obj, envir = .GlobalEnv)
        draws <- model_obj[[ spec[[param_name]] ]]
      }

      if (is.null(draws)) {
        return(c(kl = NA_real_, frob = NA_real_))
      }

      c(
        kl   = mean(kl_fun(true_values, draws)),
        frob = mean(compute_frob(true_values, draws))
      )
    })
  }
  theta_res <- compute_block(true_thetas, "theta", compute_kl_simplex)
  vartheta_res <- compute_block(true_varthetas, "vartheta", compute_kl_simplex)

  mean(compute_frob(true_values = true_thetas, draws_theta_fido), type = "l")
  mean(compute_frob(true_values = true_varthetas, draws_theta_fido))

  par(mfrow = c(1, 3))
  plot(compute_frob(true_values = true_varthetas, draws_theta_fido), type = "l", ylim = c(1.1, 1.7))
  plot(compute_frob(true_values = true_varthetas, mod_mult_ln_bart$draws_vartheta), type = "l", ylim = c(1.1, 1.7))
  plot(compute_frob(true_values = true_varthetas, mod_zanim_ln_bart$draws_abundance), type = "l", ylim = c(1.1, 1.7))


  data_theta_zanim_bart <- .summarise_draws_3d(x = mod_zanim_bart$draws_theta)
  data_theta_zanim_ln_bart <- .summarise_draws_3d(x = mod_zanim_ln_bart$draws_theta)
  data_theta_mult_bart <- .summarise_draws_3d(x = mod_mult_bart$draws_theta)
  data_theta_mult_ln_bart <- .summarise_draws_3d(x = mod_mult_ln_bart$draws_theta)
  data_theta_mult_bart$x <- data_theta_mult_ln_bart$x <- data_theta_zanim_bart$x <- data_theta_zanim_ln_bart$x <- rep(c(X), times = d)

  data_theta <- dplyr::bind_rows(
    dplyr::mutate(data_theta_zanim_bart, model = "ZANIM-BART"),
    dplyr::mutate(data_theta_mult_bart, model = "multinomial-BART"),
    dplyr::mutate(data_theta_mult_ln_bart, model = "multinomial-LN-BART"),
    dplyr::mutate(data_theta_zanim_ln_bart, model = "ZANIM-LN-BART"))

  data_theta$category_lab <- paste0("j == ", data_theta$category)
  p_theta <- ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = theta_gamma, col = "Truth", fill = "Truth"), linewidth = 0.8) +
    facet_wrap(model~category_lab, labeller = label_parsed) +
    geom_rug(data = dplyr::filter(data_sim, total == 0L),
             mapping = aes(y = NA_real_, x = x)) +
    geom_line(data = data_theta, mapping = aes(x = x, y = median, col = model)) +
    geom_ribbon(data = data_theta,
                aes(x = x, ymin = ci_lower, ymax = ci_upper, fill = model),
                alpha = 0.3) +
    labs(y = latex2exp::TeX(r'(Count probabilities, $\theta_{ij}$)'),
         x = expression(x[i]), col = "", fill = "")







})
