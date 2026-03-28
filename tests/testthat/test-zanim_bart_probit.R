library(ggplot2)
library(cowplot)

test_that("bspline probit", {
  rm(list = ls())
  devtools::load_all()
  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/zanim_probit", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate data
  set.seed(1212)
  n_trials <- 100L
  n_sample <- 400L
  d <- 4L
  tmp <- sim_data_zanim_bspline_curve(n = n_sample, d = d, n_trials = n_trials,
                                      link_zeta = "probit")
  X <- tmp$X
  Y <- tmp$Y
  data_sim <- tmp$df
  colMeans(Y == 0)
  colMeans(1 - tmp$Z)


  mod <- ZANIMBART$new(Y = Y, X_theta = X, X_zeta = X, link_zeta = "probit",
                       shared_trees = FALSE)
  mod$SetupMCMC(v0_theta = 3.5 / sqrt(2), ntrees_theta = 20L, ntrees_zeta = 20L,
                ndpost = 2000L, nskip = 200L, update_sigma_theta = TRUE,
                path = path_res, keep_draws = TRUE)#tempdir())
  mod$RunMCMC()
  mod$elapsed_time

  hist(mod$cpp_obj$sigma_mult_mcmc)

  # dim(out$draws_phi)
  par(mfrow = c(3, 3), mar = c(4, 4, 2, 1))
  for (i in sample.int(n_sample, 9)) {
    x <- mod$draws_phi[i, ]
    ess <- coda::effectiveSize(coda::mcmc(x))
    plot(x, type = "l", ylab = expression(phi[i]), main = paste0("ESS ", round(ess, 2)))
  }

  posterior_thetas <- mod$draws_theta
  dim(posterior_thetas)
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1))
  for (j in seq_len(d)) {
    plot(tmp$theta[, j], rowMeans(posterior_thetas[, j, ]), xlab = "true", ylab = "estimate")
    abline(0, 1, col = "red")
  }

  dim(tmp$theta)
  ndpost <- dim(posterior_thetas)[3]
  true_thetas <- tmp$theta
  kl_ <- numeric(ndpost)
  for (i in seq_len(ndpost)) {
    if (i %% 100 == 0) cat(i, "\n")
    kl_[i] = mean(rowSums(true_thetas * log(true_thetas / mod$draws_theta[,,i])))
  }
  coda::effectiveSize(coda::as.mcmc(kl_))
  graphics.off()
  plot(kl_, type = "l")

  # Average number of leaves
  mod$avg_leaves_theta
  mod$avg_leaves_zeta

  # Acceptance rate
  mod$accept_rate_theta
  mod$accept_rate_zeta

  mod$ComputePredictions(X = head(X), parameter = "theta")
  mod$ComputePredictions(X = head(X), parameter = "zeta")
  d_theta <- mod$LoadPredictions(parameter = "theta", ndpost = 2000L)
  d_zeta <- mod$LoadPredictions(parameter = "zeta", ndpost = 2000L)

  dim(mod$draws_theta[1:6,,])
  dim(d_theta)
  rowSums(d_theta[,,1L])
  rowSums(mod$draws_theta[1:6,,1L])

  all.equal(d_theta, mod$draws_theta[1:6,,])
  all.equal(d_zeta, mod$draws_zeta[1:6,,])


  X_pred <- matrix(seq(min(X), max(X), length.out = 100L))
  mod$ComputePredictions(X = X_pred, parameter = "theta")
  mod$ComputePredictions(X = X_pred, parameter = "zeta")
  expect_equal(mod$ndpost_pred, mod$ndpost)

  d_theta <- mod$LoadPredictions(parameter = "theta", ndpost = 2000L)
  d_zeta <- mod$LoadPredictions(parameter = "zeta", ndpost = 2000L)
  # expect_equal(dim(d_zeta), c(nrow(X), 4L, 500L))
  # expect_equal(dim(d_theta), c(nrow(X), 4L, 500L))
  range(d_theta)
  range(data_sim$theta)
  range(d_zeta)
  range(data_sim$zeta)

  data_theta <- .summarise_draws_3d(x = d_theta)
  data_theta$x <- rep(c(X_pred), times = 4L)
  head(data_theta)
  range(data_theta$mean)
  range(data_sim$theta)

  # Plotting
  p_theta <- ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = theta), linewidth = 0.4) +
    facet_wrap(~category, scales = "free_y") +
    geom_line(data = data_theta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = data_theta, aes(x = x, ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3)
  p_theta
  save_plot(filename = file.path(path_res, "theta.png"), plot = p_theta,
            bg = "white", base_height = 7.0)

  data_zeta <- .summarise_draws_3d(x = d_zeta)
  data_zeta$x <- rep(c(X_pred), times = 4L)
  head(data_zeta)

  p_zeta <- ggplot(data = data_sim) +
    geom_line(data = data_sim, mapping = aes(x = x1, y = zeta), linewidth = 0.4) +
    geom_line(data = data_zeta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = data_zeta, aes(x = x, ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3) +
    facet_wrap(~category, scales = "free_y")
  p_zeta
  save_plot(filename = file.path(path_res, "zeta.png"), plot = p_zeta,
            bg = "white", base_height = 7.0)

  y_ppc1 <- mod$GetPosteriorPredictive(in_sample = TRUE, conditional_rf = TRUE, ndpost = 1000L)
  unique(rowSums(y_ppc1[1,,]))
  y_ppc2 <- mod$GetPosteriorPredictive(in_sample = TRUE, conditional_rf = FALSE, ndpost = 1000L, relative = FALSE)

  # Average total zero
  total_zero <- mean(rowSums(Y == 0))
  total_zero_ppc <- apply(y_ppc2, c(1, 2), function(x) sum(x == 0))
  total_zero_ppc <- apply(total_zero_ppc, 1, mean)

  hist(total_zero_ppc); abline(v=total_zero, col = "red")

  dim(y_ppc2)

})


test_that("ZANIM-BART with sampling zeros", {
  rm(list = ls())
  library(ggplot2)
  devtools::load_all()

  # Simulate data
  n_sample <- 1000L
  n_trials <- sample(seq.int(100L, 500L), n_sample, replace = TRUE)
  set.seed(1212)
  d <- 4L
  dof_bs_theta <- 6L
  X <- as.matrix(seq(-1, 1, length.out = n_sample))
  X1_bs <- splines::bs(X, dof_bs_theta)
  betas_theta <- matrix(stats::rnorm(d * dof_bs_theta), dof_bs_theta, d)
  betas_theta[1L, ] <- betas_theta[1L, ] - seq(from = 3, to = 0, length.out = d)
  betas_theta[, 1] <- c(2, -1.0, -2, -3, -4, -5)
  betas_theta[, 2] <- c(-3, 1.0, 2, 3, 4, 5)
  eta_theta <- X1_bs %*% betas_theta
  eta_zeta <- matrix(nrow = n_sample, ncol = d)
  intercept <- c(1.0, 1.5, 1.75, 2.0)
  for (j in seq_len(d)) {
    eta_zeta[, j] <- sin(2 * pi * X[, 1L]) + X[, 1L]^2 - intercept[j]
  }
  true_zetas <- stats::pnorm(eta_zeta)
  alphas <- exp(eta_theta)
  Y <- Z <- true_thetas <- true_varthetas <- matrix(nrow = n_sample, ncol = d)
  for (i in seq_len(n_sample)) {
    z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
    is_zero <- z == 0L
    # Hack to avoid all zeros (it happen very rarely)
    while (all(is_zero)) {
      z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
      is_zero <- z == 0L
    }
    p_ij <- alphas[i, ] / sum(alphas[i, ])
    true_thetas[i, ] <- p_ij
    true_varthetas[i, ] <- z * p_ij / sum(z * p_ij)
    # if (all(is_zero)) {Y[i, ] <- rep(0L, d); true_varthetas[i, ] <- 0.0}
    if (sum(is_zero) == d - 1L) {
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
  ggplot(data_sim, aes(x = x, y = zeta)) + facet_wrap(~category) + geom_line()
  ggplot(data_sim, aes(x = x, y = prop)) + facet_wrap(~category) +
    geom_point() + geom_line(aes(y = theta)) +
    geom_point(data = data_sim[data_sim$total==0,], aes(col = z == 0))

  ggplot(data_sim, aes(x = x, y = zeta)) + facet_wrap(~category) + geom_line() +
    geom_point(data = data_sim[data_sim$z==0,], aes(x=x, y=prop))

  any(rowSums(Y) == 0)
  table(rowSums(Y) == 0)
  cbind(all_zeros = colMeans(Y == 0), structural_zeros = colMeans(1 - Z)
        , sampling_zeros = colMeans(Y == 0) - colMeans(1 - Z))
  # devtools::load_all()
  zanim_bart <- zanicc(Y = Y, X_count = X, X_zi = X, model = "zanim_bart",
                       ntrees_theta = 100, ntrees_zeta = 20, ndpost = 5000,
                       nskip = 5000)

  data_zeta_zanim_bart_1 <- zanicc::summarise_draws_3d(x = zanim_bart$draws_zeta)
  data_zeta_zanim_bart_2 <- zanicc::summarise_draws_3d(x = 1.0-zanim_bart$cpp_obj$draws_prob_0)
  data_zeta_zanim_bart_1$x <- rep(c(X), times = d)
  data_zeta_zanim_bart_2$x <- rep(c(X), times = d)
  data_zeta_zanim_bart_1$category_lab <- paste0("j == ", data_zeta_zanim_bart_1$category)
  data_zeta_zanim_bart_2$category_lab <- paste0("j == ", data_zeta_zanim_bart_1$category)

  data_zeta <- dplyr::bind_rows(
    dplyr::mutate(data_zeta_zanim_bart_1, model = "ZANIM-BART-zeta"),
    dplyr::mutate(data_zeta_zanim_bart_2, model = "ZANIM-BART-pi")
    )

  p_zeta <- ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = zeta, col = "Truth", fill = "Truth"), linewidth = 0.8) +
    facet_wrap(~category_lab, labeller = label_parsed) +
    geom_rug(data = dplyr::filter(data_sim, total == 0L, z == 0),
             mapping = aes(y = NA_real_, x = x), col = "red") +
    geom_rug(data = dplyr::filter(data_sim, total == 0L, z == 1),
             mapping = aes(y = NA_real_, x = x), col = "black") +
    geom_line(data = data_zeta, mapping = aes(x = x, y = median, col = model)) +
    geom_ribbon(data = data_zeta, aes(x = x, ymin = ci_lower, ymax = ci_upper,
                                      fill = model), alpha = 0.3) +
    labs(y = latex2exp::TeX(r'(Zero-inflation probabilities, $\zeta_{ij}$)'),
         x = expression(x[i]), col = "", fill = "")
  p_zeta

  # dim(zanim_bart$cpp_obj$draws_prob_0)
  # 1.0 - zanim_bart$cpp_obj$draws_prob_0[1:10,1,1]
  # Y[1:10,1,drop = FALSE]
  # Z[1:10,1,drop = FALSE]

  p_mean <- apply(1.0 - zanim_bart$cpp_obj$draws_prob_0, c(1, 2), mean)
  # p_mean <- apply(zanim_bart$draws_zeta, c(1, 2), mean)
  # j=1
  list_tabs <- list()
  for (j in seq_len(d)) {
    tab <- matrix(0.0, nrow = 2, ncol = 3)
    non_zero <- Y[, j] > 0
    sampling_zero <- (Y[, j] == 0) & (Z[, j] == 1)
    structural_zero <- (Y[, j] == 0) & (Z[, j] == 0)
    tab[1, 1] <- mean((Z[non_zero, j] == 1) == (p_mean[non_zero, j] < 0.5))
    tab[2, 1] <- mean((Z[non_zero, j] == 1) == (p_mean[non_zero, j] > 0.5))

    tab[1, 2] <- mean((Z[sampling_zero, j] == 1) == (p_mean[sampling_zero, j] < 0.5))
    tab[2, 2] <- mean((Z[sampling_zero, j] == 1) == (p_mean[sampling_zero, j] > 0.5))

    tab[1, 3] <- mean((Z[structural_zero, j] == 0) == (p_mean[structural_zero, j] < 0.5))
    tab[2, 3] <- mean((Z[structural_zero, j] == 0) == (p_mean[structural_zero, j] > 0.5))
    rownames(tab) <- c("p <= 0.5", "p > 0.5")
    colnames(tab) <- c("non-zero", "sampling-zero", "structural-zero")
    list_tabs[[j]] <- tab
  }
  list_tabs
})


test_that("test predict and ppd methods", {
  rm(list = ls())
  library(ggplot2)
  devtools::load_all()

  # Path
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/zanim_bart", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)


  # Simulate data
  n_sample <- 1000L
  n_trials <- sample(seq.int(100L, 500L), n_sample, replace = TRUE)
  set.seed(1212)
  d <- 4L
  dof_bs_theta <- 6L
  X <- as.matrix(seq(-1, 1, length.out = n_sample))
  X1_bs <- splines::bs(X, dof_bs_theta)
  betas_theta <- matrix(stats::rnorm(d * dof_bs_theta), dof_bs_theta, d)
  betas_theta[1L, ] <- betas_theta[1L, ] - seq(from = 3, to = 0, length.out = d)
  betas_theta[, 1] <- c(2, -1.0, -2, -3, -4, -5)
  betas_theta[, 2] <- c(-3, 1.0, 2, 3, 4, 5)
  eta_theta <- X1_bs %*% betas_theta
  eta_zeta <- matrix(nrow = n_sample, ncol = d)
  intercept <- c(1.0, 1.5, 1.75, 2.0)
  for (j in seq_len(d)) {
    eta_zeta[, j] <- sin(2 * pi * X[, 1L]) + X[, 1L]^2 - intercept[j]
  }
  true_zetas <- stats::pnorm(eta_zeta)
  alphas <- exp(eta_theta)
  Y <- Z <- true_thetas <- true_varthetas <- matrix(nrow = n_sample, ncol = d)
  for (i in seq_len(n_sample)) {
    z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
    is_zero <- z == 0L
    # Hack to avoid all zeros (it happen very rarely)
    while (all(is_zero)) {
      z <- stats::rbinom(n = d, size = 1L, prob = 1.0 - true_zetas[i, ])
      is_zero <- z == 0L
    }
    p_ij <- alphas[i, ] / sum(alphas[i, ])
    true_thetas[i, ] <- p_ij
    true_varthetas[i, ] <- z * p_ij / sum(z * p_ij)
    # if (all(is_zero)) {Y[i, ] <- rep(0L, d); true_varthetas[i, ] <- 0.0}
    if (sum(is_zero) == d - 1L) {
      Y[i, ] <- rep(0L, d)
      Y[i, !is_zero] <- n_trials[i]
    } else {
      Y[i, ] <- stats::rmultinom(n = 1L, size = n_trials[i],
                                 prob = true_varthetas[i, ])
    }
    Z[i, ] <- z
  }
  any(rowSums(Y) == 0)
  table(rowSums(Y) == 0)
  cbind(all_zeros = colMeans(Y == 0), structural_zeros = colMeans(1 - Z)
        , sampling_zeros = colMeans(Y == 0) - colMeans(1 - Z))
  zi_multinomial(Y)

  # Split data
  idx_train <- sample(1:n_sample, size = 800L, replace = FALSE)
  Y_train <- Y[idx_train, ]
  X_train <- X[idx_train, , drop = FALSE]
  Y_test <- Y[-idx_train, ]
  X_test <- X[-idx_train, , drop = FALSE]
  n_trials_train <- rowSums(Y_train)
  n_trials_test <- rowSums(Y_test)
  theta_train <- true_thetas[idx_train, ]
  theta_test <- true_thetas[-idx_train, ]
  Y_train_rel <- sweep(Y_train, 1, n_trials_train, "/")
  Y_test_rel <- sweep(Y_test, 1, n_trials_test, "/")

  # Fit model

  # devtools::load_all()
  zanim_bart <- zanicc(Y = Y_train, X_count = X_train, X_zi = X_train,
                       model = "zanim_bart", ntrees_theta = 20, ntrees_zeta = 20,
                       ndpost = 1000, nskip = 5000, save_trees = TRUE,
                       forests_dir = path_res)

  # Predict function
  pred_theta <- predict.ZANIMBART(object = zanim_bart, newdata = X_train, type = "theta",
                                  load = TRUE, output_dir = path_res)
  expect_equal(pred_theta, zanim_bart$draws_theta)
  pred_zeta <- predict.ZANIMBART(object = zanim_bart, newdata = X_train, type = "zeta",
                                 load = TRUE, output_dir = path_res)
  expect_equal(pred_zeta, zanim_bart$draws_zeta)

  x11(); par(mfrow = c(2,2))
  for (j in seq_len(d))  plot(theta_train[, j], rowMeans(pred_theta[,j,]))

  # PPD function
  ppd1 <- ppd(object = zanim_bart, in_sample = TRUE)
  ppd2 <- ppd(object = zanim_bart, in_sample = FALSE, draws_prob = pred_theta,
              draws_zeta = pred_zeta, n_trials = n_trials_train)
  ppd3 <- ppd(object = zanim_bart, in_sample = FALSE, n_trials = n_trials_train,
              output_dir = path_res, n_pred = nrow(Y_train))
  cowplot::plot_grid(
    bayesplot::ppc_ecdf_overlay(y = Y_train_rel[, 1L], yrep = ppd1[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_train_rel[, 1L], yrep = ppd2[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_train_rel[, 1L], yrep = ppd3[,,1L])
  )
  t_dat <- shannon_entropy(Y_train_rel)
  t_ppd <- apply(ppd2, 1, shannon_entropy)
  hist(t_ppd); abline(v = t_dat)

  t_dat <- gdi(Y_train_rel)
  t_ppd <- apply(ppd1, 1, gdi)
  hist(t_ppd); abline(v = t_dat)

  t_dat <- mean(Y_train_rel==0)
  t_ppd <- apply(ppd1, 1, function(x) mean(x == 0))
  hist(t_ppd); abline(v = t_dat)

  t_dat <- sum(cor(Y_train_rel))
  t_ppd <- apply(ppd1, 1, function(x) sum(cor(x)))
  hist(t_ppd); abline(v = t_dat)

  # Out of sample
  unlink(file.path(path_res, "theta_ij.bin"))
  unlink(file.path(path_res, "zeta_ij.bin"))
  pred2_theta <- predict.ZANIMBART(object = zanim_bart, newdata = X_test,
                                   type = "theta", load = TRUE,
                                   output_dir = path_res)
  rowSums(pred2_theta[,,1])
  pred2_zeta <- predict.ZANIMBART(object = zanim_bart, newdata = X_test,
                                  type = "zeta", load = TRUE,
                                  output_dir = path_res)
  x11()
  par(mfrow = c(2,2))
  for (j in seq_len(d))  plot(theta_test[, j], rowMeans(pred2_theta[,j,]))
  ppd2 <- ppd(object = zanim_bart, in_sample = FALSE, draws_prob = pred2_theta,
              draws_zeta = pred2_zeta, n_trials = n_trials_test)
  ppd3 <- ppd(object = zanim_bart, in_sample = FALSE, n_trials = n_trials_test,
              output_dir = path_res, n_pred = nrow(Y_test))
  cowplot::plot_grid(
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 1L], yrep = ppd2[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 1L], yrep = ppd3[,,1L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 2L], yrep = ppd2[,,2L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 2L], yrep = ppd3[,,2L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 3L], yrep = ppd2[,,3L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 3L], yrep = ppd3[,,3L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 4L], yrep = ppd2[,,4L]),
    bayesplot::ppc_ecdf_overlay(y = Y_test_rel[, 4L], yrep = ppd3[,,4L]), ncol = 4
  )

  t_dat_test <- shannon_entropy(Y_test_rel)
  t_ppd <- apply(ppd2, 1, shannon_entropy)
  hist(t_ppd); abline(v = t_dat_test)

  t_dat_test <- gdi(Y_test_rel)
  t_ppd <- apply(ppd2, 1, gdi)
  hist(t_ppd); abline(v = t_dat_test)

  # Counts
  ppd2 <- ppd(object = zanim_bart, in_sample = FALSE, relative = FALSE,
              draws_prob = pred2_theta, draws_zeta = pred2_zeta,
              n_trials = n_trials_test)

  t_dat_test <- zi_multinomial(Y_test)
  t_ppd <- apply(ppd2, 1, zi_multinomial)
  hist(t_ppd); abline(v = t_dat_test)

})



