test_that("Check if the ZANIM-LN-BART with MGP prior is working", {

  library(ggplot2)
  # Simulate data
  rm(list = ls())
  devtools::load_all()
  n_sample <- 200L
  n_trials <- 200L
  d <- 20L
  set.seed(6669)
  # Simulate data
  list_sim <- sim_data_zanim_ln_bspline_curve(n = n_sample, d = d, n_trials = n_trials,
                                              link_zeta = "probit", rho = 0.8,
                                              covariance = "fam", q_factors = 10L)
  Y <- list_sim$Y
  Z <- list_sim$Z
  X <- list_sim$X
  true_Sigma_U <- list_sim$Sigma_U
  true_thetas <- list_sim$theta1
  any(rowSums(Y) == 0)
  cbind(all = colMeans(Y == 0), struc = colMeans(1 - Z))


  data_sim <- list_sim$df
  head(data_sim)
  ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = zeta), linewidth = 0.4) +
    facet_wrap(~category, scales = "free_y")
  ggplot(data = data_sim) +
    facet_wrap(~category, scales = "free_y") +
    geom_line(mapping = aes(x = x, y = theta1), linewidth = 0.4)
  apply(true_thetas, 2, quantile)

  # Fitting models
  NDPOST <- 5000L
  NSKIP <- 2000L
  NTREES_THETA <- 20L
  NTREES_ZETA <- 10L
  # devtools::load_all()

  m_fm_q10 <- ZANIMLNBART$new(Y = Y, X_theta = X, X_zeta = X)
  m_fm_q10$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                    ndpost = NDPOST, nskip = NSKIP, covariance_type = "fa",
                    q_factors = 10, update_sigma_theta = TRUE, keep_draws = TRUE)
  m_fm_q10$RunMCMC()

  m_fm_q5 <- ZANIMLNBART$new(Y = Y, X_theta = X, X_zeta = X)
  m_fm_q5$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                    ndpost = NDPOST, nskip = NSKIP, covariance_type = "fa",
                    q_factors = 5, update_sigma_theta = TRUE, keep_draws = TRUE)
  m_fm_q5$RunMCMC()

  m_fm_mgp <- ZANIMLNBART$new(Y = Y, X_theta = X, X_zeta = X)
  m_fm_mgp$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                     ndpost = NDPOST, nskip = NSKIP, covariance_type = "fa_mgp",
                     q_factors = 10, update_sigma_theta = TRUE,
                     keep_draws = TRUE)
  m_fm_mgp$RunMCMC()


  # Check the posterior draws of the factor loading
  indices <- sample.int(n = NDPOST, size = 9)
  list_plots_mgp <- list()
  list_plots_q10 <- list()
  par(mfrow = c(3, 3))
  for (i in seq_len(9)) {
    mat_q10 <- m_fm_q10$cpp_obj$draws_Gamma[,,indices[i]]
    mat_mgp <- m_fm_mgp$cpp_obj$draws_Gamma[,,indices[i]]
    colnames(mat_mgp) <- 1:ncol(mat_mgp)
    rownames(mat_mgp) <- 1:(d - 1)
    colnames(mat_q10) <- 1:ncol(mat_q10)
    rownames(mat_q10) <- 1:(d - 1)
    list_plots_mgp[[i]] <- pheatmap::pheatmap(mat = abs(mat_mgp),
                                              main = paste0("Iteration: ", indices[i]),
                                              cluster_rows = FALSE, cluster_cols = FALSE
                                              # , color = colorRampPalette(c("blue", "white", "red"))(100)
                                              )$gtable
    list_plots_q10[[i]] <- pheatmap::pheatmap(mat = abs(mat_q10), main = paste0("Iteration: ", indices[i]),
                                              cluster_rows = FALSE, cluster_cols = FALSE)$gtable
  }
  cowplot::plot_grid(plotlist = list_plots_mgp)
  cowplot::plot_grid(plotlist = list_plots_q10)

  # Compute the log-likelihood using ZANIM distribution
  ll_mgp <- m_fm_mgp$LogPredictiveLikelihood(Y = Y)
  ll_q5 <- m_fm_q5$LogPredictiveLikelihood(Y = Y)
  ll_q10 <- m_fm_q10$LogPredictiveLikelihood(Y = Y)

  lt_waic <- list(mgp = loo::waic(ll_mgp),
                 fm_5 = loo::waic(ll_q5),
                 fm_10 = loo::waic(ll_q10))
  as.data.frame(loo::loo_compare(lt_waic))


  # Compute the log-likelihood using Y_i \mid ...) \sim Mult[N_i, \vartheta_{i}]
  lp_mult <- function(Y, draws, printevery = 100) {
    ndpost <- dim(draws)[3]
    n_samples <- dim(draws)[1]
    lpl <- matrix(nrow = ndpost, ncol = n_samples)
    for (t in seq_len(ndpost)) {
      if (t %% printevery == 0L) cat(t, "\n")
      lpl[t, ] <- .dmultinomial(x = Y, prob = draws[, ,t])
    }
    lpl
  }
  ll_q5 <- lp_mult(Y = Y, draws = m_fm_q5$draws_abundance)
  ll_q10 <- lp_mult(Y = Y, draws = m_fm_q10$draws_abundance)
  ll_mgp <- lp_mult(Y = Y, draws = m_fm_mgp$draws_abundance)

  lt_waic <- list(mgp = loo::loo(ll_mgp),
                 fm_5 = loo::loo(ll_q5),
                 fm_10 = loo::loo(ll_q10))
  as.data.frame(loo::loo_compare(lt_waic))


  # Compare the estimates of \theta_{ij}
  x11(); par(mfrow = c(4, 5))
  for (j in seq_len(d)) {
    plot(true_thetas[, j], rowMeans(m_fm_q10$draws_theta[, j, ]),
         xlab = "true", ylab = "estimate", main = paste0("j = ", j))
    abline(0, 1, col = "red")
  }
  frobs_q10 <- compute_frob_chain(true_values = true_thetas, draws = m_fm_q10$draws_theta)
  frobs_q5 <- compute_frob_chain(true_values = true_thetas, draws = m_fm_q5$draws_theta)
  frobs_mgp <- compute_frob_chain(true_values = true_thetas, draws = m_fm_mgp$draws_theta)
  graphics.off()
  yrange <- range(c(frobs_mgp, frobs_q10, frobs_q5))
  par(mfrow = c(1, 3))
  plot(frobs_mgp, type = "l", ylim = yrange, main = "mgp")
  plot(frobs_q10, type = "l", ylim = yrange, main = "q*=10")
  plot(frobs_q5, type = "l", ylim = yrange, main = "q*=5")

  df_summaries_theta <- .summarise_draws_3d(m_fm_q10$draws_theta)
  df_summaries_theta$x <- rep(c(X), times = d)
  # df_summaries_zeta <- .summarise_draws_3d(m_fm_q10$draws_zeta)
  # df_summaries_zeta$x <-

  p_zanimlnbart_q10 <- ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = theta1), linewidth = 0.4) +
    facet_wrap(~category, scales = "free_y") +
    geom_line(data = df_summaries_theta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = df_summaries_theta, aes(x = x, ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3)

  df_summaries_theta <- .summarise_draws_3d(m_fm_mgp$draws_theta)
  df_summaries_theta$x <- rep(c(X), times = d)
  p_zanimlnbart_mgp <- ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = theta1), linewidth = 0.4) +
    facet_wrap(~category, scales = "free_y") +
    geom_line(data = df_summaries_theta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = df_summaries_theta, aes(x = x, ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3)


})

test_that("Estimate theta using the ZANIM-LN-BART", {
  library(ggplot2)
  # Simulate data
  rm(list = ls())
  devtools::load_all()
  n_sample <- 200L
  n_trials <- 200L
  d <- 4L
  set.seed(1412)
  # Simulate data
  list_sim <- sim_data_zanim_ln_bspline_curve(n = n_sample, d = d, n_trials = n_trials,
                                              link_zeta = "probit", rho = 0.8,
                                              covariance = "fa")
  Y <- list_sim$Y
  Z <- list_sim$Z
  X <- list_sim$X
  cbind(all = colMeans(Y == 0), structural = colMeans(Z == 0))
  true_Sigma_U <- list_sim$Sigma_U
  true_varthetas <- list_sim$abundance
  true_thetas <- list_sim$theta
  any(rowSums(Y) == 0)

  data_sim <- list_sim$df
  head(data_sim)
  ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = zeta), linewidth = 0.4) +
    facet_wrap(~category, scales = "free_y")
  ggplot(data = data_sim) +
    facet_wrap(~category, scales = "free_y") +
    geom_line(mapping = aes(x = x, y = theta), linewidth = 0.4)
  ggplot(data = data_sim) +
    facet_wrap(~category, scales = "free_y") +
    geom_point(mapping = aes(x = x, y = total / 200)) +
    geom_line(mapping = aes(x = x, y = theta), col = "red", linewidth = 0.4)
  # apply(true_thetas, 2, quantile)

  # Fitting models

  NDPOST <- 1000L
  NSKIP <- 1000L
  NTREES_THETA <- 100L
  NTREES_ZETA <- 20L
  # devtools::load_all()
  m <- zanicc(Y, X_count = X, X_zi = X, ntrees_theta = 10, ndpost = 10, nskip =100, model ="zanim_ln_bart",
              keep_draws = TRUE, covariance_type = "fa")
  m$cpp_obj$covariance_type
  m$RunMCMC()
  mod <- ZANIMLNBART$new(Y = Y, X_theta = X, X_zeta = X)
  mod$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                ndpost = 10, nskip = 10, covariance_type = "fa_mgp",
                keep_draws = TRUE, save_trees = FALSE)
  mod$RunMCMC()
  mod$SetupMCMC(ntrees_theta = ntrees_theta, ntrees_zeta = ntrees_zeta,
                ndpost = ndpost, nskip = nskip, keep_draws = keep_draws,
                save_trees = save_trees, ...)

  colMeans(mod$avg_leaves_theta)
  colMeans(mod$avg_leaves_zeta)

  # Posterior-predictive
  yppc_cond <- mod$GetPosteriorPredictive(conditional_rf = TRUE, relative = TRUE)
  yppc_marg <- mod$GetPosteriorPredictive(conditional_rf = FALSE, relative = TRUE)
  yppc_marg[is.na(yppc_marg)] <- 0L
  Y_rel <- sweep(Y, 1, rowSums(Y), "/")
  cowplot::plot_grid(
    bayesplot::ppc_ecdf_overlay(y = Y_rel[,1], yrep = yppc_cond[,,1]),
    bayesplot::ppc_ecdf_overlay(y = Y_rel[,1], yrep = yppc_marg[,,1])
  )


  frob_vartheta <- compute_frob_chain(true_values = true_varthetas,
                                      draws = mod$draws_abundance)
  plot(frob_vartheta, type = "l")

  df_summaries_theta <- .summarise_draws_3d(mod$draws_theta)
  df_summaries_zeta <- .summarise_draws_3d(mod$draws_zeta)
  df_summaries_zeta$x <- df_summaries_theta$x <- rep(c(X), times = d)

  p_zanimlnbart <- ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = theta), linewidth = 0.4) +
    facet_wrap(~category, scales = "free_y") +
    geom_line(data = df_summaries_theta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = df_summaries_theta, aes(x = x, ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3)

  ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = zeta), linewidth = 0.4) +
    facet_wrap(~category, scales = "free_y") +
    geom_line(data = df_summaries_zeta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = df_summaries_zeta, aes(x = x, ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3)





  # mod2 <- ZANIMBART$new(Y = Y, X_theta = X, X_zeta = X)
  # mod2$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
  #               ndpost = NDPOST, nskip = NSKIP,
  #               update_sigma_theta = TRUE, keep_draws = TRUE)
  # mod2$RunMCMC()
  # colMeans(mod2$avg_leaves_theta)
  # colMeans(mod2$avg_leaves_zeta)
  # frob_vartheta <- compute_frob_chain(true_values = true_varthetas,
  #                                     draws = mod2$draws_abundance)
  # plot(frob_vartheta, type = "l")
  #
  # df_summaries_zeta <- .summarise_draws_3d(mod2$draws_zeta)
  # df_summaries_theta1 <- .summarise_draws_3d(mod2$draws_theta)
  # df_summaries_zeta$x <- df_summaries_theta1$x <- rep(c(X), times = d)
  #
  # ggplot(data = data_sim) +
  #   geom_line(mapping = aes(x = x, y = theta1), linewidth = 0.4) +
  #   facet_wrap(~category, scales = "free_y") +
  #   geom_line(data = df_summaries_theta1, mapping = aes(x = x, y = median),
  #             col = "dodgerblue") +
  #   geom_ribbon(data = df_summaries_theta1, aes(x = x, ymin = ci_lower, ymax = ci_upper),
  #               fill = "dodgerblue", alpha = 0.3)
  #
  # ggplot(data = data_sim) +
  #   geom_line(mapping = aes(x = x, y = zeta), linewidth = 0.4) +
  #   facet_wrap(~category, scales = "free_y") +
  #   geom_line(data = df_summaries_zeta, mapping = aes(x = x, y = median),
  #             col = "dodgerblue") +
  #   geom_ribbon(data = df_summaries_zeta, aes(x = x, ymin = ci_lower, ymax = ci_upper),
  #               fill = "dodgerblue", alpha = 0.3)

  mod3 <- MultinomialLNBART$new(Y = Y, X = X)
  mod3$SetupMCMC(ntrees = NTREES_THETA, ndpost = NDPOST, nskip = NSKIP,
                 covariance_type = "wishart",
                 update_sigma = TRUE, keep_draws = TRUE)
  mod3$RunMCMC()
  colMeans(mod3$avg_leaves)
  frob_vartheta <- compute_frob_chain(true_values = true_varthetas,
                                      draws = mod3$draws_abundance)
  plot(frob_vartheta, type = "l")

  df_summaries_theta3 <- .summarise_draws_3d(mod3$draws_theta)
  df_summaries_theta3$x <- rep(c(X), times = d)

  p_mlnbart <- ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = theta), linewidth = 0.4) +
    facet_wrap(~category, scales = "free_y") +
    geom_line(data = df_summaries_theta3, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = df_summaries_theta3, aes(x = x, ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3)

  # using fido
  d <- ncol(Y)
  ini <- proc.time()
  fit <- fido::basset(t(Y), t(X), Theta = function(X) matrix(0, d - 1, ncol(X)),
                      Gamma = function(X) fido::SE(X, sigma = 1))
  end <- proc.time() - ini
  # draws <- predict(fit)
  # all.equal(draws, fit$Lambda)
  fit_p <- fido::to_proportions(fit)
  draws_theta <- aperm(fit_p$Lambda, c(2, 1, 3))
  draws_vartheta <- aperm(fit_p$Eta, c(2, 1, 3))

  df_summaries_theta <- .summarise_draws_3d(draws_theta)
  df_summaries_theta$x <- rep(c(X), times = d)

  p_mlngp <- ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = theta), linewidth = 0.4) +
    facet_wrap(~category, scales = "free_y") +
    geom_line(data = df_summaries_theta, mapping = aes(x = x, y = median),
              col = "dodgerblue") +
    geom_ribbon(data = df_summaries_theta, aes(x = x, ymin = ci_lower, ymax = ci_upper),
                fill = "dodgerblue", alpha = 0.3)


  cowplot::plot_grid(p_zanimlnbart, p_mlnbart, p_mlngp)


  compute_frob(true_values = true_varthetas, apply(mod$draws_abundance, c(1, 2), mean))
  compute_frob(true_values = true_varthetas, apply(mod3$draws_abundance, c(1, 2), mean))
  compute_frob(true_values = true_varthetas, apply(draws_vartheta, c(1, 2), mean))
  compute_kl_simplex(true_values = true_thetas, apply(mod$draws_theta, c(1, 2), mean))
  compute_kl_simplex(true_values = true_thetas, apply(mod3$draws_theta, c(1, 2), mean))
  compute_kl_simplex(true_values = true_thetas, apply(draws_theta, c(1, 2), mean))

})
