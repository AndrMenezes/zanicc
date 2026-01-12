test_that("Check if the ZANIM-LN-BART with MGP prior is working", {

  library(ggplot2)
  # Simulate data
  rm(list = ls())
  devtools::load_all()
  n_sample <- 200L
  n_trials <- 200L
  d <- 20L
  set.seed(6669)
  # Simulate a covariance matrix
  list_sim <- sim_data_zanim_ln_bspline_curve(n = n_sample, d = d, n_trials = n_trials,
                                              link_zeta = "probit", rho = 0.8,
                                              covariance = "fam", q_factors = 10L)
  Y <- list_sim$Y
  X <- list_sim$X
  true_Sigma_U <- list_sim$Sigma_U
  true_thetas <- list_sim$theta
  any(rowSums(Y) == 0)

  data_sim <- list_sim$df
  head(data_sim)
  ggplot(data = data_sim) +
    geom_line(mapping = aes(x = x, y = zeta), linewidth = 0.4) +
    facet_wrap(~category, scales = "free_y")
  apply(true_thetas, 2, quantile)

  # Fitting models
  NDPOST <- 5000L
  NSKIP <- 5000L
  NTREES_THETA <- NTREES_ZETA <- 20L
  # devtools::load_all()
  # ini <- proc.time()
  m_fm_q10 <- ZANIMLogNormalBART$new(Y = Y, X_theta = X, X_zeta = X)
  m_fm_q10$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                    ndpost = NDPOST, nskip = NSKIP, covariance_type = 2L,
                    q_factors = 10, update_sigma_theta = TRUE, keep_draws = TRUE)
  m_fm_q10$RunMCMC()

  m_fm_q5 <- ZANIMLogNormalBART$new(Y = Y, X_theta = X, X_zeta = X)
  m_fm_q5$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                    ndpost = NDPOST, nskip = NSKIP, covariance_type = 2L,
                    q_factors = 5, update_sigma_theta = TRUE, keep_draws = TRUE)
  m_fm_q5$RunMCMC()

  m_fm_mgp <- ZANIMLogNormalBART$new(Y = Y, X_theta = X, X_zeta = X)
  m_fm_mgp$SetupMCMC(ntrees_theta = NTREES_THETA, ntrees_zeta = NTREES_ZETA,
                     ndpost = NDPOST, nskip = 2*NSKIP, covariance_type = 3L,
                     q_factors = .ledermann(d - 1), update_sigma_theta = TRUE,
                     keep_draws = TRUE)
  m_fm_mgp$RunMCMC()

  # Check the posterior draws of the factor loading
  indices <- sample.int(n = NDPOST, size = 9)
  list_plots_mgp <- list()
  list_plots_q3 <- list()
  par(mfrow = c(3, 3))
  for (i in seq_len(9)) {
    mat_q3 <- m_fm_q10$cpp_obj$draws_Gamma[,,indices[i]]
    mat_mgp <- m_fm_mgp$cpp_obj$draws_Gamma[,,indices[i]]
    colnames(mat_mgp) <- 1:ncol(mat_mgp)
    rownames(mat_mgp) <- 1:(d - 1)
    colnames(mat_q3) <- 1:ncol(mat_q3)
    rownames(mat_q3) <- 1:(d - 1)
    list_plots_mgp[[i]] <- pheatmap::pheatmap(mat = abs(mat_mgp),
                                              main = paste0("Iteration: ", indices[i]),
                                              cluster_rows = FALSE, cluster_cols = FALSE
                                              # , color = colorRampPalette(c("blue", "white", "red"))(100)
                                              )$gtable
    list_plots_q3[[i]] <- pheatmap::pheatmap(mat = abs(mat_q3), main = paste0("Iteration: ", indices[i]),
                                              cluster_rows = FALSE, cluster_cols = FALSE)$gtable
  }
  cowplot::plot_grid(plotlist = list_plots_mgp)
  cowplot::plot_grid(plotlist = list_plots_q3)

  # Compute the log-likelihood using ZANIM distribution
  ll_mgp <- m_fm_mgp$LogPredictiveLikelihood(Y = Y)
  ll_q5 <- m_fm_q5$LogPredictiveLikelihood(Y = Y)
  ll_q10 <- m_fm_q10$LogPredictiveLikelihood(Y = Y)

  lt_waic = list(mgp = loo::waic(ll_mgp),
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


  x11(); par(mfrow = c(4, 5))
  for (j in seq_len(d)) {
    plot(true_thetas[, j], rowMeans(m_fm_q10$draws_theta[, j, ]),
         xlab = "true", ylab = "estimate", main = paste0("j = ", j))
    abline(0, 1, col = "red")
  }
  frobs_q10 <- compute_frob(true_values = true_thetas, draws = m_fm_q10$draws_theta)
  frobs_q5 <- compute_frob(true_values = true_thetas, draws = m_fm_q5$draws_theta)
  frobs_mgp <- compute_frob(true_values = true_thetas, draws = m_fm_mgp$draws_theta)
  graphics.off()
  yrange <- range(c(frobs_mgp, frobs_q10, frobs_q5))
  par(mfrow = c(1, 3))
  plot(frobs_mgp, type = "l", ylim = yrange, main = "mgp")
  plot(frobs_q10, type = "l", ylim = yrange, main = "q*=10")
  plot(frobs_q5, type = "l", ylim = yrange, main = "q*=5")

})
