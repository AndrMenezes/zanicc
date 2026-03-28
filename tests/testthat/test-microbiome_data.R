test_that("fit models in microbiome data", {
  rm(list = ls()); gc()
  devtools::load_all()
  Y <- microbiome_data$Y
  X <- microbiome_data$X
  n <- nrow(Y)
  chosen_taxas <- names(sort(colSums(Y), decreasing = TRUE)[1:13])

  Y <- Y[, chosen_taxas]
  colnames(X)

  NDPOST <- 1000L
  NSKIP <- 5000L
  NTREES <- 100L
  # multinomial-BART
  mult_bart <- zanicc(Y = Y, X_count = X, model = "mult_bart",
                      ntrees_theta = NTREES, sparse = TRUE)
  mult_bart$avg_leaves
  colMeans(mult_bart$avg_leaves)
  ll1 <- lpd_multinomial(Y = Y, draws_vartheta = mult_bart$draws_theta)
  prob_vc <- apply(mult_bart$cpp_obj$varcount_mcmc > 0, c(1, 2), mean)
  mean(prob_vc > 0.5)
  plot(sort(prob_vc, decreasing = TRUE))

  # multinomial-LN-BART
  mult_ln_bart <- zanicc(Y = Y, X_count = X, model = "mult_ln_bart",
                         ntrees_theta = NTREES, sparse = TRUE)
  colMeans(mult_ln_bart$avg_leaves)
  ll2 <- lpd_multinomial(Y = Y, draws_vartheta = mult_ln_bart$draws_abundance)

  prob_vc <- apply(mult_ln_bart$cpp_obj$varcount_mcmc > 0, c(1, 2), mean)
  mean(prob_vc > 0.5)
  plot(sort(prob_vc, decreasing = TRUE))

  # ZANIM-BART
  zanim_bart <- zanicc(Y = Y, X_count = X, X_zi = X, model = "zanim_bart",
                       ntrees_theta = NTREES, ntrees_zeta = 20L,
                       sparse = rep(TRUE, 2))
  colMeans(zanim_bart$avg_leaves_theta)
  colMeans(zanim_bart$avg_leaves_zeta)
  ll3 <- lpd_multinomial(Y = Y, draws_vartheta = zanim_bart$draws_abundance)
  ll_zanim <- zanim_bart$LogPredictiveLikelihood(Y = Y)

  # ZANIM-LN-BART
  zanim_ln_bart <- zanicc(Y = Y, X_count = X,  X_zi = X, model = "zanim_ln_bart",
                          ntrees_theta = NTREES, ntrees_zeta = 20L,
                          sparse = rep(TRUE, 2))
  colMeans(zanim_ln_bart$avg_leaves_theta)
  colMeans(zanim_ln_bart$avg_leaves_zeta)

  ll_zanim_ln <- zanim_ln_bart$LogPredictiveLikelihood(Y = Y)
  ll4 <- lpd_multinomial(Y = Y, draws_vartheta = zanim_ln_bart$draws_abundance)

  # Need to load the zidm and dm objects
  # ll5 <- lpd_multinomial(Y = Y, draws_vartheta = zidm$varthetas)
  # ll6 <- lpd_multinomial(Y = Y, draws_vartheta = dm$varthetas)

  d <- ncol(Y)
  n <- nrow(Y)
  ini <- proc.time()
  fit <- fido::basset(t(Y), t(X), Theta = function(X) matrix(0, d - 1, n),
                      Gamma = function(X) fido::SE(X, sigma = 1),
                      n_samples = NDPOST)
  end <- proc.time() - ini
  fit_p <- fido::to_proportions(fit)
  draws_theta_gp = aperm(fit_p$Lambda, c(2, 1, 3))
  draws_abundance_gp = aperm(fit_p$Eta, c(2, 1, 3))
  ll7 <- lpd_multinomial(Y = Y, draws_vartheta = draws_abundance_gp)


  yppc_mult_bart <- posterior_predictive_multinomial(n_trials = rowSums(Y), draws_prob = mult_bart$draws_theta, relative = TRUE)
  yppc_mult_ln_bart <- posterior_predictive_multinomial(n_trials = rowSums(Y), draws_prob = mult_ln_bart$draws_abundance, relative = TRUE)
  yppc_zanim_bart <- zanim_bart$GetPosteriorPredictive()
  yppc_zanim_ln_bart <- zanim_ln_bart$GetPosteriorPredictive()

  yppc_zanim_bart <- sweep(yppc_zanim_bart, MARGIN = c(1, 2),
        STATS = apply(yppc_zanim_bart, c(1, 2), sum), FUN = "/")
  yppc_zanim_ln_bart <- sweep(yppc_zanim_ln_bart, MARGIN = c(1, 2),
        STATS = apply(yppc_zanim_ln_bart, c(1, 2), sum), FUN = "/")
  Y_rel <- sweep(Y, MARGIN = 1, STATS = rowSums(Y), FUN = "/")

  rps_mult_bart <- scoringutils::crps_sample(observed = Y_rel[, 10], predicted = t(yppc_mult_bart[,,10]))
  rps_zanim_bart <- scoringutils::crps_sample(observed = Y_rel[, 10], predicted = t(yppc_zanim_bart[,,10]))
  mean(rps_mult_bart)
  mean(rps_zanim_bart)

  # ECDF
  cowplot::plot_grid(
    bayesplot::ppc_ecdf_overlay(y = Y_rel[, 10], yrep = yppc_mult_bart[,,10]) +
      ggplot2::scale_x_continuous(limits = c(0, 1)),
    bayesplot::ppc_ecdf_overlay(y = Y_rel[, 10], yrep = yppc_zanim_bart[,,10]) +
      ggplot2::scale_x_continuous(limits = c(0, 1))
  )
  bayesplot::ppc_ecdf_overlay(y = Y[, 3], yrep = yppc_mult_ln_bart[,,3])
  bayesplot::ppc_ecdf_overlay(y = Y[, 3], yrep = yppc_zanim_ln_bart[,,3])

  #
  bayesplot::ppc_stat(y = Y[, 16], yrep = yppc_mult_bart[,,16], stat = function(x) var(x)/mean(x))
  bayesplot::ppc_stat(y = Y[, 16], yrep = yppc_mult_ln_bart[,,16], stat = function(x) var(x)/mean(x))
  bayesplot::ppc_stat(y = Y[, 16], yrep = yppc_zanim_bart[,,16], stat = function(x) var(x)/mean(x))
  bayesplot::ppc_stat(y = Y[, 16], yrep = yppc_zanim_ln_bart[,,16], stat = function(x) var(x)/mean(x))

  bayesplot::ppc_stat(y = Y[, 16], yrep = yppc_mult_bart[,,16], stat = zi_b)
  bayesplot::ppc_stat(y = Y[, 16], yrep = yppc_mult_ln_bart[,,16], stat = zi_b)
  bayesplot::ppc_stat(y = Y[, 16], yrep = yppc_zanim_bart[,,16], stat = zi_b)
  bayesplot::ppc_stat(y = Y[, 16], yrep = yppc_zanim_ln_bart[,,16], stat = zi_b)

cowplot::plot_grid(
  bayesplot::ppc_stat(y = Y_rel[, 10], yrep = yppc_mult_bart[,,10], stat = function(x) mean(x==0)) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::ggtitle("mult-BART"),
  bayesplot::ppc_stat(y = Y_rel[, 10], yrep = yppc_zanim_bart[,,10], stat = function(x) mean(x==0)) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::ggtitle("ZANIM-BART"),
  bayesplot::ppc_stat(y = Y_rel[, 10], yrep = ppd_zidm[,,14], stat = function(x) mean(x==0)) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::ggtitle("ZIDM-reg")
)

dim(zanim_bart$draws_zeta)
#auc(Y_rel[, 10] == 0, rowMeans(zanim_bart$draws_zeta[,10,]))

cowplot::plot_grid(
  bayesplot::ppc_stat(y = Y_rel[, 10], yrep = yppc_mult_bart[,,10], stat = function(x) mean(x)) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::ggtitle("mult-BART"),
  bayesplot::ppc_stat(y = Y_rel[, 10], yrep = yppc_zanim_bart[,,10], stat = function(x) mean(x)) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::ggtitle("ZANIM-BART")
)



  cbind()
  colMeans(Y == 0)


  res <- as.data.frame(  loo::loo_compare(
    list(mult_bart = loo::waic(ll1),
         mult_ln_bart = loo::waic(ll2),
         zanim_bart = loo::waic(ll3),
         zanim_bart_marg = loo::waic(ll_zanim),
         zanim_ln_bart_marg = loo::waic(ll_zanim_ln),
         zanim_ln_bart = loo::waic(ll4),
         # zidm_reg_ss = loo::waic(ll5), dm_reg_ss = loo::waic(ll6),
         mult_ln_gp = loo::waic(ll7)
         # , zidm_reg_ss_2 = loo::waic(log_like_zidm)
         )
    )
  )
  res[,c(7, 8, 5)]

  # zetas <- zanim_bart$draws_zeta
  # zetas[1,1,1]
  # for (i in seq_len(nrow(Y))) {
  #   idx <- which(Y[i, ] > 0)
  #   zanim_bart$draws_zeta[i, idx, ] <- 0.0
  # }
  # ll_zanim2 <- zanim_bart$LogPredictiveLikelihood(Y = Y)





})
