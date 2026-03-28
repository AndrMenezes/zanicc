test_that("fit models in pollen data", {
  rm(list = ls()); gc()
  devtools::load_all()
  library(ggplot2)
  library(cowplot)
  path_res <- "./tests/testthat/pollen"

  # Load data
  Y <- pollen_data$Y
  X <- pollen_data$X
  chosen_taxas <- 1:ncol(Y)#names(sort(colSums(Y), decreasing = TRUE)[1:28])
  set.seed(125)
  sample_size <- 200L
  idx <- sample(1:nrow(Y), size = sample_size, replace = FALSE)

  Y <- Y[, chosen_taxas]
  X <- X[, c(5, 6, 8)]
  Y_rel <- sweep(Y, MARGIN = 1, STATS = rowSums(Y), FUN = "/")
  Xscaled <- scale(X)
  n_trials <- rowSums(Y)
  taxa_names <- colnames(Y)

  NDPOST <- 5000L
  NSKIP <- 5000L
  NTREES <- 100L

  # ML-BART
  ml_bart <- zanicc(Y = Y, X_count = Xscaled, model = "mult_bart",
                    ntrees_theta = NTREES, ndpost = NDPOST, nskip = NSKIP)
  # MLN-BART
  mln_bart <- zanicc(Y = Y, X_count = Xscaled, model = "mult_ln_bart",
                     ntrees_theta = NTREES, ndpost = NDPOST, nskip = NSKIP)

  # ZANIM-BART
  zanim_bart <- zanicc(Y = Y, X_count = Xscaled, X_zi = Xscaled, model = "zanim_bart",
                       ntrees_theta = NTREES, ntrees_zeta = 20L,
                       ndpost = NDPOST, nskip = NSKIP)
  colMeans(zanim_bart$avg_leaves_theta)

  # ZANIM-LN-BART
  zanim_ln_bart <- zanicc(Y = Y, X_count = Xscaled, X_zi = Xscaled,
                          model = "zanim_ln_bart", ntrees_theta = NTREES,
                          ntrees_zeta = 20L, ndpost = NDPOST, nskip = NSKIP)
  colMeans(zanim_ln_bart$avg_leaves_theta)

  # DM-reg
  Xwint <- cbind(1.0, Xscaled)
  dm_reg <- zanicc(Y = Y, X_count = Xwint, model = "dm_reg",
                   ndpost = 10, nskip = 10, save_draws = TRUE)

  # ZANIDM-reg
  zanidm_reg <- zanicc(Y = Y, X_count = Xwint, X_zi = Xwint, model = "zanidm_reg",
                       ndpost = NDPOST, nskip = NSKIP)

  # ZANIM-LN-reg
  zanim_ln_reg <- zanicc(Y = Y, X_count = Xwint, X_zi = Xwint,
                         model = "zanim_ln_reg", covariance_type = "fa_mgp",
                         ndpost = NDPOST, nskip = NSKIP)

  zanim_ln_reg$draws_abundance
  zanidm_reg$draws_abundance

  # Posterior-predictive check
  y_ml_bart <- ml_bart$GetPosteriorPredictive(relative = TRUE)
  y_mln_bart <- mln_bart$GetPosteriorPredictive(relative = TRUE, conditional_rf = TRUE)
  y_zanim_bart <- zanim_bart$GetPosteriorPredictive(conditional_rf = TRUE, relative = TRUE)
  y_zanim_ln_bart <- zanim_ln_bart$GetPosteriorPredictive(conditional_rf = TRUE, relative = TRUE)
  y_zanim_ln_reg <- posterior_predictive_multinomial(n_trials = n_trials, draws_prob = zanim_ln_reg$draws_abundance)
  # y_zanidm_reg <- zanidm_reg$GetPosteriorPredictive()
  # y_zanidm_reg=sweep(y_zanidm_reg, MARGIN = c(1, 2), STATS = apply(y_zanidm_reg, c(1, 2), sum), FUN = "/")
  y_zanidm_reg <- posterior_predictive_multinomial(n_trials = n_trials, draws_prob = zanidm_reg$draws_abundance)
  y_dm_reg <- posterior_predictive_multinomial(n_trials = n_trials, draws_prob = dm_reg$draws_abundance)
  # yppc_dm_reg <- dm_reg$GetPosteriorPredictive(relative = TRUE)

  kl_ml_bart <- compute_kl_simplex(true_values = Y_rel, estimates = apply(y_ml_bart, c(2, 3), mean))
  kl_mln_bart <- compute_kl_simplex(true_values = Y_rel, estimates = apply(y_mln_bart, c(2, 3), mean))
  kl_zanim_bart <- compute_kl_simplex(true_values = Y_rel, estimates = apply(y_zanim_bart, c(2, 3), mean))
  kl_zanim_ln_bart <- compute_kl_simplex(true_values = Y_rel, estimates = apply(y_zanim_ln_bart, c(2, 3), mean), ep = 0.5)
  kl_zanim_ln_reg <- compute_kl_simplex(true_values = Y_rel, estimates = apply(y_zanim_ln_reg, c(2, 3), mean))
  kl_zanidm_reg <- compute_kl_simplex(true_values = Y_rel, estimates = apply(y_zanidm_reg, c(2, 3), mean), ep = 0.5)
  kl_dm_reg <- compute_kl_simplex(true_values = Y_rel, estimates = apply(y_dm_reg, c(2, 3), mean))
  cbind(kl_ml_bart, kl_mln_bart, kl_zanim_bart, kl_zanim_ln_bart,
        kl_zanim_ln_reg, kl_zanidm_reg, kl_dm_reg)

  kl_chain_ml_bart <- compute_kl_simplex_chain(true_values = Y_rel, draws = aperm(y_ml_bart, c(2, 3, 1)))
  kl_chain_zanim_bart <- compute_kl_simplex_chain(true_values = Y_rel, draws = aperm(y_zanim_bart, c(2, 3, 1)))
  kl_chain_zanim_ln_bart <- compute_kl_simplex_chain(true_values = Y_rel, draws = aperm(y_zanim_ln_bart, c(2, 3, 1)), ep = 1)
  kl_chain_dm_reg <- compute_kl_simplex_chain(true_values = Y_rel, draws = aperm(y_dm_reg, c(2, 3, 1)))
  kl_chain_zanidm_reg <- compute_kl_simplex_chain(true_values = Y_rel, draws = aperm(y_zanidm_reg, c(2, 3, 1)))
  kl_chain_zanim_ln_reg <- compute_kl_simplex_chain(true_values = Y_rel, draws = aperm(y_zanim_ln_reg, c(2, 3, 1)))

  library(ggplot2)
  df <- data.frame(x = c(kl_chain_ml_bart, kl_chain_zanim_bart, kl_chain_zanim_ln_bart,
                         kl_chain_zanim_ln_reg, kl_chain_dm_reg),
                   model = rep(c("ML-BART", "ZANIM-BART", "ZANIM-LN-BART",
                                 "ZANIM-LN-reg", "DM-reg"), each = NDPOST))
  ggplot(df, aes(x = model, y = x)) +
    geom_boxplot()

  kl_chain_ml_bart <- compute_frob_chain(true_values = Y_rel, draws = ml_bart$draws_theta)
  kl_chain_zanim_bart <- compute_frob_chain(true_values = Y_rel, draws = zanim_bart$draws_abundance)
  kl_chain_dm_reg <- compute_frob_chain(true_values = Y_rel, draws = dm_reg$draws_abundance)

  df <- data.frame(x = c(kl_chain_ml_bart, kl_chain_zanim_bart, kl_chain_dm_reg),
                   model = rep(c("ML-BART", "ZANIM-BART", "DM-reg"), each = NDPOST),
                   iteration = rep(seq_len(NDPOST), times = 3L))
  ggplot(df, aes(x = model, y = x)) +
    geom_boxplot()
  ggplot(df, aes(x = iteration, y = x, col = model)) +
    geom_line()


  compute_metrics <- function(true_values, draws, kl_fun = zanicc::compute_kl_simplex) {
    estimates = apply(draws, c(2, 3), mean)
    q_lo_95 <- apply(draws, c(2, 3), quantile, probs = 0.025)
    q_up_95 <- apply(draws, c(2, 3), quantile, probs = 0.975)
    q_lo_50 <- apply(draws, c(2, 3), quantile, probs = 0.25)
    q_up_50 <- apply(draws, c(2, 3), quantile, probs = 0.75)
    c(kl = mean(kl_fun(true_values, estimates)),
      frob = compute_frob(true_values, estimates),
      # abs_diff = compute_abs_diff(true_values, estimates),
      coverage_95 = compute_coverage(true_values, q_lo_95, q_up_95),
      coverage_50 = compute_coverage(true_values,  q_lo_50, q_up_50))
  }
  res <- lapply(list(y_ml_bart, y_mln_bart, y_zanim_bart, y_zanim_ln_bart, y_zanidm_reg, y_dm_reg),
                function(x) compute_metrics(Y_rel, x))
  res <- do.call(rbind, res)
  rownames(res) <- c("ml_bart", "mln_bart", "zanim_bart", "zanim_ln_bart", "zanidm_reg", "dm_reg")

  # PPC checks
  y_ml_bart <- ml_bart$GetPosteriorPredictive(relative = TRUE)
  y_mln_bart <- mln_bart$GetPosteriorPredictive(relative = TRUE, conditional_rf = FALSE)
  y_zanim_bart <- zanim_bart$GetPosteriorPredictive(conditional_rf = FALSE, relative = TRUE)
  y_zanim_ln_bart <- zanim_ln_bart$GetPosteriorPredictive(conditional_rf = FALSE, relative = TRUE)
  # y_zanim_ln_reg <- posterior_predictive_multinomial(n_trials = n_trials, draws_prob = zanim_ln_reg$draws_abundance)
  y_zanidm_reg <- zanidm_reg$GetPosteriorPredictive()
  y_zanidm_reg <- sweep(y_zanidm_reg, MARGIN = c(1, 2), STATS = apply(y_zanidm_reg, c(1, 2), sum), FUN = "/")
    # posterior_predictive_multinomial(n_trials = n_trials, draws_prob = zanidm_reg$draws_abundance)
  y_dm_reg <- dm_reg$GetPosteriorPredictive(conditional_rf = FALSE, relative = TRUE)
    # posterior_predictive_multinomial(n_trials = n_trials, draws_prob = dm_reg$draws_abundance)

  # GDI
  t_obs <- gdi(Y = Y_rel)
  t_ml_bart <- apply(y_ml_bart, 1, gdi)
  t_mln_bart <- apply(y_mln_bart, 1, gdi)
  t_zanim_bart <- apply(y_zanim_bart, 1, gdi)
  t_zanim_ln_bart <- apply(y_zanim_ln_bart, 1, gdi)
  t_zanidm_reg <- apply(y_zanidm_reg, 1, gdi)
  t_dm_reg <- apply(y_dm_reg, 1, gdi)

  df <- data.frame(x = c(t_ml_bart, t_mln_bart, t_zanim_bart,
                         t_zanim_ln_bart, t_zanidm_reg, t_dm_reg),
                   model = rep(c("ML-BART", "MLN-BART", "ZANIM-BART", "ZANIM-LN-BART",
                                 "ZIDM-reg", "DM-reg"), each = NDPOST))
  df$model <- forcats::fct_relevel(df$model,
                                   "ZANIM-BART", "ZANIM-LN-BART", "ML-BART", "MLN-BART",
                                   "ZIDM-reg", "DM-reg")
  p_gdi <- ggplot(data = df, aes(x = x)) +
    facet_wrap(~model) +
    geom_density() +
    # geom_density(aes(fill = model), alpha = 0.4, show.legend = FALSE) +
    # geom_histogram(bins = 40L, col = "white") +
    geom_vline(xintercept = t_obs, col = "red") +
    ggtitle("GDI")
  save_plot(filename = file.path(path_res, "gdi_density.png"), plot = p_gdi, base_height = 8,
            bg = "white")
  # correlation

  foo <- function(Y) sum(cor(Y), na.rm = TRUE)
  t_obs <- foo(Y_rel)
  t_ml_bart <- apply(y_ml_bart, 1, foo)
  t_mln_bart <- apply(y_mln_bart, 1, foo)
  t_zanim_bart <- apply(y_zanim_bart, 1, foo)
  t_zanim_ln_bart <- apply(y_zanim_ln_bart, 1, foo)
  t_zanidm_reg <- apply(y_zanidm_reg, 1, foo)
  t_dm_reg <- apply(y_dm_reg, 1, foo)

  df <- data.frame(x = c(t_ml_bart, t_mln_bart, t_zanim_bart,
                         t_zanim_ln_bart, t_zanidm_reg, t_dm_reg),
                   model = rep(c("ML-BART", "MLN-BART", "ZANIM-BART", "ZANIM-LN-BART",
                                 "ZIDM-reg", "DM-reg"), each = NDPOST))
  df$model <- forcats::fct_relevel(df$model,
                                   "ZANIM-BART", "ZANIM-LN-BART", "ML-BART", "MLN-BART",
                                   "ZIDM-reg", "DM-reg")
  p_sum_cor <- ggplot(data = df, aes(x = x)) +
    facet_wrap(~model) +
    # geom_density(aes(fill = model), alpha = 0.4, show.legend = FALSE) +
    geom_density() +
    # geom_histogram(bins = 40L, col = "white") +
    geom_vline(xintercept = t_obs, col = "red") +
    ggtitle("sum(cor(Y))")
  save_plot(filename = file.path(path_res, "sum_cor_Y_density.png"), plot = p_sum_cor,
            base_height = 8, bg = "white")

  # avg-zero
  foo <- function(Y) mean(Y == 0)
  t_obs <- foo(Y_rel)
  t_ml_bart <- apply(y_ml_bart, 1, foo)
  t_mln_bart <- apply(y_mln_bart, 1, foo)
  t_zanim_bart <- apply(y_zanim_bart, 1, foo)
  t_zanim_ln_bart <- apply(y_zanim_ln_bart, 1, foo)
  t_zanidm_reg <- apply(y_zanidm_reg, 1, foo)
  t_dm_reg <- apply(y_dm_reg, 1, foo)

  df <- data.frame(x = c(t_ml_bart, t_mln_bart, t_zanim_bart,
                         t_zanim_ln_bart, t_zanidm_reg, t_dm_reg),
                   model = rep(c("ML-BART", "MLN-BART", "ZANIM-BART", "ZANIM-LN-BART",
                                 "ZIDM-reg", "DM-reg"), each = NDPOST))
  df$model <- forcats::fct_relevel(df$model,
                                   "ZANIM-BART", "ZANIM-LN-BART", "ML-BART", "MLN-BART",
                                   "ZIDM-reg", "DM-reg")
  p_avg_zero <- ggplot(data = df, aes(x = x)) +
    facet_wrap(~model) +
    geom_density() +
    # geom_histogram(bins = 40L, col = "white") +
    geom_vline(xintercept = t_obs, col = "red") +
    ggtitle("mean(Y == 0)")
  save_plot(filename = file.path(path_res, "mean_Y0_density.png"), plot = p_avg_zero,
            base_height = 8, bg = "white")

  p_grid <- plot_grid(p_avg_zero, p_gdi)
  save_plot(filename = file.path(path_res, "zero_gdi.png"), plot = p_grid,
            base_height = 8, bg = "white")



  # Category-level zero count
  foo <- function(Y) colMeans(Y == 0)
  t_obs <- foo(Y_rel)
  t_ml_bart <- apply(y_ml_bart, 1, foo)
  t_mln_bart <- apply(y_mln_bart, 1, foo)
  t_zanim_bart <- apply(y_zanim_bart, 1, foo)
  t_zanim_ln_bart <- apply(y_zanim_ln_bart, 1, foo)
  t_zanidm_reg <- apply(y_zanidm_reg, 1, foo)
  t_dm_reg <- apply(y_dm_reg, 1, foo)
  for (j in seq_len(d)) {
    cat(j, "\n")
    df <- data.frame(x = c(t_ml_bart[j, ], t_mln_bart[j, ],
                            t_zanim_bart[j, ], t_zanim_ln_bart[j, ],
                            t_zanidm_reg[j, ], t_zanidm_reg[j, ]),
                      model = rep(c("ML-BART", "MLN-BART", "ZANIM-BART", "ZANIM-LN-BART",
                                    "ZIDM-reg", "DM-reg"), each = NDPOST))
    df$model <- forcats::fct_relevel(df$model,
                                     "ZANIM-BART", "ZANIM-LN-BART", "ML-BART", "MLN-BART",
                                     "ZIDM-reg", "DM-reg")
    p <- ggplot(data = df, aes(x = x)) +
      facet_wrap(~model) +
      geom_density() +
      geom_vline(xintercept = t_obs[j], col = "red") +
      ggtitle(taxa_names[j])
    save_plot(filename = file.path(path_res, paste0("prop_zero__", taxa_names[j], ".png")),
              plot = p, base_height = 10, bg = "white")
  }


  # Log-likelihood
  ll_ml_bart <- ml_bart$LogPredictiveLikelihood(Y = Y)
  ll_zanim_bart <- zanim_bart$LogPredictiveLikelihood(Y = Y)
  ll_dm_reg <- dm_reg$LogPredictiveLikelihood(Y = Y)

  # Compare log-likelihood
  as.data.frame(
    loo::loo_compare(list(ml_bart = loo::loo(ll_ml_bart),
                          zanim_bart = loo::loo(ll_zanim_bart)
                          # , dm_reg = loo::loo(ll_dm_reg)
                          )
    ))


rps_ml_bart <- scoringutils::crps_sample(observed = Y_rel[, 1], predicted = t(yppc_ml_bart[,,1]))
rps_zanim_bart <- scoringutils::crps_sample(observed = Y_rel[, 1], predicted = t(yppc_zanim_bart[,,1]))
mean(rps_ml_bart)
mean(rps_zanim_bart)

colMeans(Y)

cowplot::plot_grid(
  bayesplot::ppc_ecdf_overlay(y = Y_rel[, 4], yrep = yppc_ml_bart[,,4]) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::ggtitle("mult-BART"),
  bayesplot::ppc_ecdf_overlay(y = Y_rel[, 4], yrep = yppc_zanim_bart[,,4]) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::ggtitle("ZANIM-BART"),
  bayesplot::ppc_ecdf_overlay(y = Y_rel[, 4], yrep = yppc_dm_reg[,,4]) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::ggtitle("DM-reg")
)

colMeans(Y_rel==0)
cowplot::plot_grid(
  bayesplot::ppc_stat(y = Y_rel[, 10], yrep = yppc_ml_bart[,,10], stat = function(x) mean(x==0)) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::ggtitle("mult-BART"),
  bayesplot::ppc_stat(y = Y_rel[, 10], yrep = yppc_zanim_bart[,,10], stat = function(x) mean(x==0)) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::ggtitle("ZANIM-BART"),
  bayesplot::ppc_stat(y = Y_rel[, 10], yrep = yppc_dm_reg[,,10], stat = function(x) mean(x==0)) +
    ggplot2::scale_x_continuous(limits = c(0, 1)) +
    ggplot2::ggtitle("DM-reg")
)



})
