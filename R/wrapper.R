#' Fit Bayesian count-compositional models
#'
#' Fit different Bayesian count-compositional models, including zero-inflation,
#' overdispersion, complex dependence structure, flexible covariate effects on
#' both levels of the model via the nonparametric BART prior.
#' @export
zanicc <- function(Y, X_count, X_zi = NULL, model = c("zanim_bart",
                                                      "zanim_ln_bart",
                                                      "mult_bart",
                                                      "mult_ln_bart",
                                                      "zanim_reg",
                                                      "zanim_ln_reg",
                                                      "zanidm_reg",
                                                      "dm_reg"),
                   ntrees_theta = 100L, ntrees_zeta = 20L, ndpost = 5000L,
                   nskip = 5000L, keep_draws = TRUE, save_trees = FALSE,
                   sd_prior_beta_count = rep(1.0, ncol(X_count)),
                   sd_prior_beta_zi = diag(1.0, ncol(X_zi)),
                   S_prior_betas = diag(1.0, ncol(X_count)),
                   covariance_type = "fa_mgp", ...) {
  model <- match.arg(model)
  switch(model,
         "mult_bart" = {
           mod <- MultinomialBART$new(Y = Y, X = X_count)
           mod$SetupMCMC(ntrees = ntrees_theta, ndpost = ndpost, nskip = nskip,
                         keep_draws = keep_draws, save_trees = save_trees, ...)
         },
         "mult_ln_bart" = {
           mod <- MultinomialLNBART$new(Y = Y, X = X_count)
           mod$SetupMCMC(ntrees = ntrees_theta, ndpost = ndpost,
                         nskip = nskip, covariance_type = covariance_type,
                         keep_draws = keep_draws, save_trees = save_trees, ...)
         },
         "zanim_bart" = {
           mod <- ZANIMBART$new(Y = Y, X_theta = X_count, X_zeta = X_zi)
           mod$SetupMCMC(ntrees_theta = ntrees_theta, ntrees_zeta = ntrees_zeta,
                         ndpost = ndpost, nskip = nskip, keep_draws = keep_draws,
                         save_trees = save_trees, ...)
         },
         "zanim_ln_bart" = {
           mod <- ZANIMLNBART$new(Y = Y, X_theta = X_count, X_zeta = X_zi)
           mod$SetupMCMC(ntrees_theta = ntrees_theta, ntrees_zeta = ntrees_zeta,
                         ndpost = ndpost, nskip = nskip, keep_draws = keep_draws,
                         save_trees = save_trees, covariance_type = covariance_type,
                         ...)
         },
         "zanim_reg" = {
           mod <- ZANIMRegression$new(Y = Y, X_theta = X_count, X_zeta = X_zi)
           mod$SetupMCMC(ndpost = ndpost, nskip = nskip,
                         sd_prior_beta_theta = sd_prior_beta_count,
                         sd_prior_beta_zeta = sd_prior_beta_zi,
                         keep_draws = keep_draws, ...)
         },
         "zanidm_reg" = {
           mod <- ZANIDMRegression$new(Y = Y, X_alpha = X_count, X_zeta = X_zi)
           mod$SetupMCMC(ndpost = ndpost, nskip = nskip,
                         sd_prior_beta_alpha = sd_prior_beta_count,
                         sd_prior_beta_zeta = sd_prior_beta_zi,
                         keep_draws = keep_draws, ...)
         },
         "zanim_ln_reg" = {
           mod <- ZANIMLNRegression$new(Y = Y, X_theta = X_count, X_zeta = X_zi)
           mod$SetupMCMC(ndpost = ndpost, nskip = nskip,
                         sd_prior_beta_theta = sd_prior_beta_count,
                         sd_prior_beta_zeta = sd_prior_beta_zi,
                         covariance_type = covariance_type,
                         keep_draws = keep_draws, ...)
         },
         "dm_reg" = {
           mod <- DMRegression$new(Y = Y, X = X_count)
           mod$SetupMCMC(S_prior_betas = S_prior_betas,
                         ndpost = ndpost, nskip = nskip, keep_draws = keep_draws,
                         ...)
         }
  )
  mod$RunMCMC()

  return(mod)
}
