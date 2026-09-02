#' Bayesian (non)parametric regression models for zero-inflated multivariate count-compositional data
#'
#' @description
#' Carries out Bayesian inference, through efficient Markov chain Monte Carlo algorithms,
#' for different (non)parametric regression models for the analysis of zero-inflated
#' multivariate count-compositional data.
#'
#' @details
#' The available models differ in their treatment of the compositional
#' probabilities, structural zeros, and additional random effects.
#'
#' \subsection{Count-compositional distributions}{
#'
#' The models are based on distributions for multivariate count-compositional
#' data. Depending on the selected model, additional variation beyond the
#' multinomial sampling distribution can be accommodated using Dirichlet or
#' logistic-normal random effects.
#'
#' For multinomial and multinomial logistic-normal models, the compositional
#' probabilities are linked to covariates using the logistic (softmax) link function.
#' For Dirichlet-multinomial models, the corresponding regression parameters
#' are linked using the log link function.
#' }
#'
#' \subsection{Regression models}{
#'
#' Two types of regression specifications are available.
#'
#' \itemize{
#'   \item \strong{Nonparametric regression:} Covariates are related to
#'   category-specific compositional probabilities and, where applicable,
#'   structural-zero probabilities through independent Bayesian additive
#'   regression tree (BART) priors. This allows nonlinearities and interactions
#'   in the covariate effects to be estimated from the data.
#'
#'   \item \strong{Parametric regression:} Covariates are related to the model
#'   parameters similarly to generalised linear models, with suitable link functions
#'   and a parametric linear predictor.
#' }
#' }
#'
#' \subsection{Structural zeros}{
#'
#' Some models have a mixture-based zero-inflated representation that distinguishes
#' structural zeros from zeros arising from the sampling distribution.
#' For these models, a latent indicator
#' \eqn{z_{ij} \sim \operatorname{Bernoulli}\lbrack 1-\zeta_{ij} \rbrack}
#' determines whether the observed zero from sample \eqn{i} of category \eqn{j}
#' is structural zero with probability \eqn{1-\zeta_{ij}}.
#'
#' The structural-zero probabilities \eqn{\zeta_{ij}} can themselves depend
#' on covariates.
#' In nonparametric models, they are assigned BART priors through a probit-link
#' function, whereas probit regression is employed for the parametric models.
#' }
#'
#' \subsection{Available models}{
#'
#' The following models are available through `model`.
#'
#' \describe{
#'   \item{`"ml_bart"`}{Multinomial logistic BART model.}
#'   \item{`"mln_bart"`}{Multinomial logistic-normal BART model.}
#'   \item{`"zanim_bart"`}{Zero-and-N-inflated multinomial logistic regression model.}
#'   \item{`"zanim_ln_bart"`}{Zero-and-N-inflated multinomial logistic-normal BART model.}
#'   \item{`"dm_reg"`}{Dirichlet-multinomial regression model.}
#'   \item{`"zanidm_reg"`}{Zero-and-N-inflated Dirichlet-multinomial regression model.}
#'   \item{`"zanim_reg"`}{Zero-N-inflated multinomial logistic regression model.}
#'   \item{`"zanim_ln_reg"`}{Zero-and-N-inflated multinomial logistic-normal regression model.}
#' }
#' }
#'
#' @param Y A matrix of multivariate count-compositional data.
#' Rows correspond to observations and columns correspond to categories.
#' @param X_count A matrix of covariates used to model the count probabilities.
#' Rows must correspond to the observations in `Y`.
#' @param X_zi An optional matrix of covariates used to model the structural zero
#' probabilities. If `NULL` (the default), `X_count` is used when covariates are required
#' for the structural-zero component.
#' @param model A character string specifying the model to fit. One of
#' `"zanim_bart"`, `"zanim_ln_bart"`, `"ml_bart"`, `"mln_bart"`, `"zanim_reg"`,
#' `"zanim_ln_reg"`, `"zanidm_reg"`, or `"dm_reg"`. Defaults to `"zanim-bart"`.
#' @param ntrees_theta Number of trees used for the BART prior on the
#' count probabilities. The default is `ntrees_theta=100`.
#' @param ntrees_zeta Number of trees used for the category-specific BART prior on the
#' structural-zero probabilities. The default is `ntrees_zeta=100`.
#' @param ndpost Number of posterior MCMC draws to retain. The default is `ndpost=5000`.
#' @param nskip Number of MCMC iterations to discard as burn-in before retaining
#' posterior draws. The default is `nskip=5000`.
#' @param keep_draws Logical, defaults to `TRUE`. Governs whether to retain posterior draws.
#' @param save_trees Logical, defaults to `FALSE`. Governs whether to save the posterior draws of the BART
#' tree topologies and terminal-node parameters to `.bin` files. For BART-based
#' models, this creates files named `forests_theta_j.bin` for the
#' category-specific compositional regression trees and, for zero-inflated
#' models, `forests_zeta_j.bin` for the structural-zero regression trees.
#' Here, `j` indexes the category, and each file contains the corresponding
#' tree topologies and terminal node parameters across all `ndpost` posterior
#' draws.
#' @param forests_dir Character path indicating where to save the
#' `forests_theta_j.bin` and `forests_zeta_j.bin` files. Defaults to [tempdir()].
#' @param covariance_type Character string specifying the prior on the covariance
#' matrix for the logistic-normal random effects. Defaults to `fa_mgp`, for nonparametric factor
#'  analysis with a multiplicative gamma process shrinkage prior. Other options include `fa` (factor analysis without such a prior),
#'  `diag` (for a diagonal covariance matrix), and `wishart` (for an inverse Wishart prior).
#' This is only used for `mln_bart`, `zanim_ln_bart`, and `zanim_ln_reg` models.
#' @param sd_prior_beta_count Prior standard deviations for the regression
#' coefficients associated with `X_count`.
#' @param sd_prior_beta_zi Prior standard deviations or covariance structure for
#' the regression coefficients associated with `X_zi`.
#' @param S_prior_betas Prior covariance matrix for the regression coefficients
#' associated with the count-compositional component.
#' @param ... Catches unused arguments.
#'
#' @return An R6 object which class depends on the specified `model`.
#' @importFrom R6 "R6Class"
#'
#' @export
zanicc <- function(Y, X_count, X_zi = NULL,
                   model = c(
                     "zanim_bart", "zanim_ln_bart", "ml_bart", "mln_bart",
                     "zanim_reg", "zanim_ln_reg",
                     "zanidm_reg", "dm_reg"
                   ),
                   ntrees_theta = 100L, ntrees_zeta = 100L, ndpost = 5000L,
                   nskip = 5000L, keep_draws = TRUE, save_trees = FALSE,
                   forests_dir = tempdir(),
                   covariance_type = c("fa_mgp", "diag", "wishart", "fa"),
                   sd_prior_beta_count = rep(1.0, ncol(X_count)),
                   sd_prior_beta_zi = diag(1.0, ncol(X_zi)),
                   S_prior_betas = diag(1.0, ncol(X_count)),
                   ...) {
  model <- match.arg(model)

  switch(model,
    "ml_bart" = {
      mod <- MultinomialBART$new(Y = Y, X = X_count)
      mod$SetupMCMC(
        ntrees = ntrees_theta, ndpost = ndpost, nskip = nskip,
        keep_draws = keep_draws, save_trees = save_trees, ...
      )
    },
    "mln_bart" = {
      mod <- MultinomialLNBART$new(Y = Y, X = X_count)
      mod$SetupMCMC(
        ntrees = ntrees_theta, ndpost = ndpost,
        nskip = nskip, covariance_type = covariance_type,
        keep_draws = keep_draws, save_trees = save_trees, ...
      )
    },
    "zanim_bart" = {
      mod <- ZANIMBART$new(Y = Y, X_theta = X_count, X_zeta = X_zi)
      mod$SetupMCMC(
        ntrees_theta = ntrees_theta, ntrees_zeta = ntrees_zeta,
        ndpost = ndpost, nskip = nskip, keep_draws = keep_draws,
        save_trees = save_trees, ...
      )
    },
    "zanim_ln_bart" = {
      mod <- ZANIMLNBART$new(Y = Y, X_theta = X_count, X_zeta = X_zi)
      mod$SetupMCMC(
        ntrees_theta = ntrees_theta, ntrees_zeta = ntrees_zeta,
        ndpost = ndpost, nskip = nskip, keep_draws = keep_draws,
        save_trees = save_trees, covariance_type = covariance_type,
        ...
      )
    },
    "zanim_reg" = {
      mod <- ZANIMRegression$new(Y = Y, X_theta = X_count, X_zeta = X_zi)
      mod$SetupMCMC(
        ndpost = ndpost, nskip = nskip,
        sd_prior_beta_theta = sd_prior_beta_count,
        sd_prior_beta_zeta = sd_prior_beta_zi,
        keep_draws = keep_draws, ...
      )
    },
    "zanidm_reg" = {
      mod <- ZANIDMRegression$new(Y = Y, X_alpha = X_count, X_zeta = X_zi)
      mod$SetupMCMC(
        ndpost = ndpost, nskip = nskip,
        sd_prior_beta_alpha = sd_prior_beta_count,
        sd_prior_beta_zeta = sd_prior_beta_zi,
        keep_draws = keep_draws, ...
      )
    },
    "zanim_ln_reg" = {
      mod <- ZANIMLNRegression$new(Y = Y, X_theta = X_count, X_zeta = X_zi)
      mod$SetupMCMC(
        ndpost = ndpost, nskip = nskip,
        sd_prior_beta_theta = sd_prior_beta_count,
        sd_prior_beta_zeta = sd_prior_beta_zi,
        covariance_type = covariance_type,
        keep_draws = keep_draws, ...
      )
    },
    "dm_reg" = {
      mod <- DMRegression$new(Y = Y, X = X_count)
      mod$SetupMCMC(
        S_prior_betas = S_prior_betas,
        ndpost = ndpost, nskip = nskip, keep_draws = keep_draws,
        ...
      )
    }
  )
  mod$RunMCMC()

  return(mod)
}

# zanicc.controlBART <- function(p_theta, p_zeta, ntrees_theta, ntrees_zeta,
#                         # Terminal node prior
#                         v0_theta = 3.5 / sqrt(2), k_zeta = 3.0,
#                         # Decision rules
#                         numcut = 100L, power = 2.0, base = 0.95,
#                         proposals_prob = c(0.25, 0.25, 0.50),
#                         # Hyperprior on variance of ensembles
#                         update_sigma_theta = TRUE,
#                         s0_2_theta = 1.0 / ntrees_theta,
#                         w_ss = 1.0,
#                         # Split probabilities of each covariate
#                         splitprobs_count = rep(1.0 / p_theta, p_theta),
#                         splitprobs_zi = rep(1.0 / p_zeta, p_zeta),
#                         # Variable selection parameters
#                         sparse = c(FALSE, FALSE),
#                         sparse_parms = c(p_zeta, 0.5, 1.0,
#                                          p_theta, 0.5, 1.0),
#                         alpha_sparse = c(1.0, 1.0), alpha_random = c(FALSE, FALSE),
#                         # Breaks for decision rules
#                         xinfo = matrix()
#                         ) {
#
#   list(
#     ntrees_theta = ntrees_theta, ntrees_zeta = ntrees_zeta,
#     # Parameter controlling the regularised prior on the terminal nodes of BART
#     v0_theta = v0_theta, k_zeta = k_zeta,
#     # Hyperpriors on the terminal node parameters of BART
#     update_sigma_theta = update_sigma_theta,
#     s0_2_theta = s0_2_theta,
#     w_ss = w_ss, # parameter to control the slice sampling
#
#     # Parameters related to the decision rules in the trees
#     numcut = numcut, power = power, base = base,
#     proposals_prob = proposals_prob,
#
#     # Split probabilities of each covariate
#     splitprobs_zi = splitprobs_zi,
#     splitprobs_count = splitprobs_count,
#     # Hyperparameter for the DART prior on split-probabilities
#     sparse_parms = sparse_parms,
#     alpha_sparse = alpha_sparse,
#     alpha_random = alpha_random,
#
#     # Breaks used for each covariate in creating the decision tree rules
#     xinfo = xinfo
#   )
#
# }
#
# zanicc.controlLN <- function(d, nu_prior = d, Psi_prior = diag(d, d - 1),
#                              a_sigma = 1.0, b_sigma = 1.0,
#                              q_factors = .ledermann(d - 1L), sigma2_gamma = 1.0,
#                              a_psi = 2.5, b_psi = 1.0,
#                              shape_lsphis = 3.0, a1_gs = 2.1, a2_gs = 3.1) {
#   list(nu_prior = nu_prior,
#        Psi_prior = Psi_prior, a_sigma = a_sigma, b_sigma = b_sigma,
#        # Factor analysis
#        q_factors = q_factors,
#        sigma2_gamma = sigma2_gamma,
#        # MGP
#        a_psi = a_psi, b_psi = b_psi,
#        shape_lsphis = shape_lsphis,
#        a1_gs = a1_gs, a2_gs = a2_gs)
#
# }
