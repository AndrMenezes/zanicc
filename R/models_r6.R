#' @title ZANIM-BART
#'
#' @description
#' Carries out Bayesian inference for the zero-and-N-inflated multinomial logistic
#' BART (ZANIM-BART) model through an efficient Markov chain Monte Carlo algorithm.
#' The `R6` class is an wrapper for the underlying `C++` implementation.
#'
#' @export
ZANIMBART <- R6::R6Class(classname = "ZANIMBART", cloneable = FALSE, public = list(
  #' @field cpp_obj Internal reference to the underlying `C++` model object.
  cpp_obj = NULL,
  #' @field cpp_module_name Internal name of the `Rcpp` module used by the model.
  cpp_module_name = character(),
  #' @field n_trials Sample-specific total counts (number of trials), calculated as `rowSums(Y)`.
  n_trials = integer(),
  #' @field n Number of samples.
  n = integer(),
  #' @field d Number of categories.
  d = integer(),
  #' @field p_theta Number of covariates associated to the compositional components
  p_theta = integer(),
  #' @field p_zeta Number of covariates associated to the structural zero components
  p_zeta = integer(),
  #' @field ntrees_theta Number of trees for the structural zero components.
  ntrees_theta = integer(),
  #' @field ntrees_zeta Number of trees for the structural zero components.
  ntrees_zeta = integer(),
  #' @field ndpost Number of posterior MCMC draws to retain.
  ndpost = integer(),
  #' @field nskip Number of posterior MCMC draws to discard as burn-in before retaining
  #' posterior draws.
  nskip = integer(),
  #' @field forests_dir Character path indicating where to save the
  #' `forests_theta_j.bin` and `forests_zeta_j.bin` files.
  forests_dir = character(),
  #' @field link_zeta Structural zero link function.
  link_zeta = character(),
  #' @field shared_trees Whether the shared trees are used for the structural
  #' zero components.
  shared_trees = logical(),
  #' @field elapsed_time Elapsed time taken to run the MCMC algorithm.
  elapsed_time = NULL,
  #' @field avg_leaves_theta Average number of leaves across the posterior draws `ndpost`
  #' for the category-specific regression tree ensembles of the compositional components.
  avg_leaves_theta = NULL,
  #' @field avg_leaves_zeta Average number of leaves across the posterior draws `ndpost` for
  #' the category-specific regression tree ensembles of the structural zero components.
  avg_leaves_zeta = NULL,
  #' @field accept_rate_theta Acceptance rate of the Metropolis-Hastings proposals,
  #' `grow`, `prune`, `change`, for the category-specific regression tree ensembles
  #' of the compositional components.
  accept_rate_theta = NULL,
  #' @field accept_rate_zeta Acceptance rate of the Metropolis-Hastings proposals,
  #' `grow`, `prune`, `change`, for category-specific regression tree ensembles
  #' of the structural zero components.
  accept_rate_zeta = NULL,
  #' @field draws_theta Posterior draws of the population-level count probabilities.
  draws_theta = NULL,
  #' @field draws_zeta Posterior draws of the population-level structural zero probabilities.
  draws_zeta = NULL,
  #' @field draws_abundance Posterior draws of the individual-level structural zero probabilities.
  draws_abundance = NULL,
  #' @field keep_draws Logical indicating whether posterior draws were retained.
  keep_draws = logical(),
  #' @field save_trees Logical indicating whether the posterior forests were
  #' saved in disk.
  save_trees = logical(),
  #' @field varcount_theta A three dimensional array with dimension \eqn{p_{\theta} \times d \times m},
  #' where \eqn{p_\theta} is the number of covariates for the compositional components,
  #' \eqn{d} is the number of categories and \eqn{m} is the number of posterior draws, `ndpost`.
  #' Contains the total count of the number of times that variable is used in a
  #' tree decision rule over all category-specific trees.
  varcount_theta = NULL,
  #' @field varcount_zeta A three dimensional array with dimension \eqn{p_{\zeta} \times d \times m},
  #' where \eqn{p_\zeta} is the number of covariates for the structural zero components,
  #' \eqn{d} is the number of categories and \eqn{m} is the number of posterior draws, `ndpost`.
  #' Contains the total count of the number of times that variable is used in a
  #' tree decision rule over all category-specific trees.
  varcount_zeta = NULL,
  #' @field mppi_theta A matrix with rows being the covariates and columns the categories.
  #' It contains the posterior estimates of the marginal probability of inclusion
  #' (MPPI) for the category-specific covariates associated to the compositional components.
  mppi_theta = NULL,
  #' @field mppi_zeta A matrix with rows being the covariates and columns the categories.
  #' It contains the posterior estimates of the marginal probability of inclusion
  #' (MPPI) for the category-specific covariates associated to the compositional components.
  mppi_zeta = NULL,
  #' @field sigma_theta_hyperprior Posterior distribution of the hyperparameter related to the
  #' shrinkage prior in the compositional component.
  sigma_theta_hyperprior = NULL,

  #' Create a new `ZANIMBART` object
  #' @param Y A matrix of multivariate count-compositional data.
  #' Rows correspond to observations and columns correspond to categories.
  #' @param X_theta A matrix of covariates used to model the count probabilities.
  #' Rows must correspond to the observations in `Y`.
  #' @param X_zeta A matrix of covariates used to model the structural zero probabilities.
  #' Rows must correspond to the observations in `Y`.
  #' @param link_zeta Link function for the structural zero components.
  #' Options are `probit` and `logit`. Default and recommended are `probit`.
  #' @param shared_trees Whether the shared trees are used for the structural
  #' zero components. Only applied for the `logit` link function.
  initialize = function(Y, X_theta, X_zeta, link_zeta = c("probit", "logit"),
                        shared_trees = FALSE) {
    link_zeta <- match.arg(link_zeta)
    self$link_zeta <- link_zeta
    # Call the C++ class in R
    if (link_zeta == "logit") {
      ml <- Rcpp::Module(module = "zanim_bart_logit", PACKAGE = "zanicc")
      self$cpp_obj <- new(ml$ZANIMBARTLogit, Y, X_theta, X_zeta)
      self$cpp_module_name <- "zanim_bart_logit"
      if (shared_trees) warning("Shared trees only implemented with the probit.")
      shared_trees <- FALSE
    } else {
      if (shared_trees) {
        ml <- Rcpp::Module(module = "zanim_shared_bart_probit", PACKAGE = "zanicc")
        self$cpp_obj <- new(ml$ZANIMSharedBARTProbit, Y, X_theta, X_zeta)
        self$cpp_module_name <- "zanim_shared_bart_probit"
      } else {
        ml <- Rcpp::Module(module = "zanim_bart_probit", PACKAGE = "zanicc")
        self$cpp_obj <- new(ml$ZANIMBARTProbit, Y, X_theta, X_zeta)
        self$cpp_module_name <- "zanim_bart_probit"
      }
    }
    self$shared_trees <- shared_trees
    self$n <- nrow(Y)
    self$d <- ncol(Y)
    self$p_theta <- ncol(X_theta)
    self$p_zeta <- ncol(X_zeta)
    self$n_trials <- rowSums(Y)
  },

  #' Set up the settings for the MCMC algorithm
  #' @description
  #' Configures priors and hyperparameters of the ZANIM-BART model
  #' used by the underlying MCMC algorithm implemented in `C++`.
  #' This method must be called before \href{#method-ZANIMBART-RunMCMC}{\code{ZANIMBART$RunMCMC()}}.
  #' @param v0_theta Hyperparameter controlling the level of shrinkage of the
  #' regression trees for the compositional component. The smaller `v0_theta` is,
  #' the more shrinkage is applied, i.e., shallow trees are expected.
  #' @param k_zeta Hyperparameter controlling the level of shrinkage of the
  #' regression trees for the structural zero component. The smaller `k_zeta` is,
  #' the more shrinkage is applied, i.e., shallow trees are expected.
  #' Default is `k_zeta = 3.0`, which assigns a prior probability of 0.95 that the
  #' structural zero probability is between `qnorm(-3)` and `qnorm(3)`.
  #' @param ntrees_theta Number of trees used for the BART prior on the
  #' count probabilities. The default is `ntrees_theta=100`.
  #' @param ntrees_zeta Number of trees used for the category-specific BART prior on the
  #' structural-zero probabilities. The default is `ntrees_zeta=100`.
  #' @param ndpost Number of posterior MCMC draws to retain. The default is `ndpost=5000`.
  #' @param nskip Number of MCMC iterations to discard as burn-in before retaining
  #' posterior draws. The default is `nskip=5000`.
  #' @param numcut Total number of cut points \eqn{c_b} used to form
  #' the splitting decision rules \eqn{x_{jb} \leq c_b}. For each covariate we
  #' generate `numcut` equally space cut points, \eqn{c_b} in the range of the corresponding covariate. Default is `numcut=100`.
  #' @param power Power parameter regarding the tree prior. Default is `power=2.0`.
  #' @param base Base parameter regarding the tree prior. Default is `power=0.95`.
  #' @param proposals_prob
  #' Numeric vector of length three containing the probabilities of proposing the
  #' `grow`, `prune`, and `change` tree moves, respectively.
  #' Default probabilities are \eqn{0.25}, \eqn{0.25} and \eqn{0.50}, respectively.
  #' @param update_sigma_theta Logical indicating whether the hyperprior should be
  #' used for the shrinkage hyperparameter, `v0_theta`. If so, then we use slice
  #' sampling to update this hyperparameter during the MCMC.
  #' @param s0_2_theta Hyperprior scale parameter for the `v0_theta` hyperparameter. Default is `1/ntrees_theta`.
  #' @param w_ss Hyperprameter for stepping out method in the slice sampling algorithm.
  #' It controls the width of the slice.
  #' @param splitprobs_zi Numeric vector with the prior probabilities of each
  #' covariate in `X_zi` to generate a splitting rule. Default is `1/p_zeta`.
  #' @param splitprobs_mult Numeric vector with the prior probabilities of each
  #' covariate in `X_count` to generate a splitting rule. Default is `1/p_theta`.
  #' @param sparse Logical vector of length two indicating whether to perform
  #' variable selection based on the sparse Dirichlet prior
  #' of Linero (2018) rather than uniform prior on the splitting probabilities of the
  #' structural zero and compositional components, respectively.
  #' This prior assumes that the splitting probability vector follows
  #' \eqn{\mathbf{s} \sim \operatorname{Dirichlet}\lbrack \alpha/p, \ldots, \alpha/p \rbrack},
  #' with \eqn{\alpha} a hyperparameter and \eqn{p} number of covariates.
  #' @param alpha_sparse Numeric vector of length two with the hyperprameter values
  #' of \eqn{\alpha} which controls the level of sparsity of the Dirichlet prior on the splitting
  #' probabilities for the structural zero and compositional components, respectively.
  #' Default is `alpha_sparse = c(1, 1)`. As \eqn{\alpha \rightarrow \infty}, it recovers the
  #' default uniform prior on the splitting probabilities under BART.
  #' @param alpha_random Logical vector of length two indicating whether to put a
  #' hyperprior on \eqn{\alpha} for the structural zero and compositional components,
  #' respectively. The hyperprior is of the form
  #' \eqn{\alpha / (\alpha + \rho) \sim \operatorname{Beta}\lbrack a, b \rbrack}
  #' with further hyperprior parameters \eqn{\rho}, \eqn{a} and \eqn{b}.
  #' @param sparse_parms Numeric vector of length six for the hyperprior parameters
  #' \eqn{\rho}, \eqn{a} and \eqn{b}. The first three entries correspond to
  #' the structural-zero component and the last three to the compositional
  #' component. By default, these are `c(p_zeta, 0.5, 1.0, p_theta, 0.5, 1.0)`.
  #' @param forests_dir Character path indicating where to save the
  #' `forests_theta_j.bin` and `forests_zeta_j.bin` files. Default is to [tempdir()].
  #' @param xinfo Optional matrix containing the cut points information of each
  #' covariate supplied to the underlying `C++` implementation.
  #' An empty matrix requests that the cut points be determined internally.
  #' @param keep_draws Logical, defaults to `TRUE`. Governs whether to retain posterior draws.
  #' @param save_trees Logical, defaults to `FALSE`. Governs whether to save the posterior draws of the BART
  #' tree topologies and terminal-node parameters to `.bin` files. For BART-based
  #' models, this creates files named `forests_theta_j.bin` for the
  #' category-specific compositional regression trees and, for zero-inflated
  #' models, `forests_zeta_j.bin` for the structural-zero regression trees.
  #' Here, `j` indexes the category, and each file contains the corresponding
  #' tree topologies and terminal node parameters across all `ndpost` posterior
  #' draws.
  SetupMCMC = function(v0_theta = 1.5 / sqrt(2),
                       k_zeta = if (self$link_zeta == "logit") 3.5 / sqrt(2) else 3.0,
                       ntrees_theta = 100L, ntrees_zeta = 100L,
                       ndpost = 5000L, nskip = 5000L,
                       numcut = 100L, power = 2.0, base = 0.95,
                       proposals_prob = c(0.25, 0.25, 0.50),
                       update_sigma_theta = TRUE, s0_2_theta = 1.0 / ntrees_theta,
                       w_ss = 1.0,
                       splitprobs_zi = rep(1.0 / self$p_zeta, self$p_zeta),
                       splitprobs_mult = rep(1.0 / self$p_theta, self$p_theta),
                       sparse = c(FALSE, FALSE),
                       alpha_sparse = c(1.0, 1.0), alpha_random = c(FALSE, FALSE),
                       sparse_parms = c(self$p_zeta, 0.5, 1.0,
                                        self$p_theta, 0.5, 1.0),
                       xinfo = matrix(), forests_dir = tempdir(),
                       keep_draws = TRUE, save_trees = FALSE) {
    self$ntrees_theta <- ntrees_theta
    self$ntrees_zeta <- ntrees_zeta
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$forests_dir <- forests_dir
    self$keep_draws <- keep_draws
    self$save_trees <- save_trees
    alpha_sparse_mult <- alpha_sparse[2L]
    if (!self$shared_trees) {
      if (!is.list(splitprobs_mult)) {
        splitprobs_mult <- replicate(self$d, splitprobs_mult, simplify = FALSE)
      }
      alpha_sparse_mult <- rep(alpha_sparse[2L], self$d)
    }
    self$cpp_obj$SetMCMC(
      v0_theta, k_zeta, ntrees_theta, ntrees_zeta, ndpost, nskip,
      numcut, power, base, proposals_prob,
      as.integer(update_sigma_theta), s0_2_theta, w_ss,
      splitprobs_zi, splitprobs_mult,
      as.integer(sparse[1L]), as.integer(sparse[2L]),
      sparse_parms[1L:3L], sparse_parms[4L:6L],
      rep(alpha_sparse[1L], self$d), alpha_sparse_mult,
      as.integer(alpha_random[1L]), as.integer(alpha_random[2L]),
      xinfo, forests_dir, as.integer(keep_draws),
      as.integer(save_trees)
    )
  },
  #' Run the MCMC algorithm of ZANIM-BART
  #'
  #' @description
  #'  Runs the MCMC sampler using the settings previously configured with
  #' \href{#method-ZANIMBART-SetupMCMC}{\code{ZANIMBART$SetupMCMC()}}.
  #' Posterior draws, acceptance rates, and
  #' variable-selection statistics, are then transferred from the underlying `C++`
  #' object to the `ZANIMBART` object.
  #'
  RunMCMC = function() {
    ini <- proc.time()
    self$cpp_obj$RunMCMC()
    self$elapsed_time <- proc.time() - ini

    # Average number of leaves for theta and zeta regression trees
    self$avg_leaves_theta <- self$cpp_obj$avg_leaves_theta / self$ndpost
    self$avg_leaves_zeta <- self$cpp_obj$avg_leaves_zeta / self$ndpost
    # Avg accept rate over iteration and the trees
    self$accept_rate_theta <- self$cpp_obj$accept_rate_theta / (self$ndpost + self$nskip) / self$ntrees_theta
    self$accept_rate_zeta <- self$cpp_obj$accept_rate_zeta / (self$ndpost + self$nskip) / self$ntrees_zeta
    rownames(self$accept_rate_zeta) <- rownames(self$accept_rate_theta) <- c("grow", "prune", "change")
    # Keep the draws of the hyperprior sd
    self$sigma_theta_hyperprior <- self$cpp_obj$sigma_mult_mcmc
    # Save draws
    if (self$keep_draws) {
      self$draws_theta <- self$cpp_obj$draws_theta
      self$draws_abundance <- self$cpp_obj$draws_vartheta
      self$draws_zeta <- self$cpp_obj$draws_zeta
      if (self$link_zeta == "probit") self$draws_zeta <- stats::pnorm(self$draws_zeta)
      # self$draws_phi <- self$cpp_obj$draws_phi
      self$varcount_theta <- self$cpp_obj$varcount_mcmc_theta
      self$varcount_zeta <- self$cpp_obj$varcount_mcmc_zeta
      self$mppi_theta <- apply(self$cpp_obj$varcount_mcmc_theta > 0, c(1, 2), mean)
      self$mppi_zeta <- apply(self$cpp_obj$varcount_mcmc_zeta > 0, c(1, 2), mean)
    }
  }
))

#' @title ZANIM-LN-BART
#'
#' @description
#' Carries out Bayesian inference for the zero-and-N-inflated multinomial
#' logistic-normal BART (ZANIM-LN-BART) model through an efficient
#' Markov chain Monte Carlo algorithm.
#' The `R6` class is an wrapper for the underlying `C++` implementation.
#'
#' @export
ZANIMLNBART <- R6::R6Class(classname = "ZANIMLNBART", cloneable = FALSE,
                           public = list(
  #' @field cpp_obj Internal reference to the underlying `C++` model object.
  cpp_obj = NULL,
  #' @field cpp_module_name Internal name of the `Rcpp` module used by the model.
  cpp_module_name = character(),
  #' @field n_trials Sample-specific total counts (number of trials), calculated as `rowSums(Y)`.
  n_trials = integer(),
  #' @field n Number of samples.
  n = integer(),
  #' @field d Number of categories.
  d = integer(),
  #' @field p_theta Number of covariates associated to the compositional components
  p_theta = integer(),
  #' @field p_zeta Number of covariates associated to the structural zero components
  p_zeta = integer(),
  #' @field ntrees_theta Number of trees for the structural zero components.
  ntrees_theta = integer(),
  #' @field ntrees_zeta Number of trees for the structural zero components.
  ntrees_zeta = integer(),
  #' @field ndpost Number of posterior MCMC draws to retain.
  ndpost = integer(),
  #' @field nskip Number of posterior MCMC draws to discard as burn-in before retaining
  #' posterior draws.
  nskip = integer(),
  #' @field forests_dir Character path indicating where to save the
  #' `forests_theta_j.bin` and `forests_zeta_j.bin` files.
  forests_dir = character(),
  #' @field covariance_type Character string with the prior used for the covariance
  #' matrix for the logistic-normal random effects.
  covariance_type = NULL,
  #' @field Bt transpose of the orthogonal matrix for the sum-to-zero constraint
  #' in the logistic random effects.
  Bt = NULL,
  #' @field elapsed_time Elapsed time taken to run the MCMC algorithm.
  elapsed_time = NULL,
  #' @field avg_leaves_theta Average number of leaves across the posterior draws `ndpost`
  #' for the category-specific regression tree ensembles of the compositional components.
  avg_leaves_theta = NULL,
  #' @field avg_leaves_zeta Average number of leaves across the posterior draws `ndpost` for
  #' the category-specific regression tree ensembles of the structural zero components.
  avg_leaves_zeta = NULL,
  #' @field accept_rate_theta Acceptance rate of the Metropolis-Hastings proposals,
  #' `grow`, `prune`, `change`, for the category-specific regression tree ensembles
  #' of the compositional components.
  accept_rate_theta = NULL,
  #' @field accept_rate_zeta Acceptance rate of the Metropolis-Hastings proposals,
  #' `grow`, `prune`, `change`, for category-specific regression tree ensembles
  #' of the structural zero components.
  accept_rate_zeta = NULL,
  #' @field draws_theta Posterior draws of the population-level count probabilities.
  draws_theta = NULL,
  #' @field draws_zeta Posterior draws of the population-level structural zero probabilities.
  draws_zeta = NULL,
  #' @field draws_chol_Sigma_V Posterior draws of the Cholesky decomposition of the
  #' covariance matrix of the logistic-normal random effects.
  draws_chol_Sigma_V = NULL,
  #' @field draws_abundance Posterior draws of the individual-level structural zero probabilities.
  draws_abundance = NULL,
  #' @field keep_draws Logical indicating whether posterior draws were retained.
  keep_draws = logical(),
  #' @field save_trees Logical indicating whether the posterior forests were
  #' saved in disk.
  save_trees = logical(),
  #' @field varcount_theta A three dimensional array with dimension \eqn{p_{\theta} \times d \times m},
  #' where \eqn{p_\theta} is the number of covariates for the compositional components,
  #' \eqn{d} is the number of categories and \eqn{m} is the number of posterior draws, `ndpost`.
  #' Contains the total count of the number of times that variable is used in a
  #' tree decision rule over all category-specific trees.
  varcount_theta = NULL,
  #' @field varcount_zeta A three dimensional array with dimension \eqn{p_{\zeta} \times d \times m},
  #' where \eqn{p_\zeta} is the number of covariates for the structural zero components,
  #' \eqn{d} is the number of categories and \eqn{m} is the number of posterior draws, `ndpost`.
  #' Contains the total count of the number of times that variable is used in a
  #' tree decision rule over all category-specific trees.
  varcount_zeta = NULL,
  #' @field mppi_theta A matrix with rows being the covariates and columns the categories.
  #' It contains the posterior estimates of the marginal probability of inclusion
  #' (MPPI) for the category-specific covariates associated to the compositional components.
  mppi_theta = NULL,
  #' @field mppi_zeta A matrix with rows being the covariates and columns the categories.
  #' It contains the posterior estimates of the marginal probability of inclusion
  #' (MPPI) for the category-specific covariates associated to the compositional components.
  mppi_zeta = NULL,
  #' @field sigma_theta_hyperprior Posterior distribution of the hyperparameter related to the
  #' shrinkage prior in the compositional component.
  sigma_theta_hyperprior = NULL,

  #' Create a new `ZANIMLNBART` object
  #' @param Y A matrix of multivariate count-compositional data.
  #' Rows correspond to observations and columns correspond to categories.
  #' @param X_theta A matrix of covariates used to model the count probabilities.
  #' Rows must correspond to the observations in `Y`.
  #' @param X_zeta A matrix of covariates used to model the structural zero probabilities.
  #' Rows must correspond to the observations in `Y`.
  initialize = function(Y, X_theta, X_zeta) {
    ml <- Rcpp::Module(module = "zanim_ln_bart", PACKAGE = "zanicc")
    self$cpp_obj <- new(ml$ZANIMLNBART, Y, X_theta, X_zeta)
    self$cpp_module_name <- "zanim_ln_bart"
    self$n <- nrow(Y)
    self$d <- ncol(Y)
    self$p_theta <- ncol(X_theta)
    self$p_zeta <- ncol(X_zeta)
    self$n_trials <- rowSums(Y)
  },

  #' Set up the settings for the MCMC algorithm
  #' @description
  #' Configures priors and hyperparameters of the ZANIM-LN-BART model
  #' used by the underlying MCMC algorithm implemented in `C++`.
  #' This method must be called before
  #' \href{#method-ZANIMLNBART-RunMCMC}{\code{ZANIMLNBART$RunMCMC()}}.
  #' @param v0_theta Hyperparameter controlling the level of shrinkage of the
  #' regression trees for the compositional component. The smaller `v0_theta` is,
  #' the more shrinkage is applied, i.e., shallow trees are expected.
  #' @param k_zeta Hyperparameter controlling the level of shrinkage of the
  #' regression trees for the structural zero component. The smaller `k_zeta` is,
  #' the more shrinkage is applied, i.e., shallow trees are expected.
  #' Default is `k_zeta = 3.0`, which assigns a prior probability of 0.95 that the
  #' structural zero probability is between `qnorm(-3)` and `qnorm(3)`.
  #' @param ntrees_theta Number of trees used for the BART prior on the
  #' count probabilities. The default is `ntrees_theta=100`.
  #' @param ntrees_zeta Number of trees used for the category-specific BART prior on the
  #' structural-zero probabilities. The default is `ntrees_zeta=100`.
  #' @param ndpost Number of posterior MCMC draws to retain. The default is `ndpost=5000`.
  #' @param nskip Number of MCMC iterations to discard as burn-in before retaining
  #' posterior draws. The default is `nskip=5000`.
  #' @param covariance_type Character string specifying the prior on the covariance
  #' matrix for the logistic-normal random effects. Defaults to `fa_mgp`, for nonparametric factor
  #'  analysis with a multiplicative gamma process shrinkage prior. Other options include `fa` (factor analysis without such a prior),
  #'  `diag` (for a diagonal covariance matrix), and `wishart` (for an inverse Wishart prior).
  #' @param nu_prior Degrees of freedom for the inverse-Wishart prior on the
  #' covariance matrix of random effects, when \code{covariance_type="wishart"}.
  #' Default is number of categories, `self$d`.
  #' @param Psi_prior Prior scale matrix for the inverse-Wishart prior on the
  #' covariance matrix of random effects, when \code{covariance_type="wishart"}.
  #' Default is \eqn{\mathbf{I}_{d-1}d}, where \eqn{d} is the number of categories,
  #' `self$d`.
  #' @param a_sigma,b_sigma Shape and scale prior parameters for the independent
  #' gamma priors on the covariance matrix, i.e., when \code{covariance_type="diag"}.
  #' Default is `a_sigma=b_sigma=1.0`.
  #' @param q_factors Number of factors when the prior for the covariance matrix is
  #' has a factor-analytic representation, i.e., \code{covariance_type="fa"} or
  #' \code{covariance_type="fa_mgp"}. Default is the Ledermann bound of
  #' the dimension of the full covariance matrix.
  #' @param sigma2_gamma Scale (variance) hyperparameter of the normal prior on the
  #' factor loadings, when \code{covariance_type="fa"}.
  #' @param a_psi,b_psi Shape and rate hyperparameters, respectively for the gamma
  #' prior on the residual precisions of the error term when \code{covariance_type="fa"}
  #' or \code{covariance_type="fa_mgp"}.
  #' @param shape_lsphis Shape hyperprameter of the gamma prior on local shrinkage
  #' parameters under the multiplicative gamma process (MGP) prior, i.e.,
  #' when \code{covariance_type="fa_mgp"}. Default is `shape_lsphis=3.0`.
  #' @param a1_gs,a2_gs Shaper hyperparameters for the gamma prior on the column-wise
  #' global shrinkage paraemters under the multiplicative gamma process (MGP) prior, i.e.,
  #' when \code{covariance_type="fa_mgp"}. Default values are `a1_gs=2.1` and `a2_gs=3.1`.
  #' @param numcut Total number of cut points \eqn{c_b} used to form
  #' the splitting decision rules \eqn{x_{jb} \leq c_b}. For each covariate we
  #' generate `numcut` equally space cut points, \eqn{c_b} in the range of the corresponding covariate. Default is `numcut=100`.
  #' @param power Power parameter regarding the tree prior. Default is `power=2.0`.
  #' @param base Base parameter regarding the tree prior. Default is `power=0.95`.
  #' @param proposals_prob Numeric vector of length three containing the probabilities of proposing the
  #' `grow`, `prune`, and `change` tree moves, respectively.
  #' Default probabilities are \eqn{0.25}, \eqn{0.25} and \eqn{0.50}, respectively.
  #' @param update_sigma_theta Logical indicating whether the hyperprior should be
  #' used for the shrinkage hyperparameter, `v0_theta`. If so, then we use slice
  #' sampling to update this hyperparameter during the MCMC.
  #' @param s0_2_theta Hyperprior scale parameter for the `v0_theta` hyperparameter. Default is `1/ntrees_theta`.
  #' @param w_ss Hyperprameter for stepping out method in the slice sampling algorithm.
  #' It controls the width of the slice.
  #' @param splitprobs_zi Numeric vector with the prior probabilities of each
  #' covariate in `X_zi` to generate a splitting rule. Default is `1/p_zeta`.
  #' @param splitprobs_mult Numeric vector with the prior probabilities of each
  #' covariate in `X_count` to generate a splitting rule. Default is `1/p_theta`.
  #' @param sparse Logical vector of length two indicating whether to perform
  #' variable selection based on the sparse Dirichlet prior
  #' of Linero (2018) rather than uniform prior on the splitting probabilities of the
  #' structural zero and compositional components, respectively.
  #' This prior assumes that the splitting probability vector follows
  #' \eqn{\mathbf{s} \sim \operatorname{Dirichlet}\lbrack \alpha/p, \ldots, \alpha/p \rbrack},
  #' with \eqn{\alpha} a hyperparameter and \eqn{p} number of covariates.
  #' @param alpha_sparse Numeric vector of length two with the hyperprameter values
  #' of \eqn{\alpha} which controls the level of sparsity of the Dirichlet prior on the splitting
  #' probabilities for the structural zero and compositional components, respectively.
  #' Default is `alpha_sparse = c(1, 1)`. As \eqn{\alpha \rightarrow \infty}, it recovers the
  #' default uniform prior on the splitting probabilities under BART.
  #' @param alpha_random Logical vector of length two indicating whether to put a
  #' hyperprior on \eqn{\alpha} for the structural zero and compositional components,
  #' respectively. The hyperprior is of the form
  #' \eqn{\alpha / (\alpha + \rho) \sim \operatorname{Beta}\lbrack a, b \rbrack}
  #' with further hyperprior parameters \eqn{\rho}, \eqn{a} and \eqn{b}.
  #' @param sparse_parms Numeric vector of length six for the hyperprior parameters
  #' \eqn{\rho}, \eqn{a} and \eqn{b}. The first three entries correspond to
  #' the structural-zero component and the last three to the compositional
  #' component. By default, these are `c(p_zeta, 0.5, 1.0, p_theta, 0.5, 1.0)`.
  #' @param forests_dir Character path indicating where to save the
  #' `forests_theta_j.bin` and `forests_zeta_j.bin` files. Default is to [tempdir()].
  #' @param xinfo Optional matrix containing the cut points information of each
  #' covariate supplied to the underlying `C++` implementation.
  #' An empty matrix requests that the cut points be determined internally.
  #' @param keep_draws Logical, defaults to `TRUE`. Governs whether to retain posterior draws.
  #' @param save_trees Logical, defaults to `FALSE`. Governs whether to save the posterior draws of the BART
  #' tree topologies and terminal-node parameters to `.bin` files. For BART-based
  #' models, this creates files named `forests_theta_j.bin` for the
  #' category-specific compositional regression trees and, for zero-inflated
  #' models, `forests_zeta_j.bin` for the structural-zero regression trees.
  #' Here, `j` indexes the category, and each file contains the corresponding
  #' tree topologies and terminal node parameters across all `ndpost` posterior
  #' draws.
  SetupMCMC = function(v0_theta = 1.5 / sqrt(2), k_zeta = 3.0,
                       ntrees_theta = 100L, ntrees_zeta = 100L,
                       ndpost = 5000L, nskip = 5000L,
                       covariance_type = c("fa_mgp", "diag", "wishart", "fa"),
                       #### Related to the covariance
                       # Inv-Wishart
                       nu_prior = self$d,
                       Psi_prior = diag(self$d, self$d - 1),
                       # Independent gamma, diag
                       a_sigma = 1.0, b_sigma = 1.0,
                       # FA
                       q_factors = .ledermann(self$d - 1L),
                       a_psi = 2.5, b_psi = 1.0,
                       sigma2_gamma = 1.0,
                       # MGP
                       shape_lsphis = 3.0,
                       a1_gs = 2.1, a2_gs = 3.1,
                       ####
                       numcut = 100L, power = 2.0, base = 0.95,
                       proposals_prob = c(0.25, 0.25, 0.50),
                       update_sigma_theta = TRUE, s0_2_theta = 1 / ntrees_theta,
                       w_ss = 1.0,
                       splitprobs_zi = rep(1 / self$p_zeta, self$p_zeta),
                       splitprobs_mult = rep(1 / self$p_theta, self$p_theta),
                       sparse = c(FALSE, FALSE),
                       sparse_parms = c(self$p_zeta, 0.5, 1.0,
                                        self$p_theta, 0.5, 1.0),
                       alpha_sparse = c(1.0, 1.0), alpha_random = c(FALSE, FALSE),
                       xinfo = matrix(), forests_dir = tempdir(),
                       keep_draws = TRUE, save_trees = FALSE) {
    covariance_type <- match.arg(covariance_type)
    cov_type <- as.integer(which(covariance_type == c("diag", "wishart", "fa", "fa_mgp"))) - 1L
    if (q_factors == 0) q_factors <- self$d - 1
    self$covariance_type <- covariance_type
    self$ntrees_theta <- ntrees_theta
    self$ntrees_zeta <- ntrees_zeta
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$forests_dir <- forests_dir
    self$keep_draws <- keep_draws
    self$save_trees <- save_trees
    alpha_sparse_mult <- alpha_sparse[2L]
    if (!is.list(splitprobs_mult)) {
      splitprobs_mult <- replicate(self$d, splitprobs_mult, simplify = FALSE)
    }
    alpha_sparse_mult <- rep(alpha_sparse[2L], self$d)
    B <- qr.Q(qr(stats::contr.sum(self$d)))
    self$Bt <- t(B)
    self$cpp_obj$SetMCMC(
      v0_theta, k_zeta, ntrees_theta, ntrees_zeta,
      B, cov_type,
      a_sigma, b_sigma,
      Psi_prior, nu_prior,
      q_factors, sigma2_gamma, a_psi, b_psi,
      shape_lsphis, a1_gs, a2_gs,
      ndpost, nskip,
      numcut, power, base, proposals_prob,
      as.integer(update_sigma_theta), s0_2_theta, w_ss,
      splitprobs_zi, splitprobs_mult,
      as.integer(sparse[1L]), as.integer(sparse[2L]),
      sparse_parms[1L:3L], sparse_parms[4L:6L],
      rep(alpha_sparse[1L], self$d), alpha_sparse_mult,
      as.integer(alpha_random[1L]), as.integer(alpha_random[2L]),
      xinfo, forests_dir, as.integer(keep_draws), as.integer(save_trees)
    )
  },
  #' Run the MCMC algorithm of ZANIM-LN-BART
  #'
  #' @description
  #'  Runs the MCMC sampler using the settings previously configured with
  #' \href{#method-ZANIMLNBART-SetupMCMC}{\code{ZANIMLNBART$SetupMCMC()}}.
  #' Posterior draws, acceptance rates, and variable-selection statistics, are
  #' then transferred from the underlying `C++` object to the `ZANIMLNBART` object.
  RunMCMC = function() {
    ini <- proc.time()
    self$cpp_obj$RunMCMC()
    self$elapsed_time <- proc.time() - ini
    # Average number of leaves for theta and zeta regression trees
    self$avg_leaves_theta <- self$cpp_obj$avg_leaves_theta / self$ndpost
    self$avg_leaves_zeta <- self$cpp_obj$avg_leaves_zeta / self$ndpost
    # Avg accept rate over iteration and the trees
    self$accept_rate_theta <- self$cpp_obj$accept_rate_theta / (self$nskip + self$ndpost) / self$ntrees_theta
    self$accept_rate_zeta <- self$cpp_obj$accept_rate_zeta / (self$nskip + self$ndpost) / self$ntrees_zeta
    rownames(self$accept_rate_zeta) <- rownames(self$accept_rate_theta) <- c("grow", "prune", "change")
    self$sigma_theta_hyperprior <- self$cpp_obj$sigma_mult_mcmc
    # Save draws
    if (self$keep_draws) {
      self$draws_theta <- self$cpp_obj$draws_theta
      self$draws_abundance <- self$cpp_obj$draws_vartheta
      self$draws_zeta <- stats::pnorm(self$cpp_obj$draws_zeta)
      # self$draws_phi <- self$cpp_obj$draws_phi
      self$draws_chol_Sigma_V <- self$cpp_obj$draws_chol_Sigma_V
      self$varcount_theta <- self$cpp_obj$varcount_mcmc_theta
      self$varcount_zeta <- self$cpp_obj$varcount_mcmc_zeta
      self$mppi_theta <- apply(self$cpp_obj$varcount_mcmc_theta > 0, c(1, 2), mean)
      self$mppi_zeta <- apply(self$cpp_obj$varcount_mcmc_zeta > 0, c(1, 2), mean)
    }
  }
))


# Multinomial logistic BART
MultinomialBART <- R6::R6Class(classname = "MultinomialBART", public = list(
  cpp_obj = NULL, cpp_module_name = character(),
  n_trials = integer(), n = integer(), d = integer(), p = integer(),
  ntrees = integer(), ndpost = integer(), nskip = integer(), forests_dir = character(),
  shared_trees = logical(),
  elapsed_time = NULL, elapsed_time_log_lik = NULL, avg_leaves = NULL,
  avg_depth = NULL, accept_rate = NULL, lpl = NULL, draws_theta = NULL,
  draws_phi = NULL, keep_draws = logical(), save_trees = logical(),
  varcount = NULL, mppi = NULL,
  initialize = function(Y, X, shared_trees = FALSE) {
    self$shared_trees <- shared_trees
    # Call the C++ class in R
    if (!self$shared_trees) {
      ml <- Rcpp::Module(module = "multinomial_bart", PACKAGE = "zanicc")
      self$cpp_obj <- new(ml$MultinomialBART, Y, X)
      self$cpp_module_name <- "multinomial_bart"
    } else {
      ml <- Rcpp::Module(module = "multinomial_shared_bart", PACKAGE = "zanicc")
      self$cpp_obj <- new(ml$MultinomialSharedBART, Y, X)
      self$cpp_module_name <- "multinomial_shared_bart"
    }
    self$n <- nrow(Y)
    self$d <- ncol(Y)
    self$p <- ncol(X)
    self$n_trials <- rowSums(Y)
  },
  SetupMCMC = function(v0 = 3.5 / sqrt(2), ntrees = 20L, ndpost = 1000L,
                       nskip = 1000L, numcut = 100L,
                       power = 2.0, base = 0.95,
                       proposals_prob = c(0.25, 0.25, 0.50),
                       update_sigma = TRUE, s2_0 = 1 / ntrees, w_ss = 1.0,
                       splitprobs = rep(1 / self$p, self$p), sparse = FALSE,
                       sparse_parms = c(self$p, 0.5, 1.0), alpha_sparse = 1.0,
                       alpha_random = FALSE, xinfo = matrix(), forests_dir = tempdir(),
                       keep_draws = TRUE, save_trees = FALSE) {
    self$ntrees <- ntrees
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$forests_dir <- forests_dir
    self$keep_draws <- keep_draws
    self$save_trees <- save_trees
    if (!self$shared_trees) {
      if (!is.list(splitprobs)) splitprobs <- replicate(self$d, splitprobs, simplify = FALSE)
      alpha_sparse <- rep(alpha_sparse, self$d)
    }
    # Setup
    self$cpp_obj$SetMCMC(
      v0, ntrees, ndpost, nskip, numcut, power, base,
      proposals_prob, as.integer(update_sigma), s2_0, w_ss,
      splitprobs, as.integer(sparse), sparse_parms,
      alpha_sparse, as.integer(alpha_random), xinfo, forests_dir,
      keep_draws, save_trees
    )
  },
  RunMCMC = function() {
    ini <- proc.time()
    self$cpp_obj$RunMCMC()
    self$elapsed_time <- proc.time() - ini
    # Keep some tree diagnostics
    self$avg_leaves <- self$cpp_obj$avg_leaves / (self$ndpost) # + self$nskip
    self$avg_depth <- self$cpp_obj$avg_depth / (self$ndpost)
    self$accept_rate <- self$cpp_obj$accept_rate / (self$ndpost + self$nskip) / self$ntrees
    rownames(self$accept_rate) <- c("grow", "prune", "change")
    # Copy draws to R
    if (self$keep_draws) {
      self$draws_theta <- self$cpp_obj$draws
      # self$draws_phi <- self$cpp_obj$draws_phi
      self$varcount <- self$cpp_obj$varcount_mcmc
      self$mppi <- apply(self$cpp_obj$varcount_mcmc > 0, c(1, 2), mean)
    }
  }
))

# Multinomial logistic normal BART
MultinomialLNBART <- R6::R6Class(classname = "MultinomialLNBART", public = list(
  cpp_obj = NULL, cpp_module_name = character(),
  n_trials = integer(), n = integer(), d = integer(), p = integer(),
  ntrees = integer(), ndpost = integer(), nskip = integer(), forests_dir = character(),
  shared_trees = logical(),
  Bt = matrix(),
  covariance_type = NULL, elapsed_time = NULL, elapsed_time_log_lik = NULL,
  avg_leaves = NULL, avg_depth = NULL, accept_rate = NULL, lpl = NULL, varcount = NULL,
  draws_theta = NULL, draws_abundance = NULL, draws_chol_Sigma_V = NULL,
  draws_phi = NULL, keep_draws = logical(), save_trees = logical(),
  mppi = NULL,
  initialize = function(Y, X) {
    # Call the C++ class in R
    ml <- Rcpp::Module(module = "multinomial_ln_bart", PACKAGE = "zanicc")
    self$cpp_obj <- new(ml$MultinomialLNBART, Y, X)
    self$cpp_module_name <- "multinomial_ln_bart"
    self$n <- nrow(Y)
    self$d <- ncol(Y)
    self$p <- ncol(X)
    self$n_trials <- rowSums(Y)
  },
  SetupMCMC = function(v0 = 3.5 / sqrt(2), ntrees = 20L,
                       ndpost = 1000L, nskip = 2000L,
                       covariance_type = c("diag", "wishart", "fa", "fa_mgp"),
                       nu_prior = self$d,
                       Psi_prior = diag(self$d, self$d - 1),
                       a_sigma = 1.0, b_sigma = 1.0,
                       q_factors = .ledermann(self$d - 1L), sigma2_gamma = 1.0,
                       a_psi = 2.5, b_psi = 1.0,
                       shape_lsphis = 2.0, a1_gs = 1.5, a2_gs = 2.8,
                       numcut = 100L, power = 2.0, base = 0.95,
                       proposals_prob = c(0.25, 0.25, 0.50),
                       update_sigma = TRUE, s2_0 = 1 / ntrees, w_ss = 1.0,
                       splitprobs = rep(1 / self$p, self$p), sparse = FALSE,
                       sparse_parms = c(self$p, 0.5, 1.0), alpha_sparse = 1.0,
                       alpha_random = FALSE, xinfo = matrix(), forests_dir = tempdir(),
                       keep_draws = TRUE, save_trees = FALSE) {
    covariance_type <- match.arg(covariance_type)
    cov_type <- as.integer(which(covariance_type == c("diag", "wishart", "fa", "fa_mgp"))) - 1L
    if (q_factors == 0) q_factors <- self$d - 1
    self$covariance_type <- covariance_type
    self$ntrees <- ntrees
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$forests_dir <- forests_dir
    self$keep_draws <- keep_draws
    self$save_trees <- save_trees
    if (!is.list(splitprobs)) splitprobs <- replicate(self$d, splitprobs, simplify = FALSE)
    alpha_sparse <- rep(alpha_sparse, self$d)

    B <- qr.Q(qr(stats::contr.sum(self$d)))
    self$Bt <- t(B)

    # Setup
    self$cpp_obj$SetMCMC(
      v0, ntrees,
      B, cov_type,
      a_sigma, b_sigma,
      Psi_prior, nu_prior,
      q_factors, sigma2_gamma,
      a_psi, b_psi,
      shape_lsphis, a1_gs, a2_gs,
      ndpost, nskip, numcut, power, base,
      proposals_prob, as.integer(update_sigma), s2_0, w_ss,
      splitprobs, as.integer(sparse), sparse_parms,
      alpha_sparse, as.integer(alpha_random), xinfo, forests_dir,
      as.integer(keep_draws), as.integer(save_trees)
    )
  },
  RunMCMC = function() {
    ini <- proc.time()
    self$cpp_obj$RunMCMC()
    self$elapsed_time <- proc.time() - ini
    # Keep some tree diagnostics
    self$avg_leaves <- self$cpp_obj$avg_leaves / self$ndpost
    self$avg_depth <- self$cpp_obj$avg_depth / self$ndpost
    self$accept_rate <- self$cpp_obj$accept_rate / (self$ndpost + self$nskip) / self$ntrees
    rownames(self$accept_rate) <- c("grow", "prune", "change")
    # Copy draws to R
    if (self$keep_draws) {
      self$draws_abundance <- self$cpp_obj$draws_vartheta
      self$draws_theta <- self$cpp_obj$draws_theta
      self$draws_chol_Sigma_V <- self$cpp_obj$draws_chol_Sigma_V
      self$varcount <- self$cpp_obj$varcount_mcmc
      self$mppi <- apply(self$cpp_obj$varcount_mcmc > 0, c(1, 2), mean)
      # self$draws_phi <- self$cpp_obj$draws_phi
    }
  }
))


# ZANIM-linear regression
ZANIMRegression <- R6::R6Class(
  classname = "ZANIMRegression",
  public = list(
    cpp_obj = NULL, n_trials = integer(), n = integer(), d = integer(),
    p_theta = integer(), p_zeta = integer(),
    ndpost = integer(), nskip = integer(), nthin = integer(),
    n_pred = integer(),
    draws_theta = NULL, draws_zeta = NULL, draws_phi = NULL,
    draws_abundance = NULL, draws_betas_theta = NULL, draws_betas_zeta = NULL,
    y_rep_draws = NULL, log_lik_draws = NULL, elapsed_time = NULL,
    keep_draws = logical(), keep_draws_coef = logical(),
    initialize = function(Y, X_theta, X_zeta) {
      ml <- Rcpp::Module(module = "zanim_linear_reg", PACKAGE = "zanicc")
      self$cpp_obj <- new(ml$ZANIMReg, Y, X_theta, X_zeta)
      self$n <- nrow(Y)
      self$d <- ncol(Y)
      self$p_theta <- ncol(X_theta)
      self$p_zeta <- ncol(X_zeta)
      self$n_trials <- rowSums(Y)
    },
    SetupMCMC = function(sd_prior_beta_theta = rep(1.0, self$p_theta),
                         S_prior_beta_zeta = diag(1.0, self$p_zeta),
                         ndpost = 5000L, nskip = 5000L, nthin = 1L,
                         keep_draws = TRUE, keep_draws_coef = TRUE) {
      self$ndpost <- ndpost
      self$nskip <- nskip
      self$nthin <- nthin
      self$keep_draws <- keep_draws
      self$keep_draws_coef <- keep_draws_coef
      self$cpp_obj$SetMCMC(
        sd_prior_beta_theta, S_prior_beta_zeta, ndpost, nskip, nthin
      )
    },
    RunMCMC = function() {
      ini <- proc.time()
      self$cpp_obj$RunMCMC()
      self$elapsed_time <- proc.time() - ini
      # Save draws
      if (self$keep_draws) {
        # self$draws_abundance <- self$cpp_obj$draws_vartheta
        self$draws_theta <- self$cpp_obj$draws_thetas
        self$draws_abundance <- self$cpp_obj$draws_varthetas
        self$draws_zeta <- stats::pnorm(self$cpp_obj$draws_zetas)
        # self$draws_phi <- self$cpp_obj$draws_phi
        if (self$keep_draws_coef) {
          self$draws_betas_theta <- self$cpp_obj$draws_betas_theta
          self$draws_betas_zeta <- self$cpp_obj$draws_betas_zeta
        }
      }
    },
    PosterioMeanCoef = function(parameter = c("theta", "zeta")) {
      parameter <- match.arg(parameter)
      switch(parameter,
        "zeta" = apply(self$draws_betas_zeta, c(1, 2), mean),
        "theta" = apply(self$draws_betas_theta, c(1, 2), mean)
      )
    }
  )
)


# ZANIDM logistic regression
ZANIDMRegression <- R6::R6Class(
  classname = "ZANIDMRegression",
  public = list(
    cpp_obj = NULL, cpp_module_name = character(),
    n_trials = integer(), n = integer(), d = integer(),
    p_alpha = integer(), p_zeta = integer(),
    ndpost = integer(), nskip = integer(), nthin = integer(),
    n_pred = integer(),
    draws_alpha = NULL, draws_zeta = NULL, draws_phi = NULL, draws_theta = NULL,
    draws_abundance = NULL, draws_betas_alpha = NULL, draws_betas_zeta = NULL,
    y_rep_draws = NULL, log_lik_draws = NULL, elapsed_time = NULL,
    keep_draws = logical(), keep_draws_coef = logical(), save_draws = logical(),
    dir_draws = NULL,
    initialize = function(Y, X_alpha, X_zeta) {
      ml <- Rcpp::Module(module = "zanidm_linear_reg", PACKAGE = "zanicc")
      self$cpp_obj <- new(ml$ZANIDMReg, Y, X_alpha, X_zeta)
      self$cpp_module_name <- "zanidm_linear_reg"
      self$n <- nrow(Y)
      self$d <- ncol(Y)
      self$p_alpha <- ncol(X_alpha)
      self$p_zeta <- ncol(X_zeta)
      self$n_trials <- rowSums(Y)
    },
    SetupMCMC = function(sd_prior_beta_alpha = rep(1.0, self$p_alpha),
                         S_prior_beta_zeta = diag(1.0, self$p_zeta),
                         ndpost = 5000L, nskip = 5000L, nthin = 1L,
                         keep_draws = TRUE, keep_draws_coef = TRUE,
                         save_draws = FALSE, dir_draws = tempdir()) {
      self$ndpost <- ndpost
      self$nskip <- nskip
      self$nthin <- nthin
      self$keep_draws <- keep_draws
      self$keep_draws_coef <- keep_draws_coef
      self$dir_draws <- dir_draws
      self$save_draws <- save_draws
      self$cpp_obj$SetMCMC(
        sd_prior_beta_alpha, S_prior_beta_zeta, ndpost, nskip,
        nthin, keep_draws, save_draws, dir_draws
      )
    },
    RunMCMC = function() {
      ini <- proc.time()
      self$cpp_obj$RunMCMC()
      self$elapsed_time <- proc.time() - ini
      # Save draws
      if (self$keep_draws) {
        self$draws_abundance <- self$cpp_obj$draws_abundance
        self$draws_alpha <- self$cpp_obj$draws_alphas
        self$draws_theta <- sweep(
          x = self$cpp_obj$draws_alphas, MARGIN = c(1, 3),
          STATS = apply(self$cpp_obj$draws_alphas, c(1, 3), sum),
          FUN = "/"
        )
        self$draws_zeta <- stats::pnorm(self$cpp_obj$draws_zetas)
        # self$draws_phi <- self$cpp_obj$draws_phi
        if (self$keep_draws_coef) {
          self$draws_betas_alpha <- self$cpp_obj$draws_betas_alpha
          self$draws_betas_zeta <- self$cpp_obj$draws_betas_zeta
        }
      }
    },
    PosterioMeanCoef = function(parameter = c("alpha", "zeta")) {
      parameter <- match.arg(parameter)
      switch(parameter,
        "zeta" = apply(self$draws_betas_zeta, c(1, 2), mean),
        "alpha" = apply(self$draws_betas_alpha, c(1, 2), mean)
      )
    }
  )
)

# ZANIM logistic normal regression
ZANIMLNRegression <- R6::R6Class(
  classname = "ZANIMLNRegression",
  public = list(
    cpp_obj = NULL, cpp_module_name = NULL,
    n_trials = integer(), n = integer(), d = integer(),
    p_theta = integer(), p_zeta = integer(),
    ndpost = integer(), nskip = integer(), nthin = integer(),
    n_pred = integer(),
    draws_theta = NULL, draws_zeta = NULL, draws_phi = NULL,
    draws_abundance = NULL, draws_betas_theta = NULL, draws_betas_zeta = NULL,
    draws_chol_Sigma_V = NULL, Bt = NULL,
    y_rep_draws = NULL, log_lik_draws = NULL, elapsed_time = NULL,
    keep_draws = logical(), keep_draws_coef = logical(),
    initialize = function(Y, X_theta, X_zeta) {
      ml <- Rcpp::Module(module = "zanim_ln_reg", PACKAGE = "zanicc")
      self$cpp_obj <- new(ml$ZANIMLNReg, Y, X_theta, X_zeta)
      self$cpp_module_name <- "zanim_ln_reg"
      # self$Y <- Y
      self$n <- nrow(Y)
      self$d <- ncol(Y)
      self$p_theta <- ncol(X_theta)
      self$p_zeta <- ncol(X_zeta)
      self$n_trials <- rowSums(Y)
    },
    SetupMCMC = function(sd_prior_beta_theta = rep(1.0, self$p_theta),
                         S_prior_beta_zeta = diag(1.0, self$p_zeta),
                         ndpost = 5000L, nskip = 5000L, nthin = 1L,
                         covariance_type = c("diag", "wishart", "fa", "fa_mgp"),
                         nu_prior = self$d,
                         Psi_prior = diag(self$d, self$d - 1),
                         a_sigma = 1.0, b_sigma = 1.0,
                         q_factors = .ledermann(self$d - 1L), sigma2_gamma = 1.0,
                         a_psi = 2.5, b_psi = 1.0,
                         shape_lsphis = 2.0, a1_gs = 1.5, a2_gs = 2.8,
                         keep_draws = TRUE, keep_draws_coef = TRUE) {
      covariance_type <- match.arg(covariance_type)
      cov_type <- as.integer(which(covariance_type == c("diag", "wishart", "fa", "fa_mgp"))) - 1L
      if (q_factors == 0) q_factors <- 1

      self$ndpost <- ndpost
      self$nskip <- nskip
      self$nthin <- nthin
      self$keep_draws <- keep_draws
      self$keep_draws_coef <- keep_draws_coef
      B <- qr.Q(qr(stats::contr.sum(self$d)))
      self$Bt <- t(B)
      self$cpp_obj$SetMCMC(
        sd_prior_beta_theta, S_prior_beta_zeta, ndpost, nskip, nthin,
        B, cov_type,
        a_sigma, b_sigma,
        Psi_prior, nu_prior,
        q_factors, sigma2_gamma,
        a_psi, b_psi,
        shape_lsphis, a1_gs, a2_gs, keep_draws
      )
    },
    RunMCMC = function() {
      ini <- proc.time()
      self$cpp_obj$RunMCMC()
      self$elapsed_time <- proc.time() - ini
      # Save draws
      if (self$keep_draws) {
        self$draws_theta <- self$cpp_obj$draws_thetas
        self$draws_abundance <- self$cpp_obj$draws_varthetas
        self$draws_zeta <- stats::pnorm(self$cpp_obj$draws_zetas)
        self$draws_chol_Sigma_V <- self$cpp_obj$draws_chol_Sigma_V
        # self$draws_phi <- self$cpp_obj$draws_phi
        if (self$keep_draws_coef) {
          self$draws_betas_theta <- self$cpp_obj$draws_betas_theta
          self$draws_betas_zeta <- self$cpp_obj$draws_betas_zeta
        }
      }
    },
    PosterioMeanCoef = function(parameter = c("theta", "zeta")) {
      parameter <- match.arg(parameter)
      switch(parameter,
        "zeta" = apply(self$draws_betas_zeta, c(1, 2), mean),
        "theta" = apply(self$draws_betas_theta, c(1, 2), mean)
      )
    }
  )
)


# DM-linear regression
DMRegression <- R6::R6Class(
  classname = "DMRegression",
  public = list(
    cpp_obj = NULL, cpp_module_name = character(),
    n_trials = integer(), n = integer(), d = integer(),
    p = integer(), ndpost = integer(), nskip = integer(), nthin = integer(),
    n_pred = integer(),
    draws_alpha = NULL, draws_phi = NULL, draws_theta = NULL,
    draws_abundance = NULL, draws_betas = NULL,
    y_rep_draws = NULL, log_lik_draws = NULL, elapsed_time = NULL,
    keep_draws = logical(), keep_draws_coef = logical(), save_draws = logical(),
    dir_draws = character(),
    initialize = function(Y, X) {
      ml <- Rcpp::Module(module = "dm_linear_reg", PACKAGE = "zanicc")
      self$cpp_obj <- new(ml$DMLinearReg, Y, X)
      self$cpp_module_name <- "dm_linear_reg"
      # self$Y <- Y
      self$n <- nrow(Y)
      self$d <- ncol(Y)
      self$p <- ncol(X)
      self$n_trials <- rowSums(Y)
    },
    SetupMCMC = function(S_prior_betas = diag(1.0, self$p),
                         ndpost = 5000L, nskip = 5000L, nthin = 1L,
                         keep_draws = TRUE, keep_draws_coef = TRUE,
                         save_draws = FALSE, dir_draws = tempdir()) {
      self$ndpost <- ndpost
      self$nskip <- nskip
      self$nthin <- nthin
      self$keep_draws <- keep_draws
      self$keep_draws_coef <- keep_draws_coef
      self$dir_draws <- dir_draws
      self$save_draws <- save_draws
      if (is.matrix(S_prior_betas)) {
        S <- array(0, dim = c(self$p, self$p, self$d))
        for (j in seq_len(self$d)) S[, , j] <- S_prior_betas
      }
      self$cpp_obj$SetMCMC(
        S, ndpost, nskip, nthin, as.integer(keep_draws),
        as.integer(save_draws), dir_draws
      )
    },
    RunMCMC = function() {
      ini <- proc.time()
      self$cpp_obj$RunMCMC()
      self$elapsed_time <- proc.time() - ini
      # Save draws
      if (self$keep_draws) {
        self$draws_abundance <- self$cpp_obj$draws_abundance
        self$draws_alpha <- self$cpp_obj$draws_alphas
        self$draws_theta <- sweep(
          x = self$cpp_obj$draws_alphas, MARGIN = c(1, 3),
          STATS = apply(self$cpp_obj$draws_alphas, c(1, 3), sum),
          FUN = "/"
        )
        # self$draws_phi <- self$cpp_obj$draws_phi
        if (self$keep_draws_coef) self$draws_betas <- self$cpp_obj$draws_betas
      }
    },
    PosterioMeanCoef = function() {
      if (self$keep_draws) apply(self$draws_betas, c(1, 2), mean)
    }
  )
)
