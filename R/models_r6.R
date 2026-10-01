#' @title ZANIM-BART
#'
#' @description
#' Carries out Bayesian inference for the zero-and-N-inflated multinomial logistic
#' BART (ZANIM-BART) model through the efficient Markov chain Monte Carlo algorithm
#' proposed in Menezes et al. (2026).
#' The `R6` class is an wrapper for the underlying `C++` implementation.
#'
#' @references Menezes, A. F. B., Parnell, A. C. and Murphy, K. (2026),
#' Bayesian nonparametric models for zero-inflated count-compositional data using
#' ensembles of regression trees. \emph{arXiv preprint}, \strong{arXiv:2601.08067} <https://arxiv.org/abs/2601.08067v2>
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
  #' @field p_theta Number of covariates associated to the compositional components.
  p_theta = integer(),
  #' @field p_zeta Number of covariates associated to the structural zero components.
  p_zeta = integer(),
  #' @field ntrees_theta Number of trees for the structural zero components.
  ntrees_theta = integer(),
  #' @field ntrees_zeta Number of trees for the structural zero components.
  ntrees_zeta = integer(),
  #' @field ndpost Number of posterior MCMC draws to retain.
  ndpost = integer(),
  #' @field nskip Number of MCMC draws to discard as burn-in before retaining
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
  #' @field draws_abundance Posterior draws of the individual-level count probabilities.
  draws_abundance = NULL,
  #' @field keep_draws Logical indicating whether posterior draws were retained.
  keep_draws = logical(),
  #' @field save_trees Logical indicating whether the posterior forests were
  #' saved in disk.
  save_trees = logical(),
  #' @field varcount_theta A three dimensional array with dimension \eqn{p_{\theta} \times d \times m},
  #' where \eqn{p_\theta} is the number of covariates for the compositional components,
  #' \eqn{d} is the number of categories and \eqn{m} is the number of posterior draws, `ndpost`.
  #' It contains the total count of the number of times that a covariate is used in a
  #' tree decision rule over all category-specific trees.
  varcount_theta = NULL,
  #' @field varcount_zeta A three dimensional array with dimension \eqn{p_{\zeta} \times d \times m},
  #' where \eqn{p_\zeta} is the number of covariates for the structural zero components,
  #' \eqn{d} is the number of categories and \eqn{m} is the number of posterior draws, `ndpost`.
  #' It contains the total count of the number of times that a covariate is used in a
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

  #' @description Creates a new `ZANIMBART` object.
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
      self$cpp_obj <- methods::new(ml$ZANIMBARTLogit, Y, X_theta, X_zeta)
      self$cpp_module_name <- "zanim_bart_logit"
      if (shared_trees) warning("Shared trees only implemented with the probit.")
      shared_trees <- FALSE
    } else {
      if (shared_trees) {
        ml <- Rcpp::Module(module = "zanim_shared_bart_probit", PACKAGE = "zanicc")
        self$cpp_obj <- methods::new(ml$ZANIMSharedBARTProbit, Y, X_theta, X_zeta)
        self$cpp_module_name <- "zanim_shared_bart_probit"
      } else {
        ml <- Rcpp::Module(module = "zanim_bart_probit", PACKAGE = "zanicc")
        self$cpp_obj <- methods::new(ml$ZANIMBARTProbit, Y, X_theta, X_zeta)
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
  #' @param update_sigma_theta Logical indicating whether a hyperprior should be
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
  #' @param keep_draws Logical indicating whether to retain posterior draws. Default is `TRUE`.
  #' @param save_trees Logical indicating whether to save the posterior draws of the BART
  #' tree topologies and terminal-node parameters to `.bin` files.
  #' Default is `FALSE`.
  #' This creates files named `forests_theta_j.bin` for the category-specific
  #' compositional regression trees and, `forests_zeta_j.bin`
  #' for the structural-zero regression trees.
  #' Here, `j` indexes the category, and each file contains the corresponding
  #' tree topologies and terminal node parameters across all `ndpost` posterior
  #' draws. Such files are written in disk in the directory indicated by the argument `forests_dir`.
  #' @param forests_dir Character path indicating where to save the
  #' `forests_theta_j.bin` and `forests_zeta_j.bin` files when `save_trees=TRUE`.
  #' Default is to [tempdir()].
  #' @param xinfo Optional matrix containing the cut points information of each
  #' covariate supplied to the underlying `C++` implementation.
  #' An empty matrix implies that the cut points are determined internally.
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
                       keep_draws = TRUE, save_trees = FALSE,
                       forests_dir = tempdir(), xinfo = matrix()
                       ) {
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
#' logistic-normal BART (ZANIM-LN-BART) model through the efficient
#' Markov chain Monte Carlo algorithm proposed in Menezes et al. (2026).
#' The `R6` class is an wrapper for the underlying `C++` implementation.
#'
#' @references Menezes, A. F. B., Parnell, A. C. and Murphy, K. (2026),
#' Bayesian nonparametric models for zero-inflated count-compositional data using
#' ensembles of regression trees. \emph{arXiv preprint}, \strong{arXiv:2601.08067} <https://arxiv.org/abs/2601.08067v2>
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
  #' @field p_theta Number of covariates associated to the compositional components.
  p_theta = integer(),
  #' @field p_zeta Number of covariates associated to the structural zero components.
  p_zeta = integer(),
  #' @field ntrees_theta Number of trees for the structural zero components.
  ntrees_theta = integer(),
  #' @field ntrees_zeta Number of trees for the structural zero components.
  ntrees_zeta = integer(),
  #' @field ndpost Number of posterior MCMC draws to retain.
  ndpost = integer(),
  #' @field nskip Number of MCMC draws to discard as burn-in before retaining
  #' posterior draws.
  nskip = integer(),
  #' @field forests_dir Character path indicating where to save the
  #' `forests_theta_j.bin` and `forests_zeta_j.bin` files.
  forests_dir = character(),
  #' @field covariance_type Character string with the prior used for the covariance
  #' matrix for the logistic-normal random effects.
  covariance_type = NULL,
  #' @field Bt Transpose of the orthogonal matrix for the sum-to-zero constraint
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
  #' @field draws_abundance Posterior draws of the individual-level count probabilities.
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

  #' @description Creates a new `ZANIMLNBART` object.
  #' @param Y A matrix of multivariate count-compositional data.
  #' Rows correspond to observations and columns correspond to categories.
  #' @param X_theta A matrix of covariates used to model the count probabilities.
  #' Rows must correspond to the observations in `Y`.
  #' @param X_zeta A matrix of covariates used to model the structural zero probabilities.
  #' Rows must correspond to the observations in `Y`.
  initialize = function(Y, X_theta, X_zeta) {
    ml <- Rcpp::Module(module = "zanim_ln_bart", PACKAGE = "zanicc")
    self$cpp_obj <- methods::new(ml$ZANIMLNBART, Y, X_theta, X_zeta)
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
  #'  `diag` (for a diagonal covariance matrix with inverse-gamma priors), and `wishart` (for an inverse Wishart prior).
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
  #' @param update_sigma_theta Logical indicating whether a hyperprior should be
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
  #' @param keep_draws Logical indicating whether to retain posterior draws. Default is `TRUE`.
  #' @param save_trees Logical indicating whether to save the posterior draws of the BART
  #' tree topologies and terminal-node parameters to `.bin` files.
  #' Default is `FALSE`.
  #' This creates files named `forests_theta_j.bin` for the category-specific
  #' compositional regression trees and, `forests_zeta_j.bin`
  #' for the structural-zero regression trees.
  #' Here, `j` indexes the category, and each file contains the corresponding
  #' tree topologies and terminal node parameters across all `ndpost` posterior
  #' draws. Such files are written in disk in the directory indicated by the argument `forests_dir`.
  #' @param forests_dir Character path indicating where to save the
  #' `forests_theta_j.bin` and `forests_zeta_j.bin` files when `save_trees=TRUE`.
  #' Default is to [tempdir()].
  #' @param xinfo Optional matrix containing the cut points information of each
  #' covariate supplied to the underlying `C++` implementation.
  #' An empty matrix implies that the cut points are determined internally.
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
                       alpha_sparse = c(1.0, 1.0),
                       alpha_random = c(FALSE, FALSE),
                       keep_draws = TRUE,
                       save_trees = FALSE,
                       forests_dir = tempdir(),
                       xinfo = matrix()) {
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


#' @title Multinomial-BART
#'
#' @description
#' Carries out Bayesian inference for the multinomial logistic BART (ML-BART)
#' model through the efficient Markov chain Monte Carlo algorithm proposed by
#' Murray (2021).
#' The `R6` class is an wrapper for the underlying `C++` implementation.
#'
#' @references Murray, J. S. (2021),
#' Log-linear Bayesian additive regression trees for multinomial logistic and count regression models
#' \emph{Journal of the American Statistical Association}, \strong{116} (534), 756--769.
#'
#' @export
MultinomialBART <- R6::R6Class(classname = "MultinomialBART", cloneable = FALSE,
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
  #' @field p Number of covariates.
  p = integer(),
  #' @field ntrees Number of trees for each category-specific log-BART prior.
  ntrees = integer(),
  #' @field ndpost Number of posterior MCMC draws to retain.
  ndpost = integer(),
  #' @field nskip Number of MCMC draws to discard as burn-in before retaining
  #' posterior draws.
  nskip = integer(),
  #' @field forests_dir Character path indicating where the files,
  #' `forests_j.bin`, posterior draws for the category-specific forests are saved.
  forests_dir = character(),
  #' @field shared_trees Whether the shared trees are used across the categories.
  shared_trees = logical(),
  #' @field elapsed_time Elapsed time taken to run the MCMC algorithm.
  elapsed_time = NULL,
  #' @field avg_leaves Average number of leaves across the posterior draws `ndpost`
  #' for the category-specific regression tree ensembles.
  avg_leaves = NULL,
  #' @field avg_depth Average tree depth across the posterior draws `ndpost`
  #' for the category-specific regression tree ensembles.
  avg_depth = NULL,
  #' @field accept_rate Acceptance rate of the Metropolis-Hastings proposals,
  #' `grow`, `prune`, `change`, for the category-specific regression tree ensembles.
  accept_rate = NULL,
  #' @field draws_theta Posterior draws of the population-level count probabilities.
  draws_theta = NULL,
  #' @field keep_draws Logical indicating whether posterior draws were retained.
  keep_draws = logical(),
  #' @field save_trees Logical indicating whether the posterior forests were
  #' saved in disk.
  save_trees = logical(),
  #' @field varcount A three dimensional array with dimension \eqn{p \times d \times m},
  #' where \eqn{p} is the number of covariates,
  #' \eqn{d} is the number of categories and \eqn{m} is the number of posterior draws, `ndpost`.
  #' It contains the total count of the number of times that a covariate is used in a
  #' tree decision rule over all category-specific trees.
  varcount = NULL,
  #' @field mppi A matrix with rows being the covariates and columns the categories.
  #' It contains the posterior estimates of the marginal probability of inclusion
  #' (MPPI) for the category-specific covariates.
  mppi = NULL,

  #' @description Creates a new `MultinomialBART` object.
  #' @param Y A matrix of multivariate count-compositional data.
  #' Rows correspond to observations and columns correspond to categories.
  #' @param X A matrix of covariates used to model the category-specific
  #' count probabilities.
  #' Rows must correspond to the observations in `Y`.
  #' @param shared_trees Whether the shared trees are used across the categories.
  initialize = function(Y, X, shared_trees = FALSE) {
    self$shared_trees <- shared_trees
    # Call the C++ class in R
    if (!self$shared_trees) {
      ml <- Rcpp::Module(module = "multinomial_bart", PACKAGE = "zanicc")
      self$cpp_obj <- methods::new(ml$MultinomialBART, Y, X)
      self$cpp_module_name <- "multinomial_bart"
    } else {
      ml <- Rcpp::Module(module = "multinomial_shared_bart", PACKAGE = "zanicc")
      self$cpp_obj <- methods::new(ml$MultinomialSharedBART, Y, X)
      self$cpp_module_name <- "multinomial_shared_bart"
    }
    self$n <- nrow(Y)
    self$d <- ncol(Y)
    self$p <- ncol(X)
    self$n_trials <- rowSums(Y)
  },
  #' Set up the settings for the MCMC algorithm
  #' @description
  #' Configures priors and hyperparameters of the ML-BART model
  #' used by the underlying MCMC algorithm implemented in `C++`.
  #' This method must be called before \href{#method-MultinomialBART-RunMCMC}{\code{MultinomialBART$RunMCMC()}}.
  #'
  #' @param v0 Hyperparameter controlling the level of shrinkage of the
  #' regression trees. The smaller `v0` is, the more shrinkage is applied, i.e.,
  #' shallow trees are expected.
  #' @param ntrees Number of trees used for in the ensemble.
  #' The default is `ntrees=100`.
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
  #' @param update_sigma Logical indicating whether a hyperprior should be
  #' used for the shrinkage hyperparameter, `v0`. If so, then we use slice
  #' sampling to update this hyperparameter during the MCMC.
  #' @param s0_2 Hyperprior scale parameter for the `v0` hyperparameter. Default is `1/ntrees`.
  #' @param w_ss Hyperprameter for stepping out method in the slice sampling algorithm.
  #' It controls the width of the slice.
  #' @param splitprobs Numeric vector with the prior probabilities of each
  #' covariate in `X` to generate a splitting rule. Default is `1/p`.
  #' @param sparse Logical indicating whether to perform
  #' variable selection based on the sparse Dirichlet prior
  #' of Linero (2018) rather than uniform prior on the splitting probabilities of the
  #' structural zero and compositional components, respectively.
  #' This prior assumes that the splitting probability vector follows
  #' \eqn{\mathbf{s} \sim \operatorname{Dirichlet}\lbrack \alpha/p, \ldots, \alpha/p \rbrack},
  #' with \eqn{\alpha} a hyperparameter and \eqn{p} number of covariates.
  #' @param alpha_sparse Hyperparameter value of \eqn{\alpha} which controls the level
  #' of sparsity of the Dirichlet prior on the splitting.
  #' Default is `alpha_sparse = 1`. As \eqn{\alpha \rightarrow \infty}, it recovers the
  #' default uniform prior on the splitting probabilities under BART.
  #' @param alpha_random Logical indicating whether to put a
  #' hyperprior on \eqn{\alpha}. The hyperprior is of the form
  #' \eqn{\alpha / (\alpha + \rho) \sim \operatorname{Beta}\lbrack a, b \rbrack}
  #' with further hyperprior parameters \eqn{\rho}, \eqn{a} and \eqn{b}.
  #' @param sparse_parms Numeric vector of length three for the hyperprior parameters
  #' \eqn{\rho}, \eqn{a} and \eqn{b}. By default, these are `sparse_parms=c(p, 0.5, 1.0)`.
  #' @param keep_draws Logical indicating whether to retain posterior draws. Default is `TRUE`.
  #' @param save_trees Logical indicating whether to save the posterior draws of the BART
  #' tree topologies and terminal-node parameters to `.bin` files.
  #' Default is `FALSE`.
  #' This creates files named `forests_j.bin` for the category-specific
  #' compositional regression trees.
  #' Here, `j` indexes the category, and each file contains the corresponding
  #' tree topologies and terminal node parameters across all `ndpost` posterior
  #' draws. Such files are written in disk in the directory indicated by the argument `forests_dir`.
  #' @param forests_dir Character path indicating where to save the
  #' `forests_j.bin` files when `save_trees=TRUE`. Default is to [tempdir()].
  #' @param xinfo Optional matrix containing the cut points information of each
  #' covariate supplied to the underlying `C++` implementation.
  #' An empty matrix implies that the cut points are determined internally.
  SetupMCMC = function(v0 = 3.5 / sqrt(2), ntrees = 100L, ndpost = 5000L,
                       nskip = 5000L, numcut = 100L,
                       power = 2.0, base = 0.95,
                       proposals_prob = c(0.25, 0.25, 0.50),
                       update_sigma = TRUE,
                       s0_2 = 1 / ntrees, w_ss = 1.0,
                       splitprobs = rep(1 / self$p, self$p),
                       sparse = FALSE,
                       alpha_sparse = 1.0,
                       alpha_random = FALSE,
                       sparse_parms = c(self$p, 0.5, 1.0),
                       keep_draws = TRUE,
                       save_trees = FALSE,
                       forests_dir = tempdir(),
                       xinfo = matrix()) {
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
    self$cpp_obj$SetMCMC(
      v0, ntrees, ndpost, nskip, numcut, power, base,
      proposals_prob, as.integer(update_sigma), s0_2, w_ss,
      splitprobs, as.integer(sparse), sparse_parms,
      alpha_sparse, as.integer(alpha_random),
      xinfo, forests_dir,
      keep_draws, save_trees
    )
  },
  #' Run the MCMC algorithm of ML-BART
  #'
  #' @description
  #'  Runs the MCMC sampler using the settings previously configured with
  #' \href{#method-MultinomialBART-SetupMCMC}{\code{MultinomialBART$SetupMCMC()}}.
  #' Posterior draws, acceptance rates, and
  #' variable-selection statistics, are then transferred from the underlying `C++`
  #' object to the `MultinomialBART` object.
  #'
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


#' @title Multinomial-BART
#'
#' @description
#' Carries out Bayesian inference for the multinomial logistic-normal BART (MLN-BART)
#' model through the efficient Markov chain Monte Carlo algorithm proposed by
#' Menezes et al. (2026).
#' The `R6` class is an wrapper for the underlying `C++` implementation.
#'
#' @references Menezes, A. F. B., Parnell, A. C. and Murphy, K. (2026),
#' Bayesian nonparametric models for zero-inflated count-compositional data using
#' ensembles of regression trees. \emph{arXiv preprint}, \strong{arXiv:2601.08067} <https://arxiv.org/abs/2601.08067v2>
#'
#' @export
MultinomialLNBART <- R6::R6Class(
  classname = "MultinomialLNBART", cloneable = FALSE,
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
  #' @field p Number of covariates.
  p = integer(),
  #' @field ntrees Number of trees for each category-specific log-BART prior.
  ntrees = integer(),
  #' @field ndpost Number of posterior MCMC draws to retain.
  ndpost = integer(),
  #' @field nskip Number of MCMC draws to discard as burn-in before retaining
  #' posterior draws.
  nskip = integer(),
  #' @field forests_dir Character path indicating where the files,
  #' `forests_j.bin`, posterior draws for the category-specific forests are saved.
  forests_dir = character(),
  #' @field covariance_type Character string with the prior used for the covariance
  #' matrix for the logistic-normal random effects.
  covariance_type = NULL,
  #' @field Bt Transpose of the orthogonal matrix for the sum-to-zero constraint
  #' in the logistic random effects.
  Bt = NULL,
  #' @field elapsed_time Elapsed time taken to run the MCMC algorithm.
  elapsed_time = NULL,

  #' @field avg_leaves Average number of leaves across the posterior draws `ndpost`
  #' for the category-specific regression tree ensembles.
  avg_leaves = NULL,
  #' @field avg_depth Average tree depth across the posterior draws `ndpost`
  #' for the category-specific regression tree ensembles.
  avg_depth = NULL,
  #' @field accept_rate Acceptance rate of the Metropolis-Hastings proposals,
  #' `grow`, `prune`, `change`, for the category-specific regression tree ensembles.
  accept_rate = NULL,
  #' @field draws_theta Posterior draws of the population-level count probabilities.
  draws_theta = NULL,
  #' @field draws_abundance Posterior draws of the individual-level count probabilities.
  draws_abundance = NULL,
  #' @field draws_chol_Sigma_V Posterior draws of the Cholesky decomposition of the
  #' covariance matrix of the logistic-normal random effects.
  draws_chol_Sigma_V = NULL,
  #' @field keep_draws Logical indicating whether posterior draws were retained.
  keep_draws = logical(),
  #' @field save_trees Logical indicating whether the posterior forests were
  #' saved in disk.
  save_trees = logical(),
  #' @field varcount A three dimensional array with dimension \eqn{p \times d \times m},
  #' where \eqn{p} is the number of covariates,
  #' \eqn{d} is the number of categories and \eqn{m} is the number of posterior draws, `ndpost`.
  #' It contains the total count of the number of times that a covariate is used in a
  #' tree decision rule over all category-specific trees.
  varcount = NULL,
  #' @field mppi A matrix with rows being the covariates and columns the categories.
  #' It contains the posterior estimates of the marginal probability of inclusion
  #' (MPPI) for the category-specific covariates.
  mppi = NULL,

  #' @description Creates a new `MultinomialLNBART` object.
  #' @param Y A matrix of multivariate count-compositional data.
  #' Rows correspond to observations and columns correspond to categories.
  #' @param X A matrix of covariates used to model the category-specific
  #' count probabilities.
  #' Rows must correspond to the observations in `Y`.
  initialize = function(Y, X) {
    # Call the C++ class in R
    ml <- Rcpp::Module(module = "multinomial_ln_bart", PACKAGE = "zanicc")
    self$cpp_obj <- methods::new(ml$MultinomialLNBART, Y, X)
    self$cpp_module_name <- "multinomial_ln_bart"
    self$n <- nrow(Y)
    self$d <- ncol(Y)
    self$p <- ncol(X)
    self$n_trials <- rowSums(Y)
  },
  #' @description
  #' Configures priors and hyperparameters of the ML-BART model
  #' used by the underlying MCMC algorithm implemented in `C++`.
  #' This method must be called before \href{#method-MultinomialBART-RunMCMC}{\code{MultinomialBART$RunMCMC()}}.
  #'
  #' @param v0 Hyperparameter controlling the level of shrinkage of the
  #' regression trees. The smaller `v0` is, the more shrinkage is applied, i.e.,
  #' shallow trees are expected.
  #' @param ntrees Number of trees used for in the ensemble.
  #' The default is `ntrees=100`.
  #' @param ndpost Number of posterior MCMC draws to retain. The default is `ndpost=5000`.
  #' @param nskip Number of MCMC iterations to discard as burn-in before retaining
  #' posterior draws. The default is `nskip=5000`.
  #' @param covariance_type Character string specifying the prior on the covariance
  #' matrix for the logistic-normal random effects. Default is `fa_mgp`, for nonparametric factor
  #'  analysis with a multiplicative gamma process shrinkage prior. Other options include `fa` (factor analysis without such a prior),
  #'  `diag` (for a diagonal covariance matrix with inverse-gamma priors), and `wishart` (for an inverse Wishart prior).
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
  #' @param proposals_prob
  #' Numeric vector of length three containing the probabilities of proposing the
  #' `grow`, `prune`, and `change` tree moves, respectively.
  #' Default probabilities are \eqn{0.25}, \eqn{0.25} and \eqn{0.50}, respectively.
  #' @param update_sigma Logical indicating whether a hyperprior should be
  #' used for the shrinkage hyperparameter, `v0`. If so, then we use slice
  #' sampling to update this hyperparameter during the MCMC.
  #' @param s0_2 Hyperprior scale parameter for the `v0` hyperparameter. Default is `1/ntrees`.
  #' @param w_ss Hyperprameter for stepping out method in the slice sampling algorithm.
  #' It controls the width of the slice.
  #' @param splitprobs Numeric vector with the prior probabilities of each
  #' covariate in `X` to generate a splitting rule. Default is `1/p`.
  #' @param sparse Logical indicating whether to perform
  #' variable selection based on the sparse Dirichlet prior
  #' of Linero (2018) rather than uniform prior on the splitting probabilities of the
  #' structural zero and compositional components, respectively.
  #' This prior assumes that the splitting probability vector follows
  #' \eqn{\mathbf{s} \sim \operatorname{Dirichlet}\lbrack \alpha/p, \ldots, \alpha/p \rbrack},
  #' with \eqn{\alpha} a hyperparameter and \eqn{p} number of covariates.
  #' @param alpha_sparse Hyperparameter value of \eqn{\alpha} which controls the level
  #' of sparsity of the Dirichlet prior on the splitting.
  #' Default is `alpha_sparse = 1`. As \eqn{\alpha \rightarrow \infty}, it recovers the
  #' default uniform prior on the splitting probabilities under BART.
  #' @param alpha_random Logical indicating whether to put a
  #' hyperprior on \eqn{\alpha}. The hyperprior is of the form
  #' \eqn{\alpha / (\alpha + \rho) \sim \operatorname{Beta}\lbrack a, b \rbrack}
  #' with further hyperprior parameters \eqn{\rho}, \eqn{a} and \eqn{b}.
  #' @param sparse_parms Numeric vector of length three for the hyperprior parameters
  #' \eqn{\rho}, \eqn{a} and \eqn{b}. By default, these are `sparse_parms=c(p, 0.5, 1.0)`.
  #' @param keep_draws Logical indicating whether to retain posterior draws. Default is `TRUE`.
  #' @param save_trees Logical indicating whether to save the posterior draws of the BART
  #' tree topologies and terminal-node parameters to `.bin` files.
  #' Default is `FALSE`.
  #' This creates files named `forests_j.bin` for the category-specific
  #' compositional regression trees.
  #' Here, `j` indexes the category, and each file contains the corresponding
  #' tree topologies and terminal node parameters across all `ndpost` posterior
  #' draws. Such files are written in disk in the directory indicated by the argument `forests_dir`.
  #' @param forests_dir Character path indicating where to save the
  #' `forests_j.bin` files when `save_trees=TRUE`. Default is to [tempdir()].
  #' @param xinfo Optional matrix containing the cut points information of each
  #' covariate supplied to the underlying `C++` implementation.
  #' An empty matrix implies that the cut points are determined internally.
  SetupMCMC = function(v0 = 3.5 / sqrt(2), ntrees = 100L,
                       ndpost = 5000L, nskip = 5000L,
                       covariance_type = c("fa_mgp", "diag", "wishart", "fa"),
                       nu_prior = self$d,
                       Psi_prior = diag(self$d, self$d - 1),
                       a_sigma = 1.0, b_sigma = 1.0,
                       q_factors = .ledermann(self$d - 1L),
                       sigma2_gamma = 1.0,
                       a_psi = 2.5, b_psi = 1.0,
                       shape_lsphis = 2.0, a1_gs = 1.5, a2_gs = 2.8,
                       numcut = 100L, power = 2.0, base = 0.95,
                       proposals_prob = c(0.25, 0.25, 0.50),
                       update_sigma = TRUE, s0_2 = 1 / ntrees, w_ss = 1.0,
                       splitprobs = rep(1 / self$p, self$p), sparse = FALSE,
                       sparse_parms = c(self$p, 0.5, 1.0), alpha_sparse = 1.0,
                       alpha_random = FALSE,
                       keep_draws = TRUE, save_trees = FALSE,
                       forests_dir = tempdir(),
                       xinfo = matrix()) {
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
    self$cpp_obj$SetMCMC(
      v0, ntrees,
      B, cov_type,
      a_sigma, b_sigma,
      Psi_prior, nu_prior,
      q_factors, sigma2_gamma,
      a_psi, b_psi,
      shape_lsphis, a1_gs, a2_gs,
      ndpost, nskip, numcut, power, base,
      proposals_prob, as.integer(update_sigma), s0_2, w_ss,
      splitprobs, as.integer(sparse), sparse_parms,
      alpha_sparse, as.integer(alpha_random), xinfo, forests_dir,
      as.integer(keep_draws), as.integer(save_trees)
    )
  },
  #' @description
  #'  Runs the MCMC sampler using the settings previously configured with
  #' \href{#method-MultinomialLNBART-SetupMCMC}{\code{MultinomialLNBART$SetupMCMC()}}.
  #' Posterior draws, acceptance rates, and
  #' variable-selection statistics, are then transferred from the underlying `C++`
  #' object to the `MultinomialLNBART` object.
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


#' @title ZANIM-reg
#'
#' @description
#' Carries out Bayesian inference for the zero-and-N-inflated multinomial logistic
#' regression model (ZANIM-reg) through an efficient Markov chain Monte Carlo algorithm.
#' The `R6` class is an wrapper for the underlying `C++` implementation.
#'
#' @details
#' The MCMC algorithm leverages the data augmentation scheme proposed by
#' Menezes et al. (2025) for inference on the parameter of ZANIM distribution
#' in i.i.d settings within a linear regression context. It assumes a probit link
#' function for the structural-zero probabilities and a log-linear link
#' function for the compositional probabilities. For the category-specific
#' structural-zero components, the regression coefficients are sampled using
#' the probit data augmentation scheme of Albert and Chib (1993). For the
#' category-specific compositional probabilities, the algorithm uses the
#' elliptical slice sampling method of Murray et al. (2010).
#' For both components normal priors are assumed for the regression coefficients.
#'
#' @references
#'
#' Albert, J. H. and Chib, S. (1993),
#' Bayesian analysis of binary and polychotomous response data,
#' \emph{Journal of the American Statistical Association}, \strong{88}(442), 669--679.
#'
#' @references Menezes, A. F. B., Parnell, A. C., and Murphy, K. (2025), Finite mixture representations of
#' zero-and-N -inflated distributions for count-compositional data. \emph{Journal of Multivariate Analysis},
#' \strong{210:105492}. <https://doi.org/10.1016/j.jmva.2025.105492>.
#'
#' Murray, I., Adams, R. and MacKay, D. (2010), Elliptical slice sampling, in Y. W. Teh \& M. Titterington, eds,
#' \emph{Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics},
#' \strong{Vol. 9 of Proceedings of Machine Learning Research, PMLR}, Chia Laguna Resort, Sardinia, Italy, pp. 541--548.
#'
#' @export
ZANIMRegression <- R6::R6Class(
  classname = "ZANIMRegression", cloneable = FALSE,
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
    #' @field p_theta Number of covariates associated to the compositional components.
    p_theta = integer(),
    #' @field p_zeta Number of covariates associated to the structural zero components.
    p_zeta = integer(),
    #' @field ndpost Number of posterior MCMC draws to retain.
    ndpost = integer(),
    #' @field nskip Number of MCMC draws to discard as burn-in before retaining
    #' posterior draws.
    nskip = integer(),
    #' @field nthin Number of MCMC draws to discard as thinning before retaining
    #' posterior draws.
    nthin = integer(),
    #' @field draws_theta Posterior draws of the population-level count probabilities.
    draws_theta = NULL,
    #' @field draws_zeta Posterior draws of the population-level structural zero probabilities.
    draws_zeta = NULL,
    #' @field draws_abundance Posterior draws of the individual-level count probabilities.
    draws_abundance = NULL,
    #' @field draws_betas_theta Posterior draws of the category-specific regression
    #' coefficients associated with the compositional probabilities.
    draws_betas_theta = NULL,
    #' @field draws_betas_zeta Posterior draws of the category-specific regression
    #' coefficients associated with the structural zero probabilities.
    draws_betas_zeta = NULL,
    #' @field elapsed_time Elapsed time taken to run the MCMC algorithm.
    elapsed_time = NULL,
    #' @field keep_draws Logical indicating whether posterior draws of the
    #' population-level compositional and structural probabilities were retained.
    keep_draws = logical(),
    #' @field keep_draws_coef Logical indicating whether posterior draws of the
    #' regression coefficients were retained.
    keep_draws_coef = logical(),
    #' @description Creates a new `ZANIMRegression` object.
    #' @param Y A matrix of multivariate count-compositional data.
    #' Rows correspond to observations and columns correspond to categories.
    #' @param X_theta A matrix of covariates used to model the count probabilities.
    #' Rows must correspond to the observations in `Y`.
    #' @param X_zeta A matrix of covariates used to model the structural zero probabilities.
    #' Rows must correspond to the observations in `Y`.
    initialize = function(Y, X_theta, X_zeta) {
      ml <- Rcpp::Module(module = "zanim_linear_reg", PACKAGE = "zanicc")
      self$cpp_obj <- methods::new(ml$ZANIMReg, Y, X_theta, X_zeta)
      self$cpp_module_name <- "zanim_linear_reg"
      self$n <- nrow(Y)
      self$d <- ncol(Y)
      self$p_theta <- ncol(X_theta)
      self$p_zeta <- ncol(X_zeta)
      self$n_trials <- rowSums(Y)
    },
    #' @description
    #' Configures priors and hyperparameters of the ZANIM-reg model
    #' used by the underlying MCMC algorithm implemented in `C++`.
    #' This method must be called before \href{#method-ZANIMRegression-RunMCMC}{\code{ZANIMRegression$RunMCMC()}}.
    #' @param sd_prior_beta_theta Numeric vector of length `p_theta` with the
    #' prior standard deviation for the regression coefficients associated with `X_theta`.
    #' @param S_prior_beta_zeta A matrix `p_zeta` by `p_zeta` with the prior
    #' covariance matrix for the regression coefficients associated with `X_zeta`.
    #' Default is the identity matrix.
    #' @param ndpost Number of posterior MCMC draws to retain. The default is `ndpost=5000`.
    #' @param nskip Number of MCMC iterations to discard as burn-in before retaining
    #' posterior draws. The default is `nskip=5000`.
    #' @param nthin Currently not used.
    #' @param keep_draws Whether to keep the draws with the subject-specific predictions
    #' for the population-level count probabilities and structural zero probabilities,
    #' as well as the individual-level count probabilities. Default is `TRUE`.
    #' @param keep_draws_coef Whether to keep the draws of the regression coefficients. Default is `TRUE`.
    SetupMCMC = function(sd_prior_beta_theta = rep(1.0, self$p_theta),
                         S_prior_beta_zeta = diag(1.0, self$p_zeta),
                         ndpost = 5000L, nskip = 5000L, nthin = 1L,
                         keep_draws = TRUE, keep_draws_coef = TRUE) {
      self$ndpost <- ndpost
      self$nskip <- nskip
      self$nthin <- nthin
      self$keep_draws <- keep_draws
      self$keep_draws_coef <- keep_draws_coef
      self$cpp_obj$SetMCMC(sd_prior_beta_theta, S_prior_beta_zeta, ndpost, nskip,
                           nthin)
    },
    #' @description
    #'  Runs the MCMC sampler using the settings previously configured with
    #' \href{#method-ZANIMRegression-SetupMCMC}{\code{ZANIMRegression$SetupMCMC()}}.
    #' Posterior draws of the parameters are then transferred from the underlying `C++`
    #' object to the `ZANIMRegression` object.
    RunMCMC = function() {
      ini <- proc.time()
      self$cpp_obj$RunMCMC()
      self$elapsed_time <- proc.time() - ini
      # Save draws
      if (self$keep_draws) {
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
    #' @description Compute the posterior mean of the regression coefficients.
    #' @param parameter Character string indicating which posterior mean of
    #' regression coefficients to compute. If `parameter="theta"`, then returns posterior
    #' mean for the regression coefficients associated to the compositional probabilities.
    #' Otherwise, if `parameter="zeta"` the return the posterior mean for the regression
    #' coefficients associated to the structural zero probabilities.
    PosteriorMeanCoef = function(parameter = c("theta", "zeta")) {
      parameter <- match.arg(parameter)
      if (self$keep_draws_coef) {
        switch(parameter,
          "zeta" = apply(self$draws_betas_zeta, c(1, 2), mean),
          "theta" = apply(self$draws_betas_theta, c(1, 2), mean)
        )
      } else {
        warning("Posterior draws of the regression coefficients were not saved because you set {keep_draws_coef = FALSE}.")
      }
    }
  )
)


#' @title ZANIDM-reg
#'
#' @description
#' Carries out Bayesian inference for the zero-and-N-inflated Dirichlet multinomial
#' regression model (ZANIDM-reg) through an efficient Markov chain Monte Carlo algorithm.
#' The `R6` class is an wrapper for the underlying `C++` implementation.
#'
#' @details
#' The MCMC algorithm leverages the data augmentation scheme proposed by
#' Menezes et al. (2025) for inference on the parameter of ZANIDM distribution
#' in i.i.d settings within a linear regression context.
#' It assumes a probit link function for the structural-zero probabilities and a log-linear link
#' function for the count components. For the category-specific
#' structural-zero components, the regression coefficients are sampled using
#' the probit data augmentation scheme of Albert and Chib (1993). For the
#' category-specific count components, the algorithm uses the
#' elliptical slice sampling method of Murray et al. (2010).
#' For both components normal priors are assumed for the regression coefficients.
#'
#' @references
#'
#' Albert, J. H. and Chib, S. (1993),
#' Bayesian analysis of binary and polychotomous response data,
#' \emph{Journal of the American Statistical Association}, \strong{88}(442), 669--679.
#'
#' @references Menezes, A. F. B., Parnell, A. C., and Murphy, K. (2025), Finite mixture representations of
#' zero-and-N -inflated distributions for count-compositional data. \emph{Journal of Multivariate Analysis},
#' \strong{210:105492}. <https://doi.org/10.1016/j.jmva.2025.105492>.
#'
#' Murray, I., Adams, R. and MacKay, D. (2010), Elliptical slice sampling, in Y. W. Teh \& M. Titterington, eds,
#' \emph{Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics},
#' \strong{Vol. 9 of Proceedings of Machine Learning Research, PMLR}, Chia Laguna Resort, Sardinia, Italy, pp. 541--548.
#'
#' @export
ZANIDMRegression <- R6::R6Class(
  classname = "ZANIDMRegression",
  cloneable = FALSE,
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
    #' @field p_alpha Number of covariates associated to the compositional components.
    p_alpha = integer(),
    #' @field p_zeta Number of covariates associated to the structural zero components.
    p_zeta = integer(),
    #' @field ndpost Number of posterior MCMC draws to retain.
    ndpost = integer(),
    #' @field nskip Number of MCMC draws to discard as burn-in before retaining
    #' posterior draws.
    nskip = integer(),
    #' @field nthin Number of MCMC draws to discard as thinning before retaining
    #' posterior draws.
    nthin = integer(),
    #' @field draws_alpha Posterior draws of the population-level concentration parameters.
    draws_alpha = NULL,
    #' @field draws_theta Posterior draws of the population-level count probabilities.
    draws_theta = NULL,
    #' @field draws_zeta Posterior draws of the population-level structural zero probabilities.
    draws_zeta = NULL,
    #' @field draws_abundance Posterior draws of the individual-level count probabilities.
    draws_abundance = NULL,
    #' @field draws_betas_alpha Posterior draws of the category-specific regression
    #' coefficients associated with the compositional probabilities (concentration parameters).
    draws_betas_alpha = NULL,
    #' @field draws_betas_zeta Posterior draws of the category-specific regression
    #' coefficients associated with the structural zero probabilities.
    draws_betas_zeta = NULL,
    #' @field elapsed_time Elapsed time taken to run the MCMC algorithm.
    elapsed_time = NULL,
    #' @field keep_draws Logical indicating whether posterior draws of the
    #' population-level compositional and structural probabilities were retained.
    keep_draws = logical(),
    #' @field keep_draws_coef Logical indicating whether posterior draws of the
    #' regression coefficients were retained.
    keep_draws_coef = logical(),
    #' @field save_draws Logical indicating whether the posterior draws of the
    #' regression coefficients were saved in disk.
    save_draws = logical(),
    #' @field dir_draws Character string indicating the directory where the
    #' posterior draws of the regression coefficients were saved.
    dir_draws = NULL,
    #' @description Creates a new `ZANIDMRegression` object.
    #' @param Y A matrix of multivariate count-compositional data.
    #' Rows correspond to observations and columns correspond to categories.
    #' @param X_alpha A matrix of covariates used to model the count probabilities.
    #' Rows must correspond to the observations in `Y`.
    #' @param X_zeta A matrix of covariates used to model the structural zero probabilities.
    #' Rows must correspond to the observations in `Y`.
    initialize = function(Y, X_alpha, X_zeta) {
      ml <- Rcpp::Module(module = "zanidm_linear_reg", PACKAGE = "zanicc")
      self$cpp_obj <- methods::new(ml$ZANIDMReg, Y, X_alpha, X_zeta)
      self$cpp_module_name <- "zanidm_linear_reg"
      self$n <- nrow(Y)
      self$d <- ncol(Y)
      self$p_alpha <- ncol(X_alpha)
      self$p_zeta <- ncol(X_zeta)
      self$n_trials <- rowSums(Y)
    },
    #' @description
    #' Configures priors and hyperparameters of the ZANIDM-reg model
    #' used by the underlying MCMC algorithm implemented in `C++`.
    #' This method must be called before \href{#method-ZANIDMRegression-RunMCMC}{\code{ZANIDMRegression$RunMCMC()}}.
    #' @param sd_prior_beta_alpha Numeric vector of length `p_alpha` with the
    #' prior standard deviation for the regression coefficients associated with `X_theta`.
    #' @param S_prior_beta_zeta A matrix `p_zeta` by `p_zeta` with the prior
    #' covariance matrix for the regression coefficients associated with `X_zeta`.
    #' Default is the identity matrix.
    #' @param ndpost Number of posterior MCMC draws to retain. The default is `ndpost=5000`.
    #' @param nskip Number of MCMC iterations to discard as burn-in before retaining
    #' posterior draws. The default is `nskip=5000`.
    #' @param nthin Currently not used.
    #' @param keep_draws Whether to keep the draws with the subject-specific predictions
    #' for the population-level count probabilities and structural zero probabilities,
    #' as well as the individual-level count probabilities. Default is `TRUE`.
    #' @param keep_draws_coef Whether to keep the draws of the regression coefficients. Default is `TRUE`.
    #' @param save_draws Whether to save the draws of the regression coefficients in binary format.
    #' This creates the files `draws_betas_alpha.bin` for the count components and
    #' `draws_betas_zeta.bin` for the structural zero components. These files are
    #' created in the folder provided in the argument `dir_draws`.
    #' @param dir_draws Character string with the folder where the posterior draws
    #' of the regression coefficients should be saved, when `save_draws=TRUE`.
    #' Default is [tempdir()].
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
    #' @description
    #'  Runs the MCMC sampler using the settings previously configured with
    #' \href{#method-ZANIDMRegression-SetupMCMC}{\code{ZANIDMRegression$SetupMCMC()}}.
    #' Posterior draws of the parameters are then transferred from the underlying `C++`
    #' object to the `ZANIDMRegression` object.
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
    #' @description Compute the posterior mean of the regression coefficients.
    #' @param parameter Character string indicating which posterior mean of
    #' regression coefficients to compute. If `parameter="alpha"`, then returns posterior
    #' mean for the regression coefficients associated to the compositional probabilities.
    #' Otherwise, if `parameter="zeta"` the return the posterior mean for the regression
    #' coefficients associated to the structural zero probabilities.
    PosteriorMeanCoef = function(parameter = c("alpha", "zeta")) {
      parameter <- match.arg(parameter)
      if (self$keep_draws_coef) {
        switch(parameter,
          "zeta" = apply(self$draws_betas_zeta, c(1, 2), mean),
          "alpha" = apply(self$draws_betas_alpha, c(1, 2), mean)
        )
      } else {
        warning("Posterior draws of the regression coefficients were not saved because you set {keep_draws_coef = FALSE}.")
      }
    }
  )
)

#' @title ZANIM-LN-reg
#'
#' @description
#' Carries out Bayesian inference for the zero-and-N-inflated multinomial logistic-normal
#' regression model (ZANIM-LN-reg) through an efficient Markov chain Monte Carlo algorithm.
#' The `R6` class is an wrapper for the underlying `C++` implementation.
#'
#' @details
#' The ZANIM-LN distribution extends the ZANIM distribution with additional
#' logistic-normal random effects on the compositional probabilities.
#' The current MCMC algorithm leverages the data augmentation scheme proposed by
#' Menezes et al. (2025) for inference on the parameter of ZANIM distribution
#' in i.i.d settings within a linear regression context.
#' It assumes a probit link function for the structural-zero probabilities and a log-linear link
#' function for the count components. For the category-specific
#' structural-zero components, the regression coefficients are sampled using
#' the probit data augmentation scheme of Albert and Chib (1993). For the
#' category-specific count components, the algorithm uses the
#' elliptical slice sampling method of Murray et al. (2010).
#' For both components normal priors are assumed for the regression coefficients.
#'
#'
#' @references
#'
#' Albert, J. H. and Chib, S. (1993),
#' Bayesian analysis of binary and polychotomous response data,
#' \emph{Journal of the American Statistical Association}, \strong{88}(442), 669--679.
#'
#' @references Menezes, A. F. B., Parnell, A. C., and Murphy, K. (2025), Finite mixture representations of
#' zero-and-N -inflated distributions for count-compositional data. \emph{Journal of Multivariate Analysis},
#' \strong{210:105492}. <https://doi.org/10.1016/j.jmva.2025.105492>.
#'
#' Murray, I., Adams, R. and MacKay, D. (2010), Elliptical slice sampling, in Y. W. Teh \& M. Titterington, eds,
#' \emph{Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics},
#' \strong{Vol. 9 of Proceedings of Machine Learning Research, PMLR}, Chia Laguna Resort, Sardinia, Italy, pp. 541--548.
#'
#' @export

ZANIMLNRegression <- R6::R6Class(
  classname = "ZANIMLNRegression",
  cloneable = FALSE,
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
    #' @field p_theta Number of covariates associated to the compositional components.
    p_theta = integer(),
    #' @field p_zeta Number of covariates associated to the structural zero components.
    p_zeta = integer(),
    #' @field ndpost Number of posterior MCMC draws to retain.
    ndpost = integer(),
    #' @field nskip Number of MCMC draws to discard as burn-in before retaining
    #' posterior draws.
    nskip = integer(),
    #' @field nthin Number of MCMC draws to discard as thinning before retaining
    #' posterior draws.
    nthin = integer(),
    #' @field covariance_type Character string with the prior used for the covariance
    #' matrix for the logistic-normal random effects.
    covariance_type = NULL,
    #' @field Bt Transpose of the orthogonal matrix for the sum-to-zero constraint
    #' in the logistic random effects.
    Bt = NULL,
    #' @field draws_theta Posterior draws of the population-level count probabilities.
    draws_theta = NULL,
    #' @field draws_zeta Posterior draws of the population-level structural zero probabilities.
    draws_zeta = NULL,
    #' @field draws_abundance Posterior draws of the individual-level count probabilities.
    draws_abundance = NULL,
    #' @field draws_betas_theta Posterior draws of the category-specific regression
    #' coefficients associated with the compositional probabilities.
    draws_betas_theta = NULL,
    #' @field draws_betas_zeta Posterior draws of the category-specific regression
    #' coefficients associated with the structural zero probabilities.
    draws_betas_zeta = NULL,
    #' @field draws_chol_Sigma_V Posterior draws of the Cholesky decomposition of the
    #' covariance matrix of the logistic-normal random effects.
    draws_chol_Sigma_V = NULL,
    #' @field elapsed_time Elapsed time taken to run the MCMC algorithm.
    elapsed_time = NULL,
    #' @field keep_draws Logical indicating whether posterior draws of the
    #' population-level compositional and structural probabilities were retained.
    keep_draws = logical(),
    #' @field keep_draws_coef Logical indicating whether posterior draws of the
    #' regression coefficients were retained.
    keep_draws_coef = logical(),
    #' @description Creates a new `ZANIMLNRegression` object.
    #' @param Y A matrix of multivariate count-compositional data.
    #' Rows correspond to observations and columns correspond to categories.
    #' @param X_theta A matrix of covariates used to model the count probabilities.
    #' Rows must correspond to the observations in `Y`.
    #' @param X_zeta A matrix of covariates used to model the structural zero probabilities.
    #' Rows must correspond to the observations in `Y`.
    initialize = function(Y, X_theta, X_zeta) {
      ml <- Rcpp::Module(module = "zanim_ln_reg", PACKAGE = "zanicc")
      self$cpp_obj <- methods::new(ml$ZANIMLNReg, Y, X_theta, X_zeta)
      self$cpp_module_name <- "zanim_ln_reg"
      # self$Y <- Y
      self$n <- nrow(Y)
      self$d <- ncol(Y)
      self$p_theta <- ncol(X_theta)
      self$p_zeta <- ncol(X_zeta)
      self$n_trials <- rowSums(Y)
    },
    #' @description
    #' Configures priors and hyperparameters of the ZANIM-LN-reg model
    #' used by the underlying MCMC algorithm implemented in `C++`.
    #' This method must be called before \href{#method-ZANIDMRegression-RunMCMC}{\code{ZANIDMRegression$RunMCMC()}}.
    #' @param sd_prior_beta_theta Numeric vector of length `p_alpha` with the
    #' prior standard deviation for the regression coefficients associated with `X_theta`.
    #' @param S_prior_beta_zeta A matrix `p_zeta` by `p_zeta` with the prior
    #' covariance matrix for the regression coefficients associated with `X_zeta`.
    #' Default is the identity matrix.
    #' @param ndpost Number of posterior MCMC draws to retain. The default is `ndpost=5000`.
    #' @param nskip Number of MCMC iterations to discard as burn-in before retaining
    #' posterior draws. The default is `nskip=5000`.
    #' @param nthin Currently not used.
    #' @param covariance_type Character string specifying the prior on the covariance
    #' matrix for the logistic-normal random effects. Default is `fa_mgp`, for nonparametric factor
    #'  analysis with a multiplicative gamma process shrinkage prior. Other options include `fa` (factor analysis without such a prior),
    #'  `diag` (for a diagonal covariance matrix with inverse-gamma priors), and `wishart` (for an inverse Wishart prior).
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
    #' @param keep_draws Whether to keep the draws with the subject-specific predictions
    #' for the population-level count probabilities and structural zero probabilities,
    #' as well as the individual-level count probabilities. Default is `TRUE`.
    #' @param keep_draws_coef Whether to keep the draws of the regression coefficients. Default is `TRUE`.
    SetupMCMC = function(sd_prior_beta_theta = rep(1.0, self$p_theta),
                         S_prior_beta_zeta = diag(1.0, self$p_zeta),
                         ndpost = 5000L, nskip = 5000L, nthin = 1L,
                         covariance_type = c("fa_mgp", "diag", "wishart", "fa"),
                         nu_prior = self$d,
                         Psi_prior = diag(self$d, self$d - 1),
                         a_sigma = 1.0, b_sigma = 1.0,
                         q_factors = .ledermann(self$d - 1L),
                         sigma2_gamma = 1.0,
                         a_psi = 2.5, b_psi = 1.0,
                         shape_lsphis = 2.0, a1_gs = 2.1, a2_gs = 3.1,
                         keep_draws = TRUE, keep_draws_coef = TRUE) {
      covariance_type <- match.arg(covariance_type)
      cov_type <- as.integer(which(covariance_type == c("diag", "wishart", "fa", "fa_mgp"))) - 1L
      if (q_factors == 0) q_factors <- 1
      self$covariance_type <- covariance_type
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
    #' @description
    #' Runs the MCMC sampler using the settings previously configured with
    #' \href{#method-ZANIMLNRegression-SetupMCMC}{\code{ZANIMLNRegression$SetupMCMC()}}.
    #' Posterior draws of the parameters are then transferred from the underlying `C++`
    #' object to the `ZANIMLNRegression` object.
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
    #' @description Compute the posterior mean of the regression coefficients.
    #' @param parameter Character string indicating which posterior mean of
    #' regression coefficients to compute. If `parameter="theta"`, then returns posterior
    #' mean for the regression coefficients associated to the compositional probabilities.
    #' Otherwise, if `parameter="zeta"` the return the posterior mean for the regression
    #' coefficients associated to the structural zero probabilities.
    PosteriorMeanCoef = function(parameter = c("theta", "zeta")) {
      parameter <- match.arg(parameter)
      if (self$keep_draws_coef) {
        switch(parameter,
          "zeta" = apply(self$draws_betas_zeta, c(1, 2), mean),
          "theta" = apply(self$draws_betas_theta, c(1, 2), mean)
        )
      } else{
        warning("Posterior draws of the regression coefficients were not saved because you set {keep_draws_coef = FALSE}.")
      }
    }
  )
)


#' @title DM-reg
#'
#' @description
#' Carries out Bayesian inference for the Dirichlet multinomial
#' regression model (DM-reg) through an efficient Markov chain Monte Carlo algorithm.
#' The `R6` class is an wrapper for the underlying `C++` implementation.
#'
#' @details
#' The MCMC algorithm leverages a data augmentation scheme using the representation of
#' Dirichlet-multinomial through a vector of normalised independent gamma distributions.
#' It assumes a log-linear link
#' function for the count components, and update the regression coefficients using
#' the elliptical slice sampling method of Murray et al. (2010).
#' Normal priors are assumed for the regression coefficients.
#'
#' @references
#' Murray, I., Adams, R. and MacKay, D. (2010), Elliptical slice sampling, in Y. W. Teh \& M. Titterington, eds,
#' \emph{Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics},
#' \strong{Vol. 9 of Proceedings of Machine Learning Research, PMLR}, Chia Laguna Resort, Sardinia, Italy, pp. 541--548.
#'
#' @export
DMRegression <- R6::R6Class(
  classname = "DMRegression",
  cloneable = FALSE,
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
    #' @field p Number of covariates.
    p = integer(),
    #' @field ndpost Number of posterior MCMC draws to retain.
    ndpost = integer(),
    #' @field nskip Number of MCMC draws to discard as burn-in before retaining
    #' posterior draws.
    nskip = integer(),
    #' @field nthin Number of MCMC draws to discard as thinning before retaining
    #' posterior draws.
    nthin = integer(),
    #' @field draws_alpha Posterior draws of the population-level concentration parameters.
    draws_alpha = NULL,
    #' @field draws_theta Posterior draws of the population-level count probabilities.
    draws_theta = NULL,
    #' @field draws_abundance Posterior draws of the individual-level count probabilities.
    draws_abundance = NULL,
    #' @field draws_betas Posterior draws of the category-specific regression
    #' coefficients.
    draws_betas = NULL,
    #' @field elapsed_time Elapsed time taken to run the MCMC algorithm.
    elapsed_time = NULL,
    #' @field keep_draws Logical indicating whether posterior draws of the
    #' population-level compositional and structural probabilities were retained.
    keep_draws = logical(),
    #' @field keep_draws_coef Logical indicating whether posterior draws of the
    #' regression coefficients were retained.
    keep_draws_coef = logical(),
    #' @field save_draws Logical indicating whether the posterior draws of the
    #' regression coefficients were saved in disk.
    save_draws = logical(),
    #' @field dir_draws Character string indicating the directory where the
    #' posterior draws of the regression coefficients were saved.
    dir_draws = NULL,

    #' @description Creates a new `DMRegression` object.
    #' @param Y A matrix of multivariate count-compositional data.
    #' Rows correspond to observations and columns correspond to categories.
    #' @param X A matrix of covariates used to model the count probabilities.
    #' Rows must correspond to the observations in `Y`.
    initialize = function(Y, X) {
      ml <- Rcpp::Module(module = "dm_linear_reg", PACKAGE = "zanicc")
      self$cpp_obj <- methods::new(ml$DMLinearReg, Y, X)
      self$cpp_module_name <- "dm_linear_reg"
      self$n <- nrow(Y)
      self$d <- ncol(Y)
      self$p <- ncol(X)
      self$n_trials <- rowSums(Y)
    },
    #' @description
    #' Configures priors and hyperparameters of the DM-reg model
    #' used by the underlying MCMC algorithm implemented in `C++`.
    #' This method must be called before \href{#method-DMRegression-RunMCMC}{\code{DMRegression$RunMCMC()}}.
    #' @param S_prior_betas A matrix of dimension `p` by `p` with the prior
    #' covariance matrix for the regression coefficients associated with `X`.
    #' Default is the identity matrix.
    #' @param ndpost Number of posterior MCMC draws to retain. The default is `ndpost=5000L`.
    #' @param nskip Number of MCMC iterations to discard as burn-in before retaining
    #' posterior draws. The default is `nskip=5000L`.
    #' @param nthin Currently not used.
    #' @param keep_draws Whether to keep the draws with the subject-specific predictions
    #' for the population-level count probabilities and structural zero probabilities,
    #' as well as the individual-level count probabilities. Default is `TRUE`.
    #' @param keep_draws_coef Whether to keep the draws of the regression coefficients. Default is `TRUE`.
    #' @param save_draws Whether to save the draws of the regression coefficients in binary format.
    #' This creates the file `draws_betas.bin` in the folder provided by the argument
    #' `dir_draws`.
    #' @param dir_draws Character string with the folder where the posterior draws
    #' of the regression coefficients should be saved, when `save_draws=TRUE`.
    #' Default is [tempdir()].
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
    #' @description
    #' Runs the MCMC sampler using the settings previously configured with
    #' \href{#method-DMRegression-SetupMCMC}{\code{DMRegression$SetupMCMC()}}.
    #' Posterior draws of the parameters are then transferred from the underlying `C++`
    #' object to the `DMRegression` object.
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
    #' @description Compute the posterior mean of the regression coefficients.
    PosteriorMeanCoef = function() {
      if (self$keep_draws) apply(self$draws_betas, c(1, 2), mean)
      else warning("Posterior draws of the regression coefficients were not saved because you set {keep_draws_coef = FALSE}.")
    }
  )
)
