#' ZANIM logistic BART
#' @export
ZANIMBART <- R6::R6Class(classname = "ZANIMBART", public = list(
  cpp_obj = NULL, cpp_module_name = character(),
  n_trials = integer(), n = integer(), d = integer(), p_theta = integer(),
  p_zeta = integer(), ntrees_zeta = integer(), ntrees_theta = integer(),
  ndpost = integer(), niter = integer(), nskip = integer(), forests_dir = character(),
  n_pred = integer(), ndpost_pred = integer(), link_zeta = character(),
  shared_trees = logical(), elapsed_time = NULL,  elapsed_time_log_lik = NULL,
  avg_leaves_theta = NULL, avg_leaves_zeta = NULL, accept_rate_theta = NULL,
  accept_rate_zeta = NULL, draws_theta = NULL, draws_zeta = NULL,
  draws_abundance = NULL, draws_phi = NULL, y_rep_draws = NULL,
  keep_draws = logical(), save_trees = logical(),
  log_lik_draws = NULL, varcount_theta = NULL,
  varcount_zeta = NULL, sigma_theta_hyperprior = NULL,
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
  SetupMCMC = function(v0_theta = 1.5 / sqrt(2),
                       v0_zeta = if (self$link_zeta == "logit") 3.5 / sqrt(2) else 3.0,
                       ntrees_theta = 20L, ntrees_zeta = 20L,
                       ndpost = 1000L, nskip = 1000L,
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
    self$ntrees_theta <- ntrees_theta
    self$ntrees_zeta <- ntrees_zeta
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$niter <- nskip + ndpost
    self$forests_dir <- forests_dir
    self$keep_draws <- keep_draws
    self$save_trees <- save_trees
    alpha_sparse_mult <- alpha_sparse[2L]
    if (!self$shared_trees) {
      if (!is.list(splitprobs_mult))
        splitprobs_mult <- replicate(self$d, splitprobs_mult, simplify = FALSE)
      alpha_sparse_mult <- rep(alpha_sparse[2L], self$d)
    }
    self$cpp_obj$SetMCMC(v0_theta, v0_zeta, ntrees_theta, ntrees_zeta, ndpost, nskip,
                         numcut, power, base, proposals_prob,
                         as.integer(update_sigma_theta), s0_2_theta, w_ss,
                         splitprobs_zi, splitprobs_mult,
                         as.integer(sparse[1L]), as.integer(sparse[2L]),
                         sparse_parms[1L:3L], sparse_parms[4L:6L],
                         rep(alpha_sparse[1L], self$d), alpha_sparse_mult,
                         as.integer(alpha_random[1L]), as.integer(alpha_random[2L]),
                         xinfo, forests_dir, as.integer(keep_draws),
                         as.integer(save_trees))
  },
  RunMCMC = function() {
    ini <- proc.time()
    self$cpp_obj$RunMCMC()
    self$elapsed_time <- proc.time() - ini

    # Average number of leaves for theta and zeta regression trees
    self$avg_leaves_theta <- self$cpp_obj$avg_leaves_theta / self$ndpost
    self$avg_leaves_zeta <- self$cpp_obj$avg_leaves_zeta / self$ndpost
    # Avg accept rate over iteration and the trees
    self$accept_rate_theta <- self$cpp_obj$accept_rate_theta / self$niter / self$ntrees_theta
    self$accept_rate_zeta <- self$cpp_obj$accept_rate_zeta / self$niter / self$ntrees_zeta
    rownames(self$accept_rate_zeta) <- rownames(self$accept_rate_theta) <- c("grow", "prune", "change")
    # Keep the draws of the hyperprior sd
    self$sigma_theta_hyperprior <- self$cpp_obj$sigma_mult_mcmc
    # Save draws
    if (self$keep_draws) {
      self$draws_theta <- self$cpp_obj$draws_theta
      self$draws_abundance <- self$cpp_obj$draws_vartheta
      self$draws_zeta <- self$cpp_obj$draws_zeta
      if (self$link_zeta == "probit") self$draws_zeta <- stats::pnorm(self$draws_zeta)
      self$draws_phi <- self$cpp_obj$draws_phi
      self$varcount_theta <- self$cpp_obj$varcount_mcmc_theta
      self$varcount_zeta <- self$cpp_obj$varcount_mcmc_zeta
    }
  },
  GetPosteriorPredictive = function(in_sample = TRUE, batch_size = 100L,
                                    ndpost = self$ndpost,
                                    forests_dir = self$forests_dir,
                                    output_dir = self$forests_dir,
                                    n_pred = self$n_pred, ...) {
    if (in_sample) {
      if (!self$keep_draws)
        stop("Draws are not saved in the class. Run MCMC again with {keep_draws=TRUE}")
      ppd(self, ...)
    } else {
      # Path for the file with the predictions
      ff_theta <- file.path(forests_dir, "theta_ij.bin")
      ff_zeta <- file.path(forests_dir, "zeta_ij.bin")

      if (!file.exists(ff_theta))
        stop("File with the predictions of theta_{ij} does not exist.")
      if (!file.exists(ff_zeta))
        stop("File with the predictions of zeta_{ij} does not exist.")

      # Create a vector to load the predictions by batch
      look_head <- seq.int(from = 1L, to = ndpost, by = batch_size)
      # y_rep <- array(0L, dim = c(n_pred, self$d, ndpost))
      y_rep <- array(0L, dim = c(ndpost, n_pred, self$d))
      # Load by batch and generate the posterior-predictive
      for (i in seq_len(length(look_head))) {
        shift <- look_head[i] - 1L
        # Load \theta_ij
        thetas <- .load_bin_batch(fname = ff_theta, n = n_pred, d = self$d,
                                  k = look_head[i], m = batch_size)
        # Load \zeta_{ij}
        zetas <- .load_bin_batch(fname = ff_zeta, n = n_pred, d = self$d,
                                 k = look_head[i], m = batch_size)
        for (k in seq_len(batch_size)) {
          cat(shift + k, "\n")
          y_rep[shift + k,,] <- .rzanim_vec(n = n_pred, sizes = n_trials,
                                            probs = thetas[, , k],
                                            zetas = zetas[, , k], d = self$d)
        }
      }
      # if (relative) y_rep <- .normalize_composition(y_rep)
      return(y_rep)
    }
  },
  GetVarCount = function(parameter = c("theta", "zeta"), ndpost = self$npost) {
    if (self$shared_trees) stop("Method not available for shared trees.")
    parameter <- match.arg(parameter)
    if (!is.null(self$elapsed_time)) {
      vc <- switch(parameter,
                  "theta" = self$cpp_obj$varcount_mcmc_theta,
                  "zeta" = self$cpp_obj$varcount_mcmc_zeta)
    } else {
      ntrees <- ifelse(parameter == "theta", self$ntrees_theta, self$ntrees_zeta)
      vc <- self$cpp_obj$GetVarCount(ndpost, ntrees, parameter, self$forests_dir)
    }
    vc
  },
  ComputePredictions = function(X, ndpost = self$ndpost,
                                forests_dir = self$forests_dir,
                                output_dir = self$forests_dir,
                                parameter = c("theta", "zeta"),
                                verbose = TRUE) {
    self$n_pred <- nrow(X)
    self$ndpost_pred <- ndpost
    parameter <- match.arg(parameter)
    switch(parameter,
           "theta" = self$cpp_obj$ComputePredictProb(X, ndpost,
                                                     self$ntrees_theta,
                                                     forests_dir, output_dir,
                                                     as.integer(verbose)),
           "zeta" = self$cpp_obj$ComputePredictProbZero(X, ndpost,
                                                        self$ntrees_zeta,
                                                        forests_dir, output_dir,
                                                        as.integer(verbose)))
  },
  LoadPredictions = function(ndpost = self$ndpost_pred,
                             parameter = c("theta", "zeta"),
                             output_dir = self$forests_dir, n_pred = self$n_pred) {
    parameter <- match.arg(parameter)
    ff <- paste0(parameter, "_ij.bin")
    if (!file.exists(file.path(output_dir, ff))) stop("File with the predictions does not exist.")
    load_bin_predictions(fname = file.path(output_dir, ff), n = n_pred, d = self$d, m = ndpost)
  },
  DeleteForests = function() {
    unlink(x = self$forests_dir)
  },
  LogPredictiveLikelihood = function(in_sample = TRUE,
                                     Y = NULL, ndpost = self$ndpost,
                                     output_dir = self$forests_dir, n_pred = self$n_pred,
                                     nthin = 1L, parallel = TRUE, ncores = 10L,
                                     logfile = tempfile()) {
    if (is.null(Y)) stop("Please, provide the data matrix, Y.")

    if (in_sample) {
      if (!self$keep_draws) stop("Posterior draws are not saved in the class.")

      if (parallel) {
        seq_ind <- seq_len(self$n)
        ini <- proc.time()
        out <- parallel::mclapply(X = seq_len(ndpost), FUN = function(k) {
          sapply(seq_ind, function(i) {
            log_pmf_zanim(x = Y[i, ], prob = self$draws_theta[i,,k],
                          zeta = self$draws_zeta[i,,k])
          })
        }, mc.cores = ncores)
        self$elapsed_time_log_lik <- proc.time() - ini
        self$log_lik_draws <- do.call(rbind, out)
      } else {
        ll <- matrix(data = 0.0, nrow = self$n, ncol = ndpost)
        for (k in seq_len(ndpost)) {
          if (k %% 100L == 0) cat(k, "\n")
          ll[, k] <- sapply(X = seq_len(self$n), FUN = function(i) {
            log_pmf_zanim(x = Y[i, ], prob = self$draws_theta[i,,k],
                          zeta = self$draws_zeta[i,,k])
          })
        }
        self$log_lik_draws <- t(ll)
      }
    } else {
      # Path for the file with the predictions
      ff_theta <- file.path(output_dir, "theta_ij.bin")
      ff_zeta <- file.path(output_dir, "zeta_ij.bin")

      if (!file.exists(ff_theta))
        stop("File with the predictions of theta_{ij} does not exist.")
      if (!file.exists(ff_zeta))
        stop("File with the predictions of zeta_{ij} does not exist.")

      n_zeros <- rowSums(Y == 0)
      n_trials <- rowSums(Y)

      seq_ind <- seq_len(n_pred)
      seq_samples <- seq.int(1, ndpost, by = nthin)
      ini <- proc.time()
      out <- parallel::mclapply(X = seq_samples, FUN = function(t) {
        # load parameters at iteration t
        thetas <- .load_bin_batch(fname = ff_theta, n = n_pred, d = self$d, k = t,
                                  m = 1L, arr = FALSE)
        zetas <- .load_bin_batch(fname = ff_zeta, n = n_pred,
                                 d = self$d, k = t, m = 1L, arr = FALSE)
        # Log-file
        write.table(x = data.frame(t = t), file = file.path(output_dir, logfile),
                    append = TRUE, row.names = FALSE, col.names = FALSE)
        # compute likelihood for each observed data
        sapply(seq_ind, function(i) {
          idx <- seq.int(i, n_pred * self$d, by = n_pred)
          return(log_pmf_zanim(x = Y[i, ], prob = thetas[idx], zeta = zetas[idx]))
          # if (n_zeros[i] < 18) {
          #   return(log_pmf_zanim(x = Y[i, ], prob = thetas[idx], zeta = zetas[idx]))
          # } else {
          #   return(.log_pmf_zanim_approx(x = Y[i, ], prob = thetas[idx],
          #                                zeta = zetas[idx], scale = n_trials[i],
          #                                mc = 5000L, nskip = 1000L))
          # }
        })
      }, mc.cores = ncores)
      self$elapsed_time_log_lik <- proc.time() - ini
      self$log_lik_draws <- do.call(rbind, out)
    }
    return(self$log_lik_draws)
  }
))

#' ZANIM logistic normal BART
#' @export
ZANIMLNBART <- R6::R6Class(classname = "ZANIMLNBART", public = list(
  cpp_obj = NULL, cpp_module_name = character(),
  n_trials = integer(), n = integer(), d = integer(), p_theta = integer(),
  p_zeta = integer(), ntrees_zeta = integer(), ntrees_theta = integer(),
  ndpost = integer(), niter = integer(), nskip = integer(), forests_dir = character(),
  n_pred = integer(), ndpost_pred = integer(), covariance_type = NULL,
  elapsed_time = NULL,  elapsed_time_log_lik = NULL,
  avg_leaves_theta = NULL, avg_leaves_zeta = NULL, accept_rate_theta = NULL,
  accept_rate_zeta = NULL, y_rep_draws = NULL, log_lik_draws = NULL,
  draws_theta = NULL, Bt = NULL,
  draws_zeta = NULL, draws_phi = NULL, draws_chol_Sigma_V = NULL,
  draws_abundance = NULL, varcount_theta = NULL, varcount_zeta = NULL,
  sigma_theta_hyperprior = NULL,
  keep_draws = logical(), save_trees = logical(),
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
  SetupMCMC = function(v0_theta = 1.5 / sqrt(2), k_zeta = 3.0,
                       ntrees_theta = 50L, ntrees_zeta = 100L,
                       ndpost = 1000L, nskip = 1000L,
                       covariance_type = c("diag", "wishart", "fa", "fa_mgp"),
                       nu_prior = self$d,
                       Psi_prior = diag(self$d, self$d - 1),
                       a_sigma = 1.0, b_sigma = 1.0,
                       q_factors = .ledermann(self$d - 1L), sigma2_gamma = 1.0,
                       a_psi = 2.5, b_psi = 1.0,
                       shape_lsphis = 3.0, a1_gs = 2.1, a2_gs = 3.1,
                       # shape_lsphis = 2.0, a1_gs = 1.5, a2_gs = 2.8,
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
    self$covariance_type <- covariance_type
    self$ntrees_theta <- ntrees_theta
    self$ntrees_zeta <- ntrees_zeta
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$niter <- nskip + ndpost
    self$forests_dir <- forests_dir
    self$keep_draws <- keep_draws
    self$save_trees <- save_trees
    alpha_sparse_mult <- alpha_sparse[2L]
    if (!is.list(splitprobs_mult))
      splitprobs_mult <- replicate(self$d, splitprobs_mult, simplify = FALSE)
    alpha_sparse_mult <- rep(alpha_sparse[2L], self$d)
    B <- qr.Q(qr(stats::contr.sum(self$d)))
    self$Bt <- t(B)
    self$cpp_obj$SetMCMC(v0_theta, k_zeta, ntrees_theta, ntrees_zeta,
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
                         xinfo, forests_dir, as.integer(keep_draws), as.integer(save_trees))
  },
  RunMCMC = function() {
    ini <- proc.time()
    self$cpp_obj$RunMCMC()
    self$elapsed_time <- proc.time() - ini
    # Average number of leaves for theta and zeta regression trees
    self$avg_leaves_theta <- self$cpp_obj$avg_leaves_theta / self$ndpost
    self$avg_leaves_zeta <- self$cpp_obj$avg_leaves_zeta / self$ndpost
    # Avg accept rate over iteration and the trees
    self$accept_rate_theta <- self$cpp_obj$accept_rate_theta / self$niter / self$ntrees_theta
    self$accept_rate_zeta <- self$cpp_obj$accept_rate_zeta / self$niter / self$ntrees_zeta
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
    }
  },
  GetPosteriorPredictive = function(in_sample = TRUE, batch_size = 100L,
                                    output_dir = self$forests_dir, n_pred = self$n_pred) {
    if (in_sample) {
      if (!self$keep_draws)
        stop("Draws are not saved in the class. Run MCMC again with {keep_draws=TRUE}.")
      ppd(self, ...)
    } else {
      # Path for the file with the predictions
      ff_theta <- file.path(output_dir, "theta_ij.bin")
      ff_zeta <- file.path(output_dir, "zeta_ij.bin")

      if (!file.exists(ff_theta))
        stop("File with the predictions of theta_{ij} does not exist.")
      if (!file.exists(ff_zeta))
        stop("File with the predictions of zeta_{ij} does not exist.")

      # Create a vector to load the predictions by batch
      look_head <- seq.int(from = 1L, to = ndpost, by = batch_size)
      y_rep <- array(data = NA_real_, dim = c(n_pred, self$d, ndpost))
      # Load by batch and generate the posterior-predictive
      for (i in seq_len(length(look_head))) {
        shift <- look_head[i] - 1L
        # Load \theta_ij
        thetas <- .load_bin_batch(fname = ff_theta, n = n_pred, d = self$d,
                                  k = look_head[i], m = batch_size)
        # Load \zeta_{ij}
        zetas <- .load_bin_batch(fname = ff_zeta, n = n_pred, d = self$d,
                                 k = look_head[i], m = batch_size)
        # TODO: load chol_Sigma_V and generate random effect

        for (k in seq_len(batch_size)) {
          cat(shift + k, "\n")
          y_rep[,,shift + k] <- .rzanim_vec(n = n_pred, sizes = n_trials,
                                            probs = thetas[, , k],
                                            zetas = zetas[, , k])
        }
      }
      if (relative) y_rep <- .normalize_composition(y_rep)
      return(y_rep)
    }
  },
  ComputePredictions = function(X, ndpost = self$ndpost,
                                forests_dir = self$forests_dir,
                                output_dir = self$forests_dir,
                                parameter = c("theta", "zeta"), verbose = TRUE) {
    self$n_pred <- nrow(X)
    self$ndpost_pred <- ndpost
    parameter <- match.arg(parameter)
    switch(parameter,
           "theta" = self$cpp_obj$ComputePredictProb(X, ndpost,
                                                     self$ntrees_theta,
                                                     forests_dir, output_dir,
                                                     as.integer(verbose)),
           "zeta" = self$cpp_obj$ComputePredictProbZero(X, ndpost,
                                                        self$ntrees_zeta,
                                                        forests_dir, output_dir,
                                                        as.integer(verbose)))
  },
  LoadPredictions = function(output_dir = self$forests_dir, parameter = c("theta", "zeta")) {
    parameter <- match.arg(parameter)
    if (!self$keep_draws) {
      ff <- paste0(parameter, "_ij.bin")
      if (!file.exists(file.path(output_dir, ff)))
        stop("File with the predictions does not exist.")
      return(load_bin_predictions(fname = file.path(output_dir, ff), n = self$n_pred, d = self$d,
                       m = self$ndpost_pred))
    } else {
      if (parameter == "theta") return(self$draws_theta)
      else return(self$draws_zeta)
    }
  },
  LogPredictiveLikelihood = function(in_sample = TRUE,
                                     Y = NULL, ndpost = self$ndpost,
                                     output_dir = self$forests_dir, n_pred = self$n_pred,
                                     nthin = 1L,
                                     parallel = TRUE,
                                     ncores = 20L,
                                     logfile = tempfile()) {
    if (is.null(Y)) stop("Please, provide the data matrix, Y.")

    if (in_sample) {
      if (!self$keep_draws) stop("Posterior draws are not saved in class.")
      if (parallel) {
        seq_ind <- seq_len(self$n)
        ini <- proc.time()
        out <- parallel::mclapply(X = seq_len(ndpost), FUN = function(k) {
          sapply(seq_ind, function(i) {
            log_pmf_zanim(x = Y[i, ], prob = self$draws_theta[i,,k],
                          zeta = self$draws_zeta[i,,k])
          })
        }, mc.cores = ncores)
        self$elapsed_time_log_lik <- proc.time() - ini
        self$log_lik_draws <- do.call(rbind, out)
      } else {
        ll <- matrix(data = 0.0, nrow = self$n, ncol = ndpost)
        for (k in seq_len(ndpost)) {
          if (k %% 100 == 0) cat(k, "\n")
          ll[, k] <- sapply(X = seq_len(nrow(Y)), FUN = function(i) {
            log_pmf_zanim(x = Y[i, ], prob = self$draws_theta[i,,k],
                          zeta = self$draws_zeta[i,,k])
          })
        }
        self$log_lik_draws <- t(ll)
      }
      return(self$log_lik_draws)
    } else {
      # Path for the file with the predictions
      ff_theta <- file.path(output_dir, "theta_ij.bin")
      ff_zeta <- file.path(output_dir, "zeta_ij.bin")
      if (!file.exists(ff_theta))
        stop("File with the predictions of theta_{ij} does not exist.")
      if (!file.exists(ff_zeta))
        stop("File with the predictions of zeta_{ij} does not exist.")
      n_zeros <- rowSums(Y == 0)
      n_trials <- rowSums(Y)
      seq_ind <- seq_len(n_pred)
      seq_samples <- seq.int(1, ndpost, by = nthin)
      ini <- proc.time()
      out <- parallel::mclapply(X = seq_samples, FUN = function(t) {
        # load parameters at iteration t
        thetas <- .load_bin_batch(fname = ff_theta, n = n_pred, d = self$d, k = t,
                                  m = 1L, arr = FALSE)
        zetas <- .load_bin_batch(fname = ff_zeta, n = n_pred,
                                 d = self$d, k = t, m = 1L, arr = FALSE)
        # Log-file
        write.table(x = data.frame(t = t), file = file.path(output_dir, logfile),
                    append = TRUE, row.names = FALSE, col.names = FALSE)
        # compute likelihood for each observed data
        sapply(seq_ind, function(i) {
          idx <- seq.int(i, n_pred * self$d, by = n_pred)
            return(log_pmf_zanim(x = Y[i, ], prob = thetas[idx], zeta = zetas[idx]))
          # if (n_zeros[i] < 18) {
          #   return(log_pmf_zanim(x = Y[i, ], prob = thetas[idx], zeta = zetas[idx]))
          # } else {
          #   return(.log_pmf_zanim_approx(x = Y[i, ], prob = thetas[idx],
          #                                zeta = zetas[idx], scale = n_trials[i],
          #                                mc = 5000L, nskip = 1000L))
          # }
        })
      }, mc.cores = ncores)
      self$elapsed_time_log_lik <- proc.time() - ini
      self$log_lik_draws <- do.call(rbind, out)
    }

  },
  DeleteForests = function() unlink(x = self$forests_dir)
))


#' Multinomial logistic BART
#' @export
MultinomialBART <- R6::R6Class(classname = "MultinomialBART", public = list(
  cpp_obj = NULL, cpp_module_name = character(),
  n_trials = integer(), n = integer(), d = integer(), p = integer(),
  ntrees = integer(), ndpost = integer(), nskip = integer(), forests_dir = character(),
  n_pred = integer(), ndpost_pred = integer(), shared_trees = logical(),
  elapsed_time = NULL,  elapsed_time_log_lik = NULL, avg_leaves = NULL,
  avg_depth = NULL, accept_rate = NULL, lpl = NULL, draws_theta = NULL,
  draws_phi = NULL, keep_draws = logical(), save_trees = logical(),
  varcount = NULL,
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
    if (!self$shared_trees) {
      if (!is.list(splitprobs)) splitprobs <- replicate(self$d, splitprobs, simplify = FALSE)
      alpha_sparse <- rep(alpha_sparse, self$d)
    }
    # Setup
    self$cpp_obj$SetMCMC(v0, ntrees, ndpost, nskip, numcut, power, base,
                         proposals_prob, as.integer(update_sigma), s2_0, w_ss,
                         splitprobs, as.integer(sparse), sparse_parms,
                         alpha_sparse, as.integer(alpha_random), xinfo, forests_dir,
                         keep_draws, save_trees)
  },
  RunMCMC = function() {
    ini <- proc.time()
    self$cpp_obj$RunMCMC()
    self$elapsed_time <- proc.time() - ini
    # Keep some tree diagnostics
    self$avg_leaves <- self$cpp_obj$avg_leaves / (self$ndpost)# + self$nskip
    self$avg_depth <- self$cpp_obj$avg_depth / (self$ndpost)
    self$accept_rate <- self$cpp_obj$accept_rate / (self$ndpost + self$nskip) / self$ntrees
    rownames(self$accept_rate) <- c("grow", "prune", "change")
    # Copy draws to R
    if (self$keep_draws) {
      self$draws_theta <- self$cpp_obj$draws
      self$draws_phi <- self$cpp_obj$draws_phi
      self$varcount <- self$cpp_obj$varcount_mcmc
    }
  },
  GetPredictions = function(X, ndpost = self$ndpost,
                            forests_dir = self$forests_dir,
                            output_dir = self$forests_dir,
                            verbose = TRUE) {
    self$cpp_obj$Predict(X, self$d, ndpost, self$ntrees, forests_dir, output_dir,
                         as.integer(verbose))
  },
  DeleteForests = function() {
    unlink(x = self$forests_dir)
  },
  GetPosteriorPredictive = function(in_sample = TRUE, X = NULL,
                                    forests_dir = self$forests_dir,
                                    ndpost = self$ndpost, ...) {
    if (in_sample) {
      # Compute the predictions
      ppd(self, ...)
    } else {
      stop("Not implemented yet")
    }
  },
  LogPredictiveLikelihood = function(in_sample = TRUE,
                                     Y = NULL, X = NULL,
                                     ndpost = self$ndpost,
                                     forests_dir = self$forests_dir,
                                     printevery = 100L, nthin = 1L) {
    if (is.null(Y)) stop("Please provide the count matrix, Y.")
    if (!in_sample) {
      cat("Computing predictions....")
      # Compute the predictions
      draws <- self$GetPredictions(X, ndpost, forests_dir)
      n_pred <- nrow(X)
    } else {
      draws <- self$draws_theta
      n_pred <- self$n
    }
    idx <- seq.int(1, ndpost, by = nthin)
    ndpost <- length(idx)
    lpl <- matrix(nrow = ndpost, ncol = n_pred)
    for (k in seq_len(ndpost)) {
      if (k %% printevery == 0L) cat(k, "\n")
      lpl[k, ] <- .dmultinomial(x = Y, prob = draws[, ,idx[k]])
    }
    #self$lpl <- lpl
    lpl
  }
))

#' Multinomial logistic normal BART
#' @export
MultinomialLNBART <- R6::R6Class(classname = "MultinomialLNBART", public = list(
  cpp_obj = NULL, cpp_module_name = character(),
  n_trials = integer(), n = integer(), d = integer(), p = integer(),
  ntrees = integer(), ndpost = integer(), nskip = integer(), forests_dir = character(),
  n_pred = integer(), ndpost_pred = integer(), shared_trees = logical(),
  Bt = matrix(),
  covariance_type = NULL, elapsed_time = NULL, elapsed_time_log_lik = NULL,
  avg_leaves = NULL, avg_depth = NULL, accept_rate = NULL, lpl = NULL, varcount = NULL,
  draws_theta = NULL, draws_abundance = NULL, draws_chol_Sigma_V = NULL,
  draws_phi = NULL, keep_draws = logical(), save_trees = logical(),
  initialize = function(Y, X) {
    # Call the C++ class in R
    ml <- Rcpp::Module(module = "multinomial_lognormal_bart", PACKAGE = "zanicc")
    self$cpp_obj <- new(ml$MultinomialLNBART, Y, X)
    self$cpp_module_name <- "multinomial_lognormal_bart"
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
    self$covariance_type <- covariance_type
    self$ntrees <- ntrees
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$forests_dir <- forests_dir
    self$keep_draws <- keep_draws
    if (!is.list(splitprobs)) splitprobs <- replicate(self$d, splitprobs, simplify = FALSE)
      alpha_sparse <- rep(alpha_sparse, self$d)

    B <- qr.Q(qr(stats::contr.sum(self$d)))
    self$Bt <- t(B)

    # Setup
    self$cpp_obj$SetMCMC(v0, ntrees,
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
                         as.integer(keep_draws), as.integer(save_trees))

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
      # self$draws_phi <- self$cpp_obj$draws_phi
    }
  },
  GetPosteriorPredictive = function(...) {
    ppd(self, ...)
  },
  LogPredictiveLikelihood = function(Y = NULL, X = NULL,
                                     in_sample = TRUE, conditional = TRUE,
                                     ndpost = self$ndpost, forests_dir = self$forests_dir,
                                     printevery = 100L, MC = 100L) {
    n <- nrow(Y)
    if (is.null(Y)) stop("Please provide the count matrix, Y.")
    if (!in_sample) {
      stop("Out-of-sample log-likelihood not implemented yet")
      # cat("Computing predictions....")
      # draws <- self$GetPredictions(X, ndpost, forests_dir)
    }
    if (conditional && in_sample) {
      lpl <- matrix(nrow = ndpost, ncol = n)
      for (k in seq_len(ndpost)) {
        if (k %% printevery == 0L) cat(k, "\n")
        lpl[k, ] <- .dmultinomial(x = Y, prob = self$draws_abundance[, ,k])
      }
    } else if (!conditional && in_sample) {
      # Monte Carlo approximation (this takes time....)
      lpl <- matrix(nrow = ndpost, ncol = n)
      for (k in seq_len(ndpost)) {
        if (k %% printevery == 0L) cat(k, "\n")
        vals <- sapply(seq_len(MC), function(m) {
          # Generate random effects
          Z <- matrix(data = stats::rnorm(self$d - 1), nrow = n, ncol = self$d - 1)
          U <- Z %*% (self$draws_chol_Sigma_V[,,k] %*% self$Bt)
          # Compute the probabilities
          probs <- self$draws_theta[, ,k] * exp(U)
          probs <- sweep(probs, 1, rowSums(probs), "/")
          .dmultinomial(x = Y, prob = probs)
        }, simplify = "array")
        # log-sum-exp
        maxlog <- apply(vals, 1, max)
        lpl[k, ] <- maxlog + log(rowMeans(exp(vals - maxlog)))
      }
    } else {
      stop("Out-of-sample log-likelihood not implemented yet")
    }
    return(lpl)
  },
  GetPredictions = function(X, ndpost = self$ndpost, forests_dir = self$forests_dir) {
    self$cpp_obj$Predict(X, self$d, ndpost, self$ntrees, forests_dir)
  },
  DeleteForests = function() {
    unlink(x = self$forests_dir)
  }
))


#' ZANIM-linear regression
#' @export
ZANIMRegression <- R6::R6Class(
  classname = "ZANIMRegression",
  public = list(
  cpp_obj = NULL, n_trials = integer(), n = integer(), d = integer(),
  p_theta = integer(), p_zeta = integer(),
  ndpost = integer(), nskip = integer(), nthin = integer(),
  n_pred = integer(), ndpost_pred = integer(),
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
                       sd_prior_beta_zeta = diag(1.0, self$p_zeta),
                       ndpost = 5000L, nskip = 5000L, nthin = 1L,
                       keep_draws = TRUE, keep_draws_coef = TRUE) {
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$nthin <- nthin
    self$keep_draws <- keep_draws
    self$keep_draws_coef <- keep_draws_coef
    self$cpp_obj$SetMCMC(sd_prior_beta_theta, sd_prior_beta_zeta, ndpost, nskip,
                         nthin)
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
  },
  ComputePredictions = function(X, ndpost = self$ndpost,
                                parameter = c("theta", "zeta"),
                                verbose = TRUE) {
    stop("not implemented yet")
    self$n_pred <- nrow(X)
    self$ndpost_pred <- ndpost
    parameter <- match.arg(parameter)
  },
  GetPosteriorPredictive = function(...) {
    ppd(self, ...)
  },
  LogPredictiveLikelihood = function(Y = NULL, ndpost = self$ndpost, printevery = 100L) {
    if (is.null(Y)) stop("Please provide the count matrix in argument {Y}")
    ll <- matrix(data = 0.0, nrow = self$n, ncol = ndpost)
    for (k in seq_len(ndpost)) {
      if (k %% printevery == 0L) cat(k, "\n")
      ll[, k] <- sapply(X = seq_len(self$n), FUN = function(i) {
        log_pmf_zanim(x = Y[i, ], prob = self$draws_theta[i,,k],
                      zeta = self$draws_zeta[i,,k])
      })
    }
    self$log_lik_draws <- t(ll)
    return(self$log_lik_draws)
  }
))


#' ZANIDM logistic regression
#' @export
ZANIDMRegression <- R6::R6Class(
  classname = "ZANIDMRegression",
  public = list(
  cpp_obj = NULL,  cpp_module_name = character(),
  n_trials = integer(), n = integer(), d = integer(),
  p_alpha = integer(), p_zeta = integer(),
  ndpost = integer(), nskip = integer(), nthin = integer(),
  n_pred = integer(), ndpost_pred = integer(),
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
                       sd_prior_beta_zeta = diag(1.0, self$p_zeta),
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
    self$cpp_obj$SetMCMC(sd_prior_beta_alpha, sd_prior_beta_zeta, ndpost, nskip,
                         nthin, keep_draws, save_draws, dir_draws)
  },
  RunMCMC = function() {
    ini <- proc.time()
    self$cpp_obj$RunMCMC()
    self$elapsed_time <- proc.time() - ini
    # Save draws
    if (self$keep_draws) {
      self$draws_abundance <- self$cpp_obj$draws_abundance
      self$draws_alpha <- self$cpp_obj$draws_alphas
      self$draws_theta <- sweep(x = self$cpp_obj$draws_alphas, MARGIN = c(1, 3),
                                STATS = apply(self$cpp_obj$draws_alphas, c(1, 3), sum),
                                FUN = "/")
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
  },
  ComputePredictions = function(X, ndpost = self$ndpost,
                                parameter = c("alpha", "zeta"),
                                verbose = TRUE) {
    stop("not implemented yet")
    self$n_pred <- nrow(X)
    self$ndpost_pred <- ndpost
    parameter <- match.arg(parameter)
  },
  GetPosteriorPredictive = function(...) {
    ppd(self, ...)
  },
  LogPredictiveLikelihood = function(ndpost = self$ndpost, parallel = FALSE,
                                     ncores = 4L) {

    if (!parallel) {
      ll <- matrix(data = 0.0, nrow = self$n, ncol = ndpost)
      for (k in seq_len(ndpost)) {
        if (k %% 100 == 0) cat(k, "\n")
        ll[, k] <- sapply(X = seq_len(self$n), FUN = function(i) {
          log_pmf_zanidm(x = Y[i, ], alpha = self$draws_alpha[i,,k],
                         zeta = self$draws_zeta[i,,k])
        })
      }
      self$log_lik_draws <- t(ll)
    } else {
      seq_n <- seq_len(self$n)
      ll <- parallel::mclapply(X = seq_len(ndpost), FUN = function(k) {
        sapply(X = seq_n, FUN = function(i) {
          log_pmf_zanidm(x = Y[i, ], alpha = self$draws_alpha[i,,k],
                         zeta = self$draws_zeta[i,,k])
        })
      }, mc.cores = ncores)
      self$log_lik_draws <- do.call(rbind, ll)
    }
    return(self$log_lik_draws)
  }
))

#' ZANIM logistic normal regression
#' @export
ZANIMLNRegression <- R6::R6Class(
  classname = "ZANIMLNRegression",
  public = list(
  cpp_obj = NULL, cpp_module_name = NULL,
  n_trials = integer(), n = integer(), d = integer(),
  p_theta = integer(), p_zeta = integer(),
  ndpost = integer(), nskip = integer(), nthin = integer(),
  n_pred = integer(), ndpost_pred = integer(),
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
                       sd_prior_beta_zeta = diag(1.0, self$p_zeta),
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
    self$cpp_obj$SetMCMC(sd_prior_beta_theta, sd_prior_beta_zeta, ndpost, nskip,
                         nthin,
                         B, cov_type,
                         a_sigma, b_sigma,
                         Psi_prior, nu_prior,
                         q_factors, sigma2_gamma,
                         a_psi, b_psi,
                         shape_lsphis, a1_gs, a2_gs, keep_draws)
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
  },
  ComputePredictions = function(X, ndpost = self$ndpost,
                                parameter = c("theta", "zeta"),
                                verbose = TRUE) {
    stop("not implemented yet")
  },
  GetPosteriorPredictive = function(...) {
    ppd(self, ...)
  },
  LogPredictiveLikelihood = function(Y) {
    lpl <- matrix(nrow = self$ndpost, ncol = self$n)
    for (k in seq_len(self$ndpost)) {
      if (k %% printevery == 0L) cat(t, "\n")
      lpl[k, ] <- .dmultinomial(x = Y, prob = self$draws_abundance[, ,k])
    }
    lpl
  }
))


#' DM-linear regression
#' @export
DMRegression <- R6::R6Class(
  classname = "DMRegression",
  public = list(
  cpp_obj = NULL, cpp_module_name = character(),
  n_trials = integer(), n = integer(), d = integer(),
  p = integer(), ndpost = integer(), nskip = integer(), nthin = integer(),
  n_pred = integer(), ndpost_pred = integer(),
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
      for (j in seq_len(self$d)) S[,,j] <- S_prior_betas
    }
    self$cpp_obj$SetMCMC(S, ndpost, nskip, nthin, as.integer(keep_draws),
                         as.integer(save_draws), dir_draws)
  },
  RunMCMC = function() {
    ini <- proc.time()
    self$cpp_obj$RunMCMC()
    self$elapsed_time <- proc.time() - ini
    # Save draws
    if (self$keep_draws) {
      self$draws_abundance <- self$cpp_obj$draws_abundance
      self$draws_alpha <- self$cpp_obj$draws_alphas
      self$draws_theta <- sweep(x = self$cpp_obj$draws_alphas, MARGIN = c(1, 3),
                                STATS = apply(self$cpp_obj$draws_alphas, c(1, 3), sum),
                                FUN = "/")
      # self$draws_phi <- self$cpp_obj$draws_phi
      if (self$keep_draws_coef) self$draws_betas <- self$cpp_obj$draws_betas
    }
  },
  PosterioMeanCoef = function() {
    if (self$keep_draws) apply(self$draws_betas, c(1, 2), mean)
  },
  GetPosteriorPredictive = function(...) {
    ppd(self, ...)
  },
  LogPredictiveLikelihood = function(Y, ndpost = self$ndpost, parallel = FALSE,
                                     ncores = 4L, printevery = 100L) {
    if (!parallel) {
      ll <- matrix(data = 0.0, nrow = self$n, ncol = ndpost)
      for (k in seq_len(ndpost)) {
        if (k %% printevery == 0L) cat(k, "\n")
        ll[, k] <- ddm(x = Y, alphas = self$draws_alpha[, , k], log = TRUE)
      }
      self$log_lik_draws <- t(ll)
    } else {
      ll <- parallel::mclapply(X = seq_len(ndpost), FUN = function(k) {
        ddm(x = Y, alphas = self$draws_alpha[, , k], log = TRUE)
      }, mc.cores = ncores)
      self$log_lik_draws <- do.call(rbind, ll)
    }
    return(self$log_lik_draws)
  },
  ComputePredictions = function(X, ndpost = self$ndpost) {
    stop("not implemented yet")
  }
))
