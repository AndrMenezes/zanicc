#' ZANIM-BART
#' @export
ZANIMBART <- R6::R6Class(classname = "ZANIMBART", public = list(
  cpp_obj = NULL,
  Y_pp = NULL,
  n_trials = integer(), n = integer(), d = integer(), p_theta = integer(),
  p_zeta = integer(), ntrees_zeta = integer(), ntrees_theta = integer(),
  ndpost = integer(), niter = integer(), nskip = integer(), path = character(),
  n_pred = integer(), n_samples_pred = integer(), link_zeta = character(),
  elapsed_time = NULL,  elapsed_time_log_lik = NULL,
  avg_leaves_theta = NULL, avg_leaves_zeta = NULL, accept_rate_theta = NULL,
  accept_rate_zeta = NULL, draws_theta = NULL, draws_zeta = NULL,
  draws_abundance = NULL, draws_phi = NULL, y_rep_draws = NULL,
  keep_draws = logical(), log_lik_draws = NULL, varcount_theta = NULL,
  varcount_zeta = NULL, sigma_theta_hyperprior = NULL,
  initialize = function(Y, X_theta, X_zeta) {
    ml <- Rcpp::Module(module = "zanim_bart_probit", PACKAGE = "zanicc")
    self$cpp_obj <- new(ml$ZANIMBARTProbit, Y, X_theta, X_zeta)

    self$n <- nrow(Y)
    self$d <- ncol(Y)
    self$p_theta <- ncol(X_theta)
    self$p_zeta <- ncol(X_zeta)
    self$n_trials <- rowSums(Y)
  },
  SetupMCMC = function(v0_theta = 3.5 / sqrt(2),
                       v0_zeta = 3.0,
                       ntrees_theta = 50L, ntrees_zeta = 20L,
                       ndpost = 1000L, nskip = 1000L,
                       numcut = 100L, power = 2.0, base = 0.95,
                       proposals_prob = c(0.25, 0.25, 0.50),
                       update_sigma_theta = FALSE, s0_2_theta = 1 / ntrees_theta,
                       w_ss = 1.0,
                       splitprobs_zi = rep(1 / self$p_zeta, self$p_zeta),
                       splitprobs_mult = rep(1 / self$p_theta, self$p_theta),
                       sparse = c(FALSE, FALSE),
                       sparse_parms = c(self$p_zeta, 0.5, 1.0,
                                        self$p_theta, 0.5, 1.0),
                       alpha_sparse = c(1.0, 1.0), alpha_random = c(FALSE, FALSE),
                       xinfo = matrix(), path = tempdir(), keep_draws = FALSE) {
    self$ntrees_theta <- ntrees_theta
    self$ntrees_zeta <- ntrees_zeta
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$niter <- nskip + ndpost
    self$path <- path
    self$keep_draws <- keep_draws
    alpha_sparse_mult <- alpha_sparse[2L]
    if (!is.list(splitprobs_mult))
      splitprobs_mult <- replicate(self$d, splitprobs_mult, simplify = FALSE)
    alpha_sparse_mult <- rep(alpha_sparse[2L], self$d)
    self$cpp_obj$SetMCMC(v0_theta, v0_zeta, ntrees_theta, ntrees_zeta, ndpost, nskip,
                         numcut, power, base, proposals_prob,
                         as.integer(update_sigma_theta), s0_2_theta, w_ss,
                         splitprobs_zi, splitprobs_mult,
                         as.integer(sparse[1L]), as.integer(sparse[2L]),
                         sparse_parms[1L:3L], sparse_parms[4L:6L],
                         rep(alpha_sparse[1L], self$d), alpha_sparse_mult,
                         as.integer(alpha_random[1L]), as.integer(alpha_random[2L]),
                         xinfo, path, keep_draws)
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
      self$draws_zeta <-  stats::pnorm(self$cpp_obj$draws_zeta)
      self$draws_phi <- self$cpp_obj$draws_phi
      self$varcount_theta <- self$cpp_obj$varcount_mcmc_theta
      self$varcount_zeta <- self$cpp_obj$varcount_mcmc_zeta
    }
  },
  GetVarCount = function(parameter = c("theta", "zeta"), n_samples = self$npost) {
    parameter <- match.arg(parameter)
    if (!is.null(self$elapsed_time)) {
      vc <- switch(parameter,
                  "theta" = self$cpp_obj$varcount_mcmc_theta,
                  "zeta" = self$cpp_obj$varcount_mcmc_zeta)
    } else {
      ntrees <- ifelse(parameter == "theta", self$ntrees_theta, self$ntrees_zeta)
      vc <- self$cpp_obj$GetVarCount(n_samples, ntrees, parameter, self$path)
    }
    vc
  },
  ComputePredictions = function(X, n_samples = self$ndpost, path = self$path,
                                path_out = self$path, parameter = c("theta", "zeta"),
                                verbose = TRUE) {
    self$n_pred <- nrow(X)
    self$n_samples_pred <- n_samples
    parameter <- match.arg(parameter)
    switch(parameter,
           "theta" = self$cpp_obj$ComputePredictProb(X, n_samples,
                                                     self$ntrees_theta,
                                                     path_out, path,
                                                     as.integer(verbose)),
           "zeta" = self$cpp_obj$ComputePredictProbZero(X, n_samples,
                                                        self$ntrees_zeta,
                                                        path_out, path,
                                                        as.integer(verbose)))
  },
  LoadPredictions = function(n_samples = self$n_samples_pred,
                             parameter = c("theta", "zeta"),
                             path = self$path, n_pred = self$n_pred) {
    parameter <- match.arg(parameter)
    ff <- paste0(parameter, "_ij.bin")
    if (!file.exists(file.path(path, ff))) stop("File with the predictions does not exist.")
    .load_bin(fname = file.path(path, ff), n = n_pred, d = self$d, m = n_samples)
  },
  DeleteForests = function() {
    unlink(x = self$path)
  },
  GetPosteriorPredictive = function(in_sample = TRUE,
                                    n_trials = self$n_trials,
                                    batch_size = 100L,
                                    n_samples = self$n_samples_pred,
                                    path = self$path, n_pred = self$n_pred) {
    if (in_sample) {
      if (!self$keep_draws)
        stop("Draws are not saved in the class. Run MCMC again with {keep_draws=TRUE}")

      # Store posterior-predictive draws
      y_pp_draws <- array(0L, dim = c(self$ndpost, self$n, self$d))
      for (k in seq_len(self$ndpost)) {
        if (k %% 100 == 0) cat(k, "\n")
        y_pp_draws[k,,] <- rzanim_vec(n = self$n, sizes = self$n_trials,
                                      probs = self$draws_theta[,,k],
                                      zetas = self$draws_zeta[,,k], d = self$d)
      }
      self$y_rep_draws <- y_pp_draws
      return(self$y_rep_draws)
    } else {
      # Path for the file with the predictions
      ff_theta <- file.path(path, "theta_ij.bin")
      ff_zeta <- file.path(path, "zeta_ij.bin")

      if (!file.exists(ff_theta))
        stop("File with the predictions of theta_{ij} does not exist.")
      if (!file.exists(ff_zeta))
        stop("File with the predictions of zeta_{ij} does not exist.")

      # Create a vector to load the predictions by batch
      look_head <- seq.int(from = 1L, to = n_samples, by = batch_size)
      y_rep <- array(data = NA_real_, dim = c(n_pred, self$d, n_samples))
      # Load by batch and generate the posterior-predictive
      for (i in seq_len(length(look_head))) {
        shift <- look_head[i] - 1L
        # Load \theta_ij
        thetas <- .load_bin_batch(fname = ff_theta, n = n_pred, d = self$d,
                                  k = look_head[i], m = batch_size)
        # Load \zeta_{ij}
        zetas <- .load_bin_batch(fname = ff_zeta, n = n_pred, d = self$d,
                                 k = look_head[i], m = batch_size)
        for (t in seq_len(batch_size)) {
          cat(shift + t, "\n")
          y_rep[,,shift + t] <- rzanim_vec(n = n_pred, sizes = n_trials,
                                           probs = thetas[, , t],
                                           zetas = zetas[, , t])
        }
      }
      self$y_rep_draws <- y_rep
      return(self$y_rep_draws)
    }
  },
  LogPredictiveLikelihood = function(in_sample = TRUE,
                                     Y = NULL, n_samples = self$ndpost,
                                     path = self$path, n_pred = self$n_pred,
                                     nthin = 1L, parallel = TRUE, ncores = 10L,
                                     logfile = tempfile()) {
    if (is.null(Y)) stop("Please, provide the data matrix, Y.")

    if (in_sample) {
      if (!self$keep_draws) stop("Posterior draws are not saved in the class.")

      if (parallel) {
        seq_ind <- seq_len(self$n)
        ini <- proc.time()
        out <- parallel::mclapply(X = seq_len(n_samples), FUN = function(k) {
          sapply(seq_ind, function(i) {
            log_pmf_zanim(x = Y[i, ], prob = self$draws_theta[i,,k],
                          zeta = self$draws_zeta[i,,k])
          })
        }, mc.cores = ncores)
        self$elapsed_time_log_lik <- proc.time() - ini
        self$log_lik_draws <- do.call(rbind, out)
      } else {
        ll <- matrix(data = 0.0, nrow = self$n, ncol = n_samples)
        for (k in seq_len(n_samples)) {
          if (k %% 100L == 0) cat(k, "\n")
          ll[, k] <- sapply(X = seq_len(self$n), FUN = function(i) {
            log_pmf_zanim(x = Y[i, ],
                          prob = self$draws_theta[i,,k],
                          zeta = self$draws_zeta[i,,k])
          })
        }
        self$log_lik_draws <- t(ll)
      }
    } else {
      # Path for the file with the predictions
      ff_theta <- file.path(path, "theta_ij.bin")
      ff_zeta <- file.path(path, "zeta_ij.bin")

      if (!file.exists(ff_theta))
        stop("File with the predictions of theta_{ij} does not exist.")
      if (!file.exists(ff_zeta))
        stop("File with the predictions of zeta_{ij} does not exist.")

      n_zeros <- rowSums(Y == 0)
      n_trials <- rowSums(Y)

      seq_ind <- seq_len(n_pred)
      seq_samples <- seq.int(1, n_samples, by = nthin)
      ini <- proc.time()
      out <- parallel::mclapply(X = seq_samples, FUN = function(t) {
        # load parameters at iteration t
        thetas <- .load_bin_batch(fname = ff_theta, n = n_pred, d = self$d, k = t,
                                  m = 1L, arr = FALSE)
        zetas <- .load_bin_batch(fname = ff_zeta, n = n_pred,
                                 d = self$d, k = t, m = 1L, arr = FALSE)
        # Log-file
        write.table(x = data.frame(t = t), file = file.path(path, logfile),
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

#' ZANIM-lognormal BART
#' @export
ZANIMLogNormalBART <- R6::R6Class(classname = "ZANIMLogNormalBART", public = list(
  cpp_obj = NULL,
  Y_pp = NULL,
  n_trials = integer(), n = integer(), d = integer(), p_theta = integer(),
  p_zeta = integer(), ntrees_zeta = integer(), ntrees_theta = integer(),
  ndpost = integer(), niter = integer(), nskip = integer(), path = character(),
  n_pred = integer(), n_samples_pred = integer(), link_zeta = character(),
  elapsed_time = NULL,  elapsed_time_log_lik = NULL,
  avg_leaves_theta = NULL, avg_leaves_zeta = NULL, accept_rate_theta = NULL,
  accept_rate_zeta = NULL, y_rep_draws = NULL, log_lik_draws = NULL,
  draws_theta = NULL, draws_zeta = NULL, draws_phi = NULL, draws_Sigma_U = NULL,
  draws_abundance = NULL, varcount_theta = NULL, varcount_zeta = NULL,
  sigma_theta_hyperprior = NULL,
  keep_draws = logical(),
  initialize = function(Y, X_theta, X_zeta) {
    ml <- Rcpp::Module(module = "zanim_lognormal_bart", PACKAGE = "zanicc")
    self$cpp_obj <- new(ml$ZANIMLogNormalBART, Y, X_theta, X_zeta)
    self$n <- nrow(Y)
    self$d <- ncol(Y)
    self$p_theta <- ncol(X_theta)
    self$p_zeta <- ncol(X_zeta)
    self$n_trials <- rowSums(Y)
  },
  SetupMCMC = function(v0_theta = 3.5 / sqrt(2),
                       k_zeta = 3.0,
                       ntrees_theta = 50L, ntrees_zeta = 20L,
                       ndpost = 1000L, nskip = 1000L,
                       covariance_type = c(0L, 1L, 2L, 3L),
                       nu_prior = self$d,
                       Psi_prior = diag(self$d, self$d - 1),
                       a_sigma = 1.0, b_sigma = 1.0,
                       q_factors = .ledermann(self$d - 1L), sigma2_gamma = 1.0,
                       a_psi = 2.5, b_psi = 1.0,
                       shape_lsphis = 3.0,
                       a1_gs = 2.1, a2_gs = 3.1,
                       numcut = 100L, power = 2.0, base = 0.95,
                       proposals_prob = c(0.25, 0.25, 0.50),
                       update_sigma_theta = FALSE, s0_2_theta = 1 / ntrees_theta,
                       w_ss = 1.0,
                       splitprobs_zi = rep(1 / self$p_zeta, self$p_zeta),
                       splitprobs_mult = rep(1 / self$p_theta, self$p_theta),
                       sparse = c(FALSE, FALSE),
                       sparse_parms = c(self$p_zeta, 0.5, 1.0,
                                        self$p_theta, 0.5, 1.0),
                       alpha_sparse = c(1.0, 1.0), alpha_random = c(FALSE, FALSE),
                       xinfo = matrix(), path = tempdir(), keep_draws = TRUE) {
    self$ntrees_theta <- ntrees_theta
    self$ntrees_zeta <- ntrees_zeta
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$niter <- nskip + ndpost
    self$path <- path
    self$keep_draws <- keep_draws
    alpha_sparse_mult <- alpha_sparse[2L]
    if (!is.list(splitprobs_mult))
      splitprobs_mult <- replicate(self$d, splitprobs_mult, simplify = FALSE)
    alpha_sparse_mult <- rep(alpha_sparse[2L], self$d)
    B <- qr.Q(qr(stats::contr.sum(self$d)))
    self$cpp_obj$SetMCMC(v0_theta, k_zeta, ntrees_theta, ntrees_zeta,
                         B, as.integer(covariance_type),
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
                         xinfo, path, as.integer(keep_draws))
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
      self$draws_phi <- self$cpp_obj$draws_phi
      self$draws_Sigma_U <- self$cpp_obj$draws_Sigma_U
      self$varcount_theta <- self$cpp_obj$varcount_mcmc_theta
      self$varcount_zeta <- self$cpp_obj$varcount_mcmc_zeta
    }
  },
  ComputePredictions = function(X, n_samples = self$ndpost, path = self$path,
                                path_out = self$path,
                                parameter = c("theta", "zeta"),
                                verbose = TRUE) {
    self$n_pred <- nrow(X)
    self$n_samples_pred <- n_samples
    parameter <- match.arg(parameter)
    switch(parameter,
           "theta" = self$cpp_obj$ComputePredictProb(X, n_samples,
                                                     self$ntrees_theta,
                                                     path_out, path,
                                                     as.integer(verbose)),
           "zeta" = self$cpp_obj$ComputePredictProbZero(X, n_samples,
                                                        self$ntrees_zeta,
                                                        path_out, path,
                                                        as.integer(verbose)))
  },
  LoadPredictions = function(path = self$path, parameter = c("theta", "zeta")) {
    parameter <- match.arg(parameter)
    if (!self$keep_draws) {
      ff <- paste0(parameter, "_ij.bin")
      if (!file.exists(file.path(path, ff)))
        stop("File with the predictions does not exist.")
      return(.load_bin(fname = file.path(path, ff), n = self$n_pred, d = self$d,
                       m = self$n_samples_pred))
    } else {
      if (parameter == "theta") return(self$draws_theta)
      else return(self$draws_zeta)
    }
  },
  GetPosteriorPredictive = function(in_sample = TRUE,
                                    n_trials = self$n_trials,
                                    batch_size = 100L,
                                    n_samples = self$n_samples_pred,
                                    path = self$path, n_pred = self$n_pred) {
    if (in_sample) {
      if (!self$keep_draws)
        stop("Draws are not saved in the class. Run MCMC again with {keep_draws=TRUE}.")
      # Store posterior-predictive draws
      y_pp_draws <- array(0L, dim = c(self$ndpost, self$n, self$d))
      for (k in seq_len(self$ndpost)) {
        if (k %% 100 == 0) cat(k, "\n")
        y_pp_draws[k,,] <- rzanim_vec(n = self$n,
                                      sizes = self$n_trials,
                                      probs = self$draws_theta[,,k],
                                      zetas = self$draws_zeta[,,k],
                                      d = self$d)
      }
      self$y_rep_draws <- y_pp_draws
      return(self$y_rep_draws)
    } else {
      # Path for the file with the predictions
      ff_theta <- file.path(path, "theta_ij.bin")
      ff_zeta <- file.path(path, "zeta_ij.bin")

      if (!file.exists(ff_theta))
        stop("File with the predictions of theta_{ij} does not exist.")
      if (!file.exists(ff_zeta))
        stop("File with the predictions of zeta_{ij} does not exist.")

      # Create a vector to load the predictions by batch
      look_head <- seq.int(from = 1L, to = n_samples, by = batch_size)
      y_rep <- array(data = NA_real_, dim = c(n_pred, self$d, n_samples))
      # Load by batch and generate the posterior-predictive
      for (i in seq_len(length(look_head))) {
        shift <- look_head[i] - 1L
        # Load \theta_ij
        thetas <- .load_bin_batch(fname = ff_theta, n = n_pred, d = self$d,
                                  k = look_head[i], m = batch_size)
        # Load \zeta_{ij}
        zetas <- .load_bin_batch(fname = ff_zeta, n = n_pred, d = self$d,
                                 k = look_head[i], m = batch_size)
        for (t in seq_len(batch_size)) {
          cat(shift + t, "\n")
          y_rep[,,shift + t] <- rzanim_vec(n = n_pred, sizes = n_trials,
                                           probs = thetas[, , t],
                                           zetas = zetas[, , t])
        }
      }
      self$y_rep_draws <- y_rep
      return(self$y_rep_draws)
    }
  },
  LogPredictiveLikelihood = function(in_sample = TRUE,
                                     Y = NULL, n_samples = self$ndpost,
                                     path = self$path, n_pred = self$n_pred,
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
        out <- parallel::mclapply(X = seq_len(n_samples), FUN = function(k) {
          sapply(seq_ind, function(i) {
            log_pmf_zanim(x = Y[i, ], prob = self$draws_theta[i,,k],
                          zeta = self$draws_zeta[i,,k])
          })
        }, mc.cores = ncores)
        self$elapsed_time_log_lik <- proc.time() - ini
        self$log_lik_draws <- do.call(rbind, out)
      } else {
        ll <- matrix(data = 0.0, nrow = self$n, ncol = n_samples)
        for (k in seq_len(n_samples)) {
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
      ff_theta <- file.path(path, "theta_ij.bin")
      ff_zeta <- file.path(path, "zeta_ij.bin")
      if (!file.exists(ff_theta))
        stop("File with the predictions of theta_{ij} does not exist.")
      if (!file.exists(ff_zeta))
        stop("File with the predictions of zeta_{ij} does not exist.")
      n_zeros <- rowSums(Y == 0)
      n_trials <- rowSums(Y)
      seq_ind <- seq_len(n_pred)
      seq_samples <- seq.int(1, n_samples, by = nthin)
      ini <- proc.time()
      out <- parallel::mclapply(X = seq_samples, FUN = function(t) {
        # load parameters at iteration t
        thetas <- .load_bin_batch(fname = ff_theta, n = n_pred, d = self$d, k = t,
                                  m = 1L, arr = FALSE)
        zetas <- .load_bin_batch(fname = ff_zeta, n = n_pred,
                                 d = self$d, k = t, m = 1L, arr = FALSE)
        # Log-file
        write.table(x = data.frame(t = t), file = file.path(path, logfile),
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
  DeleteForests = function() unlink(x = self$path)
))


#' Multinomial-BART
#' @export
MultinomialBART <- R6::R6Class(classname = "MultinomialBART", public = list(
  cpp_obj = NULL,
  Y_pp = NULL, Y = NULL, X = NULL,
  n_trials = integer(), n = integer(), d = integer(), p = integer(),
  ntrees = integer(), ndpost = integer(), nskip = integer(), path = character(),
  n_pred = integer(), n_samples_pred = integer(),
  elapsed_time = NULL,  elapsed_time_log_lik = NULL, avg_leaves = NULL,
  avg_depth = NULL, accept_rate = NULL, lpl = NULL, draws_theta = NULL,
  draws_phi = NULL,
  initialize = function(Y, X) {
    # Call the C++ class in R
    ml <- Rcpp::Module(module = "multinomial_bart", PACKAGE = "zanicc")
    self$cpp_obj <- new(ml$MultinomialBART, Y, X)
    # self$Y <- Y
    # self$X <- X
    self$n <- nrow(Y)
    self$d <- ncol(Y)
    self$p <- ncol(X)
    self$n_trials <- rowSums(Y)
  },
  SetupMCMC = function(v0 = 3.5 / sqrt(2), ntrees = 20L, ndpost = 1000L,
                       nskip = 1000L, numcut = 100L,
                       power = 2.0, base = 0.95,
                       proposals_prob = c(0.25, 0.25, 0.50),
                       update_sigma = FALSE, s2_0 = 1 / ntrees, w_ss = 1.0,
                       splitprobs = rep(1 / ncol(X), ncol(X)), sparse = FALSE,
                       sparse_parms = c(ncol(X), 0.5, 1.0), alpha_sparse = 1.0,
                       alpha_random = FALSE, xinfo = matrix(), path = tempdir()) {
    self$ntrees <- ntrees
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$path <- path
    if (!is.list(splitprobs)) splitprobs <- replicate(self$d, splitprobs, simplify = FALSE)
    alpha_sparse <- rep(alpha_sparse, self$d)
    self$cpp_obj$SetMCMC(v0, ntrees, ndpost, nskip, numcut, power, base,
                         proposals_prob, as.integer(update_sigma), s2_0, w_ss,
                         splitprobs, as.integer(sparse), sparse_parms,
                         alpha_sparse, as.integer(alpha_random), xinfo, path)
  },
  RunMCMC = function() {
    ini <- proc.time()
    self$cpp_obj$RunMCMC()
    self$elapsed_time <- proc.time() - ini
    # Keep some tree diagnostics
    self$avg_leaves <- self$cpp_obj$avg_leaves / (self$ndpost + self$nskip)
    self$avg_depth <- self$cpp_obj$avg_depth / (self$ndpost + self$nskip)
    self$accept_rate <- self$cpp_obj$accept_rate / (self$ndpost + self$nskip) / self$ntrees
    rownames(self$accept_rate) <- c("grow", "prune", "change")
    self$draws_theta <- self$cpp_obj$draws
    self$draws_phi <- self$cpp_obj$draws_phi
  },
  GetPredictions = function(X, n_samples = self$ndpost, path = self$path) {
    self$cpp_obj$Predict(X, self$d, n_samples, self$ntrees, path)
  },
  DeleteForests = function() {
    unlink(x = self$path)
  },
  GetPosteriorPredictive = function(in_sample = TRUE,
                                    X = NULL, n_trials = self$n_trials,
                                    n_samples = self$ndpost, path = self$path) {

    if (!in_sample) {
      # Compute the predictions
      draws <- self$GetPredictions(X, n_samples, path)
      n_pred <- nrow(X)
    } else {
      draws <- self$draws_theta
      n_pred <- self$n
    }
    y_rep <- array(data = NA_real_, dim = c(n_samples, n_pred, self$d))
    for (t in seq_len(n_samples)) {
      for (i in seq_len(n_pred)) {
        y_rep[t,i,] <- stats::rmultinom(n = 1L, size = n_trials, prob = draws[i, , t])
      }
    }
    y_rep
  },
  LogPredictiveLikelihood = function(in_sample = TRUE,
                                     Y = NULL, X = NULL,
                                     n_samples = self$ndpost, path = self$path,
                                     printevery = 100L, nthin = 1L) {
    if (is.null(Y)) stop("Please provide the count matrix, Y.")
    if (!in_sample) {
      cat("Computing predictions....")
      # Compute the predictions
      draws <- self$GetPredictions(X, n_samples, path)
      n_pred <- nrow(X)
    } else {
      draws <- self$draws_theta
      n_pred <- self$n
    }
    idx <- seq.int(1, n_samples, by = nthin)
    ndpost <- length(idx)
    lpl <- matrix(nrow = ndpost, ncol = n_pred)
    for (t in seq_len(ndpost)) {
      if (t %% printevery == 0L) cat(t, "\n")
      lpl[t, ] <- .dmultinomial(x = Y, prob = draws[, ,idx[t]])
    }
    #self$lpl <- lpl
    lpl
  }
))


#' ZANIM-linear regression
#' @export
ZANIMLinearRegression <- R6::R6Class(
  classname = "ZANIMLinearRegression",
  public = list(
  cpp_obj = NULL,
  Y = NULL,
  n_trials = integer(), n = integer(), d = integer(),
  p_theta = integer(), p_zeta = integer(),
  ndpost = integer(), nskip = integer(), nthin = integer(),
  n_pred = integer(), n_samples_pred = integer(),
  draws_theta = NULL, draws_zeta = NULL, draws_phi = NULL,
  draws_abundance = NULL, draws_betas_theta = NULL, draws_betas_zeta = NULL,
  y_rep_draws = NULL, log_lik_draws = NULL, elapsed_time = NULL,
  keep_draws = logical(),
  initialize = function(Y, X_theta, X_zeta) {
    ml <- Rcpp::Module(module = "zanim_linear_reg", PACKAGE = "zanicc")
    self$cpp_obj <- new(ml$ZANIMLinearReg, Y, X_theta, X_zeta)
    self$Y <- Y
    self$n <- nrow(Y)
    self$d <- ncol(Y)
    self$p_theta <- ncol(X_theta)
    self$p_zeta <- ncol(X_zeta)
    self$n_trials <- rowSums(Y)
  },
  SetupMCMC = function(sd_prior_beta_theta = rep(1.0, self$p_theta),
                       sd_prior_beta_zeta = diag(10, self$p_zeta),
                       ndpost = 5000L, nskip = 5000L, nthin = 1L,
                       keep_draws = TRUE) {
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$nthin <- nthin
    self$keep_draws <- keep_draws
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
      self$draws_phi <- self$cpp_obj$draws_phi
      self$draws_betas_theta <- self$cpp_obj$draws_betas_theta
      self$draws_betas_zeta <- self$cpp_obj$draws_betas_zeta
    }
  },
  PosterioMeanCoef = function(parameter = c("theta", "zeta")) {
    parameter <- match.arg(parameter)
    switch(parameter,
      "zeta" = apply(self$draws_betas_zeta, c(1, 2), mean),
      "theta" = apply(self$draws_betas_theta, c(1, 2), mean)
    )
  },
  ComputePredictions = function(X, n_samples = self$ndpost,
                                parameter = c("theta", "zeta"),
                                verbose = TRUE) {
    stop("not implemented yet")
    self$n_pred <- nrow(X)
    self$n_samples_pred <- n_samples
    parameter <- match.arg(parameter)
  },
  GetPosteriorPredictive = function() {
    y_pp_draws <- array(0L, dim = c(self$ndpost, self$n, self$d))
    for (k in seq_len(self$ndpost)) {
      if (k %% 100 == 0) cat(k, "\n")
      y_pp_draws[k,,] <- rzanim_vec(n = self$n, sizes = self$n_trials,
                                    probs = self$draws_theta[,,k],
                                    zetas = self$draws_zeta[,,k], d = self$d)
    }
    self$y_rep_draws <- y_pp_draws
    return(self$y_rep_draws)
  },
  LogPredictiveLikelihood = function(n_samples = self$ndpost) {
    ll <- matrix(data = 0.0, nrow = self$n, ncol = n_samples)
    for (k in seq_len(n_samples)) {
      if (k %% 100 == 0) cat(k, "\n")
      ll[, k] <- sapply(X = seq_len(self$n), FUN = function(i) {
        log_pmf_zanim(x = self$Y[i, ], prob = self$draws_theta[i,,k],
                      zeta = self$draws_zeta[i,,k])
      })
    }
    self$log_lik_draws <- t(ll)
    return(self$log_lik_draws)
  }
))


#' ZANIDM-linear regression
#' @export
ZANIDMLinearRegression <- R6::R6Class(
  classname = "ZANIDMLinearRegression",
  public = list(
  cpp_obj = NULL,
  Y = NULL,
  n_trials = integer(), n = integer(), d = integer(),
  p_alpha = integer(), p_zeta = integer(),
  ndpost = integer(), nskip = integer(), nthin = integer(),
  n_pred = integer(), n_samples_pred = integer(),
  draws_alpha = NULL, draws_zeta = NULL, draws_phi = NULL, draws_theta = NULL,
  draws_abundance = NULL, draws_betas_alpha = NULL, draws_betas_zeta = NULL,
  y_rep_draws = NULL, log_lik_draws = NULL, elapsed_time = NULL,
  keep_draws = logical(),
  initialize = function(Y, X_alpha, X_zeta) {
    ml <- Rcpp::Module(module = "zanidm_linear_reg", PACKAGE = "zanicc")
    self$cpp_obj <- new(ml$ZANIDMLinearReg, Y, X_alpha, X_zeta)
    self$Y <- Y
    self$n <- nrow(Y)
    self$d <- ncol(Y)
    self$p_alpha <- ncol(X_alpha)
    self$p_zeta <- ncol(X_zeta)
    self$n_trials <- rowSums(Y)
  },
  SetupMCMC = function(sd_prior_beta_alpha = rep(1.0, self$p_alpha),
                       sd_prior_beta_zeta = diag(10, self$p_zeta),
                       ndpost = 5000L, nskip = 5000L, nthin = 1L,
                       keep_draws = TRUE) {
    self$ndpost <- ndpost
    self$nskip <- nskip
    self$nthin <- nthin
    self$keep_draws <- keep_draws
    self$cpp_obj$SetMCMC(sd_prior_beta_alpha, sd_prior_beta_zeta, ndpost, nskip,
                         nthin)
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
      self$draws_phi <- self$cpp_obj$draws_phi
      self$draws_betas_alpha <- self$cpp_obj$draws_betas_alpha
      self$draws_betas_zeta <- self$cpp_obj$draws_betas_zeta
    }
  },
  PosterioMeanCoef = function(parameter = c("alpha", "zeta")) {
    parameter <- match.arg(parameter)
    switch(parameter,
      "zeta" = apply(self$draws_betas_zeta, c(1, 2), mean),
      "alpha" = apply(self$draws_betas_alpha, c(1, 2), mean)
    )
  },
  ComputePredictions = function(X, n_samples = self$ndpost,
                                parameter = c("alpha", "zeta"),
                                verbose = TRUE) {
    stop("not implemented yet")
    self$n_pred <- nrow(X)
    self$n_samples_pred <- n_samples
    parameter <- match.arg(parameter)
  },
  GetPosteriorPredictive = function() {
    y_pp_draws <- array(0L, dim = c(self$ndpost, self$n, self$d))
    for (k in seq_len(self$ndpost)) {
      if (k %% 100 == 0) cat(k, "\n")
      y_pp_draws[k,,] <- sapply(seq_len(self$n), function(i) {
        z <- 1L - stats::rbinom(n = self$d, size = 1L,
                                prob = self$draws_zeta[i, , k])
        g <- stats::rgamma(n = self$d, shape = self$draws_alpha[i, , k], rate = 1.0)
        if (all(z == 0)) return(rep(0, self$d))
        else if (sum(z) == d - 1L) {
          x <- rep(0L, self$d)
          x[!z] <- self$n_trials[i]
          return(x)
        } else {
          return(stats::rmultinom(n = 1L, size = self$n_trials[i], prob = z * g))
        }
      })
    }
    self$y_rep_draws <- y_pp_draws
    return(self$y_rep_draws)
  },
  LogPredictiveLikelihood = function(n_samples = self$ndpost, parallel = FALSE,
                                     ncores = 4L) {

    if (!parallel) {
      ll <- matrix(data = 0.0, nrow = self$n, ncol = n_samples)
      for (k in seq_len(n_samples)) {
        if (k %% 100 == 0) cat(k, "\n")
        ll[, k] <- sapply(X = seq_len(self$n), FUN = function(i) {
          log_pmf_zanidm(x = self$Y[i, ], alpha = self$draws_alpha[i,,k],
                         zeta = self$draws_zeta[i,,k])
        })
      }
      self$log_lik_draws <- t(ll)
    } else {
      seq_n <- seq_len(self$n)
      ll <- parallel::mclapply(X = seq_len(n_samples), FUN = function(k) {
        sapply(X = seq_n, FUN = function(i) {
          log_pmf_zanidm(x = self$Y[i, ], alpha = self$draws_alpha[i,,k],
                         zeta = self$draws_zeta[i,,k])
        })
      }, mc.cores = ncores)
      self$log_lik_draws <- do.call(rbind, ll)
    }
    return(self$log_lik_draws)
  }
))
