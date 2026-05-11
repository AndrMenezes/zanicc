.ppc_stat <- function(Y, Y_ppc, stat_fun) {
  t_obs <- stat_fun(Y)
  t_ppc <- apply(Y_ppc, 1, stat_fun)
  list(t_ppc = t_ppc, t_obs = t_obs)
}

#' Plot posterior-predictive checks
#' @param Y matrix with the observed count-compositional data.
#' @param Y_ppc array with the posterior predictive distribution.
#' @param object model object to compute the posterior predictive distribution, if
#' is `NULL`.
#' @param output Whether to return the output of the posterior-predictive
#' checks.
#'
#' @export
plot_ppc <- function(Y, Y_ppc = NULL, object = NULL, output = FALSE) {
  # Generate the posterior-predictive distribution
  if (!is.null(object) && is.null(Y_ppc)) {
    cat("Generating posterior-predictive distribution\n")
    Y_ppc <- ppd(object, relative = FALSE)
  }
  # Computing
  cat("Computing the posterior predictive checks")
  res_mdi <- .ppc_stat(Y = Y, Y_ppc = Y_ppc, stat_fun = mdi)
  res_zero <- .ppc_stat(Y = Y, Y_ppc = Y_ppc, stat_fun = function(Y) mean(Y == 0))
  res_zi <- .ppc_stat(Y = Y, Y_ppc = Y_ppc, stat_fun = zi_multinomial)
  res_entropy <- .ppc_stat(Y = sweep(Y, 1, rowSums(Y), "/"),
                           Y_ppc = .normalize_composition(Y_ppc),
                           stat_fun = shannon_entropy)

  # Keep user's graphs options
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  # Plotting
  par(mar = c(4, 4, 1, 1), mfrow = c(2, 2))
  plot(density(res_entropy$t_ppc), main = "Entropy", xlab = "", ylab = "",
       xlim = range(res_entropy$t_obs, res_entropy$t_ppc))
  abline(v = res_entropy$t_obs)
  plot(density(res_mdi$t_ppc), main = "MDI", xlab = "", ylab = "",
       xlim = range(res_mdi$t_obs, res_mdi$t_ppc))
  abline(v = res_mdi$t_obs)
  plot(density(res_zero$t_ppc), main = "Prop of zero", xlab = "", ylab = "",
       xlim = range(res_zero$t_obs, res_zero$t_ppc))
  abline(v = res_zero$t_obs)
  plot(density(res_zi$t_ppc), main = "ZI", xlab = "", ylab = "",
       xlim = range(res_zi$t_obs, res_zi$t_ppc))
  abline(v = res_zi$t_obs)

  if (output)
    return(list(mdi = res_mdi, entropy = res_entropy, prop_zero = res_zero, zi = res_zi))

  invisible()
}


.compute_quantiles <- function(Y_rep, probs) {
  qs_rep <- apply(Y_rep, 1, quantile, probs = probs, names = FALSE)
  q_med <- apply(qs_rep, 1, median)
  q_lo <- apply(qs_rep, 1, quantile, probs = 0.025)
  q_up <- apply(qs_rep, 1, quantile, probs = 0.975)
  cbind(q_med, q_lo, q_up)
}
#' Plot marginal QQ-plots using the posterior-predictive distribution
#' @export
plot_qqplots <- function(Y, Y_ppc = NULL, object = NULL, relative = FALSE,
                        output = FALSE, len_probs = 100L, mfrow = NULL) {
  # Generate the posterior-predictive distribution
  if (!is.null(object) && is.null(Y_ppc)) {
    cat("Generating posterior-predictive distribution\n")
    Y_ppc <- ppd(object, relative = FALSE)
  }
  if (relative) {
    Y_ppc <- .normalize_composition(Y_ppc)
    Y <- sweep(Y, 1, rowSums(Y), "/")
  }
  # Probabilities for the quantiles
  probs <- seq(0.0, 1.0, length.out = len_probs)
  # Keep user's graphs options
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))
  d <- ncol(Y)
  if (output) list_data <- vector(mode = "list", length = d)
  if (is.null(mfrow)) mfrow <- grDevices::n2mfrow(d)
  # Plotting
  par(mfrow = mfrow, mar = c(4, 4, 1, 1))
  for (j in seq_len(d)) {
    q_obs <- quantile(Y[, j], probs = probs, names = FALSE)
    q_teo <- .compute_quantiles(Y_rep = Y_ppc[,,j], probs = probs)
    ry <- c(min(q_teo[, 2L]), max(q_teo[, 3L]))
    rx <- range(q_obs)
    plot(q_teo[, 1L], q_obs, ylim = ry, xlim = rx,
         main = sprintf("category j=%i", j),
         xlab = "Theoretical quantiles", ylab = "Empirical quantiles")
    lines(q_teo[, 2L], q_obs, lty = "dashed")
    lines(q_teo[, 3L], q_obs, lty = "dashed")
    abline(0, 1, col = "grey60", lty = "dashed")
    if (output) list_data[[j]] <- cbind(q_obs, q_teo)
  }
  if (output) return(list_data)
  invisible()
}
