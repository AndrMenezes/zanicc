#' @name ppc
#'
#' @title Posterior predictive check for count-compositional models
#'
#' @description
#' Compute and visualise posterior predictive checks for count-compositional models.
#' The function `stat_ppc()` evaluates user-defined diagnostic statistics on both
#' the observed data and the posterior predictive distribution, while `plot_ppc()`
#' summarises four built-in diagnostics through density plots.
#'
#' @param Y A matrix of observed count-compositional data with samples in rows and
#' categories in columns.
#' @param Y_ppc A three-dimensional array containing the posterior predictive
#' distribution. The first dimension indexes the posterior samples, while the
#' remaining dimensions correspond to replicated count-compositional matrices.
#' If `NULL`, then the `plot_ppc` function generates the posterior predictive
#' distribution from the model `object`.
#' @param stat_fun A function that computes a scalar diagnostic statistic from a
#' count-compositional matrix.
#' @param object A fitted model object of class `zanicc` used to generate posterior
#' predictive samples via the method [ppd()]. Only used when `Y_ppc = NULL`.
#' @param output Logical. If `TRUE`, return the computed posterior predictive
#' statistics in addition to producing the plots. The default is `FALSE`.
#'
#' @return
#' `ppc_stat()` returns a list with two elements:
#' \describe{
#'   \item{`t_ppc`}{A vector containing the diagnostic statistic evaluated on
#'   each posterior predictive replicate.}
#'   \item{`t_obs`}{The diagnostic statistic computed from the observed data.}
#' }
#'
#' `plot_ppc()` is called for its side effect of producing four posterior
#' predictive check plot:
#' Shannon entropy (`entropy`), multiple dispersion index (`mdi`),
#' proportion of zeros (`prop_zero`), and multivariate zero-inflation index (`zi`)).
#' If `output = TRUE`, it returns a named list with the corresponding results
#' from `ppc_stat()`.
#'
#'
#' @export
stat_ppc <- function(Y, Y_ppc, stat_fun) {
  t_obs <- stat_fun(Y)
  t_ppc <- apply(Y_ppc, 1, stat_fun)
  list(t_ppc = t_ppc, t_obs = t_obs)
}

#' @rdname ppc
#' @export
plot_ppc <- function(Y, Y_ppc = NULL, object = NULL, output = FALSE) {
  if (!is.null(object) && is.null(Y_ppc)) {
    cat("Generating posterior predictive distribution\n")
    Y_ppc <- ppd(object, relative = FALSE)
  }
  cat("Computing the posterior predictive checks\n")
  res_mdi <- stat_ppc(Y = Y, Y_ppc = Y_ppc, stat_fun = mdi)
  res_zero <- stat_ppc(Y = Y, Y_ppc = Y_ppc, stat_fun = function(Y) mean(Y == 0))
  res_zi <- stat_ppc(Y = Y, Y_ppc = Y_ppc, stat_fun = zi_multinomial)
  res_entropy <- stat_ppc(
    Y = sweep(Y, 1, rowSums(Y), "/"),
    Y_ppc = .normalise_composition(Y_ppc),
    stat_fun = shannon_entropy
  )

  # Keep user's graphs options
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  # Plotting
  par(mar = c(4, 4, 1, 1), mfrow = c(2, 2))
  plot(density(res_entropy$t_ppc),
    main = "Entropy", xlab = "", ylab = "",
    xlim = range(res_entropy$t_obs, res_entropy$t_ppc)
  )
  abline(v = res_entropy$t_obs)
  plot(density(res_mdi$t_ppc),
    main = "MDI", xlab = "", ylab = "",
    xlim = range(res_mdi$t_obs, res_mdi$t_ppc)
  )
  abline(v = res_mdi$t_obs)
  plot(density(res_zero$t_ppc),
    main = "Prop of zero", xlab = "", ylab = "",
    xlim = range(res_zero$t_obs, res_zero$t_ppc)
  )
  abline(v = res_zero$t_obs)
  plot(density(res_zi$t_ppc),
    main = "ZI", xlab = "", ylab = "",
    xlim = range(res_zi$t_obs, res_zi$t_ppc)
  )
  abline(v = res_zi$t_obs)

  if (output) {
    return(list(entropy = res_entropy, mdi = res_mdi, prop_zero = res_zero, zi = res_zi))
  }

  invisible()
}


#' @name marginal_qqplots_ppd
#'
#' @title Marginal QQ-plots from the posterior predictive distribution
#'
#' @description
#' Compute and visualise the marginal QQ-plots from the posterior predictive
#' distribution of a given fitted count-compositional model.
#' The function `plot_qqplots_ppd()` compares the empirical marginal quantiles
#' of each category with the corresponding posterior predictive quantiles,
#' together with pointwise 95% posterior predictive intervals.
#' The helper function `marginal_quantiles_ppd()` computes the posterior predictive
#' marginal quantiles used to construct these QQ-plots for a given category.
#'
#'
#' @param Y A matrix of observed count-compositional data with samples in rows and
#' categories in columns.
#' @param Y_ppc A three-dimensional array containing the posterior predictive
#' distribution. The first dimension indexes the posterior samples, while the
#' remaining dimensions correspond to replicated count-compositional matrices.
#' If `NULL` (the default), then the `plot_qqplots_ppd` function generates the posterior
#' predictive distribution from the model `object`.
#' @param object A fitted model object of class `zanicc` used to generate posterior
#' predictive samples via `ppd()`. Only used when `Y_ppc = NULL`.
#' @param relative Logical. If `TRUE`, QQ-plots are computed using relative
#' compositions instead of counts. The default is `FALSE`.
#' @param output Logical. If `TRUE`, return the data used to construct the
#' QQ-plots in addition to producing the plots. Default is `FALSE`.
#' @param len_probs Integer giving the number of equally spaced probabilities
#' used to compute the quantiles. The default is `100`.
#' @param mfrow A length-two integer vector passed to `par(mfrow)` specifying
#' the layout of the QQ-plots. If `NULL` (the default), a suitable layout is
#' computed using `grDevices::n2mfrow()`.
#' @param yj_ppc Matrix with the posterior predictive distribution for given category
#' dimension.
#' @param yj_ppc A matrix containing the posterior predictive samples for a single
#' category, with rows corresponding the samples and columns to observations.
#' @param probs A numeric vector of probabilities at which marginal quantiles
#' are computed.
#'
#' @return
#'
#' `plot_qqplots_ppd()` is called for its side effect of producing the marginal
#' QQ-plots for all categories. If `output = TRUE`, it returns a list whose
#' \eqn{j}-th element contains the empirical quantiles and the corresponding
#' posterior predictive median, lower (2.5%), and upper (97.5%) quantiles used
#' to construct the QQ-plot for category \eqn{j}.
#'
#' `marginal_quantiles_ppd()` returns a matrix with one row per probability in
#' `probs` and three columns containing the posterior predictive median, lower
#' (2.5%), and upper (97.5%) quantiles.
#'
#' @export
plot_qqplots_ppd <- function(Y, Y_ppc = NULL, object = NULL, relative = FALSE,
                             output = FALSE, len_probs = 100L, mfrow = NULL) {
  # Generate the posterior predictive distribution
  if (!is.null(object) && is.null(Y_ppc)) {
    cat("Generating posterior predictive distribution\n")
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
    q_teo <- marginal_quantiles_ppd(yj_ppc = Y_ppc[, , j], probs = probs)
    ry <- c(min(q_teo[, 2L]), max(q_teo[, 3L]))
    rx <- range(q_obs)
    plot(q_teo[, 1L], q_obs,
      ylim = ry, xlim = rx,
      main = sprintf("category j=%i", j),
      xlab = "Theoretical quantiles", ylab = "Empirical quantiles"
    )
    lines(q_teo[, 2L], q_obs, lty = "dashed")
    lines(q_teo[, 3L], q_obs, lty = "dashed")
    abline(0, 1, col = "grey60", lty = "dashed")
    if (output) list_data[[j]] <- cbind(q_obs, q_teo)
  }
  if (output) {
    return(list_data)
  }
  invisible()
}

#' @rdname marginal_qqplots_ppd
#' @export
marginal_quantiles_ppd <- function(yj_ppc, probs) {
  qs_ppc <- apply(yj_ppc, 1, quantile, probs = probs, names = FALSE)
  q_med <- apply(qs_ppc, 1, median)
  q_lo <- apply(qs_ppc, 1, quantile, probs = 0.025)
  q_up <- apply(qs_ppc, 1, quantile, probs = 0.975)
  cbind(q_med, q_lo, q_up)
}
