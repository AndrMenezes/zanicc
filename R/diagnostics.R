.get_legend <- function(plot, legend = NULL) {
  gt <- ggplot2::ggplotGrob(plot)
  pattern <- "guide-box"
  if (!is.null(legend)) pattern <- paste0(pattern, "-", legend)
  indices <- grep(pattern, gt$layout$name)
  not_empty <- !vapply(gt$grobs[indices], inherits, what = "zeroGrob",
                       FUN.VALUE = logical(1))
  indices <- indices[not_empty]
  if (length(indices) > 0) return(gt$grobs[[indices[1]]])
  return(NULL)
}

# Y = Y; Y_ppd = list_ppds

.plot_ppcs_list <- function(Y, Y_ppd, order_labels, output) {

  n_models <- length(Y_ppd)
  ndpost <- dim(Y_ppd[[1L]])[1L]
  models_label <- names(Y_ppd)

  # Empirical statistics
  t_mdi <- mdi(Y)
  t_zi <- zi_multinomial(Y)
  t_zero <- mean(Y == 0)
  t_entropy <- shannon_entropy(sweep(Y, 1, rowSums(Y), "/"))

  # Compute posterior predictive statistics
  ppc_mdi <- lapply(Y_ppd, function(x) apply(x, 1, mdi))
  ppc_zi <- lapply(Y_ppd, function(x) apply(x, 1, zi_multinomial))
  ppc_zero <- lapply(Y_ppd, function(x) apply(x, 1, function(u) mean(u == 0)))
  ppc_entropy <- lapply(Y_ppd, function(x) apply(.normalise_composition(x),
                                                 1, shannon_entropy))
  # Organise in data frame
  df_ppc_mdi <- data.frame(x = unlist(ppc_mdi),
                           label_model = rep(models_label, each = ndpost))
  df_ppc_zi <- data.frame(x = unlist(ppc_zi),
                          label_model = rep(models_label, each = ndpost))
  df_ppc_zero <- data.frame(x = unlist(ppc_zero),
                            label_model = rep(models_label, each = ndpost))
  df_ppc_entropy <- data.frame(x = unlist(ppc_entropy),
                               label_model = rep(models_label, each = ndpost))
  # Order the labels
  if (!is.null(order_labels)) {
    df_ppc_mdi$label_model <- factor(df_ppc_mdi$label_model, levels = order_labels)
    df_ppc_zi$label_model <- factor(df_ppc_zi$label_model, levels = order_labels)
    df_ppc_zero$label_model <- factor(df_ppc_zero$label_model, levels = order_labels)
    df_ppc_entropy$label_model <- factor(df_ppc_entropy$label_model, levels = order_labels)
  }

  # Plotting
  p_mdi <- ggplot2::ggplot(data = df_ppc_mdi, ggplot2::aes(x = x)) +
    ggplot2::geom_density(ggplot2::aes(fill = label_model), alpha = 0.4,
                          show.legend = FALSE) +
    ggplot2::geom_vline(xintercept = t_mdi, col = "black", linetype = "dashed") +
    ggplot2::labs(x = "MDI", y = "Density", fill = "") +
    ggplot2::scale_x_continuous(breaks = scales::pretty_breaks(6))
  p_zi <- ggplot2::ggplot(data = df_ppc_zi, ggplot2::aes(x = x)) +
    ggplot2::geom_density(ggplot2::aes(fill = label_model), alpha = 0.5,
                          show.legend = FALSE) +
    ggplot2::geom_vline(xintercept = t_zi, col = "black", linetype = "dashed") +
    ggplot2::labs(x = "ZI", y = "Density", fill = "") +
    ggplot2::scale_x_continuous(breaks = scales::pretty_breaks(6))
  p_zero <- ggplot2::ggplot(data = df_ppc_zero, ggplot2::aes(x = x)) +
    ggplot2::geom_density(ggplot2::aes(fill = label_model), alpha = 0.5,
                          show.legend = FALSE) +
    ggplot2::geom_vline(xintercept = t_zero, col = "black", linetype = "dashed") +
    ggplot2::labs(x = "Proportion of zeros", y = "Density", fill = "") +
    ggplot2::scale_x_continuous(breaks = scales::pretty_breaks(6))
  p_entropy <- ggplot2::ggplot(data = df_ppc_entropy, ggplot2::aes(x = x)) +
    ggplot2::geom_density(ggplot2::aes(fill = label_model), alpha = 0.5,
                          show.legend = FALSE) +
    ggplot2::geom_vline(xintercept = t_entropy, col = "black", linetype = "dashed") +
    ggplot2::labs(x = "Entropy", y = "Density", fill = "") +
    ggplot2::scale_x_continuous(breaks = scales::pretty_breaks(6))

  # Extract common legend
  p <- ggplot2::ggplot(data = df_ppc_entropy, ggplot2::aes(x = x)) +
    ggplot2::geom_density(ggplot2::aes(fill = label_model), alpha = 0.5,
                          show.legend = TRUE) +
    ggplot2::guides(fill = ggplot2::guide_legend(nrow = 1)) +
    ggplot2::labs(fill = "") +
    ggplot2::theme(text = ggplot2::element_text(size = 12))
  p_legend <- .get_legend(p)
  # Plot all diagnostics
  p_all <- cowplot::plot_grid(p_mdi, p_entropy, p_zero, p_zi,
                              align = "v", axis = "b", labels = "AUTO")
  # Include common legend
  p_out <- cowplot::plot_grid(p_legend, p_all, ncol = 1, rel_heights = c(0.05, 1))
  if (output) return(
    list(p_out, entropy = df_ppc_entropy, mdi = df_ppc_mdi,
         prop_zero = df_ppc_zero, zi = df_ppc_zi)
  )
  return(p_out)
}

.plot_ppcs <- function(Y, Y_ppd, output) {

  ndpost <- dim(Y_ppd)[1L]

  # Empirical statistics
  t_mdi <- mdi(Y)
  t_zi <- zi_multinomial(Y)
  t_zero <- mean(Y == 0)
  t_entropy <- shannon_entropy(sweep(Y, 1, rowSums(Y), "/"))
  df_empirical <- data.frame(x = c(t_mdi, t_entropy, t_zi, t_zero),
                             metric = c("MDI", "Entropy", "ZI",
                                        "Proportion of zeros"))

  # Compute posterior predictive statistics
  ppc_mdi <- apply(Y_ppd, 1, mdi)
  ppc_zi <- apply(Y_ppd, 1, zi_multinomial)
  ppc_zero <- apply(Y_ppd, 1, function(u) mean(u == 0))
  ppc_entropy <- apply(.normalise_composition(Y_ppd), 1, shannon_entropy)

  # Organise in data frame
  df_ppcs <- data.frame(x = c(ppc_mdi, ppc_entropy, ppc_zi, ppc_zero),
                        metric = rep(c("MDI", "Entropy", "ZI",
                                       "Proportion of zeros"), each = ndpost))
  # Plotting
  p_out <- ggplot2::ggplot(data = df_ppcs, ggplot2::aes(x = x)) +
    ggplot2::facet_wrap(~metric, scales = "free") +
    ggplot2::geom_density(alpha = 0.4, show.legend = FALSE) +
    ggplot2::geom_vline(data = df_empirical, mapping = ggplot2::aes(xintercept = x),
                        col = "black", linetype = "dashed") +
    ggplot2::labs(x = "", y = "Density", fill = "") +
    ggplot2::scale_x_continuous(breaks = scales::pretty_breaks(6))

  if (output) return(list(p_out, empirical = df_empirical, ppcs = df_ppcs))
  return(p_out)
}

#' @name ppc
#'
#' @title Posterior predictive check for count-compositional models
#'
#' @description
#' Compute and visualise posterior predictive checks for count-compositional models.
#'
#' `stat_ppc()` evaluates a user defined diagnostic statistic on the observed
#' data and on posterior predictive replicates.
#'
#' `plot_ppc()` visualises four built-in diagnostic statistics by comparing
#' their observed values with their posterior predictive distributions:
#' Shannon entropy, the multiple dispersion index (MDI), the proportion of
#' zeros, and the multivariate zero-inflation index (ZI).
#'
#' When posterior predictive samples from multiple models are supplied,
#' `plot_ppc()` displays the distributions for each model in the same panels,
#' allowing their diagnostic behaviour to be compared.
#'
#' @param Y A matrix of observed count-compositional data with samples in rows and
#' categories in columns.
#' @param Y_ppd Posterior predictive distribution.
#' For `stat_ppc()`, this must be a three-dimensional array, with posterior draws
#' in the first dimension and the dimensions of each replicated count matrix in
#' the remaining dimensions.
#' For `plot_ppc()`, this can be either a three-dimensional array containing
#' posterior predictive replicates from a single model or a named list of
#' such arrays, with one element per model. Each array must have posterior
#' draws in the first dimension, followed by the rows and columns of the
#' replicated count matrices.
#' @param stat_fun A function that computes a scalar diagnostic statistic from a
#' count-compositional matrix. Used by `stat_ppc()`.
#' @param output Logical. For `plot_ppc()`, if `TRUE`, return the plot together
#' with the data used to construct it. If `FALSE` (the default), return only
#' the plot.
#' @param order_labels An optional character vector specifying the order of
#' model labels in the posterior predictive density plots. Applicable when
#' `Y_ppd` is a list of arrays.
#'
#' @return
#' `stat_ppc()` returns a list with two elements:
#' \describe{
#'   \item{`t_ppc`}{A numeric vector containing the diagnostic statistic
#'   evaluated on each posterior predictive replicate.}
#'   \item{`t_obs`}{The diagnostic statistic evaluated on the observed data.}
#' }
#'
#' `plot_ppc()` returns a plot when `output = FALSE`. When `output = TRUE`,
#' it returns a list containing the plot and the data used to construct it.
#' The structure of this list depends on the input:
#' \describe{
#'   \item{Single model}{A list containing the plot (`p_out`), the observed
#'   diagnostic statistics (`empirical`), and the posterior predictive
#'   statistics (`ppcs`).}
#'   \item{Multiple models}{A list containing the plot (`p_out`) and
#'   four data frames of posterior predictive statistics: `entropy`, `mdi`,
#'   `prop_zero`, and `zi`. Each data frame contains the simulated statistic
#'   values and their corresponding model labels.}
#' }
#'
#' @export
stat_ppc <- function(Y, Y_ppd, stat_fun) {
  t_obs <- stat_fun(Y)
  t_ppc <- apply(Y_ppd, 1, stat_fun)
  list(t_ppc = t_ppc, t_obs = t_obs)
}

#' @rdname ppc
#' @export
plot_ppc <- function(Y, Y_ppd, output = FALSE, order_labels = NULL) {

  cat("Computing the posterior predictive checks\n")
  if (is.list(Y_ppd)) {
    .plot_ppcs_list(Y = Y, Y_ppd = Y_ppd, output = output,
                    order_labels = order_labels)
  } else {
    .plot_ppcs(Y = Y, Y_ppd = Y_ppd,  output = output)
  }

  # res_mdi <- stat_ppc(Y = Y, Y_ppd = Y_ppd, stat_fun = mdi)
  # res_zero <- stat_ppc(Y = Y, Y_ppd = Y_ppd, stat_fun = function(Y) mean(Y == 0))
  # res_zi <- stat_ppc(Y = Y, Y_ppd = Y_ppd, stat_fun = zi_multinomial)
  # res_entropy <- stat_ppc(Y = sweep(Y, 1, rowSums(Y), "/"),
  #                         Y_ppd = .normalise_composition(Y_ppd),
  #                         stat_fun = shannon_entropy)
  #
  # # Keep user's graphs options
  # oldpar <- graphics::par(no.readonly = TRUE)
  # on.exit(graphics::par(oldpar))
  #
  # # Plotting
  # graphics::par(mar = c(4, 4, 1, 1), mfrow = c(2, 2))
  # plot(stats::density(res_entropy$t_ppc),
  #   main = "Entropy", xlab = "", ylab = "",
  #   xlim = range(res_entropy$t_obs, res_entropy$t_ppc)
  # )
  # graphics::abline(v = res_entropy$t_obs)
  # plot(stats::density(res_mdi$t_ppc),
  #   main = "MDI", xlab = "", ylab = "",
  #   xlim = range(res_mdi$t_obs, res_mdi$t_ppc)
  # )
  # graphics::abline(v = res_mdi$t_obs)
  # plot(stats::density(res_zero$t_ppc),
  #   main = "Prop of zero", xlab = "", ylab = "",
  #   xlim = range(res_zero$t_obs, res_zero$t_ppc)
  # )
  # graphics::abline(v = res_zero$t_obs)
  # plot(stats::density(res_zi$t_ppc),
  #   main = "ZI", xlab = "", ylab = "",
  #   xlim = range(res_zi$t_obs, res_zi$t_ppc)
  # )
  # graphics::abline(v = res_zi$t_obs)
  #
  # if (output) {
  #   return(list(entropy = res_entropy, mdi = res_mdi, prop_zero = res_zero, zi = res_zi))
  # }
  # invisible()
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
    Y_ppc <- .normalise_composition(Y_ppc)
    Y <- sweep(Y, 1, rowSums(Y), "/")
  }
  # Probabilities for the quantiles
  probs <- seq(0.0, 1.0, length.out = len_probs)
  # Keep user's graphs options
  oldpar <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(oldpar))
  d <- ncol(Y)
  if (output) list_data <- vector(mode = "list", length = d)
  if (is.null(mfrow)) mfrow <- grDevices::n2mfrow(d)
  # Plotting
  graphics::par(mfrow = mfrow, mar = c(4, 4, 1, 1))
  for (j in seq_len(d)) {
    q_obs <- stats::quantile(Y[, j], probs = probs, names = FALSE)
    q_teo <- marginal_quantiles_ppd(yj_ppc = Y_ppc[, , j], probs = probs)
    ry <- c(min(q_teo[, 2L]), max(q_teo[, 3L]))
    rx <- range(q_obs)
    plot(q_teo[, 1L], q_obs,
      ylim = ry, xlim = rx,
      main = sprintf("category j=%i", j),
      xlab = "Theoretical quantiles", ylab = "Empirical quantiles"
    )
    graphics::lines(q_teo[, 2L], q_obs, lty = "dashed")
    graphics::lines(q_teo[, 3L], q_obs, lty = "dashed")
    graphics::abline(0, 1, col = "grey60", lty = "dashed")
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
  qs_ppc <- apply(yj_ppc, 1, stats::quantile, probs = probs, names = FALSE)
  q_med <- apply(qs_ppc, 1, stats::median)
  q_lo <- apply(qs_ppc, 1, stats::quantile, probs = 0.025)
  q_up <- apply(qs_ppc, 1, stats::quantile, probs = 0.975)
  cbind(q_med, q_lo, q_up)
}
