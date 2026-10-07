test_that("Check compute_metrics for p = 1", {

  n <- 100L
  sigma2 <- 25.0
  sd <- sqrt(sigma2)
  mu <- 0.5
  ndpost <- 1000L

  # Data
  x <- matrix(stats::rnorm(n = n, mean = mu, sd = sd), ncol = 1L)

  # Prior
  mu_0 <- 0.0
  sigma2_0 <- 1.0
  # Posterior and posterior-predictive parameters
  sigma2_post <- (1 / (1/sigma2_0 + n / sigma2))
  mu_post <- sigma2_post * (mu_0 / sigma2_0 + sum(x) / sigma2)
  sigma2_pp <- sigma2_post + sigma2
  sd_pp <- sqrt(sigma2_pp)


  # Simulate from the posterior predictive distribution
  x_draws <- array(stats::rnorm(n = n*ndpost, mean = mu_post, sd = sd_pp),
                   dim = c(ndpost, 1, n))
  .compute_prediction_metrics(x = x, draws = x_draws, parallel = FALSE, verbose = FALSE)
  .compute_prediction_metrics(x = x, draws = x_draws, parallel = FALSE, verbose = TRUE)
  .compute_prediction_metrics(x = x, draws = x_draws, parallel = TRUE)

})


test_that("Check compute_metrics for p > 1", {

  set.seed(54)
  ndpost <- 1000L
  p <- 3L
  n <- 100L
  Sigma <- tcrossprod(stats::rnorm(p)) + diag(stats::rgamma(n = p, shape = 2, rate = 0.5))
  Sigma_inv <- solve(Sigma)
  lower_chol_Sigma <- t(chol(Sigma))
  mu <- stats::rnorm(p, mean = 1.0)

  # Data
  x <- matrix(nrow = n, ncol = p)
  for (i in seq_len(n)) x[i, ] <- drop(lower_chol_Sigma %*% stats::rnorm(p) + mu)

  # Prior
  mu_0 <- rep(0.0, p)
  Sigma_0 <- diag(x = 1, nrow = p, ncol = p)
  Sigma_0_inv <- solve(Sigma_0)
  # Posterior and posterior-predictive parameters
  xbar <- colMeans(x)
  Sigma_post <- solve(Sigma_0_inv + n * Sigma_inv)
  mu_post <- Sigma_post %*% (Sigma_0_inv %*% mu_0 + n * Sigma_inv %*% xbar)
  Sigma_pp <- Sigma_post + Sigma

  # Simulate from the posterior predictive distribution
  x_draws <- array(dim = c(ndpost, p, n))
  lower_chol_Sigma_pp <- t(chol(Sigma_pp))
  for (k in seq_len(ndpost)) {
    for (i in seq_len(n)) {
      x_draws[k,,i] <- drop(lower_chol_Sigma_pp %*% stats::rnorm(p) + mu_post)
    }
  }

  res1 <- .compute_prediction_metrics(x = x, draws = x_draws, parallel = FALSE,
                                      verbose = FALSE, joint_coverage = FALSE)
  res2 <- .compute_prediction_metrics(x = x, draws = x_draws, parallel = FALSE,
                                      verbose = TRUE, joint_coverage = TRUE,
                                      method_kde = "kdevine")
  res3 <- .compute_prediction_metrics(x = x, draws = x_draws, parallel = TRUE,
                                      verbose = TRUE, joint_coverage = TRUE,
                                      ncores = 4L, method_kde = "kde")
  cbind(res2, res3)


})


