test_that("multiplication works", {

  devtools::load_all()
  p <- 3
  mu <- c(-2, 1.5, 3)
  Sigma <- matrix(nrow = p, ncol = p)
  diag(Sigma) <- c(1, 2, 1.5)
  Sigma[lower.tri(Sigma)] <- Sigma[upper.tri(Sigma)] <- c(-0.1, 1, 0.5)
  cov2cor(Sigma)
  Sigma_chol <- chol(Sigma)
  SigL <- t(Sigma_chol)

  n <- 5000L

  rand_norm <- matrix(stats::rnorm(p*n), ncol = p, nrow = n)

  # R implementation
  x_R <- matrix(nrow = n, ncol = p)
  for (i in seq_len(n)) x_R[i, ] <- drop(rand_norm[i, ] %*% Sigma_chol + mu)

  # R v2 implementation
  x_R2 <- matrix(nrow = n, ncol = p)
  for (i in seq_len(n)) x_R2[i, ] <- drop(SigL %*% rand_norm[i, ] + mu)

  # C++
  x_C <- replicate(n, rmvnorm_chol_22(mean = mu, L = c(Sigma_chol), p = p))
  x_C <- t(x_C)
  x_C3 <- replicate(n, rmvnorm_chol_33(mean = mu, Sigma = Sigma, p = p))
  x_C3 <- t(x_C3)

  # Comparison
  cbind(sim1 = colMeans(x_R), sim2 = colMeans(x_R2),
        simC = colMeans(x_C),
        simC3 = colMeans(x_C3),
        true = mu)
  cbind(sim1 = diag(cov(x_R)), sim2 = diag(cov(x_R2)),
        simC = diag(cov(x_C)),
        simC3 = diag(cov(x_C3)),
        true = diag(Sigma))
  cbind(sim1 = cov(x_R)[lower.tri(cov(x_R))],
        sim2 = cov(x_R2)[lower.tri(cov(x_R2))],
        simC = cov(x_C)[lower.tri(cov(x_C))],
        simC3 = cov(x_C3)[lower.tri(cov(x_C3))],
        true = Sigma[lower.tri(Sigma)])


})
