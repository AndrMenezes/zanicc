rm(list = ls())
devtools::load_all()

test_that("vectorized version of log-likelihood", {

  n_samples <- 5L
  n_trials <- 10L
  d <- 28L
  set.seed(12)
  prob <- 1:d / d
  prob <- prob / sum(prob)
  zeta <- runif(d, min = 0.5, max = 0.7)

  prob_mat <- matrix(rep(prob, each = n_samples), ncol = d)
  zeta_mat <- matrix(rep(zeta, each = n_samples), ncol = d)
  y <- .rzanim_vec(n = n, sizes = rep(n_trials, n_samples), probs = prob_mat,
                  zetas = zeta_mat)
  rowSums(y == 0)

  ini <- proc.time()
  ll1 <- sapply(seq_len(n_samples), function(i) {
    # idx <- seq.int(i, n * d, by = n)
    zanicc:::log_pmf_zanim(x = y[i, ], prob = prob_mat[i, ], zeta = zeta_mat[i, ])
  })
  end1 <- proc.time() - ini
  ini <- proc.time()
  ll2 <- log_pmf_zanim_vec(n = n_samples, d = d, x = c(y), prob = c(prob_mat),
                           zeta = c(zeta_mat))
  end2 <- proc.time() - ini
  end1[3]; end2[3]
  expect_equal(ll1, ll2)

  # microbenchmark::microbenchmark(
  #   sapply = sapply(seq_len(n), function(i) {
  #     log_pmf_zanim(x = y[i, ], prob = prob_mat[i, ], zeta = zeta_mat[i, ])
  #   }),
  #   cpp = log_pmf_zanim_vec(n = n, d = d, x = c(y), prob = c(prob_mat),
  #                           zeta = c(zeta_mat))
  # )


  # A Monte Carlo approximation using proposition 2.2
  MC <- 5000L
  pz_approx <- numeric(n_samples)
  for (i in seq_len(n_samples)) {
    y_cur <- y[i, ]
    zeta_cur <- zeta_mat[i, ]
    prob_cur <- prob_mat[i, ]
    log_like <- numeric(MC)
    for (j in seq_len(MC)) {
      z_cur <- stats::rbinom(d, size = 1, prob = 1 - zeta_cur)
      if (all(z_cur == 0)) {
        log_like[j] <- sum(z_cur * log1p(-zeta_cur) + (1 - z_cur) * log(zeta_cur))
        # if (all(y_cur == 0)) {
        #   log_like[j] <- sum(z_cur * log1p(-zeta_cur) + (1 - z_cur) * log(zeta_cur))
        # } else { # Comes from another component that we don't know...
        #   log_like[j] <- -Inf
        # }
      } else {
        s <- sum((1 - z_cur) * prob_cur)
        prob_new <- z_cur * prob_cur / (1 - s)
        is_no_zero <- z_cur > 0
        log_like[j] <- dmultinom(y_cur[is_no_zero], prob = prob_new[is_no_zero],
                                 log = TRUE) +
          sum(z_cur * log1p(-zeta_cur) + (1 - z_cur) * log(zeta_cur))
        # if (all(y_cur[!is_no_zero] == 0)) {
        #   log_like[j] <- dmultinom(y_cur[is_no_zero],
        #                            prob = prob_new[is_no_zero],
        #                            log = TRUE) +
        #     sum(z_cur * log1p(-zeta_cur) + (1 - z_cur) * log(zeta_cur))
        # } else { # Comes from another component that we don't know...
        #   log_like[j] <- -Inf
        #   #dmultinom(y_cur, prob = prob, log = TRUE) + sum(log1p(-zeta))
        # }
      }
    }
    # log-sum-exp
    m <- max(log_like)
    pz_approx[i] <- m + log(mean(exp(log_like - m)))
  }

  # Second approximation
  ldzip <- function(x, zeta, lambda, phi) {
    if (x == 0) return( log(zeta + (1 - zeta) * exp(-lambda * phi)))
    else return(log1p(-zeta) + x * log(lambda) - lambda * phi - lfactorial(x))
  }


  ini <- proc.time()

  s <- d # Or another thing...
  lambda <- prob_mat * s
  MC <- 10000L
  pz_approx_2 <- numeric(n_samples)
  for (i in seq_len(n_samples)) {
    y_cur <- y[i, ]
    n_trials2 <- sum(y_cur)
    zeta_cur <- zeta_mat[i, ]
    lambda_cur <- n_trials2 * prob_mat[i, ]
    log_like <- numeric(MC)
    for (j in seq_len(MC)) {
      z_cur <- stats::rbinom(d, size = 1, prob = 1 - zeta_cur)
      phi <- stats::rgamma(n = 1, shape = n_trials2, rate = sum(lambda_cur * z_cur))
      ll <- log(n_trials2) + (n_trials2 - 1) * log(phi)
      for (k in seq_len(d)) {
        ll <- ll + ldzip(x = y_cur[k], zeta = zeta_cur[k], lambda = lambda_cur[k],
                         phi = phi)
      }
      log_like[j] <- ll
    }
    # log-sum-exp
    ma <- max(log_like)
    pz_approx_2[i] <- ma + log(mean(exp(log_like - ma)))
  }

  end <- proc.time() - ini

  cbind(total_zeros = rowSums(y == 0),
        # approx_1 = pz_approx,
        approx = pz_approx_2, truth = ll1)

  mean((pz_approx_2 - ll1) / ll1)


})

test_that("log-likelihood", {
  # k <- 3L
  # m <- 4L
  # set.seed(12)
  # prob <- rep(1 / k, k)
  # zeta <- runif(k)

  m <- 4L
  k <- 28L
  set.seed(12)
  prob <- rep(1 / k, k)
  zeta <- runif(k, min = 0, max = 0.5)

  # Generate the support of ZANIM
  # s <- seq.int(0, m)
  # grid_all <- expand.grid(replicate(n = k, s, simplify = FALSE))
  # support <- as.matrix(grid_all[rowSums(grid_all) == m, ])
  # colnames(support) <- rownames(support) <- NULL
  # rm(grid_all)
  support <- t(combinat::xsimplex(p = k, n = m))
  # dim(support)

  p <- sapply(seq_len(nrow(support)), function(i) {
    dmultinom(x = support[i, ], size = m, prob = rep(1/k, k))
  })
  expect_equal(sum(p), 1)

  support <- t(combinat::xsimplex(p = k, n = m))
  support <- rbind(support, rep(0, k))
  pz <- sapply(seq_len(nrow(support)), function(i) {
    dzanim(x = support[i, ], prob = rep(1/k, k), zeta = zeta, log = TRUE)
  })
  expect_equal(sum(exp(pz)), 1)

  # A Monte Carlo approximation using proposition 2.2
  MC <- 5000L
  pz_approx <- numeric(nrow(support))
  for (i in seq_len(nrow(support))) {
    y_cur <- support[i, ]
    log_like <- numeric(MC)
    for (j in seq_len(MC)) {
      z_cur <- stats::rbinom(k, size = 1, prob = 1 - zeta)
      if (all(z_cur == 0)) {
        if (all(y_cur == 0)) {
          log_like[j] <- sum(z_cur * log1p(-zeta) + (1 - z_cur) * log(zeta))
        } else { # Comes from another component that we don't know...
          log_like[j] <- -Inf
        }
      } else {
        s <- sum((1 - z_cur) * prob)
        prob_new <- z_cur * prob / (1 - s)
        is_no_zero <- z_cur != 0
        # Use
        if (all(y_cur[!is_no_zero] == 0)) {
          log_like[j] <- dmultinom(y_cur[is_no_zero],
                                   prob = prob_new[is_no_zero],
                                   log = TRUE) +
            sum(z_cur * log1p(-zeta) + (1 - z_cur) * log(zeta))
        } else { # Comes from another component that we don't know...
          log_like[j] <- -Inf
            #dmultinom(y_cur, prob = prob, log = TRUE) + sum(log1p(-zeta))
        }
      }
    }
    # log-sum-exp
    m <- max(log_like)
    pz_approx[i] <- m + log(mean(exp(log_like - m)))
  }
  cbind(pz, pz_approx)
  length(pz_approx)

  # Other tests:
  k <- 3L
  m <- 10L
  zeta <- runif(k)
  zeta[sample.int(k, 3)] <- 0
  support_zanim <- rbind(t(combinat::xsimplex(p = k, n = m)), rep(0, k))
  pz <- sapply(seq_len(nrow(support_zanim)), function(i) {
    dzanim(x = support_zanim[i, ], prob = rep(1/k, k), zeta = zeta)
  })
  expect_equal(sum(exp(pz)), 1)

})
