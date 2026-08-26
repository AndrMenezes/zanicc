rm(list = ls())
devtools::load_all()

# Path
time_id <- "2026-Apr-23-11:54:17"#format(Sys.time(), "%Y-%b-%d-%X")
path_local <- "./tests/testthat/inverse_posterior/zanim_bart/one_dimension"
path_res <- file.path(path_local, time_id, "results")
forests_dir <- file.path(path_res, "forests")
if (!dir.exists(forests_dir)) dir.create(forests_dir, recursive = TRUE)

set.seed(1212)
n_sample <- 300L
d <- 3L
n_trials <- 500L
tmp <- sim_data_zanim_bspline_curve(n = n_sample, d = d, n_trials = n_trials,
                                    link_zeta = "probit")

# Split the data
n_test <- 100L
id_test <- sample.int(n_sample, n_test)
Y_test <- tmp$Y[id_test, ]
X_test <- tmp$X[id_test, , drop = FALSE]
Y_train <- tmp$Y[id_test, ]
X_train <- tmp$X[id_test, , drop = FALSE]

# Fit forward model
NDPOST <- 5000L
NSKIP <- 5000L
NTREES <- 200L

if (file.exists(file.path(path_res, "mod.rds"))) {
  zanim_bart <- load_model(model_dir = path_res)
} else {
  zanim_bart <- zanicc(Y = Y_train, X_count = X_train, X_zi = X_train,
                       model = "zanim_bart", ntrees_theta = NTREES,
                       ntrees_zeta = NTREES, ndpost = NDPOST, nskip = NSKIP,
                       save_trees = TRUE, forests_dir = forests_dir)
  save_model(object = zanim_bart, model_dir = path_res)
}

# Importance sampling
is <- .is_zanimbart(object = zanim_bart, Y = Y_test, dir_posterior_fx = path_res)
sir <- .is_zanimbart(object = zanim_bart, Y = Y_test, dir_posterior_fx = path_res, sir = TRUE)
x_proposal <- is[, -seq_len(n_test)]

# Numerical integration, Gauss-legender the domain is (-1, 1)
i <- 3L
yast <- Y_test[i, ]
x_true <- X_test[i, ]

nquad <- 20L
gh <- statmod::gauss.quad(n = nquad, kind = "legendre")
ll <- sapply(gh$nodes, function(x) cpp_obj$lmlZANIM(yast, x))
Z <- sum(exp(ll) * gh$weights)
probs <- exp(ll) / Z

# Plotting
par(mfrow = c(1, 3))
plot(gh$nodes, probs, type = "S", main = "GL")
points(x_true, y = 0.001, col = "blue", pch = 19)
plot(x_proposal, is[, i], type = "S", main = "IS")
points(x_true, y = 0.00001, col = "blue", pch = 19)
plot(density(sir[[i]]), type = "l", main = "SIR")
points(x_true, y = 0.001, col = "blue", pch = 19)


plot(density(x_draws), xlim = c(-1, 1))



ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
cpp_obj <- new(ml$InversePosterior, d, NTREES, NTREES, 500, "zanim_bart",
               forests_dir)
yast <- Y_test[3, ]
x_true <- X_test[3, ]
system.time(ll1 <- cpp_obj$lmlZANIM(yast, X_test[3,]))

# microbenchmark::microbenchmark(cpp_obj$lmlZANIM(yast, X_test[3,]))

x_cur <- stats::runif(n = 1, -1, 1)
ll_cur <- cpp_obj$lmlZANIM(yast, x_cur)
MC <- 1000L
x_draws <- numeric(MC)
accept_rate <- 0L
for (k in seq_len(MC)) {
  cat(k, "of", MC ,"\n")
  # Proposal
  x_prop <- stats::runif(n = 1, -1, 1)
  ll_prop <- cpp_obj$lmlZANIM(yast, x_prop)
  # Log-ratio
  lr <- ll_prop - ll_cur
  if (log(stats::runif(1)) < lr) {
    x_cur <- x_prop
    ll_cur <- ll_prop
    accept_rate <- accept_rate + 1L
  }
  x_draws[k] <- x_cur
}


accept_rate/MC

par(mfrow = c(1, 2))
plot(density(x_draws))
points(X_test[3, ], y = 0.001)
plot(density(x_draws2))
points(X_test[3, ], y = 0.001)
plot(x_draws, type = "l")
