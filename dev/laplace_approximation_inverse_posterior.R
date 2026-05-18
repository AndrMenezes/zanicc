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

# Laplace approximation
ml <- Rcpp::Module(module = "inverse_posterior", PACKAGE = "zanicc")
cpp_obj <- new(ml$InversePosterior, d, NTREES, NTREES, 5000, "zanim_bart",
               forests_dir)
foo <- function(x, y) -cpp_obj$lmlZANIM(y, x)
yast <- Y_test[3, ]
x_true <- X_test[3, ]
opt1 <- optim(par = 0.0, fn = foo, method = "BFGS", y = yast, hessian = TRUE)
opt2 <- optim(par = 0.0, fn = foo, method = "L-BFGS-B", y = yast, hessian = TRUE,
              lower = -1.0, upper = 1.0)
opt3 <- optimise(f = foo, interval = c(-1.0, 1.0), y = yast)

cbind(true = x_true, bfgs = opt1$par[1], lbfgsb = opt2$par[1], optimise=opt3$minimum)

v1 <- -1/opt1$hessian[1]
v2 <- 1/opt2$hessian[1]

xx1 <- stats::rnorm(1000, mean = opt1$par[1], sd = sqrt(v1))
xx2 <- stats::rnorm(1000, mean = opt2$par[1], sd = sqrt(v2))

par(mfrow = c(1, 2))
plot(density(xx1))
points(x_true, y = 0.001)
plot(density(xx2))
points(x_true, y = 0.001)
