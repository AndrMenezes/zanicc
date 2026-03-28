test_that("test save-load model", {

  rm(list = ls())
  devtools::load_all()

  # Path
  time_id <- "2026-Mar-13-11:00:42"#format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/multinomial", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  # Simulate some data
  n_trials <- 500L
  n_sample <- 500L
  d <- 4L
  set.seed(1212)
  sim_tmp <- sim_data_multinomial_bspline_curve(d = 4, n = n_sample,
                                                n_trials = n_trials)

  idx_train <- sample(1:n_sample, size = 400L)
  Y_train <- sim_tmp$Y[idx_train, ]
  X_train <- sim_tmp$X[idx_train, , drop = FALSE]
  Y_test <- sim_tmp$Y[-idx_train, ]
  X_test <- sim_tmp$X[-idx_train, , drop = FALSE]
  theta_test <- sim_tmp$theta[-idx_train, , drop = FALSE]
  data_sim_train <- sim_tmp$df[idx_train, ]
  data_sim_test <- sim_tmp$df[-idx_train, ]
  Y_train_rel <- sweep(Y_train, 1, rowSums(Y_train), "/")
  Y_test_rel <- sweep(Y_test, 1, rowSums(Y_test), "/")

  # Fit model
  ml_bart <- zanicc(Y = Y_train, X_count = X_train, model = "mult_bart",
                    ntrees_theta = 20L, ndpost = 100L, nskip = 30L,
                    forests_dir = path_res, keep_draws = TRUE, save_trees = TRUE)
  # save model object
  save_model(object = ml_bart, model_dir = path_res, file_name = "mod.rds")
  expect_true(file.exists(file.path(path_res, "mod.rds")))

  ml_bart_loaded <- load_model(model_dir = path_res, file_name = "mod.rds")

  ml_bart_loaded$cpp_obj
  pred <- predict(ml_bart_loaded, newdata = head(X_train), load = TRUE)
  expect_equal(pred, ml_bart_loaded$draws_theta[1:6,,])

  # Fit mln-bart
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/mln_bart", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  mln_bart <- zanicc(Y = Y_train, X_count = X_train, model = "mult_ln_bart",
                    ntrees_theta = 20L, ndpost = 100L, nskip = 30L,
                    keep_draws = TRUE, save_trees = TRUE,
                    forests_dir = path_res)

  # save model object
  save_model(object = mln_bart, model_dir = path_res, file_name = "mln.rds")
  expect_true(file.exists(file.path(path_res, "mln.rds")))

  mln_bart_loaded <- load_model(model_dir = path_res, file_name = "mln.rds")
  expect_type(mln_bart_loaded$cpp_obj, "S4")

  pred <- predict(mln_bart_loaded, newdata = head(X_train), load = TRUE)
  expect_equal(pred, mln_bart_loaded$draws_theta[1:6,,])

  # ZANIM-BART
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/zanim_bart", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  zanim_bart <- zanicc(Y = Y_train, X_count = X_train, X_zi = X_train,
                       model = "zanim_bart", ntrees_theta = 20L, ntrees_zeta = 20L,
                       ndpost = 100L, nskip = 30L, keep_draws = TRUE,
                       save_trees = TRUE, forests_dir = path_res)

  # save model object
  save_model(object = zanim_bart, model_dir = path_res, file_name = "mod.rds")
  expect_true(file.exists(file.path(path_res, "mod.rds")))

  zanim_bart_loaded <- load_model(model_dir = path_res, file_name = "mod.rds")
  expect_type(zanim_bart_loaded$cpp_obj, "S4")

  pred <- predict(zanim_bart_loaded, newdata = head(X_train), type = "theta",
                  output_dir = path_res, load = TRUE)
  expect_equal(pred, zanim_bart_loaded$draws_theta[1:6,,])
  predz <- predict(zanim_bart_loaded, newdata = head(X_train), type = "zeta",
                  output_dir = path_res, load = TRUE)
  expect_equal(predz, zanim_bart_loaded$draws_zeta[1:6,,])

  # ZANIM-LN-BART
  time_id <- format(Sys.time(), "%Y-%b-%d-%X")
  path_res <- file.path("./tests/testthat/zanim_ln_bart", time_id, "draws")
  if (!dir.exists(path_res)) dir.create(path_res, recursive = TRUE)

  zanim_ln_bart <- zanicc(Y = Y_train, X_count = X_train, X_zi = X_train,
                       model = "zanim_ln_bart", ntrees_theta = 20L, ntrees_zeta = 20L,
                       ndpost = 100L, nskip = 30L, keep_draws = TRUE,
                       save_trees = TRUE, forests_dir = path_res)

  # save model object
  save_model(object = zanim_ln_bart, model_dir = path_res, file_name = "mod.rds")
  expect_true(file.exists(file.path(path_res, "mod.rds")))

  zanim_ln_bart_loaded <- load_model(model_dir = path_res, file_name = "mod.rds")
  expect_type(zanim_ln_bart_loaded$cpp_obj, "S4")

  pred <- predict(zanim_ln_bart_loaded, newdata = head(X_train), type = "theta",
                  output_dir = path_res, load = TRUE)
  expect_equal(pred, zanim_ln_bart_loaded$draws_theta[1:6,,])
  predz <- predict(zanim_ln_bart_loaded, newdata = head(X_train), type = "zeta",
                  output_dir = path_res, load = TRUE)
  expect_equal(predz, zanim_ln_bart_loaded$draws_zeta[1:6,,])


})
