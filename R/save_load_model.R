#' Save and load BART-based models
#'
#' These functions provide a workflow for saving and loading BART-based models
#' implemented in `C++` classes and exposed using the `Rcpp::Module` function.
#' It includes the `C++` classes: `MultinomialBART`, `MultinomialLNBART`,
#' `ZANIMBARTProbit`, and `ZANIMLNBART`.
#' It also includes the `C++` classes for the regression-based models
#' `ZANIDMReg`, and `DMLinearReg`.
#'
#' `save_model()` serialises the R object to disk while removing the underlying
#' `C++` object (`cpp_obj`) because external pointers cannot be saved.
#' `load_model()` reads the saved .rds file and reconstructs a placeholder `C++` object.
#'
#' @param object An R6 class for the BART-based models object, which contains the
#' field `$cpp_obj` created from an `Rcpp::Module`.
#' @param model_dir Character. Directory path to save or load the model.
#' @param file_name Character. Name of the `.rds` file. Defaults to `"mod.rds"`.
#'
#' @return
#' - `save_model()`: Invisibly returns the full path to the saved `.rds` file.
#' - `load_model()`: Returns the model object with the `$cpp_obj` `C++` pointer reinitialised.
#'
#' @name save_load
NULL

#' @rdname save_load
#' @export
save_model <- function(object, model_dir, file_name = "mod.rds") {
  if (!dir.exists(model_dir)) {
    dir.create(model_dir, recursive = TRUE)
  }
  obj <- object
  obj$cpp_obj <- NULL
  saveRDS(obj, file.path(model_dir, file_name))
  invisible(file.path(model_dir, file_name))
}

#' @importFrom Rcpp "Module"
#' @rdname save_load
#' @export
load_model <- function(model_dir, file_name = "mod.rds") {
  if (!file.exists(file.path(model_dir, file_name))) {
    stop("Object doesn't exist, check the directory and file name.")
  }
  obj <- readRDS(file.path(model_dir, file_name))
  ml <- Rcpp::Module(obj$cpp_module_name, PACKAGE = "zanicc")
  obj$cpp_obj <- switch(obj$cpp_module_name,
    "multinomial_bart" = methods::new(ml$MultinomialBART, matrix(0, 1, obj$d), matrix(0, 1, obj$p)),
    "multinomial_ln_bart" = methods::new(ml$MultinomialLNBART, matrix(0, 1, obj$d), matrix(0, 1, obj$p)),
    "zanim_bart_probit" = methods::new(ml$ZANIMBARTProbit, matrix(0, 1, obj$d), matrix(0, 1, obj$p_theta), matrix(0, 1, obj$p_zeta)),
    "zanim_ln_bart" = methods::new(ml$ZANIMLNBART, matrix(0, 1, obj$d), matrix(0, 1, obj$p_theta), matrix(0, 1, obj$p_zeta)),
    "zanidm_linear_reg" = methods::new(ml$ZANIDMReg, matrix(0, 1, obj$d), matrix(0, 1, obj$p_alpha), matrix(0, 1, obj$p_zeta)),
    "dm_linear_reg" = methods::new(ml$DMLinearReg, matrix(0, 1, obj$d), matrix(0, 1, obj$p))
  )
  obj
}
