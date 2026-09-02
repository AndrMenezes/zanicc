test_that("multiplication works", {
  n <- 100L
  p <- 10L
  X <- matrix(rnorm(n = n * p), ncol = p)
  system.time(V <- geometry::delaunayn(X, options = "Qt"))
  dim(V)

  Xu <- runifconvexhull(n = 10, X = X)
})
