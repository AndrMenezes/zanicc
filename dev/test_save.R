rm(list = ls()); gc()
Rcpp::sourceCpp(file = "./dev/save.cpp")
N = 10; D = 4; M = 10L

file.remove("out.bin")
init <- proc.time()
x <- save_bin(n = N, d = D, m = M, fname = "out.bin")
end <- proc.time() - init
end[3]/60
load_bin <- function(fname, n, d, m) {
  con <- file(fname, "rb")
  on.exit(close(con))
  data <- readBin(con, what = "double", n = n * d * m)
  array(data, dim = c(n, d, m))
}
x1 <- load_bin(fname = "out.bin", n = N, d = D, m = M)
all.equal(x1, x)
x2 <- load_bin(fname = "out.bin", n = N, d = D, m = 2L)
all.equal(x2, x[,,1:2])


load_bin_batch <- function(fname, n, d, k, m) {
  con <- file(fname, "rb")
  on.exit(close(con))
  # 8 bytes per double
  offset <- (k - 1) * n * d * 8
  #  re-position the connections and read the binary data
  seek(con, where = offset, origin = "start", rw = "read")
  data <- readBin(con, what = "double", n = n * d * m)
  array(data, dim = c(n, d, m))
}

x3 <- load_bin_batch(fname = "out.bin", n = N, d = D, k = 2, m = 4)
dim(x3)

all.equal(x3[,,1L], x[,,2L])
all.equal(x3[,,2L], x[,,3L])
all.equal(x3[,,3L], x[,,4L])
all.equal(x3[,,4L], x[,,5L])


##
d = D
n = N
k = 2L
fname = "out.bin"
con <- file(fname, "rb")
offset <- (k - 1) * n * d * 8  # 8 bytes per double
seek(con, where = offset, origin = "start", rw = "read")
data <- readBin(con, what = "double", n = n * d, endian = "little")
matrix(data, nrow = n, ncol = d)
close(con)

x1[,,2]
