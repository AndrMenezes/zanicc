#include "utils.h"


// NR to compute the inverse of trigamma function. Adapted from limma package.
double trigamma_inverse(double x) {
  if (x > 1e7) return(1.0 / sqrt(x));
  if (x < 1e-6) return(1.0 / x);
  double y = 0.5 + 1.0 / x;

  double tri, dif;
  for (int i=0; i < 50; i++) {
    tri = R::trigamma(y);
    dif = tri * (1 - tri / x) / R::tetragamma(y);
    if (-dif / y < 1e-8) return(y);
  }
  return(y);
}

// log(\sum_x exp(x))
double log_sum_exp(std::vector<double> &x) {
  double x_max = *std::max_element(x.begin(), x.end());
  double sum = 0.0;
  for (double xi : x) sum += std::exp(xi - x_max);
  return x_max + std::log(sum);
}

// Convert an integer/double matrix to a row-major double vector
std::vector<int> umat_to_int_rowmajor(const arma::umat &X) {
  int n = X.n_rows, p = X.n_cols;
  std::vector<int> out(n * p);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < p; ++j)
      out[i * p + j] = static_cast<int>(X(i, j));
  return out;
}

// [[Rcpp::export]]
std::vector<double> mat_to_double_rowmajor(const arma::mat &X) {
  int n = X.n_rows, p = X.n_cols;
  std::vector<double> out(n * p);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < p; ++j)
      out[i * p + j] = static_cast<double>(X(i, j));
  return out;
}

// a*x + b*y
void axpby(double* out, double* x, double* y,
           double a, double b, int p) {
  for (int j = 0; j < p; j++) out[j] = a * x[j] + b * y[j];
}

// Transform to u = Bv, iterate over rows first then columns
void Bv(std::vector<double> &u, std::vector<double> &v, std::vector<double> &B,
        int d, int dm1) {
  std::fill(u.begin(), u.end(), 0.0);
  for (int l=0; l < d; l++) {
    for (int j=0; j < dm1; j++) u[l] += v[j] * B[l*dm1 + j];
  }
}

// Normalise log-weights (use in SIR)
std::vector<double> normalise_weights(std::vector<double> &log_weights, int n) {
  double lw_max = *std::max_element(log_weights.begin(), log_weights.end());
  double s = 0.0;
  std::vector<double> w(n, 0.0);
  for (int j=0; j < n; j++) {
    w[j] = std::exp(log_weights[j] - lw_max);
    s += w[j];
  }
  // normalise
  for (int j=0; j < n; j++) w[j] /= s;
  return w;
}

// log-Kernel Gaussian
double log_kernel_gauss(std::vector<double> &x1, std::vector<double> &x2, double h) {
  int d = x1.size();
  double u = 0;
  for (int j=0; j < d; j++)  u += std::pow(x1[j] - x2[j], 2);
  return -0.5 * std::pow(h, -2.0) * u;
  // return R::dnorm4(std::sqrt(u), 0.0, h, 1);
}

// log-Kernel exponential
double log_kernel_exp(std::vector<double> &x1, std::vector<double> &x2, double h) {
  int d = x1.size();
  double u = 0;
  for (int j=0; j < d; j++)  u += std::pow(x1[j] - x2[j], 2);
  return(-u / h);
}

// Centered log-ratio
std::vector<double> clr(const std::vector<int> &y, double pseudo) {

  int d = y.size();
  std::vector<double> logy(d, 0.0);

  double meanlogy = 0.0;
  for (int j=0; j < d; j++) {
    logy[j] = std::log(y[j] + pseudo);
    meanlogy += logy[j];
  }
  meanlogy /= d;
  for (int j=0; j < d; j++) logy[j] -= meanlogy;

  return logy;
}


