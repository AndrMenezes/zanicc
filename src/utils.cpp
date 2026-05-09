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

// Function adapted from BayesComposition::makeCRPS, to approximate the CRPS
// from samples of the distribution function F (samples).

//[[Rcpp::export]]
arma::vec compute_crps(const arma::mat &samples,
                       const arma::vec &truth,
                       const int &n_samps) {
  int n = truth.n_elem;
  arma::vec crps(n, arma::fill::zeros);
  for (int i=0; i<n; i++){
    double accuracy = 0.0;
    double precision = 0.0;
    for (int k=0; k<n_samps; k++){
      // Rcpp::checkUserInterrupt();
      accuracy += std::abs(samples(k, i) - truth(i));
      for (int j = 0; j < n_samps; j++){
        precision += std::abs(samples(k, i) - samples(j, i));
      }
    }
    crps(i) = 1.0 / n_samps * accuracy - 0.5 / pow(n_samps, 2) * precision;
  }
  return(crps);
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
  for (int j = 0; j < p;j++) out[j] = a * x[j] + b * y[j];
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



