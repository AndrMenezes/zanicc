#include "rng.h"
#include "utils.h"
#include <random>
#include <vector>
#include <Rcpp.h>

// Adapted from SoftBART package
int sample_discrete(const std::vector<double> &probs, const int &k) {
  double u = R::unif_rand();
  double cdf = 0.0;
  for(int j = 0; j < k; j++) {
    cdf += probs[j];
    if (u < cdf) return(j);
  }
  return(k - 1);
}

int sample_discrete(const int &k) {
  double u = R::unif_rand();
  double probs = 1.0 / ((double) k);
  double cdf = 0.0;
  for(int j = 0; j < k; j++) {
    cdf += probs;
    if (u < cdf) return(j);
  }
  return(k - 1);
}

// Target to sample from in order to update the sigma prior
double target_sigma_prior(double &x, int &n, double &sum_log_lambda,
                          double &sum_lambda, double &s2_0) {
  double c0 = trigamma_inverse(x * x);
  double d0 = exp(R::digamma(c0));
  return n * c0 * log(d0) - n * lgamma(c0) + c0 * sum_log_lambda - d0 * sum_lambda - 0.5 * x*x / s2_0;
}


// Adapted from BART package
// [[Rcpp::export]]
double rtnorm(const double &mean, const double &sd, const double &a) {

  double ld, z;
  double tau = (a - mean) / sd;

  if (tau <= 0.0) {
    do {z = R::norm_rand();} while (z < tau);
  } else {
    // Optimal exponential rate parameter, see Robert (1995)
    ld = 0.5 * (tau + sqrt(tau*tau + 4.0));
    // Do rejection sampling
    do {
      z = R::exp_rand() / ld + tau;
    } while (R::unif_rand() > exp(-0.5 * pow(z - ld, 2.0)));
  }

  return z * sd + mean;
}

// RGN for Dirichlet_k[a] using the Gamma stochastic representation
// This is used in the sparse Dirichlet prior
std::vector<double> UpdateSplitProbs(const arma::uvec &m, const double &a,
                                     const int &p) {

  std::vector<double> out(p);
  double s = 0.0;
  double a_p = a / ((double) p);
  for (int j = 0; j < p; j++) {
    out[j] = R::rgamma(a_p + m[j], 1.0);
    s += out[j];
  }
  // Normalise
  for (int j = 0; j < p; ++j) {
    out[j] /= s;
  }
  // std::transform(out.begin(), out.end(), out.begin(),
  //                [s](double x) { return x / s; });
  return out;
}

// Update the \alpha parameter of Dirichlet in DART prior
double UpdateAlphaDirchlet(const std::vector<double> &alphas,
                           const std::vector<double> &log_posterior_alpha,
                           const double &sum_log_splitprobs,
                           const double &p, const int &k) {
  std::vector<double> probs(k, 0.0);
  for (int i = 0; i < k; i++) {
    probs[i] = log_posterior_alpha[i] + alphas[i] / p * sum_log_splitprobs;
  }
  double lse = log_sum_exp(probs);
  // Normalising the probabilities
  for (int i = 0; i < k; i++) probs[i] = exp(probs[i] - lse);

  return alphas[sample_discrete(probs, k)];
}

// RNG for multivariate Normal
// TODO: avoid transpose
arma::vec rmvnorm(int p, arma::mat &sigma_chol) {
  arma::rowvec ep = arma::randn<arma::rowvec>(p);
  return (ep * sigma_chol).t();
  // return sigma_chol * ep; // ?
}
