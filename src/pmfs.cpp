#include <vector>
#include <algorithm>
#include <cmath>
#include "utils.h"

// Log PMF of multinomial
inline double log_pmf_mult(std::vector<int> &x, int &size,
                           std::vector<double> &prob) {
  double r = std::lgamma(size + 1);
  for(int j = 0; j < x.size(); j++) {
    r += x[j] * log(prob[j]) - std::lgamma(x[j] + 1);
  }
  return(r);
}


// Implement log_dzanim
// [[Rcpp::export]]
double log_pmf_zanim(std::vector<int> x, std::vector<double> prob,
                     std::vector<double> zeta) {

  int total_zeros = std::count(x.begin(), x.end(), 0);
  int d = x.size();
  int n_trials = std::accumulate(x.begin(), x.end(), 0.0);
  std::vector<double> logzeta(d);
  std::vector<double> log1mzeta(d);
  for (int j = 0; j < d; j ++) {
    logzeta[j] = std::log(zeta[j]);
    log1mzeta[j] = std::log1p(-zeta[j]);
  }
  if (total_zeros == d)
    return(std::accumulate(logzeta.begin(), logzeta.end(), 0.0));
  // Temporary vector to keep the contribution of each mixture component
  std::vector<double> tmp(pow(2, total_zeros));
  // Full multinomial contribution
  tmp[0] = std::accumulate(log1mzeta.begin(), log1mzeta.end(), 0.0);
  tmp[0] += log_pmf_mult(x, n_trials, prob);
  if (total_zeros > 0) {
    // Create a vector with indices for elements equal to zero
    std::vector<int> indices_zeros(total_zeros);
    int idx = 0;
    for (int i = 0; i < d; ++i) {
      if (x[i] == 0) indices_zeros[idx++] = i;
    }
    // Auxiliary variables
    double eta_K;
    double sum_prob;
    int id_zero;

    // Auxiliary Indices
    int idt = 1;
    int idk = 0;
    // Iterate over the size (total_zeros) of reduced multinomial
    for (int m = 1; m <= total_zeros; m++) {
      std::vector<int> choose(m);
      std::vector<int> cur_indices_z(m);
      std::vector<double> prob_K(d - m);
      std::vector<int> x_K(d - m);

      // Lexicographic algorithm in Knuth vol 4a book.
      // Initialize c array
      std::vector<int> c(m + 3);
      for (int j = 1; j <= m; j++) c[j] = j - 1;
      c[m + 1] = total_zeros;
      c[m + 2] = 0L;

      // Generate combinations in lexicographical order
      while (true) {
        eta_K = 0.0;
        sum_prob = 0.0;
        // Visit the combination
        for (int i = 1; i <= m; ++i) {
          id_zero = indices_zeros[c[i]];
          cur_indices_z[i - 1] = id_zero;
          eta_K += logzeta[id_zero];
          sum_prob += prob[id_zero];
          //std::cout << id_zero << " ";
        }
        //std::cout << "\n";
        idk = 0;
        for (int k = 0; k < d; k++) {
          if (std::find(cur_indices_z.begin(), cur_indices_z.end(), k) == cur_indices_z.end()) {
            eta_K += log1mzeta[k];
            prob_K[idk] = prob[k] / (1 - sum_prob);
            x_K[idk] = x[k];
            ++idk;
          }
        }
        // Save the contribution of the current reduced multinomial
        tmp[idt++] = eta_K + log_pmf_mult(x_K, n_trials, prob_K);

        // Find the rightmost index that can be incremented
        int j = 1;
        while (c[j] + 1 == c[j + 1]) {
          c[j] = j - 1;
          ++j;
        }
        // If j exceeds m, leave the loop
        if (j > m) break;
        // Increment
        ++c[j];
      }
    }
  }
  return(log_sum_exp(tmp));
}

// Vectorize version of log_pmf_zanim, meaning that x, prob and zeta should have n*d
// length.
// [[Rcpp::export]]
std::vector<double> log_pmf_zanim_vec(int n, int d,
                                      std::vector<int> x,
                                      std::vector<double> prob,
                                      std::vector<double> zeta) {
  int last = n * d;
  std::vector<double> ll(n, 0.0);
  std::vector<int> x_cur(d, 0.0);
  std::vector<double> prob_cur(d, 0.0);
  std::vector<double> zeta_cur(d, 0.0);
  for (int i = 0; i < n; i++) {
    // int m = std::floor((last - i) / n);
    // int k = 0;
    for (int j=0; j < d; j++) {
      int idx = i + j * n;
      x_cur[j] = x[idx];
      prob_cur[j] = prob[idx];
      zeta_cur[j] = zeta[idx];
      // k++;
    }
    ll[i] = log_pmf_zanim(x_cur, prob_cur, zeta_cur);
  }
  return ll;
}

// Log-pmf of "ZI-Poisson"
// //# Rcpp::export(".log_pmf_zip")
double log_pmf_zip(int &x, double &zeta, double &lambda, double &phi) {
  std::vector<double> t1(2, 0.0);
  t1[0] = log(1 - zeta) - lambda * phi;
  if (x > 0) {
    return(t1[0] + x * log(lambda) - std::lgamma( (double) x + 1.0));
  } else {
    t1[1] = log(zeta);
    return(log_sum_exp(t1));
  }
}

// Approximation of log-likelihood averaging over the augmented likelihood
// [[Rcpp::export(".log_pmf_zanim_approx")]]
double log_pmf_zanim_approx(std::vector<int> x, std::vector<double> prob,
                            std::vector<double> zeta, double scale, int mc,
                            int nskip) {

  int d = x.size();
  double n_trials = std::accumulate(x.begin(), x.end(), 0.0);
  double n_trials_m1 = n_trials - 1;
  std::vector<double> lambda(d, 0.0);
  double s_lambda = 0.0;
  for (int j = 0; j < d; j++) {
    lambda[j] = prob[j] * scale;
    s_lambda += lambda[j];
  }
  double phi = R::rgamma(n_trials, 1.0 / s_lambda);
  double z = 1;
  for (int i=0; i < nskip; i++) {
    s_lambda = 0.0;
    for (int j=0; j < d; j++) {
      // Update the latent variables z_j ~ Bernoulli[p] if x_j = 0;
      z = 1;
      if (x[j] == 0) {
        double p = (1.0 - zeta[j]) * exp(-phi * lambda[j]);
        p = p / (zeta[j] + p);
        z = R::rbinom(1, p);
      }
      s_lambda += lambda[j] * z;
    }
    // Update \phi
    phi = R::rgamma(n_trials, 1.0 / s_lambda) ;
  }
  double ll = 0.0;
  std::vector<double> loglike(mc, 0.0);
  for (int i = 0; i < mc; i++) {
    s_lambda = 0.0;
    ll = n_trials_m1 * log(phi);
    for (int j=0; j < d; j++) {
      // Update the latent variables z_j ~ Bernoulli[p] if x_j = 0;
      z = 1;
      if (x[j] == 0) {
        double p = (1.0 - zeta[j]) * exp(-phi * lambda[j]);
        p = p / (zeta[j] + p);
        z = R::rbinom(1, p);
      }
      s_lambda += lambda[j] * z;
      // Increment the log-likelihood
      ll += log_pmf_zip(x[j], zeta[j], lambda[j], phi);
    }
    // Save log-likelihood
    loglike[i] = ll;
    // Update \phi
    phi = R::rgamma(n_trials, 1.0 / s_lambda) ;
  }
  // double l_max = *std::max_element(loglike.begin(), loglike.end());
  // double sum = 0.0;
  // for (double xi : loglike) {
  //   sum += std::exp(xi - l_max);
  // }
  // return(l_max + std::log(sum) - std::log(mc) + log(n_trials));
  return log_sum_exp(loglike) - std::log(mc) + log(n_trials);
}

// Log PMF of Dirichlet-multinomial
inline double log_pmf_dm(std::vector<int> &x, int &size,
                         std::vector<double> &alpha) {
  double a0 = std::accumulate(alpha.begin(), alpha.end(), 0.0);
  double r = std::lgamma(a0) + std::lgamma(size + 1) - std::lgamma(size + a0);
  for(int j = 0; j < x.size(); j++) {
    r += lgamma(x[j] + alpha[j]) - lgamma(alpha[j]) - lgamma(x[j] + 1L);
  }
  return(r);
}

// Implement log_dzanidm
// [[Rcpp::export]]
double log_pmf_zanidm(std::vector<int> x, std::vector<double> alpha,
                      std::vector<double> zeta) {

  int total_zeros = std::count(x.begin(), x.end(), 0);
  int d = x.size();
  int n_trials = std::accumulate(x.begin(), x.end(), 0.0);
  std::vector<double> logzeta(d);
  std::vector<double> log1mzeta(d);
  for (int j = 0; j < d; j ++) {
    logzeta[j] = std::log(zeta[j]);
    log1mzeta[j] = std::log1p(-zeta[j]);
  }
  if (total_zeros == d)
    return(std::accumulate(logzeta.begin(), logzeta.end(), 0.0));
  // Temporary vector to keep the contribution of each mixture component
  std::vector<double> tmp(pow(2, total_zeros));
  // Full multinomial contribution
  tmp[0] = std::accumulate(log1mzeta.begin(), log1mzeta.end(), 0.0);
  tmp[0] += log_pmf_dm(x, n_trials, alpha);
  if (total_zeros > 0) {
    // Create a vector with indices for elements equal to zero
    std::vector<int> indices_zeros(total_zeros);
    int idx = 0L;
    for (int i = 0; i < d; ++i) {
      if (x[i] == 0L) indices_zeros[idx++] = i;
    }
    // Auxiliary variables
    double eta_K;
    int id_zero;

    // Auxiliary Indices
    int idt = 1L;
    int idk = 0L;
    // Iterate over the size (total_zeros) of reduced multinomial
    for (int m = 1; m <= total_zeros; m++) {
      std::vector<int> choose(m);
      std::vector<int> cur_indices_z(m);
      std::vector<double> alpha_K(d - m);
      std::vector<int> x_K(d - m);

      // Lexicographic algorithm in Knuth vol 4a book.
      // Initialize c array
      std::vector<int> c(m + 3);
      for (int j = 1; j <= m; j++) c[j] = j - 1;
      c[m + 1] = total_zeros;
      c[m + 2] = 0L;

      // Generate combinations in lexicographical order
      while (true) {
        eta_K = 0.0;
        // Visit the combination
        for (int i = 1; i <= m; ++i) {
          id_zero = indices_zeros[c[i]];
          cur_indices_z[i - 1L] = id_zero;
          eta_K += logzeta[id_zero];
          //std::cout << id_zero << " ";
        }
        //std::cout << "\n";
        idk = 0L;
        for (int k = 0; k < d; k++) {
          if (std::find(cur_indices_z.begin(), cur_indices_z.end(), k) == cur_indices_z.end()) {
            eta_K += log1mzeta[k];
            alpha_K[idk] = alpha[k];
            x_K[idk] = x[k];
            ++idk;
          }
        }
        // Save the contribution of the current reduced multinomial
        tmp[idt++] = eta_K + log_pmf_dm(x_K, n_trials, alpha_K);

        // Find the rightmost index that can be incremented
        int j = 1L;
        while (c[j] + 1 == c[j + 1]) {
          c[j] = j - 1;
          ++j;
        }
        // If j exceeds m, leave the loop
        if (j > m) break;
        // Increment
        ++c[j];
      }
    }
  }
  return(log_sum_exp(tmp));
}
