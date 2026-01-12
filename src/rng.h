#ifndef RNG_H
#define RNG_H

#include <vector>
#include <RcppArmadillo.h>

double trigamma_inverse(double x);
double target_sigma_prior(double &x, int &n, double &sum_log_lambda,
                          double &sum_lambda, double &s2_0);

int sample_discrete(const std::vector<double> &probs, const int &k);
int sample_discrete(const int &k);
double rtnorm(const double &mean, const double &sd, const double &a);
std::vector<double> UpdateSplitProbs(const arma::uvec &m, const double &a,
                                     const int &p);
double UpdateAlphaDirchlet(const std::vector<double> &alphas,
                           const std::vector<double> &log_posterior_alpha,
                           const double &sum_log_splitprobs,
                           const double &p, const int &k);
arma::vec rmvnorm(int p, arma::mat &sigma_chol);

#endif
