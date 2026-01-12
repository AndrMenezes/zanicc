#ifndef UTILS_H
#define UTILS_H

// #include <Rcpp.h>
#include <RcppArmadillo.h>
#include <vector>
#include <cmath>

double trigamma_inverse(double x);
double log_sum_exp(std::vector<double> &x);
arma::vec compute_crps(const arma::mat &samples, const arma::vec &truth,
                       const int &n_samps);

#endif
