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
std::vector<int> umat_to_int_rowmajor(const arma::umat &Y);
std::vector<double> mat_to_double_rowmajor(const arma::mat &X);
void rmvnorm_chol(std::vector<double> &out,
                  const std::vector<double> &mean,
                  const std::vector<double> &L, int p);
void axpby(double* out, double* x, double* y, double a, double b, int p);
#endif
