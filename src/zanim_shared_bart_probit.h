#ifndef ZANIMSHAREDPROBIT_H
#define ZANIMSHAREDPROBIT_H

#include <RcppArmadillo.h>
#include "node.h"
#include "probit.h"
#include "multinomial_shared_bart.h"

class ZANIMSharedBARTProbit {
public:
  ZANIMSharedBARTProbit(const arma::umat &Y, const arma::mat &X_theta,
                       const arma::mat &X_zeta);
  ~ZANIMSharedBARTProbit();

  // Fields
  arma::umat Y;
  arma::mat X_theta, X_zeta;
  int n, d, p_theta, p_zeta;

  // Multinomial class
  MultinomialSharedBART *bart_mult;

  // List of SharedLogitBART class
  std::vector<ProbitBART*> list_bart_zi;

  // Method to set-up the model
  void SetMCMC(double v0_theta, double k_zeta,
               int ntrees_theta_, int ntrees_zeta_, int ndpost_,
               int nskip_,
               int numcut, double power, double base,
               std::vector<double> proposals_prob_,
               int update_sd_prior_, double s2_0_, double w_ss_,
               std::vector<double> splitprobs_zi,
               std::vector<double> splitprobs_mult_,
               int sparse_zi_, int sparse_mult_,
               std::vector<double> sparse_parms_zi_,
               std::vector<double> sparse_parms_mult_,
               std::vector<double> alpha_sparse_zi_,
               double alpha_sparse_mult_,
               int alpha_random_zi_,
               int alpha_random_mult_,
               arma::mat xinfo, std::string path_out_,
               int keep_draws_, int save_trees_);
  int ntrees_theta, ntrees_zeta, ndpost, nskip, sparse_mult, alpha_random_mult,
    sparse_zi, alpha_random_zi, update_sd_prior, keep_draws, save_trees;
  std::string path_out;

  std::vector<double> proposals_prob, sparse_parms_zi, sparse_parms_mult, alpha_sparse_zi, sigma_mult_mcmc;

  // Common concentration prior parameter \alpha for the Dirichlet sparse prior
  int k_grid;
  std::vector<double> alphas_grid_mult, alphas_grid_zi, lp_alpha_sparse_mult, lp_alpha_sparse_zi;
  double alpha_sparse_mult;

  // Vector and matrices for multinomial BART
  arma::vec rt;
  arma::mat fit_h, f_mu, f_lambda, varthetas;

  // Vector and matrices for ZI probit BART
  arma::vec fit_0_h;
  arma::mat f0_mu;

  // Vector with zero indices and the latent variable z_{ij} indicator for
  // sampling or structural zero
  std::vector<std::vector<int>> zero_indices, zs;
  arma::uvec sum_col_zeros_Y;

  // Wrapper functions to perform the back-fit and update the latent variables
  int move, cur_indice;
  double zeta_ij, p_ij;
  void BackFitMultinomial(int &t);
  void BackFitZI(int &j, int &t);
  void UpdateLatentVariables();

  void RunMCMC();
  arma::cube draws_theta, draws_zeta, draws_vartheta;
  arma::mat draws_phi;

  // Storage object for number of leaves
  arma::uvec avg_leaves_theta;
  arma::umat avg_leaves_zeta;

  // Storage object for acceptance ratio
  //arma::uvec ar_grow_theta, ar_prune_theta, ar_change_theta;
  arma::uvec accept_rate_theta;
  arma::umat accept_rate_zeta;
  //ar_grow_zeta, ar_prune_zeta, ar_change_zeta;

  // Predict functions
  arma::vec GetMu(Node *tree, const arma::rowvec &x);
  void ComputePredictProb(arma::mat &X_, int n_samples, int ntrees,
                          std::string path_out, std::string path, int verbose);
  void ComputePredictProbZero(arma::mat &X_, int n_samples, int ntrees,
                              std::string path_out, std::string path, int verbose);

  void ComputeVarCount(Node *tree, arma::uvec &varcount);
  arma::ucube GetVarCount(int n_samples, int ntrees, std::string parm_name,
                          std::string path);
  // varcount
  arma::umat varcount_mcmc_theta;
  arma::ucube varcount_mcmc_zeta;

  // Compute the predictive density p(y* \mid x*, \theta^{(t)}) for given data
  // (y*, x*) over the MCMC posterior samples of \theta^{(t)}.
  std::vector<double> LogPredictiveDensity(std::vector<int> &y, arma::rowvec &x,
                                           int n_samples, int ntrees_theta,
                                           int ntrees_zeta, std::string path);
  // Compute the E_[p(y* \mid x*, \theta)], where the expectation is w.r.t. the
  // posterior distribution of \theta.
  double LogAvgPredictiveDensity(std::vector<int> &y, arma::rowvec &x, int n_samples,
                                 int ntrees_theta, int ntrees_zeta,
                                 std::string path);

  std::vector<double> GetNormaliseProbsIS(std::vector<int> &y,
                                          arma::vec &x_prior, int n_grid,
                                          int n_samples, int ntrees_theta,
                                          int ntrees_zeta, std::string path);

  std::vector<double> LogPredictiveDensitySeq(std::vector<int> &y,
                                              arma::mat X, int n_samples,
                                              int ntrees_theta, int ntrees_zeta,
                                              std::string path);

  void GetMCMCPrediction(arma::rowvec &x, std::vector<double> &theta,
                         std::vector<double> &zeta, int &d, int &ntrees_theta,
                         int &ntrees_zeta, int &np_zeta,
                         std::ifstream &file_theta,
                         std::vector<std::ifstream> &files_zeta);

  void GetMCMCPredictionLoaded(arma::rowvec &x,
                               std::vector<double> &theta, std::vector<double> &zeta,
                               int &d, int &ntrees_theta, int &ntrees_zeta,
                               const std::vector<Node*> &forest_theta,
                               const std::vector<std::vector<Node*>> &forest_zeta);

};

#endif
