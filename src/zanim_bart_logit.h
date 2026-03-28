#ifndef ZANIMBARTLOGIT_H
#define ZANIMBARTLOGIT_H

#include <RcppArmadillo.h>
#include "node.h"
#include "shared_logit_bart.h"
#include "multinomial_bart.h"

class ZANIMBARTLogit {
public:
  ZANIMBARTLogit(const arma::umat &Y, const arma::mat &X_theta,
                       const arma::mat &X_zeta);
  ~ZANIMBARTLogit();

  // Fields
  arma::umat Y;
  arma::mat X_theta, X_zeta;
  int n, d, p_theta, p_zeta;

  // Multinomial class
  MultinomialBART *bart_mult;

  // List of SharedLogitBART class
  std::vector<SharedLogitBART*> list_bart_zi;

  // Method to set-up the model
  void SetMCMC(double v0_theta, double v0_zeta,
               int ntrees_theta_, int ntrees_zeta_, int ndpost_,
               int nskip_,
               int numcut, double power, double base,
               std::vector<double> proposals_prob_,
               int update_sd_prior_, double s2_0_, double w_ss_,
               std::vector<double> splitprobs_zi,
               std::vector<std::vector<double>> splitprobs_mult,
               int sparse_zi_,
               int sparse_mult_,
               std::vector<double> sparse_parms_zi_,
               std::vector<double> sparse_parms_mult_,
               std::vector<double> alpha_sparse_zi_,
               std::vector<double> alpha_sparse_mult_,
               int alpha_random_zi_,
               int alpha_random_mult_,
               arma::mat xinfo, std::string path_out_, int keep_draws_,
               int save_trees_);
  int ntrees_theta, ntrees_zeta, ndpost, nskip, sparse_mult, alpha_random_mult,
   sparse_zi, alpha_random_zi, update_sd_prior, keep_draws, save_trees;
  std::string path_out;
  std::vector<double> proposals_prob, sparse_parms_zi, sparse_parms_mult,
   alpha_sparse_zi, alpha_sparse_mult, sigma_mult_mcmc;
  std::vector<std::vector<double>> list_splitprobs_mult;

  // Common concentration prior parameter \alpha for the Dirichlet sparse prior
  int k_grid;
  std::vector<double> alphas_grid_mult, alphas_grid_zi, lp_alpha_sparse_mult, lp_alpha_sparse_zi;

  // Vector and matrices for multinomial BART
  arma::vec rt, fit_h;
  arma::mat f_mu, f_lambda, thetas, varthetas;
  // Vector and matrices for ZI shared logit BART
  arma::vec fit_0_h, fit_1_h;
  arma::mat rt_zi, f0_mu, f1_mu, f1_lambda, zetas;

  // Vector with zero indices and the latent variable z_{ij} indicator for
  // sampling or structural zero
  std::vector<std::vector<int>> zero_indices, zs;
  arma::uvec sum_col_zeros_Y;

  // Wrapper functions to perform the back-fit and update the latent variables
  int move, cur_indice;
  double zeta_ij, p_ij;
  void BackFitMultinomial(int &j, int &t);
  void BackFitZI(int &j, int &t);
  void UpdateLatentVariables();

  void RunMCMC();
  arma::cube draws_theta, draws_zeta, draws_vartheta;
  arma::mat draws_phi;

  // Storage object for number of leaves
  arma::umat avg_leaves_theta, avg_leaves_zeta;

  // Storage object for acceptance ratio
  arma::umat accept_rate_theta, accept_rate_zeta;

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
  arma::ucube varcount_mcmc_theta;
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

  // ESS step to obtain one sample form the unobserved covariate
  arma::rowvec InversePosteriorESS(std::vector<int> &y, arma::rowvec &x_cur,
                                   double mean_prior, double sd_prior,
                                   int n_samples, int ntrees_theta,
                                   int ntrees_zeta, std::string path);

  arma::mat SampleInversePosterior(std::vector<int> &y, arma::rowvec &x_init,
                                   double mean_prior, double sd_prior,
                                   int ndpost, int nskip, int printevery,
                                   int n_samples, int ntrees_theta,
                                   int ntrees_zeta, std::string path);
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
                         int &ntrees_zeta, int &np_theta, int &np_zeta,
                         std::vector<std::ifstream> &files_theta,
                         std::vector<std::ifstream> &files_zeta);

  void GetMCMCPredictionLoaded(arma::rowvec &x,
                               std::vector<double> &theta, std::vector<double> &zeta,
                               int &d, int &ntrees_theta, int &ntrees_zeta,
                               const std::vector<std::vector<Node*>> &forest_theta,
                               const std::vector<std::vector<Node*>> &forest_zeta);

  arma::vec SampleInversePosteriorSeq(std::vector<int> &y, arma::rowvec x_cur,
                                      double mean_prior, double sd_prior,
                                      int n_rep, int n_samples, int ntrees_theta,
                                      int ntrees_zeta, std::string path);
};

#endif
