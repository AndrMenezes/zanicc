#ifndef MULTINOMIALSHAREDBART_H
#define MULTINOMIALSHAREDBART_H

#include <RcppArmadillo.h>
#include "node.h"
#include "rng.h"

class MultinomialSharedBART {
public:
  MultinomialSharedBART(const arma::umat &Y, const arma::mat &X);
  ~MultinomialSharedBART();

  // Data attributes
  arma::umat Y;
  arma::uvec n_trials;
  arma::mat X;
  int n, d, p;

  // log-marginal likelihood
  double lml(Node *leaf);
  // Update sufficient statistics
  void UpdateSuffStats(Node *leaf);
  void UpdateTree(Node *tree);
  // Draw posterior parameter
  void DrawPosterior(Node *tree);

  // Setting MCMC
  void SetMCMC(double v0, int ntrees_, int ndpost_, int nskip_,
               int numcut_, double power_ , double base_,
               std::vector<double> proposals_prob_,
               int update_sd_prior_, double s2_0_, double w_ss_,
               std::vector<double> list_splitprobs_, int sparse_,
               std::vector<double> sparse_parms_, double alpha_sparse_,
               int alpha_random_, arma::mat xinfo, std::string forests_dir_,
               int keep_draws_, int save_trees_);

  // SetMCMC attributes

  // Prior parameters for the leaf node
  double c_shape, d_rate, c_logd_lgamc, sigma, s2_0, w_ss;
  int update_sd_prior, save_trees, keep_draws;
  std::vector<double> sigma_mcmc;

  // List of trees
  std::vector<Node*> trees;

  arma::cube g_trees;
  arma::mat x_breaks;
  int ntrees, numcut, ndpost, nskip, sparse, k_grid, alpha_random;
  double power, base, logprob_grow, logprob_prune, alpha_sparse;
  std::vector<double> proposals_prob, sparse_parms;

  // Path to save the trees
  std::string forests_dir;
  std::vector<double> splitprobs;

  // Common concentration prior parameter \alpha for the Dirichlet sparse prior
  std::vector<double> alphas_grid, log_posterior_alpha_sparse;

  // flag variables to track the acceptance rate of grow, prune and change
  int flag_grow, flag_prune, flag_change;

  // Run MCMC
  void RunMCMC();
  arma::cube draws;
  // Storage object for number of leaves, depth and acceptance ratio
  arma::uvec avg_leaves, avg_depth, accept_rate;
  // Latent variable \phi and the "fit" but except the tree h times \phi
  arma::vec phi;
  arma::mat fit_h_phi;

  // Number of times that variable is used in a tree decision rule (over all trees).
  void ComputeVarCount(Node *tree, arma::uvec &varcount);
  arma::uvec GetVarCount();
  // arma::ucube GetVarCount2(int p, int d, int n_samples, int ntrees,
  //                          std::string forests_dir);
  arma::umat varcount_mcmc;

  // Predictions for a new data set
  arma::cube Predict(arma::mat &X_, int d, int n_samples, int ntrees,
                     std::string forests_dir);
  arma::vec GetMu(Node *tree, const arma::rowvec &x);
  //arma::vec PredictMu(int j);

  double UpdateSigmaPrior();


};

#endif
