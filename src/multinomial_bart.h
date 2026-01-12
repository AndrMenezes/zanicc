#ifndef MULTINOMIALBART_H
#define MULTINOMIALBART_H

#include <RcppArmadillo.h>
#include "node.h"
#include "rng.h"


class MultinomialBART {
public:
  MultinomialBART(const arma::umat &Y, const arma::mat &X);
  ~MultinomialBART();

  // Data attributes
  arma::umat Y;
  arma::uvec n_trials;
  arma::mat X;
  int n, d, p;

  // Iterator for the categories
  int j_cat;

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
               std::vector<std::vector<double>> list_splitprobs_, int sparse_,
               std::vector<double> sparse_parms_, std::vector<double> alpha_sparse_,
               int alpha_random_, arma::mat xinfo, std::string path_out_);

  // SetMCMC attributes

  // Prior parameters for the leaf node
  double c_shape, d_rate, c_logd_lgamc, sigma, s2_0, w_ss;
  int update_sd_prior;
  std::vector<double> sigma_mcmc;

  // List of trees
  std::vector<std::vector<Node*>> trees;

  arma::cube g_trees;
  arma::mat x_breaks;
  int ntrees, numcut, ndpost, nskip, sparse, k_grid, alpha_random;
  double power, base, logprob_grow, logprob_prune;
  std::vector<double> proposals_prob, sparse_parms, alpha_sparse;

  // Path to save the trees
  std::string path_out;
  // List with split probs for each category!
  std::vector<std::vector<double>> list_splitprobs;
  std::vector<double> splitprobs;

  // Common concentration prior parameter \alpha for the Dirichlet sparse prior
  std::vector<double> alphas_grid, log_posterior_alpha_sparse;

  // flag variables to track the acceptance rate of grow, prune and change
  int flag_grow, flag_prune, flag_change;

  // Run MCMC
  void RunMCMC();
  arma::cube draws;
  arma::mat draws_phi;
  // Storage object for number of leaves, depth and acceptance ratio
  arma::umat avg_leaves, avg_depth, accept_rate;
  // arma::umat ar_grow, ar_prune, ar_change;
  // Latent variable \phi and the "fit" but except the tree h times \phi
  arma::vec phi, fit_h_phi;


  // Number of times that variable is used in a tree decision rule (over all trees).
  void ComputeVarCount(Node *tree, arma::uvec &varcount);
  arma::uvec GetVarCount(int j);
  arma::ucube GetVarCount2(int p, int d, int n_samples, int ntrees,
                           std::string path_out);
  arma::ucube varcount_mcmc;

  // Predictions for a new data set
  arma::cube Predict(arma::mat &X_, int d, int n_samples, int ntrees,
                     std::string path_out);
  double GetMu(Node *tree, const arma::rowvec &x);
  //arma::vec PredictMu(int j);

  double UpdateSigmaPrior();


};

#endif
