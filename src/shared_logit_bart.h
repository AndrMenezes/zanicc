#ifndef SHAREDLOGITBART_H
#define SHAREDLOGITBART_H

#include <RcppArmadillo.h>
#include "node.h"
#include "rng.h"

struct PriorSharedLogit {
  double s0, c_shape, d_rate, c_logd_lgamc;
  PriorSharedLogit(double s0): s0(s0) {
    c_shape = trigamma_inverse(s0*s0);
    d_rate = exp(R::digamma(c_shape));
    c_logd_lgamc = c_shape * log(d_rate) - R::lgammafn(c_shape);
  }
};

class SharedLogitBART {
public:
  SharedLogitBART(const arma::uvec &y, const arma::mat &X);
  ~SharedLogitBART();

  // Data attributes
  arma::uvec y;
  arma::mat X;
  int n, p;

  // log-marginal likelihood
  double lml(Node *tree);
  // Update sufficient statistics
  void UpdateSuffStats(Node *leaf);
  void UpdateTree(Node *tree);
  // Draw posterior parameter
  void DrawPosterior(Node *tree);
  // Setting MCMC
  void SetMCMC(double v0, int ntrees_, int ndpost_, int nskip_, int printevery_,
               int numcut_, double power_ , double base_,
               std::vector<double> proposals_prob_,
               std::vector<double> splitprobs_, int sparse_,
               std::vector<double> sparse_parms_, double alpha_sparse_,
               int alpha_random_, arma::mat xinfo, std::string path_out_);
  // SetMCMC attributes
  PriorSharedLogit *prior;
  std::vector<Node*> trees;

  arma::mat g0_trees, g1_trees;
  arma::mat x_breaks;
  int ntrees, numcut, ndpost, nskip, printevery, sparse, k_grid, alpha_random;
  double power, base, logprob_grow, logprob_prune, alpha_sparse;
  std::vector<double> proposals_prob, sparse_parms, splitprobs;
  // Path to save the trees
  std::string path_out;

  // Common concentration prior parameter \alpha for the Dirichlet sparse prior
  std::vector<double> alphas_grid, log_posterior_alpha_sparse;

  // flag variables to track the acceptance rate of grow, prune and change
  int flag_grow, flag_prune, flag_change;

  // Run MCMC
  void RunMCMC();
  arma::mat draws;
  // Storage object for number of leaves, depth and acceptance ratio
  arma::uvec avg_leaves, avg_depth;
  // arma::umat ar_grow, ar_prune, ar_change;
  // Latent variable \phi and the "fit" but except the tree h times \phi
  arma::vec phi, fit_0_h_phi, fit_1_h_phi;

  arma::vec GetMu(Node *tree, const arma::rowvec &x);
  arma::mat Predict(arma::mat &X_, int n_samples, int ntrees, std::string path_out);
  // arma::vec PredictZeta();

  arma::uvec GetVarCount();
  void ComputeVarCount(Node *tree, arma::uvec &varcount);
  // arma::uvec varcount;
  arma::umat varcount_mcmc;

};


#endif
