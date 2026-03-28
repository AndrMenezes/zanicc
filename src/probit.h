#ifndef PROBIT_H
#define PROBIT_H

#include <RcppArmadillo.h>
#include <cmath>
#include "node.h"

struct PriorProbit {
  double tau_mu;
  double log_tau_mu;
  PriorProbit(int ntrees, double k) {
    double sigma_mu = 3.0 / (k * std::sqrt(ntrees));
    tau_mu = 1.0 / (sigma_mu * sigma_mu);
    log_tau_mu = std::log(tau_mu);
  }
};

class ProbitBART {

public:
  // Constructor
  ProbitBART(const arma::uvec &y, const arma::mat &X);
  // Destructor
  ~ProbitBART();

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
  void SetMCMC(double k, int ntrees_, int ndpost_, int nskip_, int printevery_,
               int numcut_, double power_ , double base_,
               std::vector<double> proposals_prob_,
               std::vector<double> splitprobs_, int sparse_,
               std::vector<double> sparse_parms_, double alpha_sparse_,
               int alpha_random_, arma::mat xinfo, std::string path_out_);

  PriorProbit *prior;
  std::vector<Node*> trees;

  // General fields/attributes
  arma::vec z; // latent variable
  arma::mat g_trees;
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

  void RunMCMC();
  // Run MCMC attributes
  arma::mat draws;
  // Storage object for number of leaves, depth and acceptance ratio
  arma::uvec avg_leaves, avg_depth;
  arma::uvec ar_grow, ar_prune, ar_change;
  // Fit but except the tree h and the partial residuals
  arma::vec fit_h, z_h;

  // Predict
  double GetMu(Node *tree, const arma::rowvec &x);
  arma::mat Predict(arma::mat &X_, int n_samples, int ntrees, std::string path_out);

  // Varcount
  arma::uvec GetVarCount();
  void ComputeVarCount(Node *tree, arma::uvec &varcount);
  // arma::uvec varcount;
  arma::umat varcount_mcmc;

// private:

};

#endif
