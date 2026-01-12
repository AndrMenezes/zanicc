#include <RcppArmadillo.h>
#include <vector>
#include "rng.h"
#include "probit_bart.h"
#include "tree_mcmc.h"
#include "write_read.h"

ProbitBART::ProbitBART(const arma::uvec &y, const arma::mat &X) : y(y), X(X) {
  n = X.n_rows;
  p = X.n_cols;
}

ProbitBART::~ProbitBART() {
  for (int h = 0; h < ntrees; h++) {
    delete trees[h];
  }
  delete prior;
}

// Log-marginal likelihood
double ProbitBART::lml(Node *tree) {
  double nt = tree->nobs + prior->tau_mu;
  return 0.5 * (std::pow(tree->ss1(0), 2) / nt - std::log(nt) + prior->log_tau_mu);
}

// Update sufficient statistics
void ProbitBART::UpdateSuffStats(Node *leaf) {
  double sum_z = 0.0;
  for (auto id : leaf->ids) sum_z += z_h(id);
  leaf->ss1(0) = sum_z;
}

void ProbitBART::UpdateTree(Node *tree) {
  // Get the leaves
  std::vector<Node*> leaves;
  tree->GetLeaves(leaves);
  for (size_t t = 0; t < leaves.size(); t++) {
    UpdateSuffStats(leaves[t]);
  }
}

// Draw from the posterior over the terminal node parameters
void ProbitBART::DrawPosterior(Node *tree) {
  double den, mu;
  const int h = tree->h;
  // Get the leaves
  std::vector<Node*> leaves;
  tree->GetLeaves(leaves);
  for (Node* leaf : leaves) {
    den = leaf->nobs + prior->tau_mu;
    mu = R::rnorm(leaf->ss1(0) / den, 1.0 / sqrt(den));
    leaf->mu(0) = mu;
    // Save the predictions at g_trees.
    for (auto id : leaf->ids) {
      g_trees(id, h) = mu;
    }
  }
}

// Get the total times a variable appear as a decision rule over ALL trees.
arma::uvec ProbitBART::GetVarCount() {
  arma::uvec varcount = arma::zeros<arma::uvec>(p);
  for (int t = 0; t < ntrees; t++) {
    ComputeVarCount(trees[t], varcount);
  }
  return varcount;
}

void ProbitBART::ComputeVarCount(Node *tree, arma::uvec &varcount) {
  if (tree->is_leaf == 0) {
    int id_j = tree->predictor;
    varcount(id_j)++;
    ComputeVarCount(tree->left, varcount);
    ComputeVarCount(tree->right, varcount);
  }
}

double ProbitBART::GetMu(Node *tree, const arma::rowvec &x) {
  if (tree->is_leaf) return tree->mu(0);
  if (x[tree->predictor] <= tree->cutoff) return GetMu(tree->left, x);
  else return GetMu(tree->right, x);
}

arma::mat ProbitBART::Predict(arma::mat &X_, int n_samples, int ntrees,
                              std::string path_out) {
  int np = 1;
  int n_ = X_.n_rows;
  arma::vec f = arma::zeros<arma::vec>(n_);
  arma::mat draws_pred = arma::zeros<arma::mat>(n_, n_samples);

  std::ifstream is(path_out + "/forests.bin", std::ios::binary);

  for (int h = 0; h < n_samples; h++) {
    f = arma::zeros<arma::vec>(n_);
    // Import a tree
    for (int t = 0; t < ntrees; t++) {
      Node *tree = deserialise_tree(is, np);
      // Do prediction for the imported tree
      for (int i = 0; i < n_; i++) {
        const arma::rowvec &xi = X_.row(i);
        f[i] += GetMu(tree, xi);
      }
    }
    // Save the predictions for the h-th posterior sample
    draws_pred.col(h) = f;
  }
  is.close();
  return draws_pred;
}

void ProbitBART::SetMCMC(double k, int ntrees_, int ndpost_, int nskip_,
                         int printevery_, int numcut_, double power_, double base_,
                         std::vector<double> proposals_prob_,
                         std::vector<double> splitprobs_, int sparse_,
                         std::vector<double> sparse_parms_, double alpha_sparse_,
                         int alpha_random_, arma::mat xinfo, std::string path_out_) {

  // Setting attributes
  ntrees = ntrees_;
  numcut = numcut_;
  ndpost = ndpost_;
  printevery = printevery_;
  nskip = nskip_;
  power = power_;
  base = base_;
  logprob_grow = log(proposals_prob_[0]);
  logprob_prune = log(proposals_prob_[1]);
  splitprobs = splitprobs_;
  sparse = sparse_;
  proposals_prob = proposals_prob_;
  sparse_parms = sparse_parms_;
  alpha_random = alpha_random_;
  alpha_sparse = alpha_sparse_;
  path_out = path_out_;

  // Define the prior
  prior = new PriorProbit(ntrees, k);

  // Setting the cut-points
  if (xinfo.n_rows == 1) {
    x_breaks = arma::zeros<arma::mat>(numcut, p);
    double min_val, max_val;
    for (int j = 0; j < p; j++) {
      min_val = X.col(j).min();
      max_val = X.col(j).max();
      x_breaks.col(j) = arma::linspace(min_val, max_val, numcut);
    }
  } else {
    x_breaks = xinfo;
  }


  // Prediction at the observational level for each tree
  g_trees = arma::zeros<arma::mat>(n, ntrees);

  // First draw for the latent variable z
  z = arma::zeros<arma::vec>(n);
  for (int i = 0; i < n; i++) {
    if (y[i] == 0) z[i] = -rtnorm(0.0, 1.0, 0.0);
    else z[i] = rtnorm(0.0, 1.0, 0.0);
  }

  // Initialize the trees
  trees.resize(0);
  for (int h = 0; h < ntrees; h++) {
    Node *nd = new Node(1);
    // Setting the tree identification
    nd->h = h;
    // Initialise the ids
    nd->ids = arma::regspace<arma::uvec>(0, n - 1);
    nd->nobs = n;
    // Compute the sufficient statistics
    nd->ss1(0) = arma::accu(z);
    trees.push_back(nd);
  }
  fit_h = arma::zeros<arma::vec>(n);
  z_h = arma::zeros<arma::vec>(n);

  // Storage object for number of leaves, depth and acceptance ratio
  avg_leaves = arma::zeros<arma::uvec>(ntrees);
  avg_depth = arma::zeros<arma::uvec>(ntrees);

  // Storage object for acceptance ratio
  ar_grow = arma::zeros<arma::uvec>(ntrees);
  ar_prune = arma::zeros<arma::uvec>(ntrees);
  ar_change = arma::zeros<arma::uvec>(ntrees);

  // Create grid for the \alpha, where \alpha = lambda * \rho / (1 - \lambda)
  // and for each \alpha compute the "constant" part of the log-posterior
  if (alpha_random) {
    k_grid = 1000;
    alphas_grid.resize(k_grid);
    log_posterior_alpha_sparse.resize(k_grid);
    double al, ld;
    for (int k = 0; k < k_grid; k++) {
      ld = (double)(k + 1)/((double)(k_grid + 1));
      al = ld * sparse_parms[0] / (1.0 - ld);
      alphas_grid[k] = al;
      log_posterior_alpha_sparse[k] = (R::lgammafn(al)
                                         - p * R::lgammafn(al / p)
                                         + (sparse_parms[1] - 1.0) * log(ld)
                                         + (sparse_parms[2] - 1.0) * log(1.0 - ld));
    }
  }

  int flag_grow = 0, flag_prune = 0, flag_change = 0;

  std::cout << "Model set up!" << "\n\n";
}

// Method to Run MCMC
void ProbitBART::RunMCMC() {

  // Storage object for draws of the probability of success parameter
  draws = arma::zeros<arma::mat>(n, ndpost);

  // Storage object for varcount
  varcount_mcmc = arma::zeros<arma::umat>(p, ndpost);

  // Define auxiliary variables
  int np = 1, niter = ndpost + nskip, move = 0;
  arma::vec f_sum_trees = arma::zeros<arma::vec>(n);
  double slp = 0.0;

  // Open file to write the forests
  std::ofstream os(path_out + "/forests.bin", std::ios::binary | std::ios::app);

  std::cout << "Starting MCMC...\n";

  for (int i=0; i < niter; i++) {
    if ((i % printevery == 0)) std::cout << "Iteration " << i << " of " << niter << "\n";
    for (int t=0; t < ntrees; t++) {
      // Back-fit
      fit_h = f_sum_trees - g_trees.col(t);
      // Compute the partial residuals
      z_h = z - fit_h;
      // Update sufficient statistics of the tree
      UpdateTree(trees[t]);
      // Sample a move
      move = 0;
      if (trees[t]->NLeaves() > 1) move = sample_discrete(proposals_prob, 3);
      switch(move) {
        case 0:
          Grow(trees[t], *this);
          break;
        case 1:
          Prune(trees[t], *this);
          break;
        case 2:
          Change(trees[t], *this);
          break;
      }
      // Draw posterior (update \mu)
      DrawPosterior(trees[t]);
      // Update the fit (Refit)
      f_sum_trees = fit_h + g_trees.col(t);
      // Update the leaves and depth
      avg_leaves[t] += trees[t]->NLeaves();
      avg_depth[t] += trees[t]->GetDepth(trees[t]);

      if (i >= nskip) serialise_tree(trees[t], os, np);
    }
    // Update latent variable z_{i}
    for (int k = 0; k < n; k++) {
      if (y[k] == 0) z[k] = -rtnorm(-f_sum_trees[k], 1.0, 0.0);
      else z[k] = rtnorm(f_sum_trees[k], 1.0, 0.0);
    }
    // Saving
    if (i >= nskip) {
      draws.col(i - nskip) = f_sum_trees;
      arma::uvec vc = GetVarCount();
      varcount_mcmc.col(i - nskip) = vc;
      if (sparse) splitprobs = UpdateSplitProbs(vc, alpha_sparse, p);
      if (alpha_random) {
        slp = 0.0;
        for (int j = 0; j < p; j++) slp += std::log(splitprobs[j]);
        alpha_sparse = UpdateAlphaDirchlet(alphas_grid,
                                           log_posterior_alpha_sparse,
                                           slp, (double)p, k_grid);
      }
    }
  }
  os.close();
}

// Exposing a C++ class in R
//using namespace Rcpp;
RCPP_MODULE(probit_bart) {

  // Expose class SharedLogitBART as "SharedLogitBART" on the R side
  Rcpp::class_<ProbitBART>("ProbitBART")

  // Exposing constructor
  .constructor<arma::uvec, arma::mat>()

  // Exposing member functions
  .method("SetMCMC", &ProbitBART::SetMCMC)
  .method("RunMCMC", &ProbitBART::RunMCMC)
  .method("Predict", &ProbitBART::Predict)
  // .method("GetVarCount", &SharedLogitBART::GetVarCount2)

  // Exposing some attributes
  .field("draws", &ProbitBART::draws)
  .field("varcount_mcmc", &ProbitBART::varcount_mcmc)
  .field("splitprobs", &ProbitBART::splitprobs)
  .field("alpha_sparse", &ProbitBART::alpha_sparse)
  .field("avg_leaves", &ProbitBART::avg_leaves)
  .field("avg_depth", &ProbitBART::avg_depth)
  // .field("ar_change", &SharedLogitBART::ar_change)
  // .field("ar_grow", &SharedLogitBART::ar_grow)
  // .field("ar_prune", &SharedLogitBART::ar_prune)

  .field("path_out", &ProbitBART::path_out)
  .field("ntrees", &ProbitBART::ntrees)
  .field("ndpost", &ProbitBART::ndpost)
  ;

}
