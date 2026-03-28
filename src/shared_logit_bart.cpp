#include "shared_logit_bart.h"
#include "tree_mcmc.h"
#include "write_read.h"

SharedLogitBART::SharedLogitBART(const arma::uvec &y, const arma::mat &X) : y(y), X(X) {
  n = X.n_rows;
  p = X.n_cols;
}

SharedLogitBART::~SharedLogitBART() {
  for (int h=0; h < ntrees; h++) delete trees[h];
  delete prior;
}

double SharedLogitBART::lml(Node *leaf) {
  double out = 2 * prior->c_logd_lgamc;
  out += R::lgammafn(leaf->ss1(0) + prior->c_shape) - (leaf->ss1(0) + prior->c_shape) * log(leaf->ss2(0) + prior->d_rate);
  out += R::lgammafn(leaf->ss1(1) + prior->c_shape) - (leaf->ss1(1) + prior->c_shape) * log(leaf->ss2(1) + prior->d_rate);
  return out;
}


void SharedLogitBART::UpdateSuffStats(Node *leaf) {
  double s0 = 0.0, r0 = 0.0, r1 = 0.0;

  for (auto id : leaf->ids) {
    s0 += y(id) == 0;
    //s1 += y(id) == 1;
    r0 += fit_0_h_phi(id);
    r1 += fit_1_h_phi(id);
  }
  leaf->ss1(0) = s0;
  leaf->ss2(0) = r0;
  leaf->ss1(1) = leaf->nobs - s0;
  leaf->ss2(1) = r1;
}

void SharedLogitBART::UpdateTree(Node *tree) {
  std::vector<Node*> leaves;
  tree->GetLeaves(leaves);
  for (size_t t = 0; t < leaves.size(); t++) {
    UpdateSuffStats(leaves[t]);
  }
}

void SharedLogitBART::DrawPosterior(Node *tree) {
  double mu0, mu1;
  const int h = tree->h;
  std::vector<Node*> leaves;
  tree->GetLeaves(leaves);
  for (Node* leaf : leaves) {
    mu0 = log(R::rgamma(leaf->ss1(0) + prior->c_shape, 1.0) / (leaf->ss2(0) + prior->d_rate));
    mu1 = log(R::rgamma(leaf->ss1(1) + prior->c_shape, 1.0) / (leaf->ss2(1) + prior->d_rate));
    leaf->mu(0) = mu0;
    leaf->mu(1) = mu1;
    // Save the predictions at g_trees.
    for (auto id : leaf->ids) {
      g0_trees(id, h) = mu0;
      g1_trees(id, h) = mu1;
    }
  }
}

// Get the total times a variable appear as a decision rule over ALL trees.
arma::uvec SharedLogitBART::GetVarCount() {
  arma::uvec vc = arma::zeros<arma::uvec>(p);
  for (int t = 0; t < ntrees; t++) {
    ComputeVarCount(trees[t], vc);
  }
  return vc;
}
void SharedLogitBART::ComputeVarCount(Node *tree, arma::uvec &varcount) {
  if (tree->is_leaf == 0) {
    int id_j = tree->predictor;
    varcount[id_j]++;
    ComputeVarCount(tree->left, varcount);
    ComputeVarCount(tree->right, varcount);
  }
}

// Function to predict
arma::vec SharedLogitBART::GetMu(Node *tree, const arma::rowvec &x) {
  if (tree->is_leaf) return tree->mu;
  if (x[tree->predictor] <= tree->cutoff) return GetMu(tree->left, x);
  else return GetMu(tree->right, x);
}

arma::mat SharedLogitBART::Predict(arma::mat &X_, int n_samples, int ntrees,
                                   std::string path_out) {
  int np = 2;
  arma::vec mus = arma::zeros<arma::vec>(np);
  int n_ = X_.n_rows;
  arma::vec f0 = arma::zeros<arma::vec>(n_);
  arma::vec f1 = arma::zeros<arma::vec>(n_);
  arma::mat draws_pred = arma::zeros<arma::mat>(n_, n_samples);

  std::ifstream is(path_out + "/forests.bin", std::ios::binary);

  for (int h = 0; h < n_samples; h++) {
    f0 = arma::zeros<arma::vec>(n_);
    f1 = arma::zeros<arma::vec>(n_);
    // Import a tree
    for (int t = 0; t < ntrees; t++) {
      Node *tree = deserialise_tree(is, np);
      // Do prediction for the imported tree
      for (int i = 0; i < n_; i++) {
        const arma::rowvec &xi = X_.row(i);
        mus = GetMu(tree, xi);
        f0[i] += mus(0);
        f1[i] += mus(1);
      }
    }
    f1 = exp(f1);
    // Save the predictions for the h-th posterior sample
    draws_pred.col(h) = f1.each_col() / (f1 + exp(f0));
  }
  is.close();
  return draws_pred;
}



void SharedLogitBART::SetMCMC(double v0, int ntrees_, int ndpost_, int nskip_,
                              int printevery_, int numcut_, double power_,
                              double base_, std::vector<double> proposals_prob_,
                              std::vector<double> splitprobs_, int sparse_,
                              std::vector<double> sparse_parms_,
                              double alpha_sparse_, int alpha_random_,
                              arma::mat xinfo, std::string path_out_) {
  // Setting attributes
  ntrees = ntrees_;
  numcut = numcut_;
  ndpost = ndpost_;
  printevery = printevery_;
  nskip = nskip_;
  power = power_;
  base = base_;
  proposals_prob = proposals_prob_;
  logprob_grow = log(proposals_prob_[0]);
  logprob_prune = log(proposals_prob_[1]);
  splitprobs = splitprobs_;
  sparse = sparse_;
  sparse_parms = sparse_parms_;
  alpha_random = alpha_random_;
  alpha_sparse = alpha_sparse_;
  path_out = path_out_;

  // Define the prior
  double s = v0 / sqrt(ntrees);
  prior = new PriorSharedLogit(s);

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
  g0_trees = arma::zeros<arma::mat>(n, ntrees);
  g1_trees = arma::zeros<arma::mat>(n, ntrees);
  // First draw for the latent variable phi
  phi = arma::zeros<arma::vec>(n);
  for (int i = 0; i < n; i++) phi(i) = R::exp_rand();
  // Initialize the trees
  trees.resize(0);
  for (int h = 0; h < ntrees; h++) {
    Node *nd = new Node(2);
    // Setting the tree identification
    nd->h = h;
    // Initialise the ids
    nd->ids = arma::regspace<arma::uvec>(0, n - 1);
    nd->nobs = n;
    // Compute the sufficient statistics
    nd->ss1(0) = arma::accu(y == 0);
    nd->ss1(1) = arma::accu(y == 1);
    nd->ss2(0) = arma::accu(phi);
    nd->ss2(1) = arma::accu(phi);
    trees.push_back(nd);
  }
  fit_0_h_phi = arma::ones<arma::vec>(n);
  fit_1_h_phi = arma::ones<arma::vec>(n);

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

  // Initialise flags for acceptance rate
  flag_grow = 0;
  flag_prune = 0;
  flag_change = 0;

  std::cout << "Model set up!" << "\n\n";
}

void SharedLogitBART::RunMCMC() {
  int np = 2;
  // Define auxiliary variables
  arma::vec rt;
  arma::vec fit_h = arma::zeros<arma::vec>(n);
  arma::vec fit_0_h = arma::zeros<arma::vec>(n);
  arma::vec fit_1_h = arma::zeros<arma::vec>(n);
  arma::vec f0_mu = arma::zeros<arma::vec>(n);
  arma::vec f1_mu = arma::zeros<arma::vec>(n);
  arma::vec f1_lambda = arma::zeros<arma::vec>(n);

  int move = 0, niter = ndpost + nskip;

  // Storage object for draws of the probability of success parameter
  draws = arma::zeros<arma::mat>(n, ndpost);

  // Varcount matrix
  varcount_mcmc = arma::zeros<arma::umat>(p, ndpost);

  // Aux variable to compute the sum(log(splitprobs))
  double slp = 0.0;

  // Open file to write the forests FOR each category
  std::ofstream os(path_out + "/forests.bin", std::ios::binary | std::ios::app);

  std::cout << "Starting MCMC...\n";

  for (int i=0; i < niter; i++) {
    if ((i % printevery == 0)) std::cout << "Iteration " << i << " of " << niter << "\n";
    // Iterate over trees categories
    for (int t = 0; t < ntrees; t++) {
      // Back-fit
      fit_0_h = f0_mu - g0_trees.col(t);
      fit_1_h = f1_mu - g1_trees.col(t);
      fit_0_h_phi = exp(fit_0_h) % phi;
      fit_1_h_phi = exp(fit_1_h) % phi;
      // Update tree
      UpdateTree(trees[t]);
      // Perform the tree movement
      // Sample a move
      move = 0;
      if (trees[t]->NLeaves() > 1L) move = sample_discrete(proposals_prob, 3.0);
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
      // Draw from the node parameters
      DrawPosterior(trees[t]);
      // Re-fit
      f0_mu = fit_0_h + g0_trees.col(t);
      f1_mu = fit_1_h + g1_trees.col(t);
      // Serialise the tree parameters
      if (i >= nskip) serialise_tree(trees[t], os, np);
    }
    // Save \lambda_j = exp(\mu_j)
    f1_lambda = exp(f1_mu);
    // Update latent variable \phi_i ~ Exp[f0 + f1]
    rt = exp(f0_mu) + f1_lambda;
    for (int k=0; k < n; k++) phi(k) = R::exp_rand() / rt(k);
    // Saving
    if (i >= nskip) {
      draws.col(i - nskip) = f1_lambda.each_col() / rt;
      // Compute var-counts, save, and update Dirichlet prior if required
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
RCPP_MODULE(shared_logit_bart) {

  // Expose class SharedLogitBART as "SharedLogitBART" on the R side
  Rcpp::class_<SharedLogitBART>("SharedLogitBART")

  // Exposing constructor
  .constructor<arma::uvec, arma::mat>()

  // Exposing member functions
  .method("SetMCMC", &SharedLogitBART::SetMCMC)
  .method("RunMCMC", &SharedLogitBART::RunMCMC)
  .method("Predict", &SharedLogitBART::Predict)
  // .method("GetVarCount", &SharedLogitBART::GetVarCount2)

  // Exposing some attributes
  .field("draws", &SharedLogitBART::draws)
  .field("varcount_mcmc", &SharedLogitBART::varcount_mcmc)
  .field("splitprobs", &SharedLogitBART::splitprobs)
  // .field("avg_leaves", &SharedLogitBART::avg_leaves)
  // .field("avg_depth", &SharedLogitBART::avg_depth)
  // .field("ar_change", &SharedLogitBART::ar_change)
  // .field("ar_grow", &SharedLogitBART::ar_grow)
  // .field("ar_prune", &SharedLogitBART::ar_prune)
   .field("alpha_sparse", &SharedLogitBART::alpha_sparse)

  .field("path_out", &SharedLogitBART::path_out)
  .field("ntrees", &SharedLogitBART::ntrees)
  .field("ndpost", &SharedLogitBART::ndpost)
  ;

}

//////// Rcpp::export
// Rcpp::List logit_bart(const arma::uvec &y, const arma::mat &X,
//                       const arma::mat &X_test,
//                       double v0, int ntrees, int ndpost, int nskip,
//                       int printevery, int numcut, double power,
//                       double base, std::vector<double> proposals_prob) {
//
//   SharedLogitBART model = SharedLogitBART(y, X, X_test);
//   model.SetMCMC(v0, ntrees, ndpost, nskip, printevery, numcut, power,
//                 base, proposals_prob);
//   arma::vec rt;
//   arma::vec fit_0_h = arma::zeros<arma::vec>(model.n);
//   arma::vec fit_1_h = arma::zeros<arma::vec>(model.n);
//   arma::vec f0_mu = arma::zeros<arma::vec>(model.n);
//   arma::vec f1_mu = arma::zeros<arma::vec>(model.n);
//   //arma::vec f0_lambda = arma::zeros<arma::vec>(model.n);
//   arma::vec f1_lambda = arma::zeros<arma::vec>(model.n);
//   int move = 0;
//
//   // Storage for the draws of \theta
//   arma::mat draws = arma::zeros<arma::mat>(model.n, model.ndpost);
//
//   int niter = model.ndpost + model.nskip;
//
//   // TODO: Add avg-leaves, var-count, and predict functions.
//   // MCMC iteration
//   std::cout << "Starting MCMC...\n";
//   for (int i=0; i < niter; i++) {
//     if ((i % printevery == 0)) std::cout << "Iteration " << i << " of " << niter << "\n";
//     // Iterate over trees categories
//     for (int t=0; t < model.ntrees; t++) {
//       // Back-fit
//       fit_0_h = f0_mu - model.g0_trees.col(t);
//       fit_1_h = f1_mu - model.g1_trees.col(t);
//       //std::cout << f_mu_h[0] << "\n\n";
//       model.fit_0_phi = exp(fit_0_h) % model.phi;
//       model.fit_1_phi = exp(fit_1_h) % model.phi;
//       // Update tree
//       model.UpdateTree(model.trees[t]);
//       // Perform the tree movement
//       // Sample a move
//       move = 0;
//       if (model.trees[t]->NLeaves() > 1L) move = sample_discrete(proposals_prob, 3.0);
//       switch(move) {
//       case 0:
//         Grow(model.trees[t], model);
//         break;
//       case 1:
//         Prune(model.trees[t], model);
//         break;
//       case 2:
//         Change(model.trees[t], model);
//         break;
//       }
//       // Draw from the node parameters
//       model.DrawPosterior(model.trees[t]);
//       // Re-fit
//       f0_mu = fit_0_h + model.g0_trees.col(t);
//       f1_mu = fit_1_h + model.g1_trees.col(t);
//     }
//     // Save \lambda_j = exp(\mu_j)
//     f1_lambda = exp(f1_mu);
//     // f1_lambda = exp(f1_mu);
//     // Update latent variable \phi_i ~ Exp[f0 + f1]
//     rt = exp(f0_mu) + f1_lambda;
//     for (int k=0; k < model.n; k++) model.phi(k) = R::exp_rand() / rt(k);
//     if (i >= nskip) draws.col(i - nskip) = f1_lambda.each_col() / rt;
//   }
//   return (Rcpp::List::create(Rcpp::Named("draws") = draws));
// }
