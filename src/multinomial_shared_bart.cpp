#include "multinomial_shared_bart.h"
#include "tree_mcmc.h"
#include "write_read.h"

MultinomialSharedBART::MultinomialSharedBART(const arma::umat &Y, const arma::mat &X) : Y(Y), X(X) {
  d = Y.n_cols;
  n = X.n_rows;
  p = X.n_cols;
  n_trials = sum(Y, 1);
}

MultinomialSharedBART::~MultinomialSharedBART() {
  for (int h=0; h < ntrees; h++) {
    delete trees[h];
  }
  // delete prior;
}

double MultinomialSharedBART::lml(Node *leaf) {
  double out = d * c_logd_lgamc;
  for (int j=0; j < d; j++) {
    out += R::lgammafn(leaf->ss1(j) + c_shape) - (leaf->ss1(j) + c_shape) * log(leaf->ss2(j) + d_rate);
  }
  return out;
}

void MultinomialSharedBART::UpdateSuffStats(Node *leaf) {

  for (int j=0; j < d; j++) {
    double s = 0.0, r = 0.0;
    for (auto id : leaf->ids) {
      s += Y(id, j);
      r += fit_h_phi(id, j);
    }
    leaf->ss1(j) = s;
    leaf->ss2(j) = r;
  }
}

void MultinomialSharedBART::UpdateTree(Node *tree) {
  std::vector<Node*> leaves;
  tree->GetLeaves(leaves);
  for (size_t t = 0; t < leaves.size(); t++) {
    UpdateSuffStats(leaves[t]);
  }
}

void MultinomialSharedBART::DrawPosterior(Node *tree) {
  double mu;
  const int h = tree->h;
  std::vector<Node*> leaves;
  tree->GetLeaves(leaves);
  for (Node* leaf : leaves) {
    for (int j=0; j < d; j++) {
      mu = log(R::rgamma(leaf->ss1(j) + c_shape, 1.0 / (leaf->ss2(j) + d_rate)) );
      leaf->mu(j) = mu;
      // Save the predictions at g_trees.
      for (auto id : leaf->ids) {
        g_trees(id, j, h) = mu;
      }
    }
  }
}

// Get the total times a variable appear as a decision rule over ALL trees.
arma::uvec MultinomialSharedBART::GetVarCount() {
  arma::uvec varcount = arma::zeros<arma::uvec>(p);
  for (int t = 0; t < ntrees; t++) {
    ComputeVarCount(trees[t], varcount);
  }
  return varcount;
}

void MultinomialSharedBART::ComputeVarCount(Node *tree, arma::uvec &varcount) {
  if (tree->is_leaf == 0) {
    int id_j = tree->predictor;
    varcount(id_j)++;
    ComputeVarCount(tree->left, varcount);
    ComputeVarCount(tree->right, varcount);
  }
}

// arma::ucube MultinomialSharedBART::GetVarCount2(int p, int d, int n_samples,
//                                                 int ntrees,
//                                                 std::string forests_dir) {
//
//   int np = d;
//   arma::ucube varcount = arma::zeros<arma::ucube>(p, d, n_samples);
//   arma::uvec vc = arma::zeros<arma::uvec>(p);
//
//   // Iterate over categories (this can be done in parallel)
//   for (int j = 0; j < d; j++) {
//     // Open file of category j
//     std::ifstream is(forests_dir + "/forests_j" + std::to_string(j) + ".bin");
//     // Iterate over the mcmc samples
//     for (int t = 0; t < n_samples; t++) {
//       // Initialise vector to keep the var-count for category j-tree
//       vc = arma::zeros<arma::uvec>(p);
//       // Iterate over trees
//       for (int h = 0; h < ntrees; h++) {
//         // Import tree
//         Node *tree = deserialise_tree(is, np);
//         // Compute var-count
//         ComputeVarCount(tree, vc);
//       }
//       // Save prediction of sample t and category j
//       varcount.slice(t).col(j) = vc;
//     }
//     // Close file for category j
//     is.close();
//   }
//
//   return varcount;
// }


// Predict function that read a tree and traverse to find the terminal node
// according to the decision rule
arma::vec MultinomialSharedBART::GetMu(Node *tree, const arma::rowvec &x) {
  if (tree->is_leaf) return tree->mu;
  if (x[tree->predictor] <= tree->cutoff) return GetMu(tree->left, x);
  else return GetMu(tree->right, x);
}

arma::cube MultinomialSharedBART::Predict(arma::mat &X_, int d, int n_samples,
                                          int ntrees, std::string forests_dir) {
  int n_ = X_.n_rows;
  arma::mat f_pred;
  arma::cube draws_pred = arma::zeros<arma::cube>(n_, d, n_samples);

  // Iterate over categories (this can be done in parallel)
  // for (int j = 0; j < d; j++) {
    // std::cout << "Compute prediction of category " << j << "\n";
  // Open file of category j
  std::ifstream is(forests_dir + "/forests.bin");
  // Iterate over the MCMC samples
  for (int t = 0; t < n_samples; t++) {
    // Initialise vector to keep the predictions of \theta_ij
    f_pred = arma::zeros<arma::vec>(n_, d);
    // Iterate over trees
    for (int h = 0; h < ntrees; h++) {
      // Import tree
      Node *tree = deserialise_tree(is, d);
      // Do the predictions
      for (int i = 0; i < n_; i++) {
        const arma::rowvec &xi = X_.row(i);
        f_pred.row(i) += GetMu(tree, xi);
      }
    }
    // Save prediction of sample t
    f_pred = exp(f_pred);
    f_pred = f_pred.each_col() / arma::sum(f_pred, 1);
    draws_pred.slice(t) = f_pred;
  }
  // Close file
  is.close();
  return draws_pred;
}

double MultinomialSharedBART::UpdateSigmaPrior() {

  double s_log_lambda = 0.0;
  double s_lambda = 0.0;
  int m_bh = 0;
  // Get the statistics
  for (int t=0; t < ntrees; t++) {
    std::vector<Node*> leaves;
    trees[t]->GetLeaves(leaves);
    for (size_t u = 0; u < leaves.size(); u++) {
      for (int j=0; j < d; j++) {
        s_log_lambda += leaves[u]->mu(j);
        s_lambda += exp(leaves[u]->mu(j));
      }
      m_bh++;
    }
  }
  //
  double y = log(R::unif_rand());
  y += target_sigma_prior(sigma, m_bh, s_log_lambda, s_lambda, s2_0);
  // Creating the lower and upper bounds
  double L = sigma - R::unif_rand() * w_ss;
  double R = L + w_ss;
  while (true) {
    if (L <= 0.0) break;
    if (target_sigma_prior(L, m_bh, s_log_lambda, s_lambda, s2_0) > y) break;
    L -= w_ss;
  }
  if (L < 0.0) L = 0.0;
  while (target_sigma_prior(R, m_bh, s_log_lambda, s_lambda, s2_0) > y) R += w_ss;
  double x_star = L + R::unif_rand() * (R - L);
  // std::cout << "Start SS \n";
  // Repeat until create an acceptable proposal
  do {
    x_star = L + R::unif_rand() * (R - L);
    // if (target_sigma_prior(x_star, m_bh, s_log_lambda, s_lambda, s2_0) >= y) break;
    if (x_star < sigma) L = x_star;
    else R = x_star;
  } while (target_sigma_prior(x_star, m_bh, s_log_lambda, s_lambda, s2_0) < y);

  return x_star;
}

void MultinomialSharedBART::SetMCMC(double v0, int ntrees_, int ndpost_, int nskip_,
                              int numcut_, double power_,
                              double base_, std::vector<double> proposals_prob_,
                              int update_sd_prior_, double s2_0_, double w_ss_,
                              std::vector<double> splitprobs_,
                              int sparse_, std::vector<double> sparse_parms_,
                              double alpha_sparse_, int alpha_random_,
                              arma::mat xinfo, std::string forests_dir_,
                              int keep_draws_, int save_trees_) {

  Rcpp::RNGScope scope;
  // Setting attributes
  ntrees = ntrees_;
  numcut = numcut_;
  ndpost = ndpost_;
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
  forests_dir = forests_dir_;
  s2_0 = s2_0_;
  w_ss = w_ss_;
  update_sd_prior = update_sd_prior_;
  keep_draws = keep_draws_;
  save_trees = save_trees_;

  // Define the prior parameters
  sigma = v0 / sqrt(ntrees);
  c_shape = trigamma_inverse(sigma*sigma);
  d_rate = exp(R::digamma(c_shape));
  c_logd_lgamc = c_shape * log(d_rate) - R::lgammafn(c_shape);

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
  g_trees = arma::zeros<arma::cube>(n, d, ntrees); // (d, ntrees, n)

  // First draw for the latent variable phi
  phi = arma::zeros<arma::vec>(n);
  for (int i = 0; i < n; i++) phi(i) = R::rgamma(n_trials(i), 1.0);

  // Initialize the trees
  trees.resize(0);
  for (int h = 0; h < ntrees; h++) {
    Node *nd = new Node(d);
    // Setting the tree identification
    nd->h = h;
    // Initialise the ids
    nd->ids = arma::regspace<arma::uvec>(0, n - 1);
    nd->nobs = n;
    // Compute the sufficient statistics
    for (int j=0; j < d; j++) {
      nd->ss1(j) = arma::accu(Y.col(j));
      nd->ss2(j) = arma::accu(phi);
    }
    trees.push_back(nd);
  }

  // Create grid for the \alpha, where \alpha = lambda * \rho / (1 - \lambda)
  // and for each \alpha compute the "constant" part of the log-posterior.
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
  if (update_sd_prior) {
    sigma_mcmc.resize(ndpost + nskip);
  }

  // Initialise matrix that is a member/field and is used in g_trees
  fit_h_phi = arma::zeros<arma::mat>(n, d);

  // Initialise flags for acceptance rate
  flag_grow = 0;
  flag_prune = 0;
  flag_change = 0;
  accept_rate = arma::zeros<arma::uvec>(3);
  avg_leaves = arma::zeros<arma::uvec>(ntrees);
  avg_depth = arma::zeros<arma::uvec>(ntrees);

  std::cout << "Model set up!" << "\n\n";
}

void MultinomialSharedBART::RunMCMC() {

  // Initialise container for keep the draws
  draws = arma::zeros<arma::cube>(n, d, ndpost);

  // Storage object for varcount
  varcount_mcmc = arma::zeros<arma::umat>(p, ndpost);

  // Define auxiliary variables
  arma::vec rt = arma::zeros<arma::vec>(n);
  arma::mat fit_h = arma::zeros<arma::mat>(n, d);
  arma::mat f_mu = arma::zeros<arma::mat>(n, d);
  arma::mat f_lambda = arma::zeros<arma::mat>(n, d);
  int move = 0, niter = ndpost + nskip;

  // Aux variable to compute the sum(log(splitprobs))
  double slp = 0.0;

  // Open file to write the forests FOR each category
  std::ofstream fout(forests_dir + "/forests.bin", std::ios::binary | std::ios::app);

  std::cout << "Starting MCMC...\n";
  double progress = 0.0;
  for (int i=0; i < niter; i++) {
    progress = (double) 100 * i / niter;
    Rprintf("\r");
    Rprintf("%3.2f%% completed", progress);
    // std::cout << i << "\n";
    // Iterate over trees categories
    for (int t=0; t < ntrees; t++) {
      // Back-fit
      fit_h = f_mu - g_trees.slice(t);
      // Multiply each row of the matrix fit_h by each element of the vector phi
      fit_h_phi = exp(fit_h);
      fit_h_phi.each_col() %= phi;
      // Update tree
      UpdateTree(trees[t]);
      // Sample a move
      move = 0;
      if (trees[t]->NLeaves() > 1) move = sample_discrete(proposals_prob, 3);
      switch(move) {
      case 0:
        Grow(trees[t], *this);
        accept_rate[0] += flag_grow;
        break;
      case 1:
        Prune(trees[t], *this);
        accept_rate[1] += flag_prune;
        break;
      case 2:
        Change(trees[t], *this);
        accept_rate[2] += flag_change;
        break;
      }
      // Draw from the node parameters
      DrawPosterior(trees[t]);
      // Re-fit
      f_mu = fit_h + g_trees.slice(t);

      // Compute avg leaves and depth
      avg_leaves[t] += trees[t]->NLeaves();
      avg_depth[t] += trees[t]->GetDepth(trees[t]);

      // Serialise the tree parameters
      if (i >= nskip) serialise_tree(trees[t], fout, d);
    }

    // Save \lambda_j = exp(\mu_j)
    f_lambda = exp(f_mu);
    // Update latent variable \phi_i ~ Gamma[N_i, \sum f_lambda]
    rt = arma::sum(f_lambda, 1);
    for (int k=0; k < n; k++) phi(k) = R::rgamma(n_trials(k), 1.0) / rt(k);

    // Update the sigma and then the leaf prior parameters
    if (update_sd_prior) {
      sigma = UpdateSigmaPrior();
      c_shape = trigamma_inverse(sigma*sigma);
      d_rate = exp(R::digamma(c_shape));
      c_logd_lgamc = c_shape * log(d_rate) - R::lgammafn(c_shape);
      sigma_mcmc[i] = sigma;
    }

    if (i >= nskip) {
      // Save draws
      draws.slice(i - nskip) = f_lambda.each_col() / rt;
      // Compute var-count, save, and update Dirichlet prior if needed
      arma::uvec vc = GetVarCount();
      varcount_mcmc.col(i - nskip) = vc;
      if (sparse) splitprobs = UpdateSplitProbs(vc, alpha_sparse, p);
      if (alpha_random) {
        slp = 0.0;
        for (int k = 0; k < p; k++) slp += std::log(splitprobs[k]);
        alpha_sparse = UpdateAlphaDirchlet(alphas_grid,
                                           log_posterior_alpha_sparse,
                                           slp, (double)p, k_grid);
      }
    }
  }

  // Close the file
  fout.close();
}

// Exposing a C++ class in R
//using namespace Rcpp;
RCPP_MODULE(multinomial_shared_bart) {

  // Expose class MultinomialSharedBART as "MultinomialSharedBART" on the R side
  Rcpp::class_<MultinomialSharedBART>("MultinomialSharedBART")

  // Exposing constructor
  .constructor<arma::umat, arma::mat>()

  // Exposing member functions
  .method("SetMCMC", &MultinomialSharedBART::SetMCMC)
  .method("RunMCMC", &MultinomialSharedBART::RunMCMC)
  .method("Predict", &MultinomialSharedBART::Predict)
  // .method("GetVarCount", &MultinomialSharedBART::GetVarCount2)

  // Exposing some attributes
  .field("draws", &MultinomialSharedBART::draws)
  // .field("draws_phi", &MultinomialSharedBART::draws_phi)
  .field("varcount_mcmc", &MultinomialSharedBART::varcount_mcmc)
  .field("splitprobs", &MultinomialSharedBART::splitprobs)
  .field("avg_leaves", &MultinomialSharedBART::avg_leaves)
  .field("avg_depth", &MultinomialSharedBART::avg_depth)
  .field("accept_rate", &MultinomialSharedBART::accept_rate)

  .field("alpha_sparse", &MultinomialSharedBART::alpha_sparse)
  .field("sigma_mcmc", &MultinomialSharedBART::sigma_mcmc)

  .field("forests_dir", &MultinomialSharedBART::forests_dir)
  .field("ntrees", &MultinomialSharedBART::ntrees)
  .field("ndpost", &MultinomialSharedBART::ndpost)
  ;

}
