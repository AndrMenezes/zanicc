#include "multinomial_bart.h"
#include "tree_mcmc.h"
#include "write_read.h"

MultinomialBART::MultinomialBART(const arma::umat &Y, const arma::mat &X) : Y(Y), X(X) {
  d = Y.n_cols;
  n = X.n_rows;
  p = X.n_cols;
  n_trials = sum(Y, 1);
}

MultinomialBART::~MultinomialBART() {
  for (int j=0; j < d; j++) {
    for (int h=0; h < ntrees; h++) {
      delete trees[j][h];
    }
  }
  // delete prior;
}

double MultinomialBART::lml(Node *leaf) {
  return (c_logd_lgamc
          + R::lgammafn(leaf->ss1(0) + c_shape)
          - (leaf->ss1(0) + c_shape) * log(leaf->ss2(0) + d_rate));
}

void MultinomialBART::UpdateSuffStats(Node *leaf) {
  double s = 0.0, r = 0.0;
  for (auto id : leaf->ids) {
    s += Y(id, j_cat);
    r += fit_h_phi(id);
  }
  leaf->ss1(0) = s;
  leaf->ss2(0) = r;
}

void MultinomialBART::UpdateTree(Node *tree) {
  std::vector<Node*> leaves;
  tree->GetLeaves(leaves);
  for (size_t t = 0; t < leaves.size(); t++) {
    UpdateSuffStats(leaves[t]);
  }
}

void MultinomialBART::DrawPosterior(Node *tree) {
  double mu;
  const int h = tree->h;
  std::vector<Node*> leaves;
  tree->GetLeaves(leaves);
  for (Node* leaf : leaves) {
    mu = log(R::rgamma(leaf->ss1(0) + c_shape, 1.0 / (leaf->ss2(0) + d_rate)) );
    leaf->mu(0) = mu;
    // Save the predictions at g_trees.
    for (auto id : leaf->ids) {
      g_trees(j_cat, h, id) = mu;
    }
  }
}

// Get the total times a variable appear as a decision rule over ALL trees.
arma::uvec MultinomialBART::GetVarCount(int j) {
  arma::uvec varcount = arma::zeros<arma::uvec>(p);
  for (int t = 0; t < ntrees; t++) {
    ComputeVarCount(trees[j][t], varcount);
  }
  return varcount;
}

void MultinomialBART::ComputeVarCount(Node *tree, arma::uvec &varcount) {
  if (tree->is_leaf == 0) {
    int id_j = tree->predictor;
    varcount(id_j)++;
    ComputeVarCount(tree->left, varcount);
    ComputeVarCount(tree->right, varcount);
  }
}

// void ComputeVarCount(Node *tree, arma::uvec &varcount) {
//   if (tree->is_leaf == 0) {
//     int id_j = tree->predictor;
//     varcount(id_j)++;
//     ComputeVarCount(tree->left, varcount);
//     ComputeVarCount(tree->right, varcount);
//   }
// }

arma::ucube MultinomialBART::GetVarCount2(int p, int d, int n_samples, int ntrees,
                                          std::string path_out) {

  int np = 1;
  arma::ucube varcount = arma::zeros<arma::ucube>(p, d, n_samples);
  arma::uvec vc = arma::zeros<arma::uvec>(p);

  // Iterate over categories (this can be done in parallel)
  for (int j = 0; j < d; j++) {
    // Open file of category j
    std::ifstream is(path_out + "/forests_j" + std::to_string(j) + ".bin");
    // Iterate over the mcmc samples
    for (int t = 0; t < n_samples; t++) {
      // Initialise vector to keep the var-count for category j-tree
      vc = arma::zeros<arma::uvec>(p);
      // Iterate over trees
      for (int h = 0; h < ntrees; h++) {
        // Import tree
        Node *tree = deserialise_tree(is, np);
        // Compute var-count
        ComputeVarCount(tree, vc);
      }
      // Save prediction of sample t and category j
      varcount.slice(t).col(j) = vc;
    }
    // Close file for category j
    is.close();
  }

  return varcount;
}


// Predict function that read a tree and traverse to find the terminal node
// according to the decision rule
double MultinomialBART::GetMu(Node *tree, const arma::rowvec &x) {
  if (tree->is_leaf) return tree->mu(0);
  if (x[tree->predictor] <= tree->cutoff) return GetMu(tree->left, x);
  else return GetMu(tree->right, x);
}

arma::cube MultinomialBART::Predict(arma::mat &X_, int d, int n_samples,
                                    int ntrees, std::string path_out) {
  int np = 1;
  int n_ = X_.n_rows;
  arma::mat f_pred;
  arma::cube draws_pred = arma::zeros<arma::cube>(n_, d, n_samples);

    // Open files to read the the forests
  std::vector<std::ifstream> files;
  for (int j=0; j < d; j++) {
    std::string ff = path_out + "/forests_" + std::to_string(j) + ".bin";
    files.emplace_back(ff, std::ios::binary);
  }

  double progress = 0.0;
  // Iterate over the MCMC samples
  for (int t = 0; t < n_samples; t++) {
    progress = (double) 100 * t / n_samples;
    Rprintf("%3.2f%% Completed", progress);
    Rprintf("\r");
    // Initialise vector to keep the predictions of \theta_ij
    f_pred = arma::zeros<arma::mat>(n_, d);
    // Iterate over categories
    for (int j = 0; j < d; j++) {
      // Iterate over trees
      for (int h = 0; h < ntrees; h++) {
        // Import tree
        Node *tree = deserialise_tree(files[j], np);
        // Do the predictions
        for (int i = 0; i < n_; i++) {
          const arma::rowvec &xi = X_.row(i);
          f_pred(i, j) += GetMu(tree, xi);
        }
        delete tree;
      }
    }
    f_pred = exp(f_pred);
    f_pred = f_pred.each_col() / arma::sum(f_pred, 1);
    draws_pred.slice(t) = f_pred;
  }
  for (int j=0; j<d; j++) files[j].close();

  return draws_pred;
}

double MultinomialBART::UpdateSigmaPrior() {

  double s_log_lambda = 0.0;
  double s_lambda = 0.0;
  int m_bh = 0;
  // Get the statistics
  for (int j=0; j < d; j++) {
    for (int t=0; t < ntrees; t++) {
      std::vector<Node*> leaves;
      trees[j][t]->GetLeaves(leaves);
      for (size_t u = 0; u < leaves.size(); u++) {
        s_log_lambda += leaves[u]->mu(0);
        s_lambda += exp(leaves[u]->mu(0));
        m_bh++;
      }
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

void MultinomialBART::SetMCMC(double v0, int ntrees_, int ndpost_, int nskip_,
                              int numcut_, double power_,
                              double base_, std::vector<double> proposals_prob_,
                              int update_sd_prior_, double s2_0_, double w_ss_,
                              std::vector<std::vector<double>> list_splitprobs_,
                              int sparse_, std::vector<double> sparse_parms_,
                              std::vector<double> alpha_sparse_, int alpha_random_,
                              arma::mat xinfo, std::string path_out_) {

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
  list_splitprobs = list_splitprobs_;
  sparse = sparse_;
  sparse_parms = sparse_parms_;
  alpha_random = alpha_random_;
  alpha_sparse = alpha_sparse_;
  path_out = path_out_;
  s2_0 = s2_0_;
  w_ss = w_ss_;
  update_sd_prior = update_sd_prior_;

  // Define the prior parameters
  sigma = v0 / sqrt(ntrees);
  c_shape = trigamma_inverse(sigma*sigma);
  d_rate = exp(R::digamma(c_shape));
  c_logd_lgamc = c_shape * log(d_rate) - R::lgammafn(c_shape);

  // prior = new PriorMultinomial(s);

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
  g_trees = arma::zeros<arma::cube>(d, ntrees, n);

  // First draw for the latent variable phi
  phi = arma::zeros<arma::vec>(n);
  for (int i = 0; i < n; i++) phi(i) = R::rgamma(n_trials(i), 1.0);

  // Initialize the trees
  std::vector<std::vector<Node*>>  my_trees(d);
  for (int j=0; j < d; j++) {
    my_trees[j].reserve(ntrees);
    // trees[j].resize(0);
    for (int h = 0; h < ntrees; h++) {
      Node *nd = new Node(1);
      // Setting the tree identification
      nd->h = h;
      // Initialise the ids
      nd->ids = arma::regspace<arma::uvec>(0, n - 1);
      nd->nobs = n;
      // Compute the sufficient statistics
      nd->ss1 = arma::accu(Y.col(j));
      nd->ss2 = arma::accu(phi);
      my_trees[j].push_back(nd);
    }
  }
  trees = my_trees;

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

  // Initialise flags for acceptance rate
  flag_grow = 0;
  flag_prune = 0;
  flag_change = 0;

  accept_rate = arma::zeros<arma::umat>(3, d);
  avg_leaves = arma::zeros<arma::umat>(ntrees, d);
  avg_depth = arma::zeros<arma::umat>(ntrees, d);

  std::cout << "Model set up!" << "\n\n";
}

void MultinomialBART::RunMCMC() {

  // Initialise container for keep the draws
  draws = arma::zeros<arma::cube>(n, d, ndpost);
  draws_phi = arma::zeros<arma::mat>(n, ndpost);
  // Storage object for varcount
  varcount_mcmc = arma::zeros<arma::ucube>(p, d, ndpost);

  int np = 1;
  // Define auxiliary variables
  arma::vec rt = arma::zeros<arma::vec>(n);
  arma::vec fit_h = arma::zeros<arma::vec>(n);
  arma::mat f_mu = arma::zeros<arma::mat>(n, d);
  arma::mat f_lambda = arma::zeros<arma::mat>(n, d);
  int move = 0, niter = ndpost + nskip;

  // Aux variable to compute the sum(log(splitprobs))
  double slp = 0.0;

  // Open file to write the forests FOR each category
  //std::vector<std::ofstream> files;
  //for (int j=0; j < d; j++) {
  //  std::string ff = path_out + "/forests_" + std::to_string(j) + ".bin";
  //  files.emplace_back(ff, std::ios::binary | std::ios::app);
  //}

  std::cout << "Starting MCMC...\n";
  double progress = 0.0;
  for (int i=0; i < niter; i++) {
    progress = (double) 100 * i / niter;
    Rprintf("\r");
    Rprintf("%3.2f%% completed", progress);

    // Save \lambda_j = exp(\mu_j)
    f_lambda = exp(f_mu);
    // Update latent variable \phi_i ~ Gamma[N_i, \sum f_lambda]
    rt = arma::sum(f_lambda, 1);
    for (int k=0; k < n; k++) phi(k) = R::rgamma(n_trials(k), 1.0) / rt(k);

    // Iterate over trees categories
    for (j_cat = 0; j_cat < d; j_cat++) {
      // Set split-probs of category j, this is used in the Grow, Prune, Change
      splitprobs = list_splitprobs[j_cat];
      // Iterate over categories
      for (int t=0; t < ntrees; t++) {
        // Back-fit
        fit_h = f_mu.col(j_cat) - arma::vec(g_trees.tube(j_cat, t));
        fit_h_phi = exp(fit_h) % phi;
        // Update tree
        UpdateTree(trees[j_cat][t]);
        // Sample a move
        move = 0;
        if (trees[j_cat][t]->NLeaves() > 1) move = sample_discrete(proposals_prob, 3);
        switch(move) {
        case 0:
          Grow(trees[j_cat][t], *this);
          accept_rate(0, j_cat) += flag_grow;
          break;
        case 1:
          Prune(trees[j_cat][t], *this);
          accept_rate(1, j_cat) += flag_prune;
          break;
        case 2:
          Change(trees[j_cat][t], *this);
          accept_rate(2, j_cat) += flag_change;
          break;
        }
        // Draw from the node parameters
        DrawPosterior(trees[j_cat][t]);
        // Re-fit
        f_mu.col(j_cat) = fit_h + arma::vec(g_trees.tube(j_cat, t));

        // Compute avg leaves and depth
        avg_leaves(t, j_cat) += trees[j_cat][t]->NLeaves();
        avg_depth(t, j_cat) += trees[j_cat][t]->GetDepth(trees[j_cat][t]);

        // Serialise the tree parameters
        //if (i >= nskip) serialise_tree(trees[j_cat][t], files[j_cat], np);
      }
    }

    // Update the sigma and then the leaf prior parameters
    if (update_sd_prior) {
      sigma = UpdateSigmaPrior();
      c_shape = trigamma_inverse(sigma*sigma);
      d_rate = exp(R::digamma(c_shape));
      c_logd_lgamc = c_shape * log(d_rate) - R::lgammafn(c_shape);
      sigma_mcmc[i] = sigma;
    }

    // Compute var-count, save, and update Dirichlet prior if needed
    if (sparse) {
      for (int j = 0; j < d; j ++) {
        arma::uvec vc = GetVarCount(j);
        list_splitprobs[j] = UpdateSplitProbs(vc, alpha_sparse[j], p);
        if (alpha_random) {
          slp = 0.0;
          for (int k = 0; k < p; k++) slp += std::log(list_splitprobs[j][k]);
          alpha_sparse[j] = UpdateAlphaDirchlet(alphas_grid,
                                                log_posterior_alpha_sparse,
                                                slp, (double)p, k_grid);

        }
      }
    }

    if (i >= nskip) {
      // Save draws
      draws.slice(i - nskip) = f_lambda.each_col() / rt;
      draws_phi.col(i - nskip) = phi;
      // Compute var-count, save, and update Dirichlet prior if needed
      for (int j = 0; j < d; j ++) {
        arma::uvec vc = GetVarCount(j);
        varcount_mcmc.slice(i - nskip).col(j) = vc;
        if (sparse) list_splitprobs[j] = UpdateSplitProbs(vc, alpha_sparse[j], p);
        if (alpha_random) {
          slp = 0.0;
          for (int k = 0; k < p; k++) slp += std::log(list_splitprobs[j][k]);
          alpha_sparse[j] = UpdateAlphaDirchlet(alphas_grid,
                                                log_posterior_alpha_sparse,
                                                slp, (double)p, k_grid);

        }
      }
    }
  }
  // Close the files
  //for (int j=0; j < d; j++) files[j].close();

}

// Exposing a C++ class in R
//using namespace Rcpp;
RCPP_MODULE(multinomial_bart) {

  // Expose class MultinomialBART as "MultinomialBART" on the R side
  Rcpp::class_<MultinomialBART>("MultinomialBART")

  // Exposing constructor
  .constructor<arma::umat, arma::mat>()

  // Exposing member functions
  .method("SetMCMC", &MultinomialBART::SetMCMC)
  .method("RunMCMC", &MultinomialBART::RunMCMC)
  .method("Predict", &MultinomialBART::Predict)
  .method("GetVarCount", &MultinomialBART::GetVarCount2)

  // Exposing some attributes
  .field("draws", &MultinomialBART::draws)
  .field("draws_phi", &MultinomialBART::draws_phi)
  .field("varcount_mcmc", &MultinomialBART::varcount_mcmc)
  .field("splitprobs", &MultinomialBART::list_splitprobs)
  .field("avg_leaves", &MultinomialBART::avg_leaves)
  .field("avg_depth", &MultinomialBART::avg_depth)
  .field("accept_rate", &MultinomialBART::accept_rate)

  .field("alpha_sparse", &MultinomialBART::alpha_sparse)
  .field("sigma_mcmc", &MultinomialBART::sigma_mcmc)

  .field("path_out", &MultinomialBART::path_out)
  .field("ntrees", &MultinomialBART::ntrees)
  .field("ndpost", &MultinomialBART::ndpost)
  ;

}
