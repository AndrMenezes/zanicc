#include "multinomial_ln_bart.h"
#include "tree_mcmc.h"
#include "write_read.h"
#include "utils.h"

constexpr double PI_2 = 6.283185307179586231996;

MultinomialLNBART::MultinomialLNBART(const arma::umat &Y, const arma::mat &X) : Y(Y), X(X) {
  d = Y.n_cols;
  dm1 = d-1;
  n = X.n_rows;
  p = X.n_cols;
  n_trials = sum(Y, 1);
}

// Constructor for loading a fitted model
MultinomialLNBART::MultinomialLNBART(int d, int p): d(d), p(p) {}

MultinomialLNBART::~MultinomialLNBART() {
  for (int j=0; j < d; j++) {
    for (int h=0; h < ntrees; h++) {
      delete trees[j][h];
    }
  }
}

double MultinomialLNBART::lml(Node *leaf) {
  return (c_logd_lgamc
          + R::lgammafn(leaf->ss1(0) + c_shape)
          - (leaf->ss1(0) + c_shape) * log(leaf->ss2(0) + d_rate));
}

void MultinomialLNBART::UpdateSuffStats(Node *leaf) {
  double s = 0.0, r = 0.0;
  for (auto id : leaf->ids) {
    s += Y(id, j_cat);
    r += fit_h_phi(id);
  }
  leaf->ss1(0) = s;
  leaf->ss2(0) = r;
}

void MultinomialLNBART::UpdateTree(Node *tree) {
  std::vector<Node*> leaves;
  tree->GetLeaves(leaves);
  for (size_t t = 0; t < leaves.size(); t++) {
    UpdateSuffStats(leaves[t]);
  }
}

void MultinomialLNBART::DrawPosterior(Node *tree) {
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
arma::uvec MultinomialLNBART::GetVarCount(int j) {
  arma::uvec varcount = arma::zeros<arma::uvec>(p);
  for (int t = 0; t < ntrees; t++) {
    ComputeVarCount(trees[j][t], varcount);
  }
  return varcount;
}

void MultinomialLNBART::ComputeVarCount(Node *tree, arma::uvec &varcount) {
  if (tree->is_leaf == 0) {
    int id_j = tree->predictor;
    varcount(id_j)++;
    ComputeVarCount(tree->left, varcount);
    ComputeVarCount(tree->right, varcount);
  }
}

// Get the var-count after fitting the model
arma::ucube MultinomialLNBART::GetVarCount2(int p, int d, int n_samples, int ntrees,
                                          std::string forests_dir) {

  int np = 1;
  arma::ucube varcount = arma::zeros<arma::ucube>(p, d, n_samples);
  arma::uvec vc = arma::zeros<arma::uvec>(p);

  // Iterate over categories (this can be done in parallel)
  for (int j = 0; j < d; j++) {
    // Open file of category j
    std::ifstream is(forests_dir + "/forests_j" + std::to_string(j) + ".bin");
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
double MultinomialLNBART::GetMu(Node *tree, const arma::rowvec &x) {
  if (tree->is_leaf) return tree->mu(0);
  if (x[tree->predictor] <= tree->cutoff) return GetMu(tree->left, x);
  else return GetMu(tree->right, x);
}

void MultinomialLNBART::Predict(arma::mat &X_, int n_samples,
                                int ntrees,
                                std::string forests_dir,
                                std::string output_dir,
                                int verbose) {
  int np = 1;
  int n_ = X_.n_rows;
  arma::mat f_pred;
  // arma::cube draws_pred = arma::zeros<arma::cube>(n_, d, n_samples);

  // Open files to read the the forests
  std::vector<std::ifstream> files;
  for (int j=0; j < d; j++) {
    std::string ff = forests_dir + "/forests_" + std::to_string(j) + ".bin";
    files.emplace_back(ff, std::ios::binary);
  }

  // Open file to save the predictions
  std::ofstream fout(output_dir + "/theta_ij.bin", std::ios::app | std::ios::binary);

  double progress = 0.0;
  // Iterate over the MCMC samples
  for (int k = 0; k < n_samples; k++) {
    if (verbose) {
      progress = (double) 100 * k / n_samples;
      Rprintf("\r");
      Rprintf("%3.2f%% Completed", progress);
    }
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
    // Save prediction of sample t and category j
    fout.write(reinterpret_cast<const char*>(f_pred.memptr()), sizeof(double)*n_*d);
    //draws_pred.slice(k) = f_pred;
  }
  fout.close();
  for (int j=0; j<d; j++) files[j].close();

  //return draws_pred;
}

double MultinomialLNBART::UpdateSigmaPrior() {

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

// Log-target of u_i
double MultinomialLNBART::LogTargetU(arma::vec &u, arma::uvec &y,
                                            arma::vec &lambda, int &n_trial) {
  double l = 0.0;
  std::vector<double> lterms(d, 0.0);
  for (int j=0; j< d; j++) {
    l += y[j] * u[j];
    lterms[j] = lambda[j] + u[j];
  }
  return l - n_trial * log_sum_exp(lterms);
}

// ESS to sample from u_i
arma::vec MultinomialLNBART::ESSDiag(arma::vec &v, arma::uvec &y,
                                            arma::vec &log_lambda,
                                            arma::vec &sigmas,
                                            int n_trial) {
  // Draw from the prior
  arma::vec nu = arma::zeros<arma::vec>(dm1);
  for (int j=0; j < dm1; j++) nu[j] = R::norm_rand() * sigmas[j];
  // log-likelihood threshold
  double logy = log(R::unif_rand());
  arma::vec u = B * v;
  logy += LogTargetU(u, y, log_lambda, n_trial);
  // Draw an angle
  double theta = R::unif_rand() * PI_2;
  double theta_max = theta;
  double theta_min = theta - PI_2;
  // Draw proposal
  arma::vec v_prop = v * cos(theta) + nu * sin(theta);
  arma::vec u_prop = B * v_prop;
  do {
    if (LogTargetU(u_prop, y, log_lambda, n_trial) > logy) break;
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    theta = theta_min + (theta_max - theta_min) * R::unif_rand();
    v_prop = v * cos(theta) + nu * sin(theta);
    u_prop = B * v_prop;
  } while (true);
  return v_prop;
}

// ESS for update v_i under the "full" covariance prior (also work for the factor model)
arma::vec MultinomialLNBART::ESSFull(arma::vec &v, arma::uvec &y,
                                            arma::vec &lambda, int n_trial) {
  // Draw from the prior
  arma::rowvec ep = arma::randn<arma::rowvec>(dm1);
  arma::vec nu = (ep * chol_Sigma_V).t();
  // log-likelihood threshold
  double logy = log(R::unif_rand());
  arma::vec u = B * v;
  logy += LogTargetU(u, y, lambda, n_trial);
  // Draw an angle
  double theta = R::unif_rand() * PI_2;
  double theta_max = theta;
  double theta_min = theta - PI_2;
  // Draw proposal
  arma::vec v_prop = v * cos(theta) + nu * sin(theta);
  arma::vec u_prop = B * v_prop;
  do {
    if (LogTargetU(u_prop, y, lambda, n_trial) > logy) break;
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    theta = theta_min + (theta_max - theta_min) * R::unif_rand();
    v_prop = v * cos(theta) + nu * sin(theta);
    u_prop = B * v_prop;
  } while (true);
  return v_prop;
}

// Update \Sigma = I_d diag(\sigma_1, ..., \sigma_{d-1}) using inv-gamma conjugate
// priors
void MultinomialLNBART::UpdateSigmaDiag() {
  // Update \mathbf{u}_{i} using ESS
  for (int k=0; k < n; k++) {
    v_cur = V.row(k).t();
    y_cur = Y.row(k).t();
    // lambda_cur = f_lambda.row(k).t();
    lambda_cur = f_mu.row(k).t();
    v_cur = ESSDiag(v_cur, y_cur, sigmas, lambda_cur, n_trials[k]);
    V.row(k) = v_cur.t();
  }
  // Transform back for other updates
  U = V * Bt;
  Sigma_U = U.t() * U / n;
  // Draw from \sigma^2_{j}
  for (int j=0; j < dm1; j++) {
    double sv = arma::dot(V.col(j), V.col(j));
    sigmas[j] = 1 / R::rgamma(shape_post_sigma, 1 / (sv/2 + b_sigma));
    Sigma_V(j, j) = sigmas[j];
    sigmas[j] = sqrt(sigmas[j]);
  }
}

// Update \Sigma using inv-Wishart conjugate prior
void MultinomialLNBART::UpdateSigmaFull() {
  // Update \mathbf{u}_{i} using ESS
  for (int k=0; k < n; k++) {
    v_cur = V.row(k).t();
    y_cur = Y.row(k).t();
    lambda_cur = f_mu.row(k).t();
    v_cur = ESSFull(v_cur, y_cur, lambda_cur, n_trials[k]);
    V.row(k) = v_cur.t();
  }
  // Draw from Sigma ~ inv-Wishart
  // TODO: use an C++ implementation that sample from the cholesky of Sigma_U directly
  Sigma_V = arma::iwishrnd(V.t()*V + Psi_prior, df_wishart);
  chol_Sigma_V = arma::chol(Sigma_V);
  // Transform back for other updates
  U = V * Bt;
  Sigma_U = B * Sigma_V * Bt;
}

// Sigma = H H' + \Psi
void MultinomialLNBART::UpdateSigmaFactorModel() {
  // Update \mathbf{u}_{i} using ESS
  for (int k=0; k < n; k++) {
    v_cur = V.row(k).t();
    y_cur = Y.row(k).t();
    lambda_cur = f_mu.row(k).t();
    v_cur = ESSFull(v_cur, y_cur, lambda_cur, n_trials[k]);
    V.row(k) = v_cur.t();
  }
  // Updates of factor model
  UpdateH();
  UpdateGamma();
  UpdatePsis();
  Sigma_V = Gamma * Gamma.t() + arma::diagmat(psis);
  chol_Sigma_V = arma::chol(Sigma_V);
  U = V * Bt;
  Sigma_U = B * Sigma_V * Bt;
}

// Sigma = H H' + \Psi, where the loading matrix H has a MGP prior
void MultinomialLNBART::UpdateSigmaFactorModelMGP() {
  // Update \mathbf{u}_{i} using ESS
  for (int k=0; k < n; k++) {
    v_cur = V.row(k).t();
    y_cur = Y.row(k).t();
    lambda_cur = f_mu.row(k).t();
    v_cur = ESSFull(v_cur, y_cur, lambda_cur, n_trials[k]);
    V.row(k) = v_cur.t();
  }
  // Updates of factor model
  UpdateH();
  UpdateGammaChol();
  UpdateLocalShrinkage();
  UpdateGlobalShrinkage();
  UpdatePsis();
  Sigma_V = Gamma * Gamma.t() + arma::diagmat(psis);
  chol_Sigma_V = arma::chol(Sigma_V);
  U = V * Bt;
  Sigma_U = B * Sigma_V * Bt;
}

// Full conditional of H
void MultinomialLNBART::UpdateH() {
  GammaPsis = Gamma;
  GammaPsis.each_col() /= psis;
  if (q_factors == 1) {
    M = (V * GammaPsis) / (arma::accu(GammaPsis % Gamma) + 1.0);
    Hmat = M + arma::randn<arma::mat>(n, 1);
  } else {
    arma::mat Z = arma::randn<arma::mat>(n, q_factors);
    // 1. Compute Cholesky factorisation
    Q_chol = arma::chol(Gamma.t() * GammaPsis + Iq);
    // 2. Compute the inverse of using backsolve
    Q_chol_inv = arma::solve(arma::trimatu(Q_chol), Iq, arma::solve_opts::fast);
    // 3. Compute the mean parameter
    M = V * (GammaPsis * Q_chol_inv * Q_chol_inv.t());
    // 4. Generate using traditional Cholesky approach
    Hmat = M + Z * Q_chol_inv.t();
  }
}

// Full conditional of Gamma
void MultinomialLNBART::UpdateGamma() {
  HtH = Hmat.t() * Hmat;
  arma::eig_sym(s_eigen, W_eigen, HtH);
  HVp = Hmat.t() * V;
  HVp.each_row() /= psis.t();
  for (int j=0; j < dm1; j++) {
    arma::vec den = s_eigen / psis(j) + 1.0 / sigma2_gamma;
    arma::vec b = (W_eigen.t() * HVp.col(j)) / den;
    for (int i=0; i < q_factors; i++) b[i] += R::norm_rand() / std::sqrt(den[i]);
    Gamma.row(j) = (W_eigen * b).t();
  }
}

// Full conditional of Gamma under MGP prior
void MultinomialLNBART::UpdateGammaChol() {
  HtH = Hmat.t() * Hmat;
  HVp = Hmat.t() * V;
  HVp.each_row() /= psis.t();
  for (int j=0; j < dm1; j++) {
    Q = HtH / psis[j];
    Q.diag() += taus_gs % Phis_ls.row(j).t();
    // 1. Compute Cholesky factorisation
    Q_chol = arma::chol(Q);
    // 2. Compute the inverse of using backsolve
    Q_chol_inv = arma::solve(arma::trimatu(Q_chol), Iq,
                             arma::solve_opts::fast);
    // 3. Compute the mean parameter
    mu_gamma_chol = (Q_chol_inv * Q_chol_inv.t()) * HVp.col(j);
    // 4. Generate using traditional approach
    Gamma.row(j) = mu_gamma_chol.t() + arma::randn<arma::rowvec>(q_factors) * Q_chol_inv;
  }
}

// Full conditional of the local shrinkage parameters, \phi_{jk}
void MultinomialLNBART::UpdateLocalShrinkage() {
  for (int j=0; j < dm1; j++) {
    for (int k=0; k < q_factors; k++) {
      Phis_ls(j, k) = R::rgamma(shape_post_lsphis, 2.0 / (shape_lsphis + taus_gs(k) * std::pow(Gamma(j, k), 2.0)));
    }
  }
}

void MultinomialLNBART::UpdateGlobalShrinkage() {
  GammaPhi = Phis_ls % arma::pow(Gamma, 2.0);
  // Rate parameter of first delta_1, arma::dot is sum of elementwise multiplication.
  rt_gs = 1.0 + 0.5 * arma::dot(taus_gs, arma::sum(GammaPhi, 0).t()) / deltas_gs[0];
  deltas_gs[0] = R::rgamma(shape_post_gs_delta1, 1.0 / rt_gs);
  taus_gs = arma::cumprod(deltas_gs);
  // Update the remaining
  for (int h=1; h < q_factors; h++) {
    rt_gs = 1.0 + 0.5 / deltas_gs[h] * arma::dot(taus_gs.subvec(h, q_factors-1), arma::sum(GammaPhi.cols(h, q_factors-1), 0).t());
    deltas_gs[h] = R::rgamma(a2_gs + 0.5*dm1*(q_factors - h + 1.0) , 1.0 / rt_gs);
    taus_gs = arma::cumprod(deltas_gs);
  }
}


// Full conditional of \psis
void MultinomialLNBART::UpdatePsis() {
  R_psis = V - Hmat * Gamma.t();
  for (int j=0; j < dm1; j++) {
    rt_psis = arma::dot(R_psis.col(j), R_psis.col(j));
    psis[j] = 1.0 / R::rgamma(shape_psis, 1.0 / (rt_psis / 2.0 + b_psi));
  }
}

// Update concentration parameter of Dirichlet prior
void MultinomialLNBART::UpdateAlphaDART(int &j) {
  slp = 0.0;
  for (int k = 0; k < p; k++) slp += std::log(list_splitprobs[j][k]);
  alpha_sparse[j] = UpdateAlphaDirchlet(alphas_grid,
                                        log_posterior_alpha_sparse,
                                        slp, (double)p, k_grid);
}


// Perform back-fitting for a given category (j) and tree (t)
void MultinomialLNBART::BackFit(int &j, int &t) {
  // Set split-probs of category j, this is used in the Grow, Prune, Change
  splitprobs = list_splitprobs[j];
  // Back-fit
  fit_h = f_mu.col(j) - arma::vec(g_trees.tube(j, t));
  fit_h_phi = exp(fit_h + log(phi) + U.col(j));
  // Update tree
  UpdateTree(trees[j][t]);
  // Sample a move
  move = 0;
  if (trees[j][t]->NLeaves() > 1) move = sample_discrete(proposals_prob, 3);
  switch(move) {
  case 0:
    Grow(trees[j][t], *this);
    accept_rate(0, j) += flag_grow;
    break;
  case 1:
    Prune(trees[j][t], *this);
    accept_rate(1, j) += flag_prune;
    break;
  case 2:
    Change(trees[j][t], *this);
    accept_rate(2, j) += flag_change;
    break;
  }
  // Draw from the node parameters
  DrawPosterior(trees[j][t]);
  // Re-fit
  f_mu.col(j) = fit_h + arma::vec(g_trees.tube(j, t));
}

// Set the model
void MultinomialLNBART::SetMCMC(double v0, int ntrees_,
                                       arma::mat B_, int covariance_type_,
                                       double a_sigma_, double b_sigma_,
                                       arma::mat Psi_prior_, double nu_prior_,
                                       int q_factors_, double sigma2_gamma_,
                                       double a_psi_, double b_psi_,
                                       double shape_lsphis_,
                                       double a1_gs_, double a2_gs_,
                                       int ndpost_, int nskip_,
                                       int numcut_, double power_ , double base_,
                                       std::vector<double> proposals_prob_,
                                       int update_sd_prior_, double s2_0_, double w_ss_,
                                       std::vector<std::vector<double>> list_splitprobs_, int sparse_,
                                       std::vector<double> sparse_parms_, std::vector<double> alpha_sparse_,
                                       int alpha_random_, arma::mat xinfo, std::string forests_dir_,
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
  list_splitprobs = list_splitprobs_;
  sparse = sparse_;
  sparse_parms = sparse_parms_;
  alpha_random = alpha_random_;
  alpha_sparse = alpha_sparse_;
  forests_dir = forests_dir_;
  s2_0 = s2_0_;
  w_ss = w_ss_;
  update_sd_prior = update_sd_prior_;
  B = B_;
  a_sigma = a_sigma_;
  b_sigma = b_sigma_;
  a_sigma = a_sigma_;
  b_sigma = b_sigma_;
  covariance_type = covariance_type_,
  Psi_prior = Psi_prior_,
  nu_prior = nu_prior_,
  q_factors = q_factors_;
  sigma2_gamma = sigma2_gamma_;
  a_psi = a_psi_;
  b_psi = b_psi_;
  shape_lsphis = shape_lsphis_;
  a1_gs = a1_gs_;
  a2_gs = a2_gs_,
  keep_draws = keep_draws_;
  save_trees = save_trees_;


  // Define the tree terminal node prior parameters
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

  // Initialise variables related to the random effect and its covariance
  Bt = B.t();
  y_cur = arma::zeros<arma::uvec>(d);
  v_cur = arma::zeros<arma::vec>(dm1);
  V = arma::randn<arma::mat>(n, dm1);
  U = V * Bt;
  Sigma_U = U.t() * U / n;

  if (covariance_type == 0) {
    Sigma_V = arma::zeros<arma::mat>(dm1, dm1);
    sigmas = arma::zeros<arma::vec>(dm1);
    shape_post_sigma = (double)n/2.0 + a_sigma;
    for (int j=0; j < dm1; j++) {
      double sv = arma::dot(V.col(j), V.col(j));
      sigmas[j] = 1.0 / R::rgamma(shape_post_sigma, 1.0 / (sv/2.0 + b_sigma));
      Sigma_V[j] = sigmas[j];
      sigmas[j] = sqrt(sigmas[j]);
    }
    chol_Sigma_V = arma::chol(Sigma_V);
  } else if (covariance_type == 1){
    Sigma_V = V.t() * V / n;
    chol_Sigma_V = arma::chol(Sigma_V);
    df_wishart = n + nu_prior;
  } else {
    Sigma_V = V.t() * V / n;
    chol_Sigma_V = arma::chol(Sigma_V);
    Gamma = arma::randn<arma::mat>(dm1, q_factors);
    Hmat = arma::randn<arma::mat>(n, q_factors);
    psis = arma::zeros<arma::vec>(dm1);
    for (int j=0; j < dm1; j++) psis[j] = 1.0 / R::rgamma(a_psi, 1.0 / b_psi);
    // Initialise some variables used in the update of factor model
    // (this avoid need to define then during the loop)
    Iq = arma::eye(q_factors, q_factors);
    GammaPsis = Gamma;
    GammaPsis.each_col() /= psis;
    M = arma::zeros<arma::mat>(n, q_factors);
    Q = arma::zeros<arma::mat>(q_factors, q_factors);
    Q_chol = arma::zeros<arma::mat>(q_factors, q_factors);
    Q_chol_inv = arma::zeros<arma::mat>(q_factors, q_factors);
    HtH = Hmat.t() * Hmat;
    HVp = Hmat.t() * V;
    s_eigen = arma::zeros<arma::vec>(q_factors);
    W_eigen = arma::zeros<arma::mat>(q_factors, q_factors);
    shape_psis = (double)(n) / 2.0 + a_psi;
    R_psis = arma::zeros<arma::mat>(n, dm1);
    // Initialise variables MGP prior specific variables
    if (covariance_type == 3) {
      // taus_gs, deltas_gs, mu
      mu_gamma_chol = arma::zeros<arma::vec>(q_factors);
      Phis_ls = arma::zeros<arma::mat>(dm1, q_factors);
      GammaPhi = arma::zeros<arma::mat>(dm1, q_factors);
      deltas_gs = arma::zeros<arma::vec>(q_factors);
      // arma::vec taus = arma::zeros<arma::vec>(q_factors);
      for (int k=0; k < q_factors; k++) {
        for (int j=0; j < dm1; j++) {
          // local-shrinkage
          Phis_ls(j, k) = R::rgamma(shape_lsphis / 2, 2 / shape_lsphis);
        }
      }
      // global-shrinkage
      deltas_gs[0] = R::rgamma(a1_gs, 1.0);
      for (int j=1; j < q_factors; j++) deltas_gs[j] = R::rgamma(a2_gs, 1.0);
      taus_gs = arma::cumprod(deltas_gs);
      // Compute the posterior shape parameters of local and shrinkage
      shape_post_lsphis = (shape_lsphis + 1.0) / 2.0;
      shape_post_gs_delta1 = a1_gs + dm1 * q_factors/2;
    }
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

  // Initialise flags for acceptance rate
  flag_grow = 0;
  flag_prune = 0;
  flag_change = 0;

  // Acceptance rate and average
  accept_rate = arma::zeros<arma::umat>(3, d);
  avg_leaves = arma::zeros<arma::umat>(ntrees, d);
  avg_depth = arma::zeros<arma::umat>(ntrees, d);

  // flag for the MH move and aux for slp = sum(log(splitprobs))
  move = 0; slp = 0.0;
  // Initialise the aux variable to keep the varcount
  vc = arma::zeros<arma::uvec>(p);

  std::cout << "Model set up!" << "\n\n";
}

void MultinomialLNBART::RunMCMC() {

  // Use the serialise the trees (indicates how many parameters in the terminal nodes)
  int np = 1;

  // Initialise container for keep the draws
  if (keep_draws) {
    draws_theta = arma::zeros<arma::cube>(n, d, ndpost);
    draws_vartheta = arma::zeros<arma::cube>(n, d, ndpost);
    draws_chol_Sigma_V = arma::zeros<arma::cube>(dm1, dm1, ndpost);
    //draws_phi = arma::zeros<arma::mat>(n, ndpost);
  }
  // Storage object for varcount
  varcount_mcmc = arma::zeros<arma::ucube>(p, d, ndpost);

  // Initialise variables for the backfitting
  rt = arma::zeros<arma::vec>(n);
  fit_h = arma::zeros<arma::vec>(n);
  f_mu = arma::zeros<arma::mat>(n, d);
  f_lambda = arma::zeros<arma::mat>(n, d);

  // Start burn-in
  std::cout << "Doing the warm-up (burn-in) of " << nskip << "\n\n";
  double progress = 0;
  for (int i = 0; i < nskip; i++) {
    progress = (double) 100 * i / nskip;
    Rprintf("%3.2f%% Warm-up completed", progress);
    Rprintf("\r");
    // Save \lambda_j = exp(\mu_j)
    f_lambda = exp(f_mu + U);
    // Update latent variable \phi_i ~ Gamma[N_i, \sum f_lambda]
    rt = arma::sum(f_lambda, 1);
    for (int k=0; k < n; k++) phi(k) = R::rgamma(n_trials(k), 1.0) / rt(k);
    // Iterate over categories
    for (int j = 0; j < d; j++) {
      // "Gambiarra" to pass the category index inside an attribute of the class
      j_cat = j;
      // Iterate over trees
      for (int t=0; t < ntrees; t++) BackFit(j, t);
      // Update the DART parameters if required
      if (sparse) {
        vc = GetVarCount(j);
        list_splitprobs[j] = UpdateSplitProbs(vc, alpha_sparse[j], p);
        if (alpha_random) UpdateAlphaDART(j);
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
    // Update the random effects U's and its covariance Sigma
    switch (covariance_type) {
      case 0: UpdateSigmaDiag(); break;
      case 1: UpdateSigmaFull(); break;
      case 2: UpdateSigmaFactorModel(); break;
      case 3: UpdateSigmaFactorModelMGP(); break;
    }
  }

  // Open file to write the forests FOR each category
  std::vector<std::ofstream> files;
  for (int j=0; j < d; j++) {
   std::string ff = forests_dir + "/forests_" + std::to_string(j) + ".bin";
   files.emplace_back(ff, std::ios::binary | std::ios::app);
  }
  // Save the cholesky of covariance matrix
  std::ofstream ff_Sigma_V(forests_dir + "/chol_Sigma_V.bin",
                           std::ios::binary | std::ios::app);

  std::cout << "Starting post-burn-in iterations of " << ndpost << "\n\n";
  progress = 0.0;
  for (int i=0; i < ndpost; i++) {
    progress = (double) 100 * i / ndpost;
    Rprintf("%3.2f%% Posterior samples completed", progress);
    Rprintf("\r");

    // \lambda_j = exp(\mu_j)
    f_lambda = exp(f_mu + U);
    // Update latent variable \phi_i ~ Gamma[N_i, \sum f_lambda]
    rt = arma::sum(f_lambda, 1);
    for (int k=0; k < n; k++) phi(k) = R::rgamma(n_trials(k), 1.0) / rt(k);

    // Iterate over categories
    for (int j = 0; j < d; j++) {
      // "Gambiarra" to pass the category index inside an attribute of the class
      j_cat = j;
      // Iterate over trees
      for (int t=0; t < ntrees; t++) {
        BackFit(j, t);
        // Compute avg leaves and depth
        avg_leaves(t, j) += trees[j][t]->NLeaves();
        avg_depth(t, j) += trees[j][t]->GetDepth(trees[j][t]);
        // Serialise the tree parameters
        if (save_trees) serialise_tree(trees[j][t], files[j], np);
      }
      // Get varcount
      vc = GetVarCount(j);
      varcount_mcmc.slice(i).col(j) = vc;
      // Update the DART parameters if required
      if (sparse) {
        list_splitprobs[j] = UpdateSplitProbs(vc, alpha_sparse[j], p);
        if (alpha_random) UpdateAlphaDART(j);
      }
    }
    // Update the random effects and its covariance
    switch (covariance_type) {
      case 0: UpdateSigmaDiag(); break;
      case 1: UpdateSigmaFull(); break;
      case 2: UpdateSigmaFactorModel(); break;
      case 3: UpdateSigmaFactorModelMGP(); break;
    }
    // Update the sigma and then the leaf prior parameters
    if (update_sd_prior) {
      sigma = UpdateSigmaPrior();
      c_shape = trigamma_inverse(sigma*sigma);
      d_rate = exp(R::digamma(c_shape));
      c_logd_lgamc = c_shape * log(d_rate) - R::lgammafn(c_shape);
      sigma_mcmc[i] = sigma;
    }
    // Save draws
    if (keep_draws) {
      // with the random effect
      f_lambda = exp(f_mu + U);
      draws_vartheta.slice(i) = f_lambda.each_col() / arma::sum(f_lambda, 1);
      // without random effect
      f_lambda = exp(f_mu);
      draws_theta.slice(i) = f_lambda.each_col() / arma::sum(f_lambda, 1);
      draws_chol_Sigma_V.slice(i) = chol_Sigma_V;
      //draws_phi.col(i) = phi;
    }
    if (save_trees) {
      ff_Sigma_V.write(reinterpret_cast<const char*>(chol_Sigma_V.memptr()), sizeof(double)*dm1*dm1);
    }
  }
  // Close the files
  for (int j=0; j < d; j++) files[j].close();

}

// Exposing a C++ class in R
//using namespace Rcpp;
RCPP_MODULE(multinomial_lognormal_bart) {

  // Expose class MultinomialLNBART as "MultinomialLNBART" on the R side
  Rcpp::class_<MultinomialLNBART>("MultinomialLNBART")

  // Exposing constructor
  .constructor<arma::umat, arma::mat>()
  .constructor<int, int>()

  // Exposing member functions
  .method("SetMCMC", &MultinomialLNBART::SetMCMC)
  .method("RunMCMC", &MultinomialLNBART::RunMCMC)
  .method("Predict", &MultinomialLNBART::Predict)
  .method("GetVarCount", &MultinomialLNBART::GetVarCount2)

  // Exposing some attributes
  .field("draws_theta", &MultinomialLNBART::draws_theta)
  .field("draws_vartheta", &MultinomialLNBART::draws_vartheta)
  .field("draws_phi", &MultinomialLNBART::draws_phi)
  .field("draws_chol_Sigma_V", &MultinomialLNBART::draws_chol_Sigma_V)
  .field("varcount_mcmc", &MultinomialLNBART::varcount_mcmc)
  .field("splitprobs", &MultinomialLNBART::list_splitprobs)
  .field("avg_leaves", &MultinomialLNBART::avg_leaves)
  .field("avg_depth", &MultinomialLNBART::avg_depth)
  .field("accept_rate", &MultinomialLNBART::accept_rate)

  .field("alpha_sparse", &MultinomialLNBART::alpha_sparse)
  .field("sigma_mcmc", &MultinomialLNBART::sigma_mcmc)

  .field("forests_dir", &MultinomialLNBART::forests_dir)
  .field("ntrees", &MultinomialLNBART::ntrees)
  .field("ndpost", &MultinomialLNBART::ndpost)
  ;

}
