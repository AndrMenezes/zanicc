#include "zanim_shared_bart_probit.h"
#include "multinomial_bart.h"
#include "probit.h"
#include "tree_mcmc.h"
#include "write_read.h"
#include "pmfs.h"
#include "utils.h"

constexpr double PI_2 = 6.283185307179586231996;

ZANIMSharedBARTProbit::ZANIMSharedBARTProbit(const arma::umat &Y,
                                           const arma::mat &X_theta,
                                           const arma::mat &X_zeta)
  : Y(Y), X_theta(X_theta), X_zeta(X_zeta) {

  n = Y.n_rows;
  d = Y.n_cols;
  p_theta = X_theta.n_cols;
  p_zeta = X_zeta.n_cols;

}

ZANIMSharedBARTProbit::~ZANIMSharedBARTProbit(){};

void ZANIMSharedBARTProbit::SetMCMC(double v0_theta, double k_zeta,
                              int ntrees_theta_, int ntrees_zeta_, int ndpost_,
                              int nskip_,
                              int numcut, double power, double base,
                              std::vector<double> proposals_prob_,
                              int update_sd_prior_, double s2_0_, double w_ss_,
                              std::vector<double> splitprobs_zi,
                              std::vector<double> splitprobs_mult,
                              int sparse_zi_,
                              int sparse_mult_,
                              std::vector<double> sparse_parms_zi_,
                              std::vector<double> sparse_parms_mult_,
                              std::vector<double> alpha_sparse_zi_,
                              double alpha_sparse_mult_,
                              int alpha_random_zi_,
                              int alpha_random_mult_,
                              arma::mat xinfo, std::string path_out_,
                              int keep_draws_, int save_trees_) {
  Rcpp::RNGScope scope;                              
  ntrees_theta = ntrees_theta_;
  ntrees_zeta = ntrees_zeta_;
  path_out = path_out_;
  proposals_prob = proposals_prob_;
  ndpost = ndpost_;
  nskip = nskip_;
  //splitprobs_mult = splitprobs_mult_;
  sparse_zi = sparse_zi_;
  sparse_mult = sparse_mult_;
  sparse_parms_zi = sparse_parms_zi_;
  sparse_parms_mult = sparse_parms_mult_;
  alpha_sparse_zi = alpha_sparse_zi_;
  alpha_sparse_mult = alpha_sparse_mult_;
  alpha_random_zi = alpha_random_zi_;
  alpha_random_mult = alpha_random_mult_;
  update_sd_prior = update_sd_prior_;
  keep_draws = keep_draws_;
  save_trees = save_trees_;

  // Initialize multinomial BART
  bart_mult = new MultinomialSharedBART(Y, X_theta);
  bart_mult->SetMCMC(v0_theta, ntrees_theta_, ndpost, nskip, numcut,
                     power, base, proposals_prob_, update_sd_prior_, s2_0_, w_ss_,
                     splitprobs_mult, sparse_mult, sparse_parms_mult,
                     alpha_sparse_mult, alpha_random_mult, xinfo, path_out_,
                     keep_draws, save_trees);

  // For each category initialize the BART zero-inflation model
  list_bart_zi.reserve(d);
  for (int j = 0; j < d; j++) {
    arma::uvec yy = Y.col(j) == 0;
    ProbitBART *m = new ProbitBART(yy, X_zeta);
    m->SetMCMC(k_zeta, ntrees_zeta_, ndpost, nskip, 1, numcut, power,
               base, proposals_prob_, splitprobs_zi, sparse_zi, sparse_parms_zi,
               alpha_sparse_zi[j], alpha_random_zi, xinfo, path_out_);
    list_bart_zi.push_back(m);
  }

  // Scale parameter of Gamma latent variable \phi_i
  rt = arma::zeros<arma::vec>(n);
  // Auxiliary vectors/matrices for the back-fitting in Multinomial
  // fit_h = arma::zeros<arma::vec>(n);
  fit_h = arma::zeros<arma::mat>(n, d);
  f_mu = arma::zeros<arma::mat>(n, d);
  f_lambda = arma::zeros<arma::mat>(n, d);
  varthetas = arma::zeros<arma::mat>(n, d);

  // Auxiliary vectors/matrices for the back-fitting in shared logit BART
  fit_0_h = arma::zeros<arma::vec>(n);
  f0_mu = arma::zeros<arma::mat>(n, d);
  //zetas = arma::zeros<arma::mat>(n, d);

  // z_{ij} ~ Bernoulli[1 - zeta_j] for y_{ij} = 0, else z_{ij} = 1.
  // Vectors with the index of observations
  zero_indices.resize(d);
  zs.resize(d);
  sum_col_zeros_Y = arma::sum(Y == 0, 0).t();
  for (int j = 0; j < d; ++j) {
    zero_indices[j].resize(sum_col_zeros_Y(j));
    zs[j].resize(sum_col_zeros_Y(j));
    zero_indices[j] = arma::conv_to<std::vector<int>>::from(arma::find(Y.col(j) == 0));
    // First draw of the latent variable z_{ij}~ Bernoulli[\pi_{ij}] IF y_{ij}=0
    for (int k = 0; k < sum_col_zeros_Y(j); k++) {
      zs[j][k] = R::rbinom(1, 0.5);
    }
  }
  // Aux variable to sample a movement in the back-fitting
  move = 0;

  // Create grid for the \alpha, where \alpha = lambda * \rho / (1 - \lambda)
  // and for each \alpha compute the "constant" part of the log-posterior.
  k_grid = 1000;
  if (alpha_random_mult) {
    alphas_grid_mult.resize(k_grid);
    lp_alpha_sparse_mult.resize(k_grid);
    double al, ld;
    for (int k = 0; k < k_grid; k++) {
      ld = (double)(k + 1)/((double)(k_grid + 1));
      al = ld * sparse_parms_mult[0] / (1.0 - ld);
      alphas_grid_mult[k] = al;
      lp_alpha_sparse_mult[k] = (R::lgammafn(al)
                                   - p_theta * R::lgammafn(al / p_theta)
                                   + (sparse_parms_mult[1] - 1.0) * log(ld)
                                   + (sparse_parms_mult[2] - 1.0) * log(1.0 - ld));
    }
  }
  if (alpha_random_zi) {
    alphas_grid_zi.resize(k_grid);
    lp_alpha_sparse_zi.resize(k_grid);
    double al, ld;
    for (int k = 0; k < k_grid; k++) {
      ld = (double)(k + 1)/((double)(k_grid + 1));
      al = ld * sparse_parms_zi[0] / (1.0 - ld);
      alphas_grid_zi[k] = al;
      lp_alpha_sparse_zi[k] = (R::lgammafn(al)
                                 - p_zeta * R::lgammafn(al / p_zeta)
                                 + (sparse_parms_zi[1] - 1.0) * log(ld)
                                 + (sparse_parms_zi[2] - 1.0) * log(1.0 - ld));
    }
  }

  if (update_sd_prior) sigma_mult_mcmc.resize(ndpost);

  // Cube to keep the var-count for each iteration-category/forest
  varcount_mcmc_theta = arma::zeros<arma::umat>(p_theta, ndpost);
  varcount_mcmc_zeta = arma::zeros<arma::ucube>(p_zeta, d, ndpost);

  // Average number of leaves
  avg_leaves_theta = arma::zeros<arma::uvec>(ntrees_theta);
  avg_leaves_zeta = arma::zeros<arma::umat>(ntrees_zeta, d);

  // Acceptance rate
  accept_rate_theta = arma::zeros<arma::uvec>(3);
  accept_rate_zeta = arma::zeros<arma::umat>(3, d);

  std::cout << "ZANIM BART set up!! \n\n";
}

void ZANIMSharedBARTProbit::BackFitMultinomial(int &t) {

    // Back-fit
  fit_h = f_mu - bart_mult->g_trees.slice(t);

  // std::cout << f_mu(0, j) << "\n";

  bart_mult->fit_h_phi = exp(fit_h);
  bart_mult->fit_h_phi.each_col() %= bart_mult->phi;

  //  bart_mult->fit_h_phi = exp(fit_h) % bart_mult->phi;

  for (int j = 0; j < d; j++) {
    for (int k = 0; k < sum_col_zeros_Y(j); k++) {
      bart_mult->fit_h_phi(zero_indices[j][k], j) *= zs[j][k];
    }
  }
  // Update tree
  bart_mult->UpdateTree(bart_mult->trees[t]);
  // Perform the tree movement
  move = 0;
  if (bart_mult->trees[t]->NLeaves() > 1L)
    move = sample_discrete(proposals_prob, 3.0);
  switch (move) {
  case 0:
    Grow(bart_mult->trees[t], *bart_mult);
    accept_rate_theta[0] += bart_mult->flag_grow;
    break;
  case 1:
    Prune(bart_mult->trees[t], *bart_mult);
    accept_rate_theta[1] += bart_mult->flag_prune;
    break;
  case 2:
    Change(bart_mult->trees[t], *bart_mult);
    accept_rate_theta[2] += bart_mult->flag_change;
    break;
  }
  // Draw from the node parameters
  bart_mult->DrawPosterior(bart_mult->trees[t]);
  // Re-fit
  f_mu = fit_h + bart_mult->g_trees.slice(t);
}

void ZANIMSharedBARTProbit::BackFitZI(int &j, int &t) {
  // Back-fit with partial residuals
  fit_0_h = f0_mu.col(j) - list_bart_zi[j]->g_trees.col(t);
  list_bart_zi[j]->z_h = list_bart_zi[j]->z - fit_0_h;
  // Update tree
  list_bart_zi[j]->UpdateTree(list_bart_zi[j]->trees[t]);
  // Perform the tree movement
  move = 0;
  if (list_bart_zi[j]->trees[t]->NLeaves() > 1L)
    move = sample_discrete(proposals_prob, 3.0);
  switch(move) {
  case 0:
    Grow(list_bart_zi[j]->trees[t], *list_bart_zi[j]);
    accept_rate_zeta(0, j) += list_bart_zi[j]->flag_grow;
    break;
  case 1:
    Prune(list_bart_zi[j]->trees[t], *list_bart_zi[j]);
    accept_rate_zeta(1, j) += list_bart_zi[j]->flag_prune;
    break;
  case 2:
    Change(list_bart_zi[j]->trees[t], *list_bart_zi[j]);
    accept_rate_zeta(2, j) += list_bart_zi[j]->flag_change;
    break;
  }
  // Draw from the node parameters
  list_bart_zi[j]->DrawPosterior(list_bart_zi[j]->trees[t]);
  // Re-fit
  f0_mu.col(j) = fit_0_h + list_bart_zi[j]->g_trees.col(t);
}

void ZANIMSharedBARTProbit::UpdateLatentVariables() {
  f_lambda = exp(f_mu);
  for (int j=0; j < d; j++) {
    // Update (z_{ij} | \phi_i, \lambda_j, zeta_j) ~ Bernoulli[p_{ij}] for y_{ij}=0
    for(int k = 0; k < sum_col_zeros_Y(j); k++) {
      // Get the index and the zero-inflated probability
      cur_indice = zero_indices[j][k];
      zeta_ij = R::pnorm5(f0_mu(cur_indice, j), 0.0, 1.0, 1.0, 0.0);
      // Update z_j | \phi_i
      p_ij = (1.0 - zeta_ij) * exp(-f_lambda(cur_indice, j) * bart_mult->phi(cur_indice));
      p_ij /= (zeta_ij + p_ij);
      zs[j][k] = R::rbinom(1, p_ij);
      // Zs(cur_indice, j) = zs[j][k];
      // Update the response!
      list_bart_zi[j]->y[cur_indice] = 1 - zs[j][k];
    }

    // Update latent variables for the probit-BART
    for (int k = 0; k < n; k++) {
      if (list_bart_zi[j]->y[k] == 0)
        list_bart_zi[j]->z[k] = -rtnorm(-f0_mu(k, j), 1.0, 0.0);
      else list_bart_zi[j]->z[k] = rtnorm(f0_mu(k, j), 1.0, 0.0);
    }
  }
  // Update \phi_i
  for (int k = 0; k < n; k++) {
    rt(k) = 0;
    for (int j=0; j < d; j++) {
      varthetas(k, j) = f_lambda(k, j) * (1 - list_bart_zi[j]->y[k]);
      rt(k) += varthetas(k, j);
    }
    if (bart_mult->n_trials(k) == 0) bart_mult->phi(k) = 0.0;
    else bart_mult->phi(k) = R::rgamma(bart_mult->n_trials(k), 1.0) / rt(k);
  }
  // Normalise vartheta (this is what we call abundance)
  varthetas.each_col() /= rt;
}

void ZANIMSharedBARTProbit::RunMCMC() {

  // Aux to keep the varcount
  arma::uvec vc_mult = arma::zeros<arma::uvec>(p_theta);
  arma::uvec vc_zi = arma::zeros<arma::uvec>(p_zeta);

  if (keep_draws) {
    draws_vartheta = arma::zeros<arma::cube>(n, d, ndpost);
    draws_theta = arma::zeros<arma::cube>(n, d, ndpost);
    draws_zeta = arma::zeros<arma::cube>(n, d, ndpost);
    draws_phi = arma::zeros<arma::mat>(n, ndpost);
  }

  // Aux to save the forests
  int np_t = 1, np_z = 1;
  // Aux to compute sum(log(splitprobs))
  double slp = 0.0;

  std::cout << "Doing the warm-up (burn-in) of " << nskip << "\n\n";
  double progress = 0;
  for (int i = 0; i < nskip; i++) {
    progress = (double) 100 * i / nskip;
    Rprintf("%3.2f%% Warm-up completed", progress);
    Rprintf("\r");
    // Update latent variables
    UpdateLatentVariables();
    // Update trees and leaf node parameters
    for (int t = 0; t < ntrees_theta; t++) BackFitMultinomial(t);
    for (int j = 0; j < d; j++) {
      for (int t = 0; t < ntrees_zeta; t++) BackFitZI(j, t);
    }
    // Update the sigma and then the leaf prior parameters
    if (update_sd_prior) {
      bart_mult->sigma = bart_mult->UpdateSigmaPrior();
      bart_mult->c_shape = trigamma_inverse(bart_mult->sigma*bart_mult->sigma);
      bart_mult->d_rate = exp(R::digamma(bart_mult->c_shape));
      bart_mult->c_logd_lgamc = bart_mult->c_shape * log(bart_mult->d_rate) - R::lgammafn(bart_mult->c_shape);
    }
  }

  // Open file to write the forests FOR each category
  std::ofstream file_theta(path_out + "/forests_theta.bin", std::ios::binary | std::ios::app);
  std::vector<std::ofstream> files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff2 = path_out + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_zeta.emplace_back(ff2, std::ios::binary | std::ios::app);
  }

  // Run the actual posterior samples
  std::cout << "Starting post-burn-in iterations...\n\n";
  progress = 0;
  for (int i = 0; i < ndpost; i++) {
    progress = (double) 100 * i / ndpost;
    Rprintf("%3.2f%% Posterior samples completed", progress);
    Rprintf("\r");
    // Update latent variables
    UpdateLatentVariables();
    // Update trees and leaf parameters
    for (int t = 0; t < ntrees_theta; t++) {
      BackFitMultinomial(t);
      if (save_trees) serialise_tree(bart_mult->trees[t], file_theta, d);
      avg_leaves_theta[t] += bart_mult->trees[t]->NLeaves();
    }
    // Compute varcount and update DART parameters if required
    vc_mult = bart_mult->GetVarCount();
    varcount_mcmc_theta.col(i) = vc_mult;
    if (sparse_mult)
      bart_mult->splitprobs = UpdateSplitProbs(vc_mult, alpha_sparse_mult,
                                               p_theta);
    if (alpha_random_mult) {
      slp = 0.0;
      for (int k = 0; k < p_theta; k++)
        slp += std::log(bart_mult->splitprobs[k]);
      alpha_sparse_mult = UpdateAlphaDirchlet(alphas_grid_mult,
                                              lp_alpha_sparse_mult,
                                              slp, (double)p_theta, k_grid);
    }

    // Update the \zeta_j trees
    for (int j = 0; j < d; j++) {
      for (int t = 0; t < ntrees_zeta; t++) {
        BackFitZI(j, t);
        if (save_trees) serialise_tree(list_bart_zi[j]->trees[t], files_zeta[j], np_z);
        avg_leaves_zeta(t, j) += list_bart_zi[j]->trees[t]->NLeaves();
      }
      // Compute varcount and update DART parameters if required
      vc_zi = list_bart_zi[j]->GetVarCount();
      varcount_mcmc_zeta.slice(i).col(j) = vc_zi;
      if (sparse_zi)
        list_bart_zi[j]->splitprobs = UpdateSplitProbs(vc_zi, alpha_sparse_zi[j],
                                                       p_zeta);
      // Update concentration parameter of Dirichlet prior
      if (alpha_random_zi) {
        slp = 0.0;
        for (int k = 0; k < p_zeta; k++)
          slp += std::log(list_bart_zi[j]->splitprobs[k]);

        alpha_sparse_zi[j] = UpdateAlphaDirchlet(alphas_grid_zi, lp_alpha_sparse_zi,
                                                 slp, (double)p_zeta, k_grid);
      }
    }
    // Update the sigma prior and then the leaf prior parameters
    if (update_sd_prior) {
      bart_mult->sigma = bart_mult->UpdateSigmaPrior();
      bart_mult->c_shape = trigamma_inverse(bart_mult->sigma*bart_mult->sigma);
      bart_mult->d_rate = exp(R::digamma(bart_mult->c_shape));
      bart_mult->c_logd_lgamc = bart_mult->c_shape * log(bart_mult->d_rate) - R::lgammafn(bart_mult->c_shape);
      sigma_mult_mcmc[i] = bart_mult->sigma;
    }
    // Save draws
    if (keep_draws) {
      f_lambda = exp(f_mu);
      draws_theta.slice(i) = f_lambda.each_col() / arma::sum(f_lambda, 1);
      draws_vartheta.slice(i) = varthetas;
      draws_zeta.slice(i) = f0_mu;
      draws_phi.col(i) = bart_mult->phi;
    }
  }
  // Close the files
  file_theta.close();
  for (int j=0; j < d; j++) {
    files_zeta[j].close();
  }
}

// Predict functions
arma::vec ZANIMSharedBARTProbit::GetMu(Node *tree, const arma::rowvec &x) {
  if (tree->is_leaf) return tree->mu;
  if (x[tree->predictor] <= tree->cutoff) return GetMu(tree->left, x);
  else return GetMu(tree->right, x);
}

void ZANIMSharedBARTProbit::ComputePredictProb(arma::mat &X_, int n_samples,
                                              int ntrees, std::string path_out,
                                              std::string path, int verbose) {
  int n_ = X_.n_rows;
  arma::mat f_pred = arma::zeros<arma::mat>(n_, d);

  // Open file to read the the forests
  std::ifstream is(path + "/forests_theta.bin", std::ios::binary);

  // Open file to save the predictions
  std::ofstream fout(path_out + "/theta_ij.bin", std::ios::app | std::ios::binary);

  // Iterate over the MCMC samples
  for (int t = 0; t < n_samples; t++) {
    if (verbose) {
      double progress = (double) 100 * t / n_samples;
      Rprintf("\r");
      Rprintf("%3.2f%% Completed", progress);
    }
    // Initialise vector to keep the predictions of \theta_ij
    f_pred = arma::zeros<arma::mat>(n_, d);
    // Iterate over trees
    for (int h = 0; h < ntrees; h++) {
      // Import tree
      Node *tree = deserialise_tree(is, d);
      // std::cout << tree->NLeaves() << "\n\n";
      // Do the predictions
      for (int i = 0; i < n_; i++) {
        const arma::rowvec &xi = X_.row(i);
        f_pred.row(i) += GetMu(tree, xi).t();
        //std::cout << "j=" << j << " h=" << h << " mu_ij=" << mu << "\n";
      }
      delete tree;
    }
    f_pred = exp(f_pred);
    f_pred = f_pred.each_col() / arma::sum(f_pred, 1);
    // Save prediction of sample t and category j
    fout.write(reinterpret_cast<const char*>(f_pred.memptr()), sizeof(double)*n_*d);
  }
  fout.close();
  is.close();
}

void ZANIMSharedBARTProbit::ComputePredictProbZero(arma::mat &X_, int n_samples,
                                                  int ntrees, std::string path_out,
                                                  std::string path, int verbose) {
  int np = 1;
  int n_ = X_.n_rows;
  arma::vec f = arma::zeros<arma::vec>(n_);

  // Open files to read the the forests
  std::vector<std::ifstream> files;
  for (int j=0; j < d; j++) {
    std::string ff = path + "/forests_zeta_" + std::to_string(j) + ".bin";
    files.emplace_back(ff, std::ios::binary);
  }

  // Open file to save the predictions
  std::ofstream fout(path_out + "/zeta_ij.bin", std::ios::app | std::ios::binary);

  // Iterate over posterior samples/draws
  double progress = 0.0;

  for (int t = 0; t < n_samples; t++) {
    if (verbose) {
      progress = (double) 100 * t / n_samples;
      Rprintf("\r");
      Rprintf("%3.2f%% Completed", progress);
      // std::cout << "Posterior draw " << t << " of " << n_samples << "\n";
    }
    // Iterate over categories
    for (int j = 0; j < d; j++) {
      f = arma::zeros<arma::vec>(n_);
      // Import a tree
      for (int h = 0; h < ntrees; h++) {
        Node *tree = deserialise_tree(files[j], np);
        // Do prediction for the imported tree
        for (int i = 0; i < n_; i++) {
          const arma::rowvec &xi = X_.row(i);
          f[i] += GetMu(tree, xi)[0];
        }
        delete tree;
      }
      for (int k=0; k < n_; k++) f[k] = R::pnorm5(f[k], 0.0, 1.0, 1.0, 0.0);
      // Save the predictions of sample h-th and category j
      fout.write(reinterpret_cast<const char*>(f.memptr()), sizeof(double)*n_);
    }
  }
  fout.close();
  for (int j=0; j<d; j++) files[j].close();
}

// Count number of times a variable appears in the decision rule across all the
// trees
void ZANIMSharedBARTProbit::ComputeVarCount(Node *tree, arma::uvec &varcount) {
  if (tree->is_leaf == 0) {
    int id_j = tree->predictor;
    varcount(id_j)++;
    ComputeVarCount(tree->left, varcount);
    ComputeVarCount(tree->right, varcount);
  }
}

// TODO: FIX, since now theta is shared tree,
arma::ucube ZANIMSharedBARTProbit::GetVarCount(int n_samples, int ntrees,
                                              std::string parm_name,
                                              std::string path) {
  int p = parm_name == "theta" ? p_theta : p_zeta;
  int np = parm_name == "theta" ? d : 2;
  arma::uvec vc = arma::zeros<arma::uvec>(p);
  arma::ucube varcount = arma::zeros<arma::ucube>(p, d, n_samples);

  // Iterate over categories (this can be done in parallel)
  for (int j = 0; j < d; j++) {
    std::cout << "Computing varcount for " << j << "\n";
    // Open file of category j
    std::ifstream is(path + "/forests_" + parm_name + "_" + std::to_string(j) + ".bin");
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
        delete tree;
      }
      // Save prediction of sample t and category j
      varcount.slice(t).col(j) = vc;
    }
    // Close file for category j
    is.close();
  }
  return varcount;
}


std::vector<double> ZANIMSharedBARTProbit::LogPredictiveDensity(std::vector<int> &y,
                                                               arma::rowvec &x,
                                                               int n_samples,
                                                               int ntrees_theta,
                                                               int ntrees_zeta,
                                                               std::string path) {
  int np_zeta = 2;
  // Open files to read the the forests
  std::vector<std::ifstream> files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff2 = path + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_zeta.emplace_back(ff2, std::ios::binary);
  }
  std::ifstream file_theta(path + "/forests_theta.bin", std::ios::binary);

  // Get the total of zeros and number of trials
  int total_zeros = 0, n_trials = 0;
  for (int j=0; j<d; j++) {
    n_trials += y[j];
    if (y[j] == 0) total_zeros++;
  }

  // Define which likelihood should use according to the number of zeros
  std::function<double(const std::vector<int>&, const std::vector<double>&,
                       const std::vector<double>&)> log_pmf;
  if (total_zeros < 10) {
    log_pmf = [](const std::vector<int>& y,
                 const std::vector<double>& theta,
                 const std::vector<double>& zeta) {
      return log_pmf_zanim(y, theta, zeta);
    };
  } else {
    int n_mc = 500;
    int n_skip = 100;
    log_pmf = [n_trials, n_mc, n_skip](const std::vector<int>& y,
                                       const std::vector<double>& theta,
                                       const std::vector<double>& zeta) {
      return log_pmf_zanim_approx(y, theta, zeta, n_trials, n_mc, n_skip);
    };
  }

  std::vector<double> ll(n_samples, 0.0);
  // Iterate over the MCMC samples
  for (int t = 0; t < n_samples; t++) {
    // Compute predictions of \theta_ij and \zeta_{ij}
    std::vector<double> theta(d, 0.0);
    std::vector<double> zeta(d, 0.0);
    // Iterate over trees
    for (int h = 0; h < ntrees_theta; h++) {
      // Import tree
      Node *tree = deserialise_tree(file_theta, d);
      // Do the predictions
      arma::vec mu = GetMu(tree, x);
      for (int j=0; j< d; j++) theta[j] += mu[j];
      delete tree;
    }

    // Iterate over categories
    for (int j = 0; j < d; j++) {
      for (int h = 0; h < ntrees_zeta; h++) {
        // Import tree
        Node *tree = deserialise_tree(files_zeta[j], np_zeta);
        // Do the predictions
        zeta[j] += GetMu(tree, x)[0];
        delete tree;
      }
    }
    // Normalise the parameters
    double s_theta = 0.0;
    for (int j=0; j < d; j++) {
      theta[j] = exp(theta[j]);
      s_theta += theta[j];
      zeta[j] = R::pnorm5(zeta[j], 0.0, 1.0, 1.0, 0.0);
    }
    for (auto &u : theta) u /= s_theta;
    // Compute the likelihood
    ll[t] = log_pmf(y, theta, zeta);
  }
  for (int j=0; j<d; j++) files_zeta[j].close();
  file_theta.close();
  return ll;
}

double ZANIMSharedBARTProbit::LogAvgPredictiveDensity(std::vector<int> &y,
                                                     arma::rowvec &x, int n_samples,
                                                     int ntrees_theta,
                                                     int ntrees_zeta,
                                                     std::string path) {
  std::vector<double> ll = LogPredictiveDensity(y, x, n_samples, ntrees_theta,
                                                ntrees_zeta, path);
  return log_sum_exp(ll) - log(n_samples);
}


std::vector<double> ZANIMSharedBARTProbit::GetNormaliseProbsIS(
    std::vector<int> &y, arma::vec &x_prior, int n_grid,
    // double mean_prior, double sd_prior,
    int n_samples, int ntrees_theta, int ntrees_zeta, std::string path) {
  // Create a sample and evaluate the log-target
  // arma::vec x_prior = arma::linspace(mean_prior - 2 * sd_prior,
  //                                    mean_prior + 2 * sd_prior, n_grid);
  std::vector<double> probs(n_grid, 0.0);
  double progress = 0.0;
  // Compute the likelihood over for each x_prior.
  for (int i=0; i < n_grid; i++) {
    Rprintf("\r");
    progress = (double) 100 * i / n_grid;
    Rprintf("%3.2f%% samples completed", progress);
    arma::rowvec x_cur = x_prior.row(i);
    probs[i] = LogAvgPredictiveDensity(y, x_cur, n_samples, ntrees_theta,
                                       ntrees_zeta, path);
  }
  double lse = log_sum_exp(probs);
  // Normalising the probabilities
  for (int i = 0; i < n_grid; i++) probs[i] = exp(probs[i] - lse);

  return probs;
}

void ZANIMSharedBARTProbit::GetMCMCPrediction(arma::rowvec &x,
                       std::vector<double> &theta, std::vector<double> &zeta,
                       int &d, int &ntrees_theta, int &ntrees_zeta,
                       int &np_zeta,
                       std::ifstream &file_theta,
                       std::vector<std::ifstream> &files_zeta) {

  arma::vec mus = arma::zeros<arma::vec>(np_zeta);
  std::vector<double> f0(d, 0.0);

  // Iterate over trees of \theta
  for (int h = 0; h < ntrees_theta; h++) {
    // Import tree
    Node *tree = deserialise_tree(file_theta, d);
    // Do the predictions
    arma::vec mu = GetMu(tree, x);
    for (int j=0; j < d; j++) theta[j] += mu[j];
    delete tree;
  }


  // Iterate over categories for zeta_j
  for (int j = 0; j < d; j++) {
    // Iterate over trees
    for (int h = 0; h < ntrees_zeta; h++) {
      // Import tree
      Node *tree = deserialise_tree(files_zeta[j], np_zeta);
      // Do the predictions
      mus = GetMu(tree, x);
      zeta[j] += mus(1);
      delete tree;
    }
  }
  // Normalise the parameters
  double s_theta = 0.0;
  for (int j=0; j < d; j++) {
    theta[j] = exp(theta[j]);
    s_theta += theta[j];
    zeta[j] = R::pnorm5(zeta[j], 0.0, 1.0, 1.0, 0.0);
  }
  for (auto &u : theta) u /= s_theta;
}

void ZANIMSharedBARTProbit::GetMCMCPredictionLoaded(arma::rowvec &x,
                       std::vector<double> &theta, std::vector<double> &zeta,
                       int &d, int &ntrees_theta, int &ntrees_zeta,
                       const std::vector<Node*> &forest_theta,
                       const std::vector<std::vector<Node*>> &forest_zeta) {

  //arma::vec mus = arma::zeros<arma::vec>(2);
  std::vector<double> f0(d, 0.0);
  // Predictions for \theta_ij
  for (int h = 0; h < ntrees_theta; h++) {
    // Do the predictions
    arma::vec mu = GetMu(forest_theta[h], x);
    for (int j=0; j < d; j++) theta[j] += mu[j];
  }

  // Iterate over categories
  for (int j = 0; j < d; j++) {
    // Iterate over trees
    for (int h = 0; h < ntrees_zeta; h++) {
      // Do the predictions
      zeta[j] += GetMu(forest_zeta[j][h], x)[0];
    }
  }
  // Normalise the parameters
  double s_theta = 0.0;
  for (int j=0; j < d; j++) {
    theta[j] = exp(theta[j]);
    s_theta += theta[j];
    zeta[j] = R::pnorm5(zeta[j], 0.0, 1.0, 1.0, 0.0);
  }
  for (auto &u : theta) u /= s_theta;
}

std::vector<double> ZANIMSharedBARTProbit::LogPredictiveDensitySeq(std::vector<int> &y,
                                                arma::mat X,
                                                int n_samples,
                                                int ntrees_theta,
                                                int ntrees_zeta,
                                                std::string path) {
  int np_zeta = 2;
  // Open files to read the the forests
  std::vector<std::ifstream> files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff2 = path + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_zeta.emplace_back(ff2, std::ios::binary);
  }
  std::ifstream file_theta(path + "/forests_theta.bin", std::ios::binary);

  int total_zeros = 0, n_trials = 0;
  for (int j=0; j<d; j++) {
    n_trials += y[j];
    if (y[j] == 0) total_zeros++;
  }

  std::function<double(const std::vector<int>&, const std::vector<double>&,
                       const std::vector<double>&)> log_pmf;
  if (total_zeros < 10) {
    log_pmf = [](const std::vector<int>& y,
                 const std::vector<double>& theta,
                 const std::vector<double>& zeta) {
      return log_pmf_zanim(y, theta, zeta);
    };
  } else {
    int n_mc = 1000;
    int n_skip = 500;
    log_pmf = [n_trials, n_mc, n_skip](const std::vector<int>& y,
                                       const std::vector<double>& theta,
                                       const std::vector<double>& zeta) {
      return log_pmf_zanim_approx(y, theta, zeta, n_trials, n_mc, n_skip);
    };
  }

  std::vector<double> ll(n_samples, 0.0);
  // Iterate over the MCMC samples
  for (int t = 0; t < n_samples; t++) {
    std::vector<double> theta(d, 0.0);
    std::vector<double> zeta(d, 0.0);
    // Compute the prediction for the proposal x*
    arma::rowvec x = X.row(t);
    GetMCMCPrediction(x, theta, zeta, d, ntrees_theta, ntrees_zeta,
                      np_zeta, file_theta, files_zeta);

    // Compute the likelihood
    ll[t] = log_pmf(y, theta, zeta);
  }
  for (int j=0; j<d; j++) files_zeta[j].close();
  file_theta.close();
  //log_sum_exp(ll) - log(n_samples);
  return ll;
}

// Exposing a C++ class in R
//using namespace Rcpp;
RCPP_MODULE(zanim_shared_bart_probit) {

  // Expose class MultinomialBART as "MultinomialBART" on the R side
  Rcpp::class_<ZANIMSharedBARTProbit>("ZANIMSharedBARTProbit")

  // Exposing constructor
  .constructor<arma::umat, arma::mat, arma::mat>()

  // Exposing member functions
  .method("SetMCMC", &ZANIMSharedBARTProbit::SetMCMC)
  .method("RunMCMC", &ZANIMSharedBARTProbit::RunMCMC)
  .method("ComputePredictProb", &ZANIMSharedBARTProbit::ComputePredictProb)
  .method("ComputePredictProbZero", &ZANIMSharedBARTProbit::ComputePredictProbZero)
  .method("GetVarCount", &ZANIMSharedBARTProbit::GetVarCount)

  .method("LogPredictiveDensity", &ZANIMSharedBARTProbit::LogPredictiveDensity)
  .method("LogAvgPredictiveDensity", &ZANIMSharedBARTProbit::LogAvgPredictiveDensity)


  .method("GetNormaliseProbsIS", &ZANIMSharedBARTProbit::GetNormaliseProbsIS)
  .method("LogPredictiveDensitySeq", &ZANIMSharedBARTProbit::LogPredictiveDensitySeq)

  // Exposing some attributes
  .field("draws_theta", &ZANIMSharedBARTProbit::draws_theta)
  .field("draws_zeta", &ZANIMSharedBARTProbit::draws_zeta)
  .field("draws_phi", &ZANIMSharedBARTProbit::draws_phi)
  .field("draws_phi", &ZANIMSharedBARTProbit::draws_phi)

  .field("varcount_mcmc_theta", &ZANIMSharedBARTProbit::varcount_mcmc_theta)
  .field("varcount_mcmc_zeta", &ZANIMSharedBARTProbit::varcount_mcmc_zeta)
  .field("alpha_sparse_mult", &ZANIMSharedBARTProbit::alpha_sparse_mult)
  .field("alpha_sparse_zi", &ZANIMSharedBARTProbit::alpha_sparse_zi)
  .field("sigma_mult_mcmc", &ZANIMSharedBARTProbit::sigma_mult_mcmc)
  .field("avg_leaves_theta", &ZANIMSharedBARTProbit::avg_leaves_theta)
  .field("avg_leaves_zeta", &ZANIMSharedBARTProbit::avg_leaves_zeta)

  // Acceptance rate
  .field("accept_rate_theta", &ZANIMSharedBARTProbit::accept_rate_theta)
  .field("accept_rate_zeta", &ZANIMSharedBARTProbit::accept_rate_zeta)

  ;

}
