#include "zanim_ln_bart.h"
#include "multinomial_bart.h"
#include "probit_bart.h"
#include "tree_mcmc.h"
#include "write_read.h"
#include "pmfs.h"
#include "utils.h"

constexpr double PI_2 = 6.283185307179586231996;

ZANIMLogNormalBART::ZANIMLogNormalBART(const arma::umat &Y,
                                           const arma::mat &X_theta,
                                           const arma::mat &X_zeta)
  : Y(Y), X_theta(X_theta), X_zeta(X_zeta) {

  n = Y.n_rows;
  d = Y.n_cols;
  dm1 = d - 1;
  p_theta = X_theta.n_cols;
  p_zeta = X_zeta.n_cols;

}

ZANIMLogNormalBART::~ZANIMLogNormalBART(){};

// Set up the model
void ZANIMLogNormalBART::SetMCMC(double v0_theta, double k_zeta,
                                 int ntrees_theta_, int ntrees_zeta_,
                                 arma::mat B_, int covariance_type_,
                                 double a_sigma_, double b_sigma_,
                                 arma::mat Psi_prior_, double nu_prior_,
                                 int q_factors_, double sigma2_gamma_,
                                 double a_psi_, double b_psi_,
                                 double shape_lsphis_,
                                 double a1_gs_, double a2_gs_,
                                 int ndpost_, int nskip_,
                                 int numcut, double power, double base,
                                 std::vector<double> proposals_prob_,
                                 int update_sd_prior_, double s2_0_,
                                 double w_ss_, std::vector<double> splitprobs_zi,
                                 std::vector<std::vector<double>> splitprobs_mult,
                                 int sparse_zi_, int sparse_mult_,
                                 std::vector<double> sparse_parms_zi_,
                                 std::vector<double> sparse_parms_mult_,
                                 std::vector<double> alpha_sparse_zi_,
                                 std::vector<double> alpha_sparse_mult_,
                                 int alpha_random_zi_, int alpha_random_mult_,
                                 arma::mat xinfo, std::string path_out_,
                                 int keep_draws_) {

  // Setting up fields
  ntrees_theta = ntrees_theta_;
  ntrees_zeta = ntrees_zeta_;
  path_out = path_out_;
  keep_draws = keep_draws_;
  proposals_prob = proposals_prob_;
  ndpost = ndpost_;
  nskip = nskip_;
  list_splitprobs_mult = splitprobs_mult;
  sparse_zi = sparse_zi_;
  sparse_mult = sparse_mult_;
  sparse_parms_zi = sparse_parms_zi_;
  sparse_parms_mult = sparse_parms_mult_;
  alpha_sparse_zi = alpha_sparse_zi_;
  alpha_sparse_mult = alpha_sparse_mult_;
  alpha_random_zi = alpha_random_zi_;
  alpha_random_mult = alpha_random_mult_;
  update_sd_prior = update_sd_prior_;
  B = B_;
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

  // Initialize multinomial BART
  bart_mult = new MultinomialBART(Y, X_theta);
  bart_mult->SetMCMC(v0_theta, ntrees_theta_, ndpost, nskip, numcut,
                     power, base, proposals_prob_, update_sd_prior_, s2_0_, w_ss_,
                     splitprobs_mult, sparse_mult, sparse_parms_mult,
                     alpha_sparse_mult, alpha_random_mult, xinfo, path_out_);
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
  fit_h = arma::zeros<arma::vec>(n);
  f_mu = arma::zeros<arma::mat>(n, d);
  f_lambda = arma::zeros<arma::mat>(n, d);
  varthetas = arma::zeros<arma::mat>(n, d);

  // Auxiliary vectors/matrices for the back-fitting in shared logit BART
  fit_0_h = arma::zeros<arma::vec>(n);
  f0_mu = arma::zeros<arma::mat>(n, d);
  //zetas = arma::zeros<arma::mat>(n, d);

  // z_{ij} ~ Bernoulli[1 - zeta_j] for y_{ij} = 0, else z_{ij} = 1.
  // Vectors with the index of observations that are zero
  zero_indices.resize(d);
  zs.resize(d);
  sum_col_zeros_Y = arma::sum(Y == 0, 0).t();
  // arma::umat Z(n, d, arma::fill::ones);
  for (int j = 0; j < d; ++j) {
    zero_indices[j].resize(sum_col_zeros_Y(j));
    zs[j].resize(sum_col_zeros_Y(j));
    zero_indices[j] = arma::conv_to<std::vector<int>>::from(arma::find(Y.col(j) == 0));
    // First draw of the latent variable z_{ij}~ Bernoulli[\pi_{ij}] IF y_{ij}=0
    for (int k = 0; k < sum_col_zeros_Y(j); k++) {
      zs[j][k] = R::rbinom(1, 0.5);
      // Z(zero_indices[j][k], j) = zs[j][k];
    }
  }
  // Aux variable to sample a movement in the back-fitting
  move = 0;

  // Initialise latent variables for the random effect
  Bt = B.t();
  y_cur = arma::zeros<arma::uvec>(d);
  v_cur = arma::zeros<arma::vec>(dm1);
  z_cur = arma::zeros<arma::uvec>(d);
  V = arma::randn<arma::mat>(n, dm1);
  U = V * Bt;
  Sigma_U = U.t() * U / n;

  if (covariance_type == 0) {
    sigmas = arma::zeros<arma::vec>(dm1);
    shape_post_sigma = (double)n/2.0 + a_sigma;
    for (int j=0; j < dm1; j++) {
      double sv = arma::dot(V.col(j), V.col(j));
      sigmas[j] = 1.0 / R::rgamma(shape_post_sigma, 1.0 / (sv/2.0 + b_sigma));
      sigmas[j] = sqrt(sigmas[j]);
    }
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

  list_splitprobs_zi.resize(d);
  alpha_zi_post_mean = arma::zeros<arma::mat>(d);
  alpha_mult_post_mean = arma::zeros<arma::mat>(d);

  if (update_sd_prior) sigma_mult_mcmc.resize(ndpost);

  // Keep the var-count for each iteration-category/forest
  varcount_mcmc_theta = arma::zeros<arma::ucube>(p_theta, d, ndpost);
  varcount_mcmc_zeta = arma::zeros<arma::ucube>(p_zeta, d, ndpost);

  // Average number of leaves
  avg_leaves_theta = arma::zeros<arma::umat>(ntrees_theta, d);
  avg_leaves_zeta = arma::zeros<arma::umat>(ntrees_zeta, d);

  // Acceptance rate
  accept_rate_theta = arma::zeros<arma::umat>(3, d);
  accept_rate_zeta = arma::zeros<arma::umat>(3, d);

  std::cout << "ZANIMLogNormal BART model set up!! \n\n";
}

// Back-fit for the multinomial part
void ZANIMLogNormalBART::BackFitMultinomial(int &j, int &t) {
  // Set split-probs of category j, this is used in the Grow, Prune, Change
  bart_mult->splitprobs = list_splitprobs_mult[j];
  // Back-fit
  fit_h = f_mu.col(j) - arma::vec(bart_mult->g_trees.tube(j, t));
  // std::cout << f_mu(0, j) << "\n";
  // bart_mult->fit_h_phi = exp(fit_h) % bart_mult->phi % exp_U.col(j);
  bart_mult->fit_h_phi = exp(fit_h + log(bart_mult->phi) + U.col(j));
  for (int k = 0; k < sum_col_zeros_Y(j); k++) {
    bart_mult->fit_h_phi[zero_indices[j][k]] *= zs[j][k];
  }
  // Update tree
  bart_mult->UpdateTree(bart_mult->trees[j][t]);
  // Perform the tree movement
  move = 0;
  if (bart_mult->trees[j][t]->NLeaves() > 1L)
    move = sample_discrete(proposals_prob, 3.0);
  switch (move) {
  case 0:
    Grow(bart_mult->trees[j][t], *bart_mult);
    accept_rate_theta(0, j) += bart_mult->flag_grow;
    break;
  case 1:
    Prune(bart_mult->trees[j][t], *bart_mult);
    accept_rate_theta(1, j) += bart_mult->flag_prune;
    break;
  case 2:
    Change(bart_mult->trees[j][t], *bart_mult);
    accept_rate_theta(2, j) += bart_mult->flag_change;
    break;
  }
  // Draw from the node parameters
  bart_mult->DrawPosterior(bart_mult->trees[j][t]);
  // Re-fit
  f_mu.col(j) = fit_h + arma::vec(bart_mult->g_trees.tube(j, t));
}

// Back-fit for the zero-inflation part
void ZANIMLogNormalBART::BackFitZI(int &j, int &t) {
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

// Update the latent variables
void ZANIMLogNormalBART::UpdateLatentVariables() {
  // f_lambda = exp(f_mu);
  f_lambda = exp(f_mu + U);
  for (int j=0; j < d; j++) {
    // Update (z_{ij} | \phi_i, \lambda_j, zeta_j) ~ Bernoulli[p_{ij}] for y_{ij}=0
    for(int k = 0; k < sum_col_zeros_Y(j); k++) {
      cur_indice = zero_indices[j][k];
      zeta_ij = R::pnorm5(f0_mu(cur_indice, j), 0.0, 1.0, 1.0, 0.0);
      // p_ij = (1 - zeta_ij) * exp(-f_lambda(cur_indice, j) * exp_U(cur_indice, j) * bart_mult->phi(cur_indice));
      p_ij = (1 - zeta_ij) * exp(-f_lambda(cur_indice, j) * bart_mult->phi(cur_indice));
      p_ij /= (zeta_ij + p_ij);
      zs[j][k] = R::rbinom(1, p_ij);
      // Update the response (recall we are modelling the prob. of zero-inflation)
      list_bart_zi[j]->y[cur_indice] = 1 - zs[j][k];
    }
    // Update latent variable of probit model z_{i}
    for (int k = 0; k < n; k++) {
      if (list_bart_zi[j]->y[k] == 0) list_bart_zi[j]->z[k] = -rtnorm(-f0_mu(k, j), 1.0, 0.0);
      else list_bart_zi[j]->z[k] = rtnorm(f0_mu(k, j), 1.0, 0.0);
    }
  }
  // Update \phi_i
  for (int k = 0; k < n; k++) {
    rt(k) = 0;
    for (int j=0; j < d; j++) {
      varthetas(k, j) = f_lambda(k, j) * (1 - list_bart_zi[j]->y[k]);
      rt(k) += varthetas(k, j); //f_lambda(k, j) * (1 - list_bart_zi[j]->y[k]);
    }
    if (bart_mult->n_trials(k) == 0) bart_mult->phi(k) = 0.0;
    else bart_mult->phi(k) = R::rgamma(bart_mult->n_trials(k), 1.0) / rt(k);
  }
  // Normalise theta (this is called abundance)
  varthetas.each_col() /= rt;

}


// Update split probabilities
void ZANIMLogNormalBART::UpdateSplitProbsMult(int &j) {
  list_splitprobs_mult[j] = UpdateSplitProbs(vc_mult, alpha_sparse_mult[j], p_theta);
}
void ZANIMLogNormalBART::UpdateSplitProbsZI(int &j) {
  list_bart_zi[j]->splitprobs = UpdateSplitProbs(vc_zi, alpha_sparse_zi[j], p_zeta);
}


// Update concentration parameter of Dirichlet prior
void ZANIMLogNormalBART::UpdateAlphaMult(int &j) {
  slp = 0.0;
  for (int k = 0; k < p_theta; k++) slp += std::log(list_splitprobs_mult[j][k]);
  alpha_sparse_mult[j] = UpdateAlphaDirchlet(alphas_grid_mult, lp_alpha_sparse_mult,
                                             slp, (double)p_theta, k_grid);
}
void ZANIMLogNormalBART::UpdateAlphaZI(int &j) {
  slp = 0.0;
  for (int k = 0; k < p_zeta; k++) slp += std::log(list_bart_zi[j]->splitprobs[k]);
  alpha_sparse_zi[j] = UpdateAlphaDirchlet(alphas_grid_zi, lp_alpha_sparse_zi,
                                           slp, (double)p_zeta, k_grid);
}

// Log-target of u_i without marginalised phi_i
double ZANIMLogNormalBART::LogTargetU_phi(arma::vec &u, arma::uvec &y,
                                      arma::uvec &z, double &phi,
                                      arma::vec &lambda) {
  double l = 0.0;
  for (int j=0; j < d; j++)
    l += z[j] *  (y[j] * u[j] - phi * lambda[j] * exp(u[j]));
  return l;
}

// Log-target of u_i with marginalised phi_i
double ZANIMLogNormalBART::LogTargetU(arma::vec &u, arma::uvec &y,
                                      arma::uvec &z, arma::vec &lambda) {

  int idx = 0;
  int k=arma::sum(z > 0);
  std::vector<double> lterms(k, 0.0);
  double l = 0.0, n_trials = 0.0;
  for (int j=0; j < d; j++) {
    n_trials += y[j];
    l += y[j] * u[j];
    if (z[j] > 0) lterms[idx++] = lambda[j] + u[j];
    // Rprintf("%i, %i, %i, %f, %f", j, y[j], z[j], lambda[j], u[j]);
    // Rprintf("\r");
  }
  if (lterms.empty()) return l;
  return l - n_trials * log_sum_exp(lterms);
}


// ESS for update v_i under the diagonal covariance prior
arma::vec ZANIMLogNormalBART::ESSDiag(arma::vec &v, arma::uvec &y, arma::vec &sigmas,
                                      arma::uvec &z, arma::vec &lambda, double phi) {
  // Draw from the prior
  arma::vec nu = arma::zeros<arma::vec>(dm1);
  for (int j=0; j < dm1; j++) nu[j] = R::norm_rand() * sigmas[j];

  // log-likelihood threshold
  double logy = log(R::unif_rand());
  arma::vec u = B * v;
  // logy += LogTargetU(u, y, z, phi, lambda);
  logy += LogTargetU(u, y, z, lambda);
  // Draw an angle
  double theta = R::unif_rand() * PI_2;
  double theta_max = theta;
  double theta_min = theta - PI_2;
  // Draw proposal
  arma::vec v_prop = v * cos(theta) + nu * sin(theta);
  arma::vec u_prop = B * v_prop;
  do {
    // if (LogTargetU(u_prop, y, z, phi, lambda) > logy) break;
    if (LogTargetU(u_prop, y, z, lambda) > logy) break;
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    theta = theta_min + (theta_max - theta_min) * R::unif_rand();
    v_prop = v * cos(theta) + nu * sin(theta);
    u_prop = B * v_prop;
  } while (true);
  return v_prop;
}

// ESS for update v_i under the "full" covariance prior (also work for the factor model)
arma::vec ZANIMLogNormalBART::ESSFull(arma::vec &v, arma::uvec &y, arma::uvec &z,
                                      arma::vec &lambda) {
  // Draw from the prior
  arma::rowvec ep = arma::randn<arma::rowvec>(dm1);
  arma::vec nu = (ep * chol_Sigma_V).t();
  // log-likelihood threshold
  double logy = log(R::unif_rand());
  arma::vec u = B * v;
  logy += LogTargetU(u, y, z, lambda);
  // Draw an angle
  double theta = R::unif_rand() * PI_2;
  double theta_max = theta;
  double theta_min = theta - PI_2;
  // Draw proposal
  arma::vec v_prop = v * cos(theta) + nu * sin(theta);
  arma::vec u_prop = B * v_prop;
  do {
    if (LogTargetU(u_prop, y, z, lambda) > logy) break;
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
void ZANIMLogNormalBART::UpdateSigmaDiag() {
  // Update \mathbf{u}_{i} using ESS
  for (int k=0; k < n; k++) {
    v_cur = V.row(k).t();
    y_cur = Y.row(k).t();
    for (int j=0; j < d; j++) z_cur[j] = 1 - list_bart_zi[j]->y[k];
    // lambda_cur = f_lambda.row(k).t();
    lambda_cur = f_mu.row(k).t();
    v_cur = ESSDiag(v_cur, y_cur, sigmas, z_cur, lambda_cur, bart_mult->phi(k));
    V.row(k) = v_cur.t();
  }
  // Transform back for other updates
  U = V * Bt;
  Sigma_U = U.t() * U / n;
  // Draw from \sigma^2_{j}
  for (int j=0; j < dm1; j++) {
    double sv = arma::dot(V.col(j), V.col(j));
    sigmas[j] = 1 / R::rgamma(shape_post_sigma, 1 / (sv/2 + b_sigma));
    sigmas[j] = sqrt(sigmas[j]);
  }

}

// Update \Sigma using inv-Wishart conjugate prior
void ZANIMLogNormalBART::UpdateSigmaFull() {
  // Update \mathbf{u}_{i} using ESS
  for (int k=0; k < n; k++) {
    v_cur = V.row(k).t();
    y_cur = Y.row(k).t();
    for (int j=0; j < d; j++) z_cur[j] = 1 - list_bart_zi[j]->y[k];
    lambda_cur = f_mu.row(k).t();
    v_cur = ESSFull(v_cur, y_cur, z_cur, lambda_cur);
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

void ZANIMLogNormalBART::UpdateSigmaFactorModel() {
  // Update \mathbf{u}_{i} using ESS
  for (int k=0; k < n; k++) {
    v_cur = V.row(k).t();
    y_cur = Y.row(k).t();
    for (int j=0; j < d; j++) z_cur[j] = 1 - list_bart_zi[j]->y[k];
    lambda_cur = f_mu.row(k).t();
    v_cur = ESSFull(v_cur, y_cur, z_cur, lambda_cur);
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

void ZANIMLogNormalBART::UpdateSigmaFactorModelMGP() {
  // Update \mathbf{u}_{i} using ESS
  for (int k=0; k < n; k++) {
    v_cur = V.row(k).t();
    y_cur = Y.row(k).t();
    for (int j=0; j < d; j++) z_cur[j] = 1 - list_bart_zi[j]->y[k];
    lambda_cur = f_mu.row(k).t();
    v_cur = ESSFull(v_cur, y_cur, z_cur, lambda_cur);
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

// Full conditional of F
void ZANIMLogNormalBART::UpdateH() {
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
void ZANIMLogNormalBART::UpdateGamma() {
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
void ZANIMLogNormalBART::UpdateGammaChol() {
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
void ZANIMLogNormalBART::UpdateLocalShrinkage() {
  for (int j=0; j < dm1; j++) {
    for (int k=0; k < q_factors; k++) {
      Phis_ls(j, k) = R::rgamma(shape_post_lsphis, 2.0 / (shape_lsphis + taus_gs(k) * std::pow(Gamma(j, k), 2.0)));
    }
  }
}

void ZANIMLogNormalBART::UpdateGlobalShrinkage() {
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
void ZANIMLogNormalBART::UpdatePsis() {
  R_psis = V - Hmat * Gamma.t();
  for (int j=0; j < dm1; j++) {
    rt_psis = arma::dot(R_psis.col(j), R_psis.col(j));
    psis[j] = 1.0 / R::rgamma(shape_psis, 1.0 / (rt_psis / 2.0 + b_psi));
  }
}



// Run MCMC for ZANIM-LogNormal BART model
void ZANIMLogNormalBART::RunMCMC() {

  // Aux to keep the varcount
  vc_mult = arma::zeros<arma::uvec>(p_theta);
  vc_zi = arma::zeros<arma::uvec>(p_zeta);

  if (keep_draws) {
    draws_vartheta = arma::zeros<arma::cube>(n, d, ndpost);
    draws_theta = arma::zeros<arma::cube>(n, d, ndpost);
    draws_zeta = arma::zeros<arma::cube>(n, d, ndpost);
    draws_Sigma_U = arma::zeros<arma::cube>(d, d, ndpost);
    draws_Gamma = arma::zeros<arma::cube>(dm1, q_factors, ndpost);
    draws_phi = arma::zeros<arma::mat>(n, ndpost);
  }
  //draws_rt = arma::zeros<arma::mat>(n, ndpost);

  // Aux to save the forests
  int np_t = 1, np_z = 1;

  std::cout << "Doing the warm-up (burn-in) of " << nskip << "\n\n";
  double progress = 0;
  for (int i = 0; i < nskip; i++) {
    progress = (double) 100 * i / nskip;
    Rprintf("%3.2f%% Warm-up completed", progress);
    Rprintf("\r");
    // Update latent variables
    UpdateLatentVariables();
    for (int j = 0; j < d; j++) {
      // "Gambiarra" to pass the category index inside an attribute of the class
      bart_mult->j_cat = j;
      for (int t = 0; t < ntrees_theta; t++) BackFitMultinomial(j, t);
      for (int t = 0; t < ntrees_zeta; t++) BackFitZI(j, t);
      // Update the DART parameters if required
      // if (sparse_mult) {
      //   vc_mult = bart_mult->GetVarCount(j);
      //   UpdateSplitProbsMult(j);
      //   if (alpha_random_mult) UpdateAlphaMult(j);
      // }
      // if (sparse_zi) {
      //   vc_zi = list_bart_zi[j]->GetVarCount();
      //   UpdateSplitProbsZI(j);
      //   if (alpha_random_zi) UpdateAlphaZI(j);
      // }
    }
    switch (covariance_type) {
      case 0: UpdateSigmaDiag(); break;
      case 1: UpdateSigmaFull(); break;
      case 2: UpdateSigmaFactorModel(); break;
      case 3: UpdateSigmaFactorModelMGP(); break;
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
  std::vector<std::ofstream> files_theta, files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff1 = path_out + "/forests_theta_" + std::to_string(j) + ".bin";
    files_theta.emplace_back(ff1, std::ios::binary | std::ios::app);
    std::string ff2 = path_out + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_zeta.emplace_back(ff2, std::ios::binary | std::ios::app);
  }
  // std::ofstream ff_Sigma_U(path_out + "/Sigma_U.bin", std::ios::binary | std::ios::app);
  std::ofstream ff_Sigma_V(path_out + "/chol_Sigma_V.bin", std::ios::binary | std::ios::app);

  // Run the actual posterior samples
  std::cout << "Starting post-burn-in iterations of " << ndpost << "\n\n";
  progress = 0;
  for (int i = 0; i < ndpost; i++) {
    progress = (double) 100 * i / ndpost;
    Rprintf("%3.2f%% Posterior samples completed", progress);
    Rprintf("\r");
    // Update latent variables (\phi_i, z_{ij} and probit-TN's)
    UpdateLatentVariables();
    for (int j = 0; j < d; j++) {
      // "Gambiarra" to pass the category index inside an attribute of the class
      bart_mult->j_cat = j;
      for (int t = 0; t < ntrees_theta; t++) {
        BackFitMultinomial(j, t);
        serialise_tree(bart_mult->trees[j][t], files_theta[j], np_t);
        avg_leaves_theta(t, j) += bart_mult->trees[j][t]->NLeaves();
      }
      for (int t = 0; t < ntrees_zeta; t++) {
        BackFitZI(j, t);
        serialise_tree(list_bart_zi[j]->trees[t], files_zeta[j], np_z);
        avg_leaves_zeta(t, j) += list_bart_zi[j]->trees[t]->NLeaves();
      }
      vc_mult = bart_mult->GetVarCount(j);
      vc_zi = list_bart_zi[j]->GetVarCount();
      // Update the DART parameters if required
      if (sparse_mult) {
        UpdateSplitProbsMult(j);
        if (alpha_random_mult) {
          UpdateAlphaMult(j);
          alpha_mult_post_mean[j] += alpha_sparse_mult[j];
        }
      }
      if (sparse_zi) {
        UpdateSplitProbsZI(j);
        if (alpha_random_zi) {
          UpdateAlphaZI(j);
          alpha_zi_post_mean[j] += alpha_sparse_zi[j];
        }
      }
      // Keep the computed var-count
      varcount_mcmc_theta.slice(i).col(j) = vc_mult;
      varcount_mcmc_zeta.slice(i).col(j) = vc_zi;
    }
    // Update U/V's and the hyper-prior on \Sigma
    switch (covariance_type) {
      case 0: UpdateSigmaDiag(); break;
      case 1: UpdateSigmaFull(); break;
      case 2: UpdateSigmaFactorModel(); break;
      case 3: UpdateSigmaFactorModelMGP(); break;
    }
    // Update the sigma prior and then the leaf prior parameters
    if (update_sd_prior) {
      bart_mult->sigma = bart_mult->UpdateSigmaPrior();
      bart_mult->c_shape = trigamma_inverse(bart_mult->sigma*bart_mult->sigma);
      bart_mult->d_rate = exp(R::digamma(bart_mult->c_shape));
      bart_mult->c_logd_lgamc = bart_mult->c_shape * log(bart_mult->d_rate) - R::lgammafn(bart_mult->c_shape);
      sigma_mult_mcmc[i] = bart_mult->sigma;
    }

    // Save the draws of cholesky of Sigma_V
    if (covariance_type > 0) ff_Sigma_V.write(reinterpret_cast<const char*>(chol_Sigma_V.memptr()), sizeof(double)*dm1*dm1);
    // Save draws
    if (keep_draws) {
      draws_Sigma_U.slice(i) = Sigma_U;
      draws_theta.slice(i) = f_lambda.each_col() / arma::sum(f_lambda, 1);
      draws_vartheta.slice(i) = varthetas; // abundance
      draws_zeta.slice(i) = f0_mu;
      // draws_Gamma.slice(i) = Gamma; // factor loadings
      draws_phi.col(i) = bart_mult->phi;
    }
  }
  // Close the files
  ff_Sigma_V.close();
  for (int j=0; j < d; j++) {
    files_theta[j].close();
    files_zeta[j].close();
    // Keep the last split probability
    list_splitprobs_zi[j] = list_bart_zi[j]->splitprobs;
    if (alpha_random_mult) alpha_mult_post_mean[j] /= ndpost;
    if (alpha_random_zi) alpha_zi_post_mean[j] /= ndpost;
  }
}

// Predict functions
arma::vec ZANIMLogNormalBART::GetMu(Node *tree, const arma::rowvec &x) {
  if (tree->is_leaf) return tree->mu;
  if (x[tree->predictor] <= tree->cutoff) return GetMu(tree->left, x);
  else return GetMu(tree->right, x);
}

void ZANIMLogNormalBART::ComputePredictProb(arma::mat &X_, int n_samples,
                                            int ntrees, std::string path_out,
                                            std::string path, int verbose) {
  int np = 1;
  int n_ = X_.n_rows;
  arma::mat f_pred = arma::zeros<arma::mat>(n_, d);

  // Open files to read the the forests
  std::vector<std::ifstream> files;
  for (int j=0; j < d; j++) {
    std::string ff = path + "/forests_theta_" + std::to_string(j) + ".bin";
    files.emplace_back(ff, std::ios::binary);
  }
  // Open file for the Sigma_U
  std::ifstream ff_Sigma_V(path + "/chol_Sigma_V.bin", std::ios::binary);
  arma::cube chol_Sigma_V_(dm1, dm1, n_samples);
  arma::vec V_(dm1);
  // arma::cube U_ = arma::zeros<arma::cube>(n_, d, n_samples);
  arma::mat U_ = arma::zeros<arma::mat>(n_, d);
  // arma::cube draws_out = arma::zeros<arma::cube>(n_, d, n_samples);

  // Import the chol_Sigma_V
  ff_Sigma_V.read(reinterpret_cast<char*>(chol_Sigma_V_.memptr()),
                  sizeof(double) * dm1 * dm1 * n_samples);

  // Open file to save the predictions
  std::ofstream fout(path_out + "/theta_ij.bin", std::ios::app | std::ios::binary);

  // Iterate over the MCMC samples
  for (int t = 0; t < n_samples; t++) {
    if (verbose) {
      double progress = (double) 100 * t / n_samples;
      Rprintf("\r");
      Rprintf("%3.2f%% Completed", progress);
    }

    // from v_i ~ N_{d-1}[0, \Sigma_V], then u_i = B v_i
    for (int i=0; i < n_; i++) {
      V_ = (arma::randn<arma::rowvec>(dm1) * chol_Sigma_V_.slice(t)).t();
      U_.row(i) = (B * V_).t();
    }

    // Initialise vector to keep the predictions of \theta_ij
    f_pred = arma::zeros<arma::mat>(n_, d);
    // Iterate over categories
    for (int j = 0; j < d; j++) {
      // Iterate over trees
      for (int h = 0; h < ntrees; h++) {
        // Import tree
        Node *tree = deserialise_tree(files[j], np);
        // std::cout << tree->NLeaves() << "\n\n";
        // Do the predictions
        for (int i = 0; i < n_; i++) {
          const arma::rowvec &xi = X_.row(i);
          f_pred(i, j) += GetMu(tree, xi)[0];
          //std::cout << "j=" << j << " h=" << h << " mu_ij=" << mu << "\n";
        }
        delete tree;
      }
    }

    f_pred = exp(f_pred + U_); //
    f_pred = f_pred.each_col() / arma::sum(f_pred, 1);
    // Save prediction of sample t and category j
    fout.write(reinterpret_cast<const char*>(f_pred.memptr()), sizeof(double)*n_*d);
    //draws_out.slice(t) = f_pred;
  }


  // Close files
  ff_Sigma_V.close();
  //fout.close();
  for (int j=0; j<d; j++) files[j].close();
  // return draws_out;
}

void ZANIMLogNormalBART::ComputePredictProbZero(arma::mat &X_, int n_samples,
                                             int ntrees,
                                             std::string path_out,
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
void ZANIMLogNormalBART::ComputeVarCount(Node *tree, arma::uvec &varcount) {
  if (tree->is_leaf == 0) {
    int id_j = tree->predictor;
    varcount(id_j)++;
    ComputeVarCount(tree->left, varcount);
    ComputeVarCount(tree->right, varcount);
  }
}

arma::ucube ZANIMLogNormalBART::GetVarCount(int n_samples, int ntrees,
                                              std::string parm_name,
                                              std::string path) {
  int p = parm_name == "theta" ? p_theta : p_zeta;
  int np = parm_name == "theta" ? 1 : 2;
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


// Exposing a C++ class in R
//using namespace Rcpp;
RCPP_MODULE(zanim_lognormal_bart) {

  // Expose class
  Rcpp::class_<ZANIMLogNormalBART>("ZANIMLogNormalBART")

  // Exposing constructor
  .constructor<arma::umat, arma::mat, arma::mat>()

  // Exposing member functions
  .method("SetMCMC", &ZANIMLogNormalBART::SetMCMC)
  .method("RunMCMC", &ZANIMLogNormalBART::RunMCMC)
  .method("ComputePredictProb", &ZANIMLogNormalBART::ComputePredictProb)
  .method("ComputePredictProbZero", &ZANIMLogNormalBART::ComputePredictProbZero)
  .method("GetVarCount", &ZANIMLogNormalBART::GetVarCount)

  // Exposing some attributes
  .field("draws_phi", &ZANIMLogNormalBART::draws_phi)
  .field("draws_Sigma_U", &ZANIMLogNormalBART::draws_Sigma_U)
  .field("draws_theta", &ZANIMLogNormalBART::draws_theta)
  .field("draws_zeta", &ZANIMLogNormalBART::draws_zeta)
  .field("draws_vartheta", &ZANIMLogNormalBART::draws_vartheta)
  // .field("draws_Gamma", &ZANIMLogNormalBART::draws_Gamma)

  .field("varcount_mcmc_theta", &ZANIMLogNormalBART::varcount_mcmc_theta)
  .field("varcount_mcmc_zeta", &ZANIMLogNormalBART::varcount_mcmc_zeta)
  .field("alpha_sparse_mult", &ZANIMLogNormalBART::alpha_mult_post_mean)
  .field("alpha_sparse_zi", &ZANIMLogNormalBART::alpha_zi_post_mean)
  .field("splitprobs_mult", &ZANIMLogNormalBART::list_splitprobs_mult)

  .field("sigma_mult_mcmc", &ZANIMLogNormalBART::sigma_mult_mcmc)
  .field("avg_leaves_theta", &ZANIMLogNormalBART::avg_leaves_theta)
  .field("avg_leaves_zeta", &ZANIMLogNormalBART::avg_leaves_zeta)

  // Acceptance rate
  .field("accept_rate_theta", &ZANIMLogNormalBART::accept_rate_theta)
  .field("accept_rate_zeta", &ZANIMLogNormalBART::accept_rate_zeta)
  ;

}
