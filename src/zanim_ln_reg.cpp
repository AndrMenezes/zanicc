#include "zanim_ln_reg.h"
#include "rng.h"
#include "utils.h"

constexpr double PI_2 = 6.283185307179586231996;

ZANIMLNReg::ZANIMLNReg(const arma::umat &Y, const arma::mat &X_theta,
                       const arma::mat &X_zeta) : Y(Y), X_theta(X_theta), X_zeta(X_zeta) {
  d = Y.n_cols;
  n = Y.n_rows;
  dm1 = d - 1;
  p_theta = X_theta.n_cols;
  p_zeta = X_zeta.n_cols;
  n_trials = arma::sum(Y, 1);
}

ZANIMLNReg::~ZANIMLNReg(){};


void ZANIMLNReg::SetMCMC(std::vector<double> sd_prior_beta_theta_,
                         arma::mat sigma_prior_beta_zeta_,
                         int ndpost_, int nskip_, int nthin_,
                         arma::mat B_, int covariance_type_,
                         double a_sigma_, double b_sigma_,
                         arma::mat Psi_prior_, double nu_prior_,
                         int q_factors_, double sigma2_gamma_,
                         double a_psi_, double b_psi_,
                         double shape_lsphis_,
                         double a1_gs_, double a2_gs_, int keep_draws_) {
  Rcpp::RNGScope scope;
  sd_prior_beta_theta = sd_prior_beta_theta_;
  sigma_prior_beta_zeta = sigma_prior_beta_zeta_;
  ndpost = ndpost_;
  nskip = nskip_;
  nthin = nthin_;
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
  keep_draws_ = keep_draws;

  // Initialise the reg coefficients
  betas_theta = arma::randn<arma::mat>(p_theta, d);
  betas_zeta = arma::randn<arma::mat>(p_zeta, d);

  // Initialise the linear predictions coefficients
  alphas = arma::zeros<arma::mat>(n, d);
  lambdas = arma::zeros<arma::mat>(n, d);
  eta_zetas = arma::zeros<arma::mat>(n, d);
  varthetas = arma::zeros<arma::mat>(n, d);

  // Define the response for zero-inflation (it will change over the iterations)
  // Z = arma::zeros<arma::umat>(n, d);
  Y_zi = arma::zeros<arma::umat>(n, d);
  for (int j=0; j < d; j++) Y_zi.col(j) = Y.col(j) == 0;

  // Compute the cholesky decomposition of the covariance matrix of the full conditional
  // distribution of the probit-reg coefficients for all categories
  sigma_inv_probit = arma::inv_sympd(X_zeta.t() * X_zeta + arma::inv_sympd(sigma_prior_beta_zeta));
  sigma_chol_inv_probit = arma::chol(sigma_inv_probit);

  // First draw for the latent variable phi
  phi = arma::zeros<arma::vec>(n);
  for (int i = 0; i < n; i++) phi(i) = R::rgamma(n_trials(i), 1.0);

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

  // Latent variables for the probit Albert-Chib DA
  Z_probit = arma::zeros<arma::mat>(n, d);
  // Update latent variable for the probit part
  for (int j = 0; j < d; j++) {
    for (int i = 0; i < n; i++) {
      if (Y_zi(i, j) == 0) Z_probit(i, j) = -rtnorm(0, 1.0, 0.0);
      else Z_probit(i, j) = rtnorm(0, 1.0, 0.0);
    }
  }

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
      // inverse-gamma
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
      mu_gamma_chol = arma::zeros<arma::vec>(q_factors);
      Phis_ls = arma::zeros<arma::mat>(dm1, q_factors);
      GammaPhi = arma::zeros<arma::mat>(dm1, q_factors);
      deltas_gs = arma::zeros<arma::vec>(q_factors);
      arma::vec taus = arma::zeros<arma::vec>(q_factors);
      for (int k=0; k < q_factors; k++) {
        for (int j=0; j < dm1; j++) {
          // local-shrinkage
          Phis_ls(j, k) = R::rgamma(shape_lsphis / 2.0, 2.0 / shape_lsphis);
        }
      }
      // global-shrinkage
      deltas_gs[0]= R::rgamma(a1_gs, 1.0);
      for (int j=1; j < q_factors; j++) deltas_gs[j] = R::rgamma(a2_gs, 1.0);
      taus_gs = arma::cumprod(deltas_gs);
      // Compute the posterior shape parameters of local and shrinkage
      shape_post_lsphis = (shape_lsphis + 1.0) / 2.0;
      shape_post_gs_delta1 = a1_gs + dm1 * q_factors / 2.0;
    }
  }
}

double ZANIMLNReg::LogTargetBetasTheta(arma::vec &beta_cur, int &j) {

  arma::vec eta = X_theta * beta_cur;
  arma::vec e_eta_phi_u = exp(eta + U.col(j)) % phi;
  double t1 = arma::sum(Y.col(j) % eta);
  double t2 = arma::sum(e_eta_phi_u);

  // std::cout << "t1 " << t1 << " t2 " << t2 << "\n";

  // Remove the contribution of zero counts such that z_{ij} = 0
  for (int k = 0; k < sum_col_zeros_Y(j); k++) {
    if (zs[j][k] == 0) {
      int cur_indice = zero_indices[j][k];
      t2 -= e_eta_phi_u(cur_indice);
    }
  }
  return t1 - t2;
}



// Update linear regression coefficients for the \theta_{ij} part
arma::vec ZANIMLNReg::UpdateBetasThetaESS(int &j) {

  arma::vec beta_cur = betas_theta.col(j);
  // Draw from the prior
  arma::vec nu = arma::zeros<arma::vec>(p_theta);
  for (int k=0; k < p_theta; k++) nu[k] = sd_prior_beta_theta[k] * R::norm_rand();
  // Set a log-likelihood threshold
  double u_s = log(R::unif_rand());
  u_s += LogTargetBetasTheta(beta_cur, j);
  // Draw an angle
  double theta = R::unif_rand() * PI_2;
  double theta_max = theta;
  double theta_min = theta - PI_2;
  // Draw proposal
  arma::vec beta_star = beta_cur * cos(theta) + nu * sin(theta);
  // Start slicing
  do {
    if (LogTargetBetasTheta(beta_star, j) > u_s) break;
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    theta = theta_min + (theta_max - theta_min) * R::unif_rand();
    beta_star = beta_cur * cos(theta) + nu * sin(theta);
  } while (true);

  return beta_star;
}


arma::vec ZANIMLNReg::UpdateBetasZetaChib(int &j) {
  arma::vec beta_tilde = sigma_inv_probit * (X_zeta.t() * Z_probit.col(j));
  return beta_tilde + rmvnorm(p_zeta, sigma_chol_inv_probit);
}

void ZANIMLNReg::UpdateLatentVariables() {

  for (int j=0; j < d; j++) {
    // Compute the linear predictions, eta^{\lambda}_{ij} and eta^{\zeta}_{ij}
    eta_zetas.col(j) = X_zeta * betas_zeta.col(j);
    alphas.col(j) = X_theta * betas_theta.col(j);
    lambdas.col(j) = exp(alphas.col(j) + U.col(j));
    alphas.col(j) = exp(alphas.col(j));

    //for (int i=0; i<n; i++) if (alphas(i, j) < 1e-10) alphas(i, j) = 1e-10;

    // Update (z_{ij} | \phi_i, \lambda_j, zeta_j) ~ Bernoulli[p_{ij}] for y_{ij}=0
    for(int k = 0; k < sum_col_zeros_Y(j); k++) {
      int cur_indice = zero_indices[j][k];
      double zeta_ij = R::pnorm5(eta_zetas(cur_indice, j), 0.0, 1.0, 1.0, 0.0);
      double p_ij = (1 - zeta_ij) * exp(-lambdas(cur_indice, j) * phi(cur_indice));
      p_ij /= (zeta_ij + p_ij);
      zs[j][k] = R::rbinom(1, p_ij);
      // Update the response
      Y_zi(cur_indice, j) = 1 - zs[j][k];
    }

    // Update latent variable for the probit part
    for (int i = 0; i < n; i++) {
      // Right truncation
      if (Y_zi(i, j) == 0) Z_probit(i, j) = -rtnorm(-eta_zetas(i, j), 1.0, 0.0);
      else Z_probit(i, j) = rtnorm(eta_zetas(i, j), 1.0, 0.0);
    }
  }

  // Compute the \sum_{j=1}^d z_{ij}\lambda_{ij}
  rt = arma::sum(lambdas, 1);
  varthetas = lambdas;
  // Remove the contribution when z_{ij} = 0 in rt
  for (int j = 0; j < d; j++) {
    for (int k = 0; k < sum_col_zeros_Y(j); k++) {
      if (zs[j][k] == 0) {
        int cur_indice = zero_indices[j][k];
        rt(cur_indice) -= lambdas(cur_indice, j);
        varthetas(cur_indice, j) = 0.0;
      }
    }
  }
  // Normalise theta (this is what we call abundance)
  varthetas.each_col() /= rt;

  // Update \phi_i \sim Gamma[N_i, \sum_j \lambda_{ij} * z_{ij}]
  for (int i=0; i < n; i++) {
    if (n_trials(i) == 0) phi(i) = 0.0;
    else phi(i) = R::rgamma(n_trials(i), 1.0) / rt(i);
    // std::cout << phi(i) << "\n";
  }

}

// Log-target of u_i with marginalised phi_i
double ZANIMLNReg::LogTargetU(arma::vec &u, arma::uvec &y, arma::uvec &z,
                              arma::vec &alpha) {

  int idx = 0;
  int k = arma::sum(z > 0);
  std::vector<double> lterms(k, 0.0);
  double l = 0.0, n_trials = 0.0;
  for (int j=0; j < d; j++) {
    n_trials += y[j];
    l += y[j] * u[j];
    if (z[j] > 0) lterms[idx++] = alpha[j] + u[j];
  }
  if (lterms.empty()) return l; // Need to return prob = 1, i.e. l=0.
  return l - n_trials * log_sum_exp(lterms);
}


// ESS for update v_i under the diagonal covariance prior
arma::vec ZANIMLNReg::ESSDiag(arma::vec &v, arma::uvec &y, arma::vec &sigmas,
                              arma::uvec &z, arma::vec &alpha) {
  // Draw from the prior
  arma::vec nu = arma::zeros<arma::vec>(dm1);
  for (int j=0; j < dm1; j++) nu[j] = R::norm_rand() * sigmas[j];

  // log-likelihood threshold
  double logy = log(R::unif_rand());
  arma::vec u = B * v;
  logy += LogTargetU(u, y, z, alpha);
  // alpha.print();
  // std::cout << "\n\n";
  // Draw an angle
  double theta = R::unif_rand() * PI_2;
  double theta_max = theta;
  double theta_min = theta - PI_2;
  // Draw proposal
  arma::vec v_prop = v * cos(theta) + nu * sin(theta);
  arma::vec u_prop = B * v_prop;
  do {
    // double l=LogTargetU(u_prop, y, z, alpha);
    // std::cout << l << " " << LogTargetU(u, y, z, alpha) << "\n";
    if (LogTargetU(u_prop, y, z, alpha) > logy) break;
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    theta = theta_min + (theta_max - theta_min) * R::unif_rand();
    v_prop = v * cos(theta) + nu * sin(theta);
    u_prop = B * v_prop;
  } while (true);
  return v_prop;
}

// ESS for update v_i under the "full" covariance prior (also work for the factor model)
arma::vec ZANIMLNReg::ESSFull(arma::vec &v, arma::uvec &y, arma::uvec &z,
                              arma::vec &alpha) {
  // Draw from the prior
  arma::rowvec ep = arma::randn<arma::rowvec>(dm1);
  arma::vec nu = (ep * chol_Sigma_V).t();
  // log-likelihood threshold
  double logy = log(R::unif_rand());
  arma::vec u = B * v;
  logy += LogTargetU(u, y, z, alpha);
  // Draw an angle
  double theta = R::unif_rand() * PI_2;
  double theta_max = theta;
  double theta_min = theta - PI_2;
  // Draw proposal
  arma::vec v_prop = v * cos(theta) + nu * sin(theta);
  arma::vec u_prop = B * v_prop;
  do {
    if (LogTargetU(u_prop, y, z, alpha) > logy) break;
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
void ZANIMLNReg::UpdateSigmaDiag() {
  // Update \mathbf{u}_{i} using ESS
  for (int k=0; k < n; k++) {
    v_cur = V.row(k).t();
    y_cur = Y.row(k).t();
    for (int j=0; j < d; j++) z_cur[j] = 1 - Y_zi(k, j);
    alphas_cur = log(alphas.row(k).t());
    v_cur = ESSDiag(v_cur, y_cur, sigmas, z_cur, alphas_cur);
    V.row(k) = v_cur.t();
  }
  // Transform back for other updates
  U = V * Bt;
  Sigma_U = U.t() * U / n;
  // Draw from \sigma^2_{j}
  for (int j=0; j < dm1; j++) {
    double sv = arma::dot(V.col(j), V.col(j));
    sigmas[j] = 1.0 / R::rgamma(shape_post_sigma, 1.0 / (sv/2.0 + b_sigma));
    sigmas[j] = sqrt(sigmas[j]);
  }
}

// Update \Sigma using inv-Wishart conjugate prior
void ZANIMLNReg::UpdateSigmaFull() {
  // Update \mathbf{u}_{i} using ESS
  for (int k=0; k < n; k++) {
    v_cur = V.row(k).t();
    y_cur = Y.row(k).t();
    for (int j=0; j < d; j++) z_cur[j] = 1 - Y_zi(k, j);
    alphas_cur = log(alphas.row(k).t());
    v_cur = ESSFull(v_cur, y_cur, z_cur, alphas_cur);
    V.row(k) = v_cur.t();
  }
  // Draw from Sigma ~ inv-Wishart
  Sigma_V = arma::iwishrnd(V.t()*V + Psi_prior, df_wishart);
  chol_Sigma_V = arma::chol(Sigma_V);
  // Transform back for other updates
  U = V * Bt;
  Sigma_U = B * Sigma_V * Bt;
}

void ZANIMLNReg::UpdateSigmaFactorModel() {
  // Update \mathbf{u}_{i} using ESS
  for (int k=0; k < n; k++) {
    v_cur = V.row(k).t();
    y_cur = Y.row(k).t();
    for (int j=0; j < d; j++) z_cur[j] = 1 - Y_zi(k, j);
    alphas_cur = log(alphas.row(k).t());
    v_cur = ESSFull(v_cur, y_cur, z_cur, alphas_cur);
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

void ZANIMLNReg::UpdateSigmaFactorModelMGP() {
  // Update \mathbf{u}_{i} using ESS
  for (int k=0; k < n; k++) {
    v_cur = V.row(k).t();
    y_cur = Y.row(k).t();
    for (int j=0; j < d; j++) z_cur[j] = 1 - Y_zi(k, j);
    alphas_cur = log(alphas.row(k).t());
    v_cur = ESSFull(v_cur, y_cur, z_cur, alphas_cur);
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
void ZANIMLNReg::UpdateH() {
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
void ZANIMLNReg::UpdateGamma() {
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
void ZANIMLNReg::UpdateGammaChol() {
  HtH = Hmat.t() * Hmat;
  HVp = Hmat.t() * V;
  HVp.each_row() /= psis.t();
  for (int j=0; j < dm1; j++) {
    Q = HtH / psis[j];
    Q.diag() += taus_gs % Phis_ls.row(j).t();
    // 1. Compute Cholesky factorisation
    Q_chol = arma::chol(Q);
    // 2. Compute the inverse of using backsolve
    Q_chol_inv = arma::solve(arma::trimatu(Q_chol), Iq, arma::solve_opts::fast);
    // 3. Compute the mean parameter
    mu_gamma_chol = (Q_chol_inv * Q_chol_inv.t()) * HVp.col(j);
    // 4. Generate using traditional approach
    Gamma.row(j) = mu_gamma_chol.t() + arma::randn<arma::rowvec>(q_factors) * Q_chol_inv;
  }
}

// Full conditional of the local shrinkage parameters, \phi_{jk}
void ZANIMLNReg::UpdateLocalShrinkage() {
  for (int j=0; j < dm1; j++) {
    for (int k=0; k < q_factors; k++) {
      Phis_ls(j, k) = R::rgamma(shape_post_lsphis, 2.0 / (shape_lsphis + taus_gs(k) * std::pow(Gamma(j, k), 2.0)));
    }
  }
}

void ZANIMLNReg::UpdateGlobalShrinkage() {
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
void ZANIMLNReg::UpdatePsis() {
  R_psis = V - Hmat * Gamma.t();
  for (int j=0; j < dm1; j++) {
    rt_psis = arma::dot(R_psis.col(j), R_psis.col(j));
    psis[j] = 1.0 / R::rgamma(shape_psis, 1.0 / (rt_psis / 2.0 + b_psi));
  }
}


void ZANIMLNReg::RunMCMC() {

  std::cout << "Doing the warm-up (burn-in) of " << nskip << "\n\n";
  double progress = 0;
  for (int t=0; t < nskip; t++) {
    progress = (double) 100 * t / nskip;
    Rprintf("%3.2f%% Warm-up completed", progress);
    Rprintf("\r");
    // Update latent variables
    UpdateLatentVariables();
    for (int j = 0; j < d; j++) {
      // Update \beta^{(\lambda)}_{j,k} using ESS
      betas_theta.col(j) = UpdateBetasThetaESS(j);
      // Update \beta^{(\zeta)}_{j,k} using Abert-Chib probit DA.
      betas_zeta.col(j) = UpdateBetasZetaChib(j);
    }
    // Sum-to-zero (gambiarra), for identifiability
    // for (int l=0; l < p_theta; l++) {
    //   double mu = arma::accu(betas_theta.row(l)) / d;
    //   for (int j = 0; j < d; j ++) {
    //     betas_theta.row(l).col(j) -= mu;
    //   }
    // }

    switch (covariance_type) {
      case 0: UpdateSigmaDiag(); break;
      case 1: UpdateSigmaFull(); break;
      case 2: UpdateSigmaFactorModel(); break;
      case 3: UpdateSigmaFactorModelMGP(); break;
    }
  }

  // Initialise container for keep the draws
  if (keep_draws) {
    draws_betas_theta = arma::zeros<arma::cube>(p_theta, d, ndpost);
    draws_betas_zeta = arma::zeros<arma::cube>(p_zeta, d, ndpost);
    draws_thetas = arma::zeros<arma::cube>(n, d, ndpost);
    draws_varthetas = arma::zeros<arma::cube>(n, d, ndpost);
    draws_zetas = arma::zeros<arma::cube>(n, d, ndpost);
    draws_chol_Sigma_V = arma::zeros<arma::cube>(dm1, dm1, ndpost);
    // draws_Sigma_U = arma::zeros<arma::cube>(d, d, ndpost);
    // draws_phi = arma::zeros<arma::mat>(n, ndpost);
  }

  // Run the post-burn in iterations
  std::cout << "Starting post-burn-in iterations of " << ndpost << "\n\n";
  progress = 0;

  for (int t=0; t < ndpost; t++) {
    progress = (double) 100 * t / ndpost;
    Rprintf("%3.2f%% Posterior samples completed", progress);
    Rprintf("\r");
    // Update latent variables
    UpdateLatentVariables();
    // Iterate over the categories
    for (int j = 0; j < d; j++) {
      // Update \beta^{(\theta)}_{j,k} using ESS
      betas_theta.col(j) = UpdateBetasThetaESS(j);
      // Update \beta^{(\zeta)}_{j,k} using Abert-Chib probit DA.
      betas_zeta.col(j) = UpdateBetasZetaChib(j);
    }
    // Sum-to-zero (gambiarra), for identifiability
    // for (int l=0; l < p_theta; l++) {
    //   double mu = arma::accu(betas_theta.row(l)) / d;
    //   for (int j = 0; j < d; j ++) {
    //     betas_theta.row(l).col(j) -= mu;
    //   }
    // }

    // Update random effects and its covariance matrix
    switch (covariance_type) {
      case 0: UpdateSigmaDiag(); break;
      case 1: UpdateSigmaFull(); break;
      case 2: UpdateSigmaFactorModel(); break;
      case 3: UpdateSigmaFactorModelMGP(); break;
    }
    // Save draws
    if (keep_draws) {
      // draws_phi.col(t) = phi;
      draws_chol_Sigma_V.slice(t) = chol_Sigma_V;
      draws_betas_theta.slice(t) = betas_theta;
      draws_betas_zeta.slice(t) = betas_zeta;
      // draws_Sigma_U.slice(t) = Sigma_U;
      // without random effect u's
      draws_thetas.slice(t) = alphas.each_col() / arma::sum(alphas, 1);

      draws_varthetas.slice(t) = varthetas;
      draws_zetas.slice(t) = eta_zetas;
    }
  }

}

// Exposing a C++ class in R
//using namespace Rcpp;
RCPP_MODULE(zanim_ln_reg) {

  // Expose class on the R side
  Rcpp::class_<ZANIMLNReg>("ZANIMLNReg")

  // Exposing constructor
  .constructor<arma::umat, arma::mat, arma::mat>()

  // Exposing member functions
  .method("SetMCMC", &ZANIMLNReg::SetMCMC)
  .method("RunMCMC", &ZANIMLNReg::RunMCMC)

  // Exposing some attributes
  .field("draws_betas_theta", &ZANIMLNReg::draws_betas_theta)
  .field("draws_betas_zeta", &ZANIMLNReg::draws_betas_zeta)
  .field("draws_phi", &ZANIMLNReg::draws_phi)
  .field("draws_thetas", &ZANIMLNReg::draws_thetas)
  .field("draws_varthetas", &ZANIMLNReg::draws_varthetas)
  .field("draws_zetas", &ZANIMLNReg::draws_zetas)
  .field("draws_chol_Sigma_V", &ZANIMLNReg::draws_chol_Sigma_V)
  .field("Sigma_U", &ZANIMLNReg::Sigma_U)

  ;

}
