#include "zanim_reg.h"
#include "rng.h"

constexpr double PI_2 = 6.283185307179586231996;

ZANIMLinearReg::ZANIMLinearReg(const arma::umat &Y, const arma::mat &X_theta,
                               const arma::mat &X_zeta) : Y(Y), X_theta(X_theta), X_zeta(X_zeta) {
  d = Y.n_cols;
  n = Y.n_rows;
  p_theta = X_theta.n_cols;
  p_zeta = X_zeta.n_cols;
  n_trials = arma::sum(Y, 1);
}

ZANIMLinearReg::~ZANIMLinearReg(){};


void ZANIMLinearReg::SetMCMC(std::vector<double> sd_prior_beta_theta_,
                             arma::mat sigma_prior_beta_zeta_,
                             int ndpost_, int nskip_, int nthin_) {
  sd_prior_beta_theta = sd_prior_beta_theta_;
  sigma_prior_beta_zeta = sigma_prior_beta_zeta_;
  ndpost = ndpost_;
  nskip = nskip_;
  nthin = nthin_;

  // Initialise the reg coefficients
  betas_theta = arma::randn<arma::mat>(p_theta, d);
  betas_zeta = arma::randn<arma::mat>(p_zeta, d);

  // Initialise the linear predictions coefficients
  eta_lambdas = arma::zeros<arma::mat>(n, d);
  varthetas = arma::zeros<arma::mat>(n, d);
  eta_zetas = arma::zeros<arma::mat>(n, d);

  // Define the response for zero-inflation (it will change over the iterations)
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
    for (int k = 0; k < n; k++) {
      if (Y_zi(k, j) == 0) Z_probit(k, j) = -rtnorm(0, 1.0, 0.0);
      else Z_probit(k, j) = rtnorm(0, 1.0, 0.0);
    }
  }

}

double ZANIMLinearReg::LogTargetBetasTheta(arma::vec &beta_cur, int &j) {

  arma::vec eta = X_theta * beta_cur;
  arma::vec e_eta_phi = exp(eta) % phi;
  double t1 = arma::sum(Y.col(j) % eta);
  double t2 = arma::sum(e_eta_phi);
  // Remove the contribution of z_{ij} = 0 in rt
  for (int k = 0; k < sum_col_zeros_Y(j); k++) {
    if (zs[j][k] == 0) {
      int cur_indice = zero_indices[j][k];
      t2 -= e_eta_phi(cur_indice);
    }
  }
  return t1 - t2;
}

arma::vec ZANIMLinearReg::UpdateBetasThetaESS(int &j) {

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


arma::vec ZANIMLinearReg::UpdateBetasZetaChib(int &j) {
  arma::vec beta_tilde = sigma_inv_probit * (X_zeta.t() * Z_probit.col(j));
  return beta_tilde + rmvnorm(p_zeta, sigma_chol_inv_probit);
}

void ZANIMLinearReg::UpdateLatentVariables() {
  for (int j=0; j < d; j++) {
    // Compute the linear predictions, eta^{\lambda}_{ij} and eta^{\zeta}_{ij}
    eta_zetas.col(j) = X_zeta * betas_zeta.col(j);
    eta_lambdas.col(j) = exp(X_theta * betas_theta.col(j));
    // Update (z_{ij} | \phi_i, \lambda_j, zeta_j) ~ Bernoulli[p_{ij}] for y_{ij}=0
    for(int k = 0; k < sum_col_zeros_Y(j); k++) {
      int cur_indice = zero_indices[j][k];
      double zeta_ij = R::pnorm5(eta_zetas(cur_indice, j), 0.0, 1.0, 1.0, 0.0);
      double p_ij = (1 - zeta_ij) * exp(-eta_lambdas(cur_indice, j) * phi(cur_indice));
      p_ij /= (zeta_ij + p_ij);
      zs[j][k] = R::rbinom(1, p_ij);
      // Update the response
      Y_zi(cur_indice, j) = 1 - zs[j][k];
    }

    // Update latent variable for the probit part
    for (int k = 0; k < n; k++) {
      // Right truncation
      if (Y_zi(k, j) == 0) Z_probit(k, j) = -rtnorm(-eta_zetas(k, j), 1.0, 0.0);
      else Z_probit(k, j) = rtnorm(eta_zetas(k, j), 1.0, 0.0);
    }
  }

  // Compute the \sum_{j=1}^d z_{ij}\lambda_{ij}
  rt = arma::sum(eta_lambdas, 1);
  varthetas = eta_lambdas;
  // Remove the contribution when z_{ij} = 0 in rt
  for (int j = 0; j < d; j++) {
    for (int k = 0; k < sum_col_zeros_Y(j); k++) {
      if (zs[j][k] == 0) {
        int cur_indice = zero_indices[j][k];
        rt(cur_indice) -= eta_lambdas(cur_indice, j);
        varthetas(cur_indice, j) = 0.0;
      }
    }
  }
  // Normalise theta (this is what we call abundance)
  varthetas.each_col() /= rt;

  // Update \phi_i \sim Gamma[N_i, \sum_j \lambda_{ij} * z_{ij}]
  for (int k=0; k < n; k++) {
    if (n_trials(k) == 0) phi(k) = 0.0;
    else phi(k) = R::rgamma(n_trials(k), 1.0) / rt(k) ;
  }

}

void ZANIMLinearReg::RunMCMC() {

  std::cout << "Doing the warm-up (burn-in) of " << nskip << "\n\n";
  double progress = 0;
  for (int t=0; t < nskip; t++) {
    progress = (double) 100 * t / nskip;
    Rprintf("%3.2f%% Warm-up completed", progress);
    Rprintf("\r");
    // Update latent variables
    UpdateLatentVariables();
    // Update regression coefficients
    for (int j = 0; j < d; j++) {
      // Update \beta^{(\lambda)}_{j,k} using ESS
      betas_theta.col(j) = UpdateBetasThetaESS(j);

      // Update \beta^{(\zeta)}_{j,k} using Abert-Chib probit DA.
      betas_zeta.col(j) = UpdateBetasZetaChib(j);
    }
  }

  // // Initialise container for keep the draws
  draws_betas_theta = arma::zeros<arma::cube>(p_theta, d, ndpost);
  draws_betas_zeta = arma::zeros<arma::cube>(p_zeta, d, ndpost);
  draws_thetas = arma::zeros<arma::cube>(n, d, ndpost);
  draws_varthetas = arma::zeros<arma::cube>(n, d, ndpost);
  draws_zetas = arma::zeros<arma::cube>(n, d, ndpost);
  draws_phi = arma::zeros<arma::mat>(n, ndpost);

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
      // Update \beta^{(\lambda)}_{j,k} using ESS
      betas_theta.col(j) = UpdateBetasThetaESS(j);
      // Update \beta^{(\zeta)}_{j,k} using Abert-Chib probit DA.
      betas_zeta.col(j) = UpdateBetasZetaChib(j);
    }
    draws_phi.col(t) = phi;
    // Save draws
    draws_betas_theta.slice(t) = betas_theta;
    draws_betas_zeta.slice(t) = betas_zeta;
    draws_thetas.slice(t) = eta_lambdas.each_col() / arma::sum(eta_lambdas, 1);
    draws_varthetas.slice(t) = varthetas;
    draws_zetas.slice(t) = eta_zetas;
  }
}

// Exposing a C++ class in R
//using namespace Rcpp;
RCPP_MODULE(zanim_linear_reg) {

  // Expose class on the R side
  Rcpp::class_<ZANIMLinearReg>("ZANIMLinearReg")

  // Exposing constructor
  .constructor<arma::umat, arma::mat, arma::mat>()

  // Exposing member functions
  .method("SetMCMC", &ZANIMLinearReg::SetMCMC)
  .method("RunMCMC", &ZANIMLinearReg::RunMCMC)

  // Exposing some attributes
  .field("draws_betas_theta", &ZANIMLinearReg::draws_betas_theta)
  .field("draws_betas_zeta", &ZANIMLinearReg::draws_betas_zeta)
  .field("draws_phi", &ZANIMLinearReg::draws_phi)
  .field("draws_thetas", &ZANIMLinearReg::draws_thetas)
  .field("draws_varthetas", &ZANIMLinearReg::draws_varthetas)
  .field("draws_zetas", &ZANIMLinearReg::draws_zetas)
  ;

}
