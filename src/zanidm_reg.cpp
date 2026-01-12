#include "zanidm_reg.h"
#include "rng.h"

constexpr double PI_2 = 6.283185307179586231996;

ZANIDMLinearReg::ZANIDMLinearReg(const arma::umat &Y, const arma::mat &X_alpha,
                               const arma::mat &X_zeta) : Y(Y), X_alpha(X_alpha), X_zeta(X_zeta) {
  d = Y.n_cols;
  n = Y.n_rows;
  p_alpha = X_alpha.n_cols;
  p_zeta = X_zeta.n_cols;
  n_trials = arma::sum(Y, 1);
}

ZANIDMLinearReg::~ZANIDMLinearReg(){};


void ZANIDMLinearReg::SetMCMC(std::vector<double> sd_prior_beta_alpha_,
                             arma::mat sigma_prior_beta_zeta_,
                             int ndpost_, int nskip_, int nthin_) {
  sd_prior_beta_alpha = sd_prior_beta_alpha_;
  sigma_prior_beta_zeta = sigma_prior_beta_zeta_;
  ndpost = ndpost_;
  nskip = nskip_;
  nthin = nthin_;

  // Initialise the reg coefficients
  betas_alpha = arma::randn<arma::mat>(p_alpha, d);
  betas_zeta = arma::randn<arma::mat>(p_zeta, d);

  // Initialise the linear predictions coefficients
  alphas = arma::zeros<arma::mat>(n, d);
  eta_zetas = arma::zeros<arma::mat>(n, d);
  lambdas = arma::zeros<arma::mat>(n, d);

  // Define the response for zero-inflation (it will change over the iterations)
  Y_zi = arma::zeros<arma::umat>(n, d);
  for (int j=0; j < d; j++) Y_zi.col(j) = Y.col(j) == 0;

  // Compute the cholesky decomposition of the covariance matrix of the full conditional
  // distribution of the probit-reg coefficients for all categories

  sigma_inv_probit = arma::inv_sympd(X_zeta.t() * X_zeta + arma::inv_sympd(sigma_prior_beta_zeta));
  sigma_chol_inv_probit = arma::chol(sigma_inv_probit);


  // First draw for the latent variable phi_i
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

  // First draw for the latent variable \lambda_{ij} ~ Gamma[\alpha_{ij} + y_{ij}, 1 + \phi_i]
  // if z_{ij}=1 otherwise is zero

  for (int j = 0; j < d; j++) {
    for (int i=0; i < n; i ++) {
      if (Y_zi(i, j) == 1) lambdas(i, j) = 0.0;
      else lambdas(i, j) = R::rgamma(Y(i, j), 1.0);
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

double ZANIDMLinearReg::LogTargetBetasAlpha(arma::vec &beta_cur, int &j) {

  arma::vec alphas = exp(X_alpha * beta_cur);
  double t1 = 0.0;
  for (int i=0; i < n; i++) {
    if (lambdas(i, j) > 0) t1 += alphas(i) * log(lambdas(i, j)) - R::lgammafn(alphas(i));
  }
  return t1;
}

arma::vec ZANIDMLinearReg::UpdateBetasAlphaESS(int &j) {

  arma::vec beta_cur = betas_alpha.col(j);
  // Draw from the prior
  arma::vec nu = arma::zeros<arma::vec>(p_alpha);
  for (int k=0; k < p_alpha; k++) nu[k] = sd_prior_beta_alpha[k] * R::norm_rand();
  // Set a log-likelihood threshold
  double u_s = log(R::unif_rand());
  u_s += LogTargetBetasAlpha(beta_cur, j);
  // Draw an angle
  double theta = R::unif_rand() * PI_2;
  double theta_max = theta;
  double theta_min = theta - PI_2;
  // Draw proposal
  arma::vec beta_star = beta_cur * cos(theta) + nu * sin(theta);
  // Start slicing
  do {
    if (LogTargetBetasAlpha(beta_star, j) > u_s) break;
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    theta = theta_min + (theta_max - theta_min) * R::unif_rand();
    beta_star = beta_cur * cos(theta) + nu * sin(theta);
  } while (true);

  return beta_star;
}


arma::vec ZANIDMLinearReg::UpdateBetasZetaChib(int &j) {
  arma::vec beta_tilde = sigma_inv_probit * (X_zeta.t() * Z_probit.col(j));
  return beta_tilde + rmvnorm(p_zeta, sigma_chol_inv_probit);
}

void ZANIDMLinearReg::UpdateLatentVariables() {
  for (int j=0; j < d; j++) {
    // Compute the linear predictions
    eta_zetas.col(j) = X_zeta * betas_zeta.col(j);
    alphas.col(j) = exp(X_alpha * betas_alpha.col(j));
    // Update z_{ij}, then \lambda_{ij} | z_{ij} (marginal first, conditional later)
    for (int i = 0; i < n; i++) {
      lambdas(i, j) = R::rgamma(Y(i, j) + alphas(i, j), 1.0) / (1.0 + phi(i));
    }
    for(int k = 0; k < sum_col_zeros_Y(j); k++) {
      int cur_indice = zero_indices[j][k];
      double zeta_ij = R::pnorm5(eta_zetas(cur_indice, j), 0.0, 1.0, 1.0, 0.0);
      double p_ij = (1.0 - zeta_ij) *  std::pow(1.0 + phi(cur_indice), -alphas(cur_indice, j));
      p_ij /= (zeta_ij + p_ij);
      zs[j][k] = R::rbinom(1, p_ij);
      // Update lambdas'
      lambdas(cur_indice, j) *= zs[j][k];
      // Update the response when y_{ij}=0 (the "success" is y_{ij} being structural zero)
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
  rt = arma::sum(lambdas, 1);
  // Update \phi_i \sim Gamma[N_i, \sum_j \lambda_{ij} * z_{ij}]
  for (int k=0; k < n; k++) {
    if (n_trials(k) == 0) phi(k) = 0.0;
    else phi(k) = R::rgamma(n_trials(k), 1.0) / rt(k) ;
  }

}

void ZANIDMLinearReg::RunMCMC() {

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
      // Update \beta^{(\alpha)}_{j,k} using ESS
      betas_alpha.col(j) = UpdateBetasAlphaESS(j);

      // Update \beta^{(\zeta)}_{j,k} using Abert-Chib probit DA.
      betas_zeta.col(j) = UpdateBetasZetaChib(j);
    }
  }

  // // Initialise container for keep the draws
  draws_betas_alpha = arma::zeros<arma::cube>(p_alpha, d, ndpost);
  draws_betas_zeta = arma::zeros<arma::cube>(p_zeta, d, ndpost);
  draws_varthetas = arma::zeros<arma::cube>(n, d, ndpost);
  draws_alphas = arma::zeros<arma::cube>(n, d, ndpost);
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
      betas_alpha.col(j) = UpdateBetasAlphaESS(j);
      // Update \beta^{(\zeta)}_{j,k} using Abert-Chib probit DA.
      betas_zeta.col(j) = UpdateBetasZetaChib(j);
    }
    draws_phi.col(t) = phi;
    // Save draws
    draws_betas_alpha.slice(t) = betas_alpha;
    draws_betas_zeta.slice(t) = betas_zeta;
    draws_varthetas.slice(t) = lambdas.each_col() / arma::sum(lambdas, 1);
    draws_alphas.slice(t) = alphas;
    draws_zetas.slice(t) = eta_zetas;
  }
}

// Exposing a C++ class in R
//using namespace Rcpp;
RCPP_MODULE(zanidm_linear_reg) {

  // Expose class on the R side
  Rcpp::class_<ZANIDMLinearReg>("ZANIDMLinearReg")

  // Exposing constructor
  .constructor<arma::umat, arma::mat, arma::mat>()

  // Exposing member functions
  .method("SetMCMC", &ZANIDMLinearReg::SetMCMC)
  .method("RunMCMC", &ZANIDMLinearReg::RunMCMC)

  // Exposing some attributes
  .field("draws_betas_alpha", &ZANIDMLinearReg::draws_betas_alpha)
  .field("draws_betas_zeta", &ZANIDMLinearReg::draws_betas_zeta)
  .field("draws_phi", &ZANIDMLinearReg::draws_phi)
  .field("draws_abundance", &ZANIDMLinearReg::draws_varthetas)
  .field("draws_alphas", &ZANIDMLinearReg::draws_alphas)
  .field("draws_zetas", &ZANIDMLinearReg::draws_zetas)
  ;

}
