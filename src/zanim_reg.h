#ifndef ZANIMLINEARREG
#define ZANIMLINEARREG

#include <RcppArmadillo.h>

class ZANIMLinearReg {

public:
  ZANIMLinearReg(const arma::umat &Y, const arma::mat &X_theta,
                 const arma::mat &X_zeta);
  ~ZANIMLinearReg();

  // Fields
  arma::umat Y;
  arma::uvec n_trials, sum_col_zeros_Y;
  arma::mat X_theta, X_zeta;
  int n, d, p_theta, p_zeta;

  void SetMCMC(std::vector<double> sd_prior_beta_theta,
               arma::mat sigma_prior_beta_zeta,
               int ndpost, int nskip, int nthin);
  std::vector<double> sd_prior_beta_theta;
  arma::mat sigma_prior_beta_zeta;
  int ndpost, nskip,  nthin;

  double LogTargetBetasTheta(arma::vec &beta_cur, int &j);
  arma::vec UpdateBetasThetaESS(int &j);
  arma::vec UpdateBetasZetaChib(int &j);

  void UpdateLatentVariables();
  void RunMCMC();

  // Vector with zero indices and the latent variable z_{ij} indicator for
  // sampling or structural zero
  std::vector<std::vector<int>> zero_indices, zs;

  // Latent variables \phi_i \sim Gamma
  arma::vec phi, rt;
  arma::mat eta_lambdas, eta_zetas, Z_probit, varthetas;
  arma::umat Y_zi;

  // Parameters
  arma::mat betas_theta, betas_zeta;

  // Posterior parameters
  arma::mat sigma_inv_probit, sigma_chol_inv_probit;
  // Keep the draws
  arma::cube draws_betas_theta, draws_betas_zeta, draws_zetas, draws_thetas, draws_varthetas;
  arma::mat draws_phi;
};
#endif
