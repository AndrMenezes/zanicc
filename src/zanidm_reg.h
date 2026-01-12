#ifndef ZANIDMLINEARREG
#define ZANIDMLINEARREG

#include <RcppArmadillo.h>

class ZANIDMLinearReg {

public:
  ZANIDMLinearReg(const arma::umat &Y, const arma::mat &X_alpha,
                  const arma::mat &X_zeta);
  ~ZANIDMLinearReg();

  // Fields
  arma::umat Y;
  arma::uvec n_trials, sum_col_zeros_Y;
  arma::mat X_alpha, X_zeta;
  int n, d, p_alpha, p_zeta;

  void SetMCMC(std::vector<double> sd_prior_beta_alpha_,
               arma::mat sigma_prior_beta_zeta_,
               int ndpost_, int nskip_, int nthin_);
  std::vector<double> sd_prior_beta_alpha;
  arma::mat sigma_prior_beta_zeta;
  int ndpost, nskip,  nthin;

  double LogTargetBetasAlpha(arma::vec &beta_cur, int &j);
  arma::vec UpdateBetasAlphaESS(int &j);
  arma::vec UpdateBetasZetaChib(int &j);

  void UpdateLatentVariables();
  void RunMCMC();

  // Vector with zero indices and the latent variable z_{ij} indicator for
  // sampling or structural zero
  std::vector<std::vector<int>> zero_indices, zs;

  // Latent variables \phi_i \sim Gamma
  arma::vec phi, rt;
  arma::mat lambdas, alphas, eta_zetas, Z_probit;
  arma::umat Y_zi;

  // Parameters
  arma::mat betas_alpha, betas_zeta;

  // Posterior parameters
  arma::mat sigma_inv_probit, sigma_chol_inv_probit;
  // Keep the draws
  arma::cube draws_betas_alpha, draws_betas_zeta, draws_zetas, draws_varthetas, draws_alphas;
  arma::mat draws_phi;
};
#endif
