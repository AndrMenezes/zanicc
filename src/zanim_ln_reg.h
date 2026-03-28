#ifndef ZANIMLNREG_h
#define ZANIMLNREG_h

#include <RcppArmadillo.h>

class ZANIMLNReg {
public:
  ZANIMLNReg(const arma::umat &Y, const arma::mat &X_theta, const arma::mat &X_zeta);
  ~ZANIMLNReg();

  // Fields
  arma::umat Y;
  arma::uvec n_trials, sum_col_zeros_Y;
  arma::mat X_theta, X_zeta;
  int n, d, dm1, p_theta, p_zeta;

  void SetMCMC(std::vector<double> sd_prior_beta_theta_,
               arma::mat sigma_prior_beta_zeta_,
               int ndpost_, int nskip_, int nthin_,
               arma::mat B_, int covariance_type_,
               double a_sigma_, double b_sigma_,
               arma::mat Psi_prior_, double nu_prior_,
               int q_factors_, double sigma2_gamma_,
               double a_psi_, double b_psi_,
               double shape_lsphis_,
               double a1_gs_, double a2_gs_, int keep_draws_);
  std::vector<double> sd_prior_beta_theta;
  arma::mat sigma_prior_beta_zeta;
  int ndpost, nskip, nthin, covariance_type, q_factors, keep_draws ;

  // Factor analytic hyper-prior on the covariance of random effect
  double a_sigma, b_sigma, sigma2_gamma, a_psi, b_psi;

  double LogTargetBetasTheta(arma::vec &beta_cur, int &j);
  arma::vec UpdateBetasThetaESS(int &j);
  arma::vec UpdateBetasZetaChib(int &j);
  // Regression coefficients
  arma::mat betas_theta, betas_zeta;

  void UpdateLatentVariables();
  // Vector with zero indices and the latent variable z_{ij} indicator for
  // sampling or structural zero
  std::vector<std::vector<int>> zero_indices, zs;
  // Latent variables \phi_i \sim Gamma
  arma::vec phi, rt;
  arma::mat alphas, lambdas, eta_zetas, varthetas, Z_probit;
  arma::umat Y_zi;
  // Latent variables where each row u_i \sim N_d[0, \Sigma]
  arma::umat Z;

  // Method to run the MCMC
  void RunMCMC();

  // Posterior parameters
  arma::mat sigma_inv_probit, sigma_chol_inv_probit;
  // Keep the draws
  arma::cube draws_betas_theta, draws_betas_zeta, draws_chol_Sigma_V, draws_zetas,
  draws_thetas, draws_varthetas;
  arma::mat draws_phi;

  // Update random effect and the covariance
  // Internal function to compute the log-target of u_i
  double LogTargetU(arma::vec &u, arma::uvec &y, arma::uvec &z, arma::vec &lambda);
  arma::mat B, Bt, U, exp_U, V, Psi_prior, Sigma_U, Sigma_V, chol_Sigma_V;
  arma::vec v_cur, alphas_cur, sigmas;
  arma::uvec y_cur, z_cur;
  double nu_prior;

  // ESS algorithm to update the latent variables u_i
  arma::vec ESSDiag(arma::vec &v, arma::uvec &y, arma::vec &sigmas, arma::uvec &z, arma::vec &lambda);
  arma::vec ESSFull(arma::vec &v, arma::uvec &y, arma::uvec &z, arma::vec &lambda);
  // Different methods to update the \Sigma
  void UpdateSigmaDiag();
  void UpdateSigmaFull();
  void UpdateSigmaFactorModel();
  void UpdateSigmaFactorModelMGP();
  // Functions to update the factor model, factor loadings, factor scores, and the precisions.
  void UpdateH();
  void UpdateGamma();
  void UpdatePsis();
  // Functions to update the parameters of the MGP prior
  void UpdateGammaChol();
  void UpdateLocalShrinkage();
  void UpdateGlobalShrinkage();

  // Auxiliary variables used for the random effects and its covariance
  double shape_post_sigma, df_wishart, shape_psis, rt_psis, shape_lsphis,
  shape_post_lsphis, a1_gs, a2_gs, rt_gs, shape_post_gs_delta1;
  arma::mat Gamma, GammaPsis, Hmat, Iq, M, Q, Q_chol, Q_chol_inv, HtH, HVp,
  W_eigen, R_psis, Phis_ls, GammaPhi;
  arma::vec psis, s_eigen, taus_gs, deltas_gs, mu_gamma_chol;


};
#endif
