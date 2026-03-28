#ifndef DMLINEARREG
#define DMLINEARREG

#include <RcppArmadillo.h>
#include "rng2.h"

class DMLinearReg {

private:
  RNG2 rng;

public:
  DMLinearReg(const arma::umat &Y, const arma::mat &X);
  ~DMLinearReg();


  // Fields

  arma::umat Y;
  arma::uvec n_trials, sum_col_zeros_Y;
  arma::mat X;
  int n, d, p;

  void SetMCMC(arma::cube V_prior_betas_, int ndpost_, int nskip_, int nthin_,
               int keep_draws, int save_draws, std::string dir_draws);
  arma::cube V_prior_betas, chol_V_prior;
  int ndpost, nskip, nthin, keep_draws, save_draws;
  std::string dir_draws;

  double LogTargetBetas(arma::vec &beta_cur, int &j);
  arma::vec UpdateBetasESS(int &j);
  arma::rowvec ep;
  arma::vec nu;

  void UpdateLatentVariables();
  void RunMCMC();
  void Predict();

  // Latent variables \phi_i \sim Gamma
  arma::vec phi, rt;
  arma::mat lambdas, alphas;

  // Parameters
  arma::mat betas;

  // Posterior parameters
  arma::mat sigma_inv_probit, sigma_chol_inv_probit;

  // Keep the draws
  arma::cube draws_betas, draws_varthetas, draws_alphas;
  arma::mat draws_phi;
};
#endif
