#ifndef ZANIMLOGNORMALBART_H
#define ZANIMLOGNORMALBART_H

#include <RcppArmadillo.h>
#include "node.h"
#include "probit_bart.h"
#include "multinomial_bart.h"

class ZANIMLogNormalBART {
public:
  ZANIMLogNormalBART(const arma::umat &Y, const arma::mat &X_theta,
                       const arma::mat &X_zeta);
  ~ZANIMLogNormalBART();

  // Fields
  arma::umat Y;
  arma::mat X_theta, X_zeta;
  int n, d, dm1, p_theta, p_zeta;

  // Multinomial class
  MultinomialBART *bart_mult;

  // List of ProbitBART class
  std::vector<ProbitBART*> list_bart_zi;

  // Method to set-up the model
  void SetMCMC(double v0_theta, double k_zeta,
               int ntrees_theta_, int ntrees_zeta_,
               arma::mat B_, int covariance_type_,
               double a_sigma_, double b_sigma_,
               arma::mat Psi_prior_, double nu_prior_,
               int q_factors_, double sigma2_gamma_,
               double a_psi_, double b_psi_,
               double shape_lsphis_, double a1_gs_, double a2_gs_,
               int ndpost_, int nskip_,
               int numcut, double power, double base,
               std::vector<double> proposals_prob_,
               int update_sd_prior_, double s2_0_, double w_ss_,
               std::vector<double> splitprobs_zi,
               std::vector<std::vector<double>> splitprobs_mult,
               int sparse_zi_,
               int sparse_mult_,
               std::vector<double> sparse_parms_zi_,
               std::vector<double> sparse_parms_mult_,
               std::vector<double> alpha_sparse_zi_,
               std::vector<double> alpha_sparse_mult_,
               int alpha_random_zi_,
               int alpha_random_mult_,
               arma::mat xinfo, std::string path_out_, int keep_draws_);
  int ntrees_theta, ntrees_zeta, ndpost, nskip, sparse_mult,
   alpha_random_mult, sparse_zi, alpha_random_zi, update_sd_prior, covariance_type,
   q_factors, keep_draws;
  std::string path_out;
  std::vector<double> proposals_prob, sparse_parms_zi, sparse_parms_mult,
   alpha_sparse_zi, alpha_sparse_mult, sigma_mult_mcmc;
  std::vector<std::vector<double>> list_splitprobs_mult, list_splitprobs_zi;

  // hyper-prior on the variance
  double a_sigma, b_sigma, sigma2_gamma, a_psi, b_psi;

  // Common concentration prior parameter \alpha for the Dirichlet sparse prior
  int k_grid;
  std::vector<double> alphas_grid_mult, alphas_grid_zi, lp_alpha_sparse_mult,
  lp_alpha_sparse_zi;

  // Vector and matrices for multinomial BART
  arma::vec rt, fit_h;
  arma::mat f_mu, f_lambda, varthetas;
  // Vector and matrices for ZI probit BART
  // arma::umat Z;
  arma::vec fit_0_h;
  arma::mat f0_mu;
  arma::mat draws_phi;
  arma::uvec vc_mult, vc_zi;

  // Vector with zero indices and the latent variable z_{ij} indicator for
  // sampling or structural zero
  std::vector<std::vector<int>> zero_indices, zs;
  arma::uvec sum_col_zeros_Y;

  // Aux to compute sum(log(splitprobs))
  double slp;

  // Wrapper functions to perform the back-fit and update the latent variables
  int move, cur_indice;
  double zeta_ij, p_ij;
  void BackFitMultinomial(int &j, int &t);
  void BackFitZI(int &j, int &t);
  void UpdateLatentVariables();
  void UpdateSplitProbsMult(int &j);
  void UpdateSplitProbsZI(int &j);
  void UpdateAlphaMult(int &j);
  void UpdateAlphaZI(int &j);

  // Internal function to compute the log-target of u_i
  double LogTargetU_phi(arma::vec &u, arma::uvec &y, arma::uvec &z, double &phi,
                        arma::vec &lambda);
  double LogTargetU(arma::vec &u, arma::uvec &y, arma::uvec &z, arma::vec &lambda);
  arma::mat B, Bt, U, exp_U, V, Psi_prior, Sigma_U, Sigma_V, chol_Sigma_V;
  arma::vec v_cur, lambda_cur, sigmas;
  arma::uvec y_cur, z_cur;
  double nu_prior;

  // ESS algorithm to update the latent variables u_i
  arma::vec ESSDiag(arma::vec &v, arma::uvec &y, arma::vec &sigmas,
                    arma::uvec &z, arma::vec &lambda, double phi);
  arma::vec ESSFull(arma::vec &u, arma::uvec &y, arma::uvec &z, arma::vec &lambda);
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

  double shape_post_sigma, df_wishart, shape_psis, rt_psis, shape_lsphis,
  shape_post_lsphis, a1_gs, a2_gs, rt_gs, shape_post_gs_delta1;
  arma::mat Gamma, GammaPsis, Hmat, Iq, M, Q, Q_chol, Q_chol_inv, HtH, HVp, W_eigen, R_psis, Phis_ls, GammaPhi;
  arma::vec psis, s_eigen, taus_gs, deltas_gs, mu_gamma_chol;

  // Cube to keep the draws
  arma::cube draws_Sigma_U, draws_theta, draws_zeta, draws_vartheta, draws_Gamma;

  // Function to run the MCMC
  void RunMCMC();
  //arma::cube draws_theta, draws_zeta;

  // Storage object for number of leaves
  arma::umat avg_leaves_theta, avg_leaves_zeta;

  // Storage object for acceptance ratio
  arma::umat accept_rate_theta, accept_rate_zeta;

  // Predict functions
  arma::vec GetMu(Node *tree, const arma::rowvec &x);
  void ComputePredictProb(arma::mat &X_, int n_samples, int ntrees,
                          std::string path_out, std::string path, int verbose);
  void ComputePredictProbZero(arma::mat &X_, int n_samples, int ntrees,
                              std::string path_out, std::string path, int verbose);

  void ComputeVarCount(Node *tree, arma::uvec &varcount);
  arma::ucube GetVarCount(int n_samples, int ntrees, std::string parm_name,
                          std::string path);
  // varcount
  arma::ucube varcount_mcmc_theta;
  arma::ucube varcount_mcmc_zeta;

  // Keep the posterior mean of concentration parameter \alpha
  arma::vec alpha_mult_post_mean, alpha_zi_post_mean;
  // std::vector<double> split_post_mean;
};

#endif
