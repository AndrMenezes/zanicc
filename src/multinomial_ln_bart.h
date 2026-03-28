#ifndef MultinomialLNBART_H
#define MultinomialLNBART_H

#include <RcppArmadillo.h>
#include "node.h"
#include "rng.h"

class MultinomialLNBART {
public:
  MultinomialLNBART(const arma::umat &Y, const arma::mat &X);
  MultinomialLNBART(int d, int p);
  ~MultinomialLNBART();

  // Data attributes
  arma::umat Y;
  arma::uvec n_trials;
  arma::mat X;
  int n, d, dm1, p;

  // Iterator for the categories
  int j_cat;

  // log-marginal likelihood
  double lml(Node *leaf);
  // Update sufficient statistics
  void UpdateSuffStats(Node *leaf);
  void UpdateTree(Node *tree);
  // Draw posterior parameter
  void DrawPosterior(Node *tree);

  // Setting MCMC
  void SetMCMC(double v0, int ntrees_,
               arma::mat B_, int covariance_type_,
               double a_sigma_, double b_sigma_,
               arma::mat Psi_prior_, double nu_prior_,
               int q_factors_, double sigma2_gamma_,
               double a_psi_, double b_psi_,
               double shape_lsphis_,
               double a1_gs_, double a2_gs_,
               int ndpost_, int nskip_,
               int numcut_, double power_ , double base_,
               std::vector<double> proposals_prob_,
               int update_sd_prior_, double s2_0_, double w_ss_,
               std::vector<std::vector<double>> list_splitprobs_, int sparse_,
               std::vector<double> sparse_parms_, std::vector<double> alpha_sparse_,
               int alpha_random_, arma::mat xinfo, std::string forests_dir_,
               int keep_draws_, int save_trees_);

  // Prior parameters for the leaf node
  double c_shape, d_rate, c_logd_lgamc, sigma, s2_0, w_ss;
  int update_sd_prior;
  std::vector<double> sigma_mcmc;

  // List of trees
  std::vector<std::vector<Node*>> trees;
  // Keep the individual trees predictions
  arma::cube g_trees;
  // If user want to give the x_breaks
  arma::mat x_breaks;
  // Many hyper-parameters
  int ntrees, numcut, ndpost, nskip, sparse, k_grid, alpha_random, covariance_type,
    q_factors, keep_draws, save_trees;
  double power, base, logprob_grow, logprob_prune, a_sigma, b_sigma, slp;
  std::vector<double> proposals_prob, sparse_parms, alpha_sparse;

  // Path to save the trees
  std::string forests_dir;
  // List with split probs for each category!
  std::vector<std::vector<double>> list_splitprobs;
  std::vector<double> splitprobs;

  // Common concentration prior parameter \alpha for the Dirichlet sparse prior
  std::vector<double> alphas_grid, log_posterior_alpha_sparse;

  // flag variables to track the acceptance rate of grow, prune and change
  int flag_grow, flag_prune, flag_change, move;

  // Backfit
  void BackFit(int &j, int &t);
  // Latent variable \phi and the "fit" but except the tree h times \phi
  arma::vec phi, fit_h_phi, fit_h, rt;
  arma::mat f_mu, f_lambda;

  // Run MCMC
  void RunMCMC();
  arma::cube draws_theta, draws_vartheta;
  arma::mat draws_phi;
  // Storage object for number of leaves, depth and acceptance ratio
  arma::umat avg_leaves, avg_depth, accept_rate;
  // arma::umat ar_grow, ar_prune, ar_change;

  // Functions/fields related to the random effects and the factor analytic
  // hyper-prior on the covariance matrix
  double LogTargetU(arma::vec &u, arma::uvec &y,
                    arma::vec &lambda, int &n_trial);
  arma::mat B, Bt, U, exp_U, V, Psi_prior, Sigma_U, Sigma_V, chol_Sigma_V;
  arma::vec v_cur, lambda_cur, sigmas;
  arma::uvec y_cur;
  arma::cube draws_chol_Sigma_V;

  // ESS algorithm to update the latent variables u_i
  arma::vec ESSDiag(arma::vec &v, arma::uvec &y, arma::vec &lambda,
                    arma::vec &sigmas, int n_trial);
  arma::vec ESSFull(arma::vec &v, arma::uvec &y, arma::vec &lambda, int n_trial);
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
  // Parameters related to prior or posterior
  double nu_prior, sigma2_gamma, a_psi, b_psi, shape_lsphis, a1_gs, a2_gs,
    shape_post_sigma, df_wishart, shape_psis, rt_psis, shape_post_lsphis,
    rt_gs, shape_post_gs_delta1;
  // Matrix used for the factor-analysis
  arma::mat Gamma, GammaPsis, Hmat, Iq, M, Q, Q_chol, Q_chol_inv, HtH, HVp,
  W_eigen, R_psis, Phis_ls, GammaPhi;
  arma::vec psis, s_eigen, taus_gs, deltas_gs, mu_gamma_chol;


  // Number of times that variable is used in a tree decision rule (over all trees).
  void ComputeVarCount(Node *tree, arma::uvec &varcount);
  arma::uvec GetVarCount(int j);
  arma::ucube GetVarCount2(int p, int d, int n_samples, int ntrees,
                           std::string forests_dir);
  arma::ucube varcount_mcmc;
  arma::uvec vc;
  // Update the concentration parameter of the DART prior
  void UpdateAlphaDART(int &j);

  // Predictions for a new data set
  void Predict(arma::mat &X_, int n_samples, int ntrees, std::string forests_dir,
               std::string output_dir, int verbose);
  double GetMu(Node *tree, const arma::rowvec &x);
  //arma::vec PredictMu(int j);

  double UpdateSigmaPrior();

};

#endif
