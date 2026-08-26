#ifndef INVERSEPOSTERIOR
#define INVERSEPOSTERIOR

#include "node.h"

class InversePosterior {

public:
  // Constructor
  InversePosterior(int d, int ntrees_theta, int ntrees_zeta,
                   std::string forests_dir);

  int d, ntrees_theta, ntrees_zeta, p, n_samples, dm1;

  // Path with the posterior draws of the latent field (BART)
  std::string forests_dir;
  // Field to keep the effective sample size of SIR
  std::vector<double> ess_sir;
  // Row-major vector with the indices of proposal values of SIR
  std::vector<int> indices_sir;

  // Mean and Cholesky decomposition of prior covariance matrix used in the (c)ESS
  std::vector<double> mu_prior, chol_S_prior;
  // Mean and sd used in univariate cESS
  double mu_prior_1, sd_prior_1;

  //////////////////////////////////////////////////////////////////////////////////
  // Stable methods for the ZANIM-LN-BART

  // Get the tree-specific prediction by traversing the tree
  double GetMu(Node *tree, std::vector<double> &x);

  // Compute the ZANIM-(LN)-BART predictions for a given x
  void GetBARTPredictions(std::vector<double> &x, std::vector<double> &theta,
                          std::vector<double> &zeta,
                          const std::vector<std::vector<Node*>> &forest_theta,
                          const std::vector<std::vector<Node*>> &forest_zeta);

  // Effective sample size for IS-based algorithms
  double EffectiveSampleSize(std::vector<double> &probs);

  // Sampling importance resampling with multiple imputation
  void SIR(arma::umat Y, int n_proposal, int ndpost, arma::mat B,
           std::string dir_posterior_fx);
  // pseudo-marginal ESS
  std::vector<double> UpdateESS(std::vector<double> &x_cur,
                                std::vector<int> &y,
                                std::vector<double> &chol_Sigma_V,
                                std::vector<double> &B,
                                std::vector<double> &theta,
                                std::vector<double> &zeta,
                                const std::vector<std::vector<Node*>> &forest_theta,
                                const std::vector<std::vector<Node*>> &forest_zeta,
                                int n_particles);
  std::vector<double> ESS(arma::umat Y, arma::mat X_ini,
                          int ndpost, int nburnin, int n_particles,
                          std::vector<double> mean_prior,
                          arma::mat S_prior,
                          arma::mat B);

  // pseudo-marginal constrained ESS
  std::vector<double> UpdateCESS(std::vector<double> &x_cur,
                                 std::vector<int> &y,
                                 std::vector<double> &chol_Sigma_V,
                                 std::vector<double> &B,
                                 std::vector<double> &Amat,
                                 std::vector<double> &bvec, double &eta,
                                 std::vector<double> &theta,
                                 std::vector<double> &zeta,
                                 const std::vector<std::vector<Node*>> &forest_theta,
                                 const std::vector<std::vector<Node*>> &forest_zeta,
                                 int n_particles);

  std::vector<double> CESS(arma::umat Y, arma::mat X_ini, int ndpost,
                           int nburnin, int n_particles,
                           std::vector<double> mean_prior,
                           arma::mat S_prior, arma::mat B, arma::mat A,
                           std::vector<double> bvec, double eta);
  // Similar as above, but specific for univariate target
  double UpdateESS1p(double &x_cur,
                      std::vector<int> &y,
                      double &mu_prior,
                      double &sd_prior,
                      std::vector<double> &chol_Sigma_V,
                      std::vector<double> &B,
                      std::vector<double> &theta,
                      std::vector<double> &zeta,
                      const std::vector<std::vector<Node*>> &forest_theta,
                      const std::vector<std::vector<Node*>> &forest_zeta,
                      int n_particles);

  std::vector<double> ESS1p(arma::umat Y, std::vector<double> X_ini, int ndpost,
                            int nburnin, int n_particles,
                            double mean_prior, double sd_prior,
                            arma::mat B);
  // Similar as above, but specific for univariate target
  double UpdateCESS1p(double &x_cur,
                      std::vector<int> &y,
                      std::vector<double> &chol_Sigma_V,
                      std::vector<double> &B,
                      double &lower,
                      double &upper,
                      double &eta,
                      std::vector<double> &theta,
                      std::vector<double> &zeta,
                      const std::vector<std::vector<Node*>> &forest_theta,
                      const std::vector<std::vector<Node*>> &forest_zeta,
                      int n_particles);

  std::vector<double> CESS1p(arma::umat Y, std::vector<double> X_ini, int ndpost,
                             int nburnin, int n_particles,
                             double mean_prior, double sd_prior,
                             arma::mat B,
                             double lower, double upper, double eta);

};
#endif
