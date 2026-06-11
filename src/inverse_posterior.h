#ifndef INVERSEPOSTERIOR
#define INVERSEPOSTERIOR

#include "node.h"

class InversePosterior {

public:
  // Constructor
  InversePosterior(int d, int ntrees_theta, int ntrees_zeta, std::string forests_dir);

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
  double mu_prior_1, sd_prior;

  //////////////////////////////////////////////////////////////////////////////////
  // Implemented and stable methods for the ZANIM-LN-BART

  // Get the tree-specific prediction by traversing the tree
  double GetMu(Node *tree, std::vector<double> &x);

  // Compute the ZANIM-(LN)-BART predictions for a given x
  void GetBARTPredictions(std::vector<double> &x, std::vector<double> &theta,
                          std::vector<double> &zeta,
                          const std::vector<std::vector<Node*>> &forest_theta,
                          const std::vector<std::vector<Node*>> &forest_zeta);

  // Sampling importance resampling with multiple imputation
  void SIR(arma::umat Y, int n_proposal, int ndpost, arma::mat B,
           std::string draws_dir);
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


  //////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////
  // Experimental methods for other models or different samplers

  // Elliptical slice sampling for ML-BART and ZANIM-BART models
  std::vector<double> SamplerMLBARTeSS(arma::umat Y, arma::mat X_ini, int ndpost,
                                       std::vector<double> mean_prior,
                                       arma::mat S_prior,
                                       int n_rep);

  std::vector<double> SamplerZANIMBARTeSS(arma::umat Y, arma::mat X_ini, int ndpost,
                                          std::vector<double> mean_prior,
                                          arma::mat S_prior, int nburnin,
                                          int conditional);

  // SIR
  std::vector<int> SIRZANIMLNBART(std::vector<int> y, int n_proposal,
                                  int ndpost, arma::mat B,
                                  std::string draws_dir, int n_particles, int mixture);
  std::vector<int> SIRMLBART(std::vector<int> y, int n_proposal,
                             int ndpost, std::string draws_dir);
  std::vector<int> SIRZANIMBART(std::vector<int> y, int n_proposal, int ndpost,
                                std::string draws_dir, int conditional);
  std::vector<int> ABCSIRZANIMLNBART(std::vector<int> y, int n_proposal,
                                     int ndpost, arma::mat B,
                                     std::string draws_dir,
                                     int kernel,
                                     double h,
                                     int n_particles);

  // Posterior draws and the probabilities of Population MC scheme
  std::vector<double> x_posterior, x_probs;

  // Implementation of adaptive Population Monte Carlo method
  double ComputeEfSS(std::vector<double> &x);
  void WeightedMeanVar(double &mu, double &s2, std::vector<double> &x,
                       std::vector<double> &probs);
  void PopulationMC(std::vector<int> y,
                    int ndpost, int n_particles_x,
                    arma::mat B,
                    std::vector<double> range_prior,
                    double scale_prop,
                    double prob_level, double ep);

  // Get the ML-BART predictions for a given x
  void GetPredictionMLBART(std::vector<double> &x, std::vector<double> &theta,
                           const std::vector<std::vector<Node*>> &forest_theta);

  // Test different implementations of the marginal log-likelihood for the ZANIM-LN
  double LogLikelihoodZANIMLN(std::vector<int> &y,
                              std::vector<double> &x,
                              int ndpost,
                              int chain_index, arma::mat B);
  std::vector<double> LogLikelihoodZANIMLN_2(std::vector<int> &y,
                                             std::vector<double> &x, int ndpost,
                                             arma::mat B);
  double lmlZANIM(std::vector<int> &y, std::vector<double> &x, int n_particles);

  // Get BART predictions without normalise the composition probabilities
  void GetTreesPredictionsZANIMBART(std::vector<double> &x,
                                    std::vector<double> &lambda,
                                    std::vector<double> &zeta,
                                    const std::vector<std::vector<Node*>> &forest_theta,
                                    const std::vector<std::vector<Node*>> &forest_zeta);

  // Run one update of ESS using the Poisson-type likelihood
  std::vector<double> UpdateESSZANIMLNBART2(
      std::vector<double> &x_cur,
      std::vector<int> &y,
      std::vector<double> &z, std::vector<double> &u,
      double &phi,
      std::vector<double> &lambda, std::vector<double> &zeta,
      const std::vector<std::vector<Node*>> &forest_theta,
      const std::vector<std::vector<Node*>> &forest_zeta);

  // ESS2
  std::vector<double> ESSZANIMLNBART2(arma::umat Y, arma::mat X_ini,
                                     int ndpost, int nburnin,
                                     std::vector<double> mean_prior,
                                     arma::mat S_prior,
                                     arma::mat B);

  // Functions to sample from the full conditional of u_i = (u_i1, ..., u_id) under
  // the ZANIM-LN-BART model
  double LogTargetU(std::vector<double> &u, std::vector<int> &y,
                    std::vector<double> &z,
                    std::vector<double> &lambda);
  std::vector<double> UpdateESSV(std::vector<double> &v,
                       std::vector<double> &chol_Sigma_V,
                       std::vector<double> &B,
                       std::vector<int> &y, std::vector<double> &z,
                       std::vector<double> &lambda);
  //////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////

};
#endif
