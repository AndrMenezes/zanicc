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
  // Which forward model to use
  std::string forward_model;

  // Inverse posterior sampler using elliptical slice sampling
  std::vector<double> SamplerMLBARTeSS(arma::umat Y, arma::mat X_ini, int ndpost,
                                       std::vector<double> mean_prior, arma::mat S_prior,
                                       int n_rep);

  std::vector<double> SamplerZANIMBARTeSS(arma::umat Y, arma::mat X_ini, int ndpost,
                                          std::vector<double> mean_prior,
                                          arma::mat S_prior, int nburnin,
                                          int conditional);

  std::vector<double> SamplerZANIMLNBARTceSS(arma::umat Y, arma::mat X_ini,
                                             int ndpost, int nburnin,
                                             std::vector<double> mean_prior,
                                             arma::mat S_prior, arma::mat A,
                                             arma::mat B, std::vector<double> bvec,
                                             double eta);
  // SIR
  std::vector<int> SIRZANIMLNBART(std::vector<int> y, int n_proposal,
                                  int ndpost, arma::mat B,
                                  std::string draws_dir, int mc);
  std::vector<int> SIRMLBART(std::vector<int> y, int n_proposal,
                             int ndpost, std::string draws_dir);
  std::vector<int> SIRZANIMBART(std::vector<int> y, int n_proposal, int ndpost,
                                std::string draws_dir, int conditional);

  // Get the tree-specific prediction by traversing the tree
  double GetMu(Node *tree, std::vector<double> &x);

  // Get the ML-BART predictions for a given x
  void GetPredictionMLBART(std::vector<double> &x, std::vector<double> &theta,
                           const std::vector<std::vector<Node*>> &forest_theta);
  // Get the ZANIM-BART predictions for a given x
  void GetPredictionZANIMBART(std::vector<double> &x, std::vector<double> &theta,
                              std::vector<double> &zeta,
                              const std::vector<std::vector<Node*>> &forest_theta,
                              const std::vector<std::vector<Node*>> &forest_zeta);
  // Log-likelihoods
  double LogLikelihoodZANIMLN(std::vector<int> &y,
                              std::vector<double> &x,
                              int ndpost,
                              int chain_index, arma::mat B);
  std::vector<double> LogLikelihoodZANIMLN_2(std::vector<int> &y,
                                             std::vector<double> &x, int ndpost,
                                             arma::mat B);

  double lmlZANIM(std::vector<int> &y, std::vector<double> &x, int n_particles);

  //----- New implementations

  // Internal function to set-up variables
  // void Set();

  // Common fields for the ESS
  std::vector<double> chol_S_prior, mu_prior;


  // Update ESS using the multinomial likelihood
  std::vector<double> UpdateESSZANIMLNBART(
      std::vector<double> &x_cur,
      std::vector<int> &y,
      std::vector<double> &chol_Sigma_V,
      std::vector<double> &B,
      std::vector<double> &theta, std::vector<double> &zeta,
      const std::vector<std::vector<Node*>> &forest_theta,
      const std::vector<std::vector<Node*>> &forest_zeta,
      int mc);

  std::vector<double> ESSZANIMLNBART(arma::umat Y, arma::mat X_ini,
                                     int ndpost, int nburnin,
                                     std::vector<double> mean_prior,
                                     arma::mat S_prior,
                                     arma::mat B, int mc);


  // Get BART predictions
  void GetTreesPredictionsZANIMBART(std::vector<double> &x,
                                    std::vector<double> &lambda,
                                    std::vector<double> &zeta,
                                    const std::vector<std::vector<Node*>> &forest_theta,
                                    const std::vector<std::vector<Node*>> &forest_zeta);

  // Run one update of ESS using the Poisson-type likelihood
  std::vector<double> UpdateESSZANIMLNBART2(
      std::vector<double> &x_cur,
      // std::vector<double> &mean_prior,
      // std::vector<double> &chol_S_prior,
      std::vector<int> &y,
      std::vector<double> &z, std::vector<double> &u,
      double &phi,
      std::vector<double> &lambda, std::vector<double> &zeta,
      const std::vector<std::vector<Node*>> &forest_theta,
      const std::vector<std::vector<Node*>> &forest_zeta);

  // Run ESS
  std::vector<double> ESSZANIMLNBART2(arma::umat Y, arma::mat X_ini,
                                     int ndpost, int nburnin,
                                     std::vector<double> mean_prior,
                                     arma::mat S_prior,
                                     arma::mat B);

};
#endif
