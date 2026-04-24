#ifndef INVERSEPOSTERIOR
#define INVERSEPOSTERIOR

#include "node.h"

class InversePosterior {

public:
  // Constructor
  InversePosterior(int d, int ntrees_theta, int ntrees_zeta,
                   std::string forward_model, std::string forests_dir);

  int d, ntrees_theta, ntrees_zeta;

  // Path with the posterior draws of the latent field (BART)
  std::string forests_dir;
  // Which forward model to use
  std::string forward_model;

  // Inverse posterior sampler
  std::vector<double> SamplerMLBARTeSS(arma::umat Y, arma::mat X_ini, int ndpost,
                                       std::vector<double> mean_prior, arma::mat S_prior,
                                       int n_rep);

  // void SamplerISMLBART();
  //  void SamplerMHMLBART();

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




};
#endif
