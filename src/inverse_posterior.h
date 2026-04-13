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
  arma::mat SamplerMLBARTeSS(std::vector<int> &y, arma::rowvec x_cur,
                             int ndpost, arma::rowvec mean_prior, arma::mat S_prior,
                             int n_rep);
  arma::mat SamplerMLBARTuiMH(std::vector<int> &y, arma::rowvec x_cur,
                             int ndpost,
                             arma::rowvec mean_prior, arma::mat S_prior,
                             int n_rep);

  // void SamplerISMLBART();
  //  void SamplerMHMLBART();

  // Get the tree-specific prediction by traversing the tree
  double GetMu(Node *tree, const arma::rowvec &x);

  // Get the ML-BART predictions for a given x and return in theta
  void GetPredictionMLBART(arma::rowvec &x, std::vector<double> &theta,
                           const std::vector<std::vector<Node*>> &forest_theta);




};
#endif
