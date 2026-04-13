#include "inverse_posterior.h"
#include "pmfs.h"
#include "write_read.h"

constexpr double PI_2 = 6.283185307179586231996;

// Constructor
InversePosterior::InversePosterior(int d, int ntrees_theta, int ntrees_zeta,
                                   std::string forward_model,
                                   std::string forests_dir) :
                                   d(d),
                                   ntrees_theta(ntrees_theta),
                                   ntrees_zeta(ntrees_zeta),
                                   forward_model(forward_model),
                                   forests_dir(forests_dir) {
}

// Traverse the tree to get tree-specific the prediction
double InversePosterior::GetMu(Node *tree, const arma::rowvec &x) {
  if (tree->is_leaf) return tree->mu[0];
  if (x[tree->predictor] <= tree->cutoff) return GetMu(tree->left, x);
  else return GetMu(tree->right, x);
}

// Compute the ML-BART prediction for a given x
void InversePosterior::GetPredictionMLBART(arma::rowvec &x,
                                           std::vector<double> &theta,
                                           const std::vector<std::vector<Node*>> &forest_theta) {
  // Iterate over categories
  for (int j = 0; j < d; j++) {
    // Iterate over trees
    for (int h = 0; h < ntrees_theta; h++) {
      // Do the predictions
      theta[j] += GetMu(forest_theta[j][h], x);
    }
  }
  // Normalise the parameters
  double s_theta = 0.0;
  for (int j=0; j < d; j++) {
    theta[j] = exp(theta[j]);
    s_theta += theta[j];
  }
  for (auto &u : theta) u /= s_theta;
}


// Eliptical slice sampling for ML-BART
arma::mat InversePosterior::SamplerMLBARTeSS(std::vector<int> &y,
                                             arma::rowvec x_cur,
                                             int ndpost,
                                             arma::rowvec mean_prior,
                                             arma::mat S_prior,
                                             int n_rep) {

  int p = x_cur.n_cols;

  // Total count
  int total = std::accumulate(y.begin(), y.end(), 0);

  int np_theta = 1;
  // Open files to read the the forests
  std::vector<std::ifstream> files_theta;
  for (int j=0; j < d; j++) {
    std::string ff1 = forests_dir + "/forests_" + std::to_string(j) + ".bin";
    files_theta.emplace_back(ff1, std::ios::binary);
  }

  // Vector to keep the posterior draws
  arma::mat X_posterior = arma::zeros<arma::mat>(ndpost, p);

  // Define objects use inside the loop
  arma::rowvec nu = arma::zeros<arma::rowvec>(p);
  arma::rowvec x_star = arma::zeros<arma::rowvec>(p);
  arma::mat chol_S = arma::chol(S_prior);
  std::vector<double> theta(d, 0.0);
  double u_s, nu_angle, nu_max, nu_min;

  // Iterate over the MCMC samples
  double progress = 0.0;
  for (int t = 0; t < ndpost; t++) {

    progress = (double) 100 * t / ndpost;
    Rprintf("%3.2f%% Warm-up completed", progress);
    Rprintf("\r");

    // Load all forests in memory (safer)
    std::vector<std::vector<Node*>> forest_theta(d);
    for (int j = 0; j < d; j++) {
      for (int h = 0; h < ntrees_theta; h++) {
        forest_theta[j].push_back(deserialise_tree(files_theta[j], np_theta));
      }
    }

    // Run for "n_rep" iterations per posterior sample to guarantee convergence
    for (int k=0; k < n_rep; k++) {
      // Draw from the prior
      nu = mean_prior + arma::randn<arma::rowvec>(p) * chol_S;
      // Get the predictions for theta and zeta given the x_cur
      std::fill(theta.begin(), theta.end(), 0.0);
      GetPredictionMLBART(x_cur, theta, forest_theta);
      // Set a log-likelihood threshold
      u_s = log(R::unif_rand());
      u_s += log_pmf_mult(y, total, theta);
      // Draw an angle and the proposal
      nu_angle = R::unif_rand() * PI_2;
      nu_max = nu_angle;
      nu_min = nu_angle - PI_2;
      x_star = x_cur * cos(nu_angle) + nu * sin(nu_angle);
      // Start slice sampling
      do {
        std::fill(theta.begin(), theta.end(), 0.0);
        // Get the predictions for theta  given the x_star
        GetPredictionMLBART(x_star, theta, forest_theta);
        // double ll = log_pmf_mult(y, total, theta);
        // std::cout << counter << " " << ll << " " << u_s << " nu_angle " << nu_angle << " nu_min: " << nu_min << " nu_max: " << nu_max << "\n";
        if (log_pmf_mult(y, total, theta) > u_s) break;
        if (nu_angle < 0) nu_min = nu_angle;
        else nu_max = nu_angle;
        // Update the angle and the proposal
        nu_angle = nu_min + (nu_max - nu_min) * R::unif_rand();
        x_star = x_cur * cos(nu_angle) + nu * sin(nu_angle);
      } while (true);
      // Update x_cur
      x_cur = x_star;
    }
    // Save the posterior draw
    X_posterior.row(t) = x_cur; // + mean_prior;

    // Remove the trees (to free the memory usage)
    for (int j = 0; j < d; ++j) {
      for (auto *tree : forest_theta[j]) delete tree;
    }

  }
  // Close the files
  for (int j=0; j<d; j++) files_theta[j].close();


  return X_posterior;
}


// Exposing a C++ class in R
RCPP_MODULE(inverse_posterior) {

  // Expose class InversePosterior as "InversePosterior" on the R side
  Rcpp::class_<InversePosterior>("InversePosterior")

  // Constructor
  .constructor<int, int, int, std::string, std::string>()

  // Methods
  .method("SamplerMLBARTeSS", &InversePosterior::SamplerMLBARTeSS)

  ;

}
