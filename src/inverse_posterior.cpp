#include "inverse_posterior.h"
#include "pmfs.h"
#include "utils.h"
#include "write_read.h"

constexpr double PI_2 = 6.283185307179586231996;

// Constructor
InversePosterior::InversePosterior(int d, int ntrees_theta, int ntrees_zeta,
                                   int n_particles,
                                   std::string forward_model,
                                   std::string forests_dir) :
                                   d(d),
                                   ntrees_theta(ntrees_theta),
                                   ntrees_zeta(ntrees_zeta),
                                   n_particles(n_particles),
                                   forward_model(forward_model),
                                   forests_dir(forests_dir) {
}

// Traverse the tree to get tree-specific the prediction
double InversePosterior::GetMu(Node *tree, std::vector<double> &x) {
  if (tree->is_leaf) return tree->mu[0];
  if (x[tree->predictor] <= tree->cutoff) return GetMu(tree->left, x);
  else return GetMu(tree->right, x);
}

// Compute the ML-BART prediction for a given x
void InversePosterior::GetPredictionMLBART(std::vector<double> &x,
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

void InversePosterior::GetPredictionZANIMBART(std::vector<double> &x,
                                              std::vector<double> &theta,
                                              std::vector<double> &zeta,
                                              const std::vector<std::vector<Node*>> &forest_theta,
                                              const std::vector<std::vector<Node*>> &forest_zeta) {

  std::vector<double> f0(d, 0.0);
  // Iterate over categories
  for (int j = 0; j < d; j++) {
    // Iterate over trees
    for (int h = 0; h < ntrees_theta; h++) {
      // Do the predictions
      theta[j] += GetMu(forest_theta[j][h], x);
    }
    for (int h = 0; h < ntrees_zeta; h++) {
      // Do the predictions
      zeta[j] += GetMu(forest_zeta[j][h], x);
    }
  }
  // Map the regression trees predictions to the model parameters
  double s_theta = 0.0;
  for (int j=0; j < d; j++) {
    theta[j] = exp(theta[j]);
    s_theta += theta[j];
    zeta[j] = R::pnorm5(zeta[j], 0.0, 1.0, 1.0, 0.0);
  }
  for (auto &u : theta) u /= s_theta;
}

// Elliptical slice sampling for ML-BART
std::vector<double> InversePosterior::SamplerMLBARTeSS(arma::umat Y,
                                                       arma::mat X_ini,
                                                       int ndpost,
                                                       std::vector<double> mean_prior,
                                                       arma::mat S_prior,
                                                       int n_rep) {

  int p = X_ini.n_cols;
  int n_samples = Y.n_rows;
  int np_theta = 1;

  // Transform data into row-major vectors
  std::vector<int> Yrm = umat_to_int_rowmajor(Y);
  std::vector<double> Xrm = mat_to_double_rowmajor(X_ini);

  // Open files to read the forests
  std::vector<std::ifstream> files_theta;
  for (int j=0; j < d; j++) {
    std::string ff1 = forests_dir + "/forests_" + std::to_string(j) + ".bin";
    files_theta.emplace_back(ff1, std::ios::binary);
  }

  // Vector to keep the posterior draws
  std::vector<double> x_posterior(ndpost*p*n_samples, 0.0);
  //arma::mat X_posterior = arma::zeros<arma::mat>(ndpost, p);

  // Define objects use inside the loop
  // arma::rowvec nu = arma::zeros<arma::rowvec>(p);
  // arma::rowvec x_star = arma::zeros<arma::rowvec>(p);
  std::vector<double> nu(p, 0.0);
  std::vector<double> x_cur(p, 0.0);
  std::vector<double> x_star(p, 0.0);
  std::vector<double> theta(d, 0.0);
  double u_s, nu_angle, nu_max, nu_min;

  // Compute the Cholesky and transform it to row-major
  arma::mat chol_S = arma::chol(S_prior);
  std::vector<double> chol_Srm = mat_to_double_rowmajor(chol_S);
  std::cout << chol_Srm[0] << "\n";

  // Create vector to allocate the counts for a given i
  std::vector<int> y(d, 0);
  int ntrial = 0;
  double progress = 0.0;


  // Load all forests in memory
  // std::cout << "Loading forests...\n";
  // std::vector<std::vector<std::vector<Node*>>> forest_theta(ndpost);
  // for (int t = 0; t < ndpost; ++t) {
  //   forest_theta[t].resize(d);
  //   for (int j = 0; j < d; ++j) {
  //     forest_theta[t][j].reserve(ntrees_theta);
  //     for (int h = 0; h < ntrees_theta; ++h) {
  //       Node* tree = deserialise_tree(files_theta[j], np_theta);
  //       forest_theta[t][j].push_back(tree);
  //       // delete tree;
  //     }
  //   }
  // }
  // std::cout << "All forests are loaded, star MCMC...\n";

  // Iterate over the observations samples
  for (int i=0; i < n_samples; i++) {

    int base_i = i * ndpost * p;

    // Copy Y_i and compute the total
    ntrial = 0;
    for (int j = 0; j < d; j++) {
      y[j] = Yrm[i * d + j];
      ntrial += y[j];
    }
    // Get the initial value for X
    for (int k = 0; k < p; k++) x_cur[k] = Xrm[i * p + k];

    // Iterate over the MCMC samples
    progress = 0.0;
    for (int t = 0; t < ndpost; t++) {

      progress = (double) 100 * t / ndpost;
      Rprintf("%3.2f%% Sampling completed for observation %i of %i", progress, i+1, n_samples);
      Rprintf("\r");

      // Load all category-specific forests for the current MCMC iteration in memory
      std::vector<std::vector<Node*>> forest_theta(d);
      for (int j = 0; j < d; j++) {
        for (int h = 0; h < ntrees_theta; h++) {
          forest_theta[j].push_back(deserialise_tree(files_theta[j], np_theta));
        }
      }

      // Run for "n_rep" iterations per posterior sample to guarantee convergence
      for (int k=0; k < n_rep; k++) {
        // Draw from the prior
        rmvnorm_chol(nu, mean_prior, chol_Srm, p);
        // nu = mean_prior + arma::randn<arma::rowvec>(p) * chol_S;

        // Get the predictions for theta and zeta given the x_cur
        std::fill(theta.begin(), theta.end(), 0.0);
        GetPredictionMLBART(x_cur, theta, forest_theta);
        // Set a log-likelihood threshold
        u_s = log(R::unif_rand());
        u_s += log_pmf_mult(y, ntrial, theta);
        // Draw an angle and the proposal
        nu_angle = R::unif_rand() * PI_2;
        nu_max = nu_angle;
        nu_min = nu_angle - PI_2;
        // x_star = x_cur * cos(nu_angle) + nu * sin(nu_angle);
        // for (int j = 0; j < p; j++) x_star[j] = cos(nu_angle) * x_cur[j] + sin(nu_angle) * nu[j];
        axpby(x_star.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
        // Start slice sampling
        do {
          std::fill(theta.begin(), theta.end(), 0.0);
          // Get the predictions for theta  given the x_star
          GetPredictionMLBART(x_star, theta, forest_theta);
          // double ll = log_pmf_mult(y, total, theta);
          // std::cout << counter << " " << ll << " " << u_s << " nu_angle " << nu_angle << " nu_min: " << nu_min << " nu_max: " << nu_max << "\n";
          if (log_pmf_mult(y, ntrial, theta) > u_s) break;
          if (nu_angle < 0) nu_min = nu_angle;
          else nu_max = nu_angle;
          // Update the angle and the proposal
          nu_angle = nu_min + (nu_max - nu_min) * R::unif_rand();
          // for (int j = 0; j < p;j++) x_star[j] = cos(nu_angle) * x_cur[j] + sin(nu_angle) * nu[j];
          axpby(x_star.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
          // x_star = x_cur * cos(nu_angle) + nu * sin(nu_angle);
        } while (true);
        // Update x_cur
        x_cur = x_star;
      }
      // Save the posterior draw
      for (int k = 0; k < p; k++) x_posterior[base_i + t * p + k] = x_cur[k];
      // Remove the trees (to free the memory usage)
      for (int j = 0; j < d; ++j)
        for (auto *tree : forest_theta[j]) delete tree;
    }
    // Rewind the forests files (go backward to an earlier point)
    for (int j = 0; j < d; ++j) {
      files_theta[j].clear();
      files_theta[j].seekg(0); // go back to beginning
    }

  }

  // Remove the trees (to free the memory usage)
  // for (int t=0; t<ndpost; t++) {
  //   for (int j = 0; j < d; ++j)
  //     for (auto *tree : forest_theta[t][j]) delete tree;
  // }

  // Close the files
  for (int j=0; j<d; j++) files_theta[j].close();


  return x_posterior;
}

// Elliptical slice sampling for ZANIM-BART
std::vector<double> InversePosterior::SamplerZANIMBARTeSS(arma::umat Y,
                                                          arma::mat X_ini,
                                                          int ndpost,
                                                          std::vector<double> mean_prior,
                                                          arma::mat S_prior,
                                                          int n_rep, int conditional) {

  // Dimension
  int p = X_ini.n_cols, n_samples = Y.n_rows, np = 1;

  // Function pointer for which likelihood function to use, depending on the
  // conditional argument
  std::function<double(const std::vector<int>&, const std::vector<double>&,
                       const std::vector<double>&)> log_pmf;
  if (conditional) {
    log_pmf = [](const std::vector<int>& y,
                 const std::vector<double>& theta,
                 const std::vector<double>& zeta) {
      return log_pmf_zanim_conditional(y, theta, zeta);
    };
  } else {
    log_pmf = [](const std::vector<int>& y,
                 const std::vector<double>& theta,
                 const std::vector<double>& zeta) {
      return log_pmf_zanim(y, theta, zeta);
    };
  }

  // Transform data into row-major vectors
  std::vector<int> Yrm = umat_to_int_rowmajor(Y);
  std::vector<double> Xrm = mat_to_double_rowmajor(X_ini);

  // Open files to read the forests
  std::vector<std::ifstream> files_theta, files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff1 = forests_dir + "/forests_theta_" + std::to_string(j) + ".bin";
    std::string ff2 = forests_dir + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_theta.emplace_back(ff1, std::ios::binary);
    files_zeta.emplace_back(ff2, std::ios::binary);
  }

  // Vector to keep the posterior draws
  std::vector<double> x_posterior(ndpost*p*n_samples, 0.0);

  // Define objects use inside the loop
  std::vector<double> nu(p, 0.0), x_cur(p, 0.0), x_star(p, 0.0), theta(d, 0.0), zeta(d, 0.0);
  double u_s, nu_angle, nu_max, nu_min;

  // Compute the Cholesky and transform it to row-major
  arma::mat chol_S = arma::chol(S_prior);
  std::vector<double> chol_Srm = mat_to_double_rowmajor(chol_S);

  // Create vector to allocate the counts for a given i
  std::vector<int> y(d, 0);
  int ntrial = 0;
  double progress = 0.0;

  // Iterate over the observations samples
  for (int i=0; i < n_samples; i++) {

    int base_i = i * ndpost * p;

    // Get current values of Y_i and compute the N_i=\sum_j(y_{ij})
    ntrial = 0;
    for (int j = 0; j < d; j++) {
      y[j] = Yrm[i * d + j];
      ntrial += y[j];
    }
    // Get the initial value for X
    for (int k = 0; k < p; k++) x_cur[k] = Xrm[i * p + k];

    // Iterate over the MCMC samples
    progress = 0.0;
    for (int t = 0; t < ndpost; t++) {

      progress = (double) 100 * t / ndpost;
      Rprintf("%3.2f%% Sampling completed for observation %i of %i", progress, i+1, n_samples);
      Rprintf("\r");

      // Load all category-specific forests for the current MCMC iteration in memory
      std::vector<std::vector<Node*>> forest_theta(d);
      std::vector<std::vector<Node*>> forest_zeta(d);
      for (int j = 0; j < d; j++) {
        for (int h = 0; h < ntrees_theta; h++) {
          forest_theta[j].push_back(deserialise_tree(files_theta[j], np));
        }
        for (int h = 0; h < ntrees_zeta; h++) {
          forest_zeta[j].push_back(deserialise_tree(files_zeta[j], np));
        }
      }

      // Run for "n_rep" iterations per posterior sample to guarantee convergence
      for (int k=0; k < n_rep; k++) {
        // Draw from the prior
        rmvnorm_chol(nu, mean_prior, chol_Srm, p);
        // Get the predictions for theta and zeta given the x_cur
        std::fill(theta.begin(), theta.end(), 0.0);
        std::fill(zeta.begin(), zeta.end(), 0.0);
        GetPredictionZANIMBART(x_cur, theta, zeta, forest_theta, forest_zeta);
        // Set a log-likelihood threshold
        u_s = log(R::unif_rand());
        // double ll_cur = log_pmf(y, theta, zeta);
        u_s += log_pmf(y, theta, zeta);
        // Draw an angle and the proposal
        nu_angle = R::unif_rand() * PI_2;
        nu_max = nu_angle;
        nu_min = nu_angle - PI_2;
        axpby(x_star.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
        // Start slice sampling
        do {
          std::fill(theta.begin(), theta.end(), 0.0);
          std::fill(zeta.begin(), zeta.end(), 0.0);
          // Get the predictions for theta  given the x_star
          GetPredictionZANIMBART(x_star, theta, zeta, forest_theta, forest_zeta);
          // double ll = log_pmf(y, theta, zeta);
          //std::cout << "PROPOSAL and CURRENT log-likelihood " << ll << " " << ll_cur << " " << u_s << "\n";
          if (log_pmf(y, theta, zeta) > u_s) break;
          if (nu_angle < 0) nu_min = nu_angle;
          else nu_max = nu_angle;
          // Update the angle and the proposal
          nu_angle = nu_min + (nu_max - nu_min) * R::unif_rand();
          axpby(x_star.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
        } while (true);
        // Update x_cur
        x_cur = x_star;
      }
      // Save the posterior draw
      for (int k = 0; k < p; k++) x_posterior[base_i + t * p + k] = x_cur[k];
      // Delete the trees (to free the memory usage)
      for (int j = 0; j < d; ++j){
        for (auto *tree : forest_theta[j]) delete tree;
        for (auto *tree : forest_zeta[j]) delete tree;
      }
    }
    // Rewind the forests files (go backward to an earlier point)
    for (int j = 0; j < d; ++j) {
      files_theta[j].clear(); files_theta[j].seekg(0);
      files_zeta[j].clear(); files_zeta[j].seekg(0);
    }

  }

  // Remove the trees (to free the memory usage)
  // for (int t=0; t<ndpost; t++) {
  //   for (int j = 0; j < d; ++j)
  //     for (auto *tree : forest_theta[t][j]) delete tree;
  // }

  // Close the files
  for (int j=0; j<d; j++) files_theta[j].close();


  return x_posterior;
}

// Elliptical slice sampling for ZANIM-LN-BART
std::vector<double> InversePosterior::SamplerZANIMLNBARTeSS(arma::umat Y,
                                                            arma::mat X_ini,
                                                            int ndpost,
                                                            std::vector<double> mean_prior,
                                                            arma::mat S_prior,
                                                            int n_rep,
                                                            arma::mat B) {

  // Dimension
  int p = X_ini.n_cols, n_samples = Y.n_rows, np = 1, dm1 = d - 1;

  // Transform data into row-major vectors
  std::vector<int> Yrm = umat_to_int_rowmajor(Y);
  std::vector<double> Xrm = mat_to_double_rowmajor(X_ini);

  // Open files to read the forests
  std::vector<std::ifstream> files_theta, files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff1 = forests_dir + "/forests_theta_" + std::to_string(j) + ".bin";
    std::string ff2 = forests_dir + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_theta.emplace_back(ff1, std::ios::binary);
    files_zeta.emplace_back(ff2, std::ios::binary);
  }
  // Open file for the the posterior draws of chol(Sigma_V)
  std::ifstream ff_Sigma_V(forests_dir + "chol_Sigma_V.bin", std::ios::binary);
  // Create placeholder vector for dynamic read the posterior draws
  std::vector<double> chol_Sigma_V(dm1*dm1, 0.0);
  // std::vector<double> chol_Sigma_V(dm1*dm1*ndpost);
  // ff_Sigma_V.read(reinterpret_cast<char*>(chol_Sigma_V.data()), sizeof(double) * dm1 * dm1 * ndpost);


  // Transform the B (sum-to-zero) vector into row-major
  std::vector<double> Brm = mat_to_double_rowmajor(B);

  // Vector to keep the posterior draws
  std::vector<double> x_posterior(ndpost*p*n_samples, 0.0);

  // Define objects use inside the loop
  std::vector<double> nu(p, 0.0), x_cur(p, 0.0), x_star(p, 0.0), theta(d, 0.0), zeta(d, 0.0);
  double u_s, nu_angle, nu_max, nu_min;

  // Compute the Cholesky and transform it to row-major
  arma::mat chol_S = arma::chol(S_prior);
  std::vector<double> chol_Srm = mat_to_double_rowmajor(chol_S);

  // Create vector to allocate the counts for a given sample unit i
  std::vector<int> y(d, 0);
  int ntrial = 0;
  double progress = 0.0;

  // Iterate over the observations samples
  for (int i=0; i < n_samples; i++) {

    int base_i = i * ndpost * p;

    // Get current values of Y_i and compute the N_i=\sum_j(y_{ij})
    ntrial = 0;
    for (int j = 0; j < d; j++) {
      y[j] = Yrm[i * d + j];
      ntrial += y[j];
    }
    // Get the initial value for X
    for (int k = 0; k < p; k++) x_cur[k] = Xrm[i * p + k];

    // Iterate over the MCMC samples
    progress = 0.0;
    for (int t = 0; t < ndpost; t++) {

      progress = (double) 100 * t / ndpost;
      Rprintf("%3.2f%% Sampling completed for observation %i of %i", progress, i+1, n_samples);
      Rprintf("\r");

      // Load all category-specific forests for the current MCMC iteration in memory
      std::vector<std::vector<Node*>> forest_theta(d);
      std::vector<std::vector<Node*>> forest_zeta(d);
      for (int j = 0; j < d; j++) {
        for (int h = 0; h < ntrees_theta; h++) {
          forest_theta[j].push_back(deserialise_tree(files_theta[j], np));
        }
        for (int h = 0; h < ntrees_zeta; h++) {
          forest_zeta[j].push_back(deserialise_tree(files_zeta[j], np));
        }
      }
      // Load current posterior draw of chol_Sigma_V
      ff_Sigma_V.read(reinterpret_cast<char*>(chol_Sigma_V.data()),
                      sizeof(double) * dm1 * dm1);

      // Run for "n_rep" iterations per posterior sample to guarantee convergence
      for (int k=0; k < n_rep; k++) {
        // Draw from the prior
        rmvnorm_chol(nu, mean_prior, chol_Srm, p);
        // Get the predictions for theta and zeta given the x_cur
        std::fill(theta.begin(), theta.end(), 0.0);
        std::fill(zeta.begin(), zeta.end(), 0.0);
        GetPredictionZANIMBART(x_cur, theta, zeta, forest_theta, forest_zeta);
        // Set a log-likelihood threshold
        u_s = log(R::unif_rand());
        // double ll_cur = log_pmf(y, theta, zeta);
        u_s += log_pmf_zanim_ln_conditional(y, theta, zeta, chol_Sigma_V, Brm);
        // Draw an angle and the proposal
        nu_angle = R::unif_rand() * PI_2;
        nu_max = nu_angle;
        nu_min = nu_angle - PI_2;
        axpby(x_star.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
        // Start slice sampling
        do {
          std::fill(theta.begin(), theta.end(), 0.0);
          std::fill(zeta.begin(), zeta.end(), 0.0);
          // Get the predictions for theta  given the x_star
          GetPredictionZANIMBART(x_star, theta, zeta, forest_theta, forest_zeta);
          // double ll = log_pmf(y, theta, zeta);
          //std::cout << "PROPOSAL and CURRENT log-likelihood " << ll << " " << ll_cur << " " << u_s << "\n";
          if (log_pmf_zanim_ln_conditional(y, theta, zeta, chol_Sigma_V, Brm) > u_s) break;
          if (nu_angle < 0) nu_min = nu_angle;
          else nu_max = nu_angle;
          // Update the angle and the proposal
          nu_angle = nu_min + (nu_max - nu_min) * R::unif_rand();
          axpby(x_star.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
        } while (true);
        // Update x_cur
        x_cur = x_star;
      }
      // Save the posterior draw
      for (int k = 0; k < p; k++) x_posterior[base_i + t * p + k] = x_cur[k];
      // Delete the trees (to free the memory usage)
      for (int j = 0; j < d; ++j){
        for (auto *tree : forest_theta[j]) delete tree;
        for (auto *tree : forest_zeta[j]) delete tree;
      }
    }
    // Rewind the forests files (go backward to an earlier point)
    for (int j = 0; j < d; ++j) {
      files_theta[j].clear(); files_theta[j].seekg(0);
      files_zeta[j].clear(); files_zeta[j].seekg(0);
    }
    ff_Sigma_V.clear(); ff_Sigma_V.seekg(0);

  }
  // Close the files
  for (int j=0; j<d; j++) files_theta[j].close();
  ff_Sigma_V.close();

  return x_posterior;
}

// Compute the Log-likelihood of ZANIM-BART given y and x,
double InversePosterior::lmlZANIM(std::vector<int> &y, std::vector<double> &x) {

  int np = 1;

  // Open files to read the forests
  std::vector<std::ifstream> files_theta, files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff1 = forests_dir + "/forests_theta_" + std::to_string(j) + ".bin";
    std::string ff2 = forests_dir + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_theta.emplace_back(ff1, std::ios::binary);
    files_zeta.emplace_back(ff2, std::ios::binary);
  }

  std::vector<std::vector<Node*>> forest_theta(d);
  std::vector<std::vector<Node*>> forest_zeta(d);
  for (int j = 0; j < d; j++) {
    forest_theta[j].reserve(ntrees_theta);
    forest_zeta[j].reserve(ntrees_zeta);
  }

  std::vector<double> lml(n_particles, 0.0), theta(d, 0.0), zeta(d, 0.0);

  double progress = 0.0;
  // Iterate over the MCMC samples
  for (int k = 0; k < n_particles; k++) {
    // Load the forests (f^{(c)}_j and f^{(0)}_j) for the given particle in memory
    for (int j = 0; j < d; j++) {
      forest_theta[j].clear();
      forest_zeta[j].clear();
      for (int h = 0; h < ntrees_theta; h++) {
        forest_theta[j].push_back(deserialise_tree(files_theta[j], np));
      }
      for (int h = 0; h < ntrees_zeta; h++) {
        forest_zeta[j].push_back(deserialise_tree(files_zeta[j], np));
      }
    }
    // Compute the regression trees predictions for the
    std::fill(theta.begin(), theta.end(), 0.0);
    std::fill(zeta.begin(), zeta.end(), 0.0);
    GetPredictionZANIMBART(x, theta, zeta, forest_theta, forest_zeta);

    // Compute the likelihood
    lml[k] = log_pmf_zanim_conditional(y, theta, zeta);

    // progress = (double) 100 * k / n_particles;
    // Rprintf("%3.2f%% Computation completed with %f", progress, lml[k]);
    // Rprintf("\r");

    // Remove the trees (to free the memory usage)
    for (int j = 0; j < d; ++j){
      for (auto *tree : forest_theta[j]) delete tree;
      for (auto *tree : forest_zeta[j]) delete tree;
    }
  }
  for (int j=0; j<d; j++) {
    files_theta[j].close(); files_zeta[j].close();
  }

  return log_sum_exp(lml) - log(n_particles);
}


// TODO: Finish....
void InversePosterior::ComputeWeightsIS(arma::umat Y, int n_proposal, arma::mat B,
                                        std::string draws_dir) {

  // Dimension
  int n_samples = Y.n_rows, d = Y.n_cols, dm1 = d - 1;

  // Transform data into row-major vectors
  std::vector<int> Yrm = umat_to_int_rowmajor(Y);
  std::vector<double> Brm = mat_to_double_rowmajor(B);

  // Open file for the the posterior draws of chol(Sigma_V)
  std::ifstream ff_Sigma_V(forests_dir + "chol_Sigma_V.bin", std::ios::binary);
  std::ifstream ff_theta(draws_dir + "theta_ij.bin", std::ios::binary);
  std::ifstream ff_zeta(draws_dir + "zeta_ij.bin", std::ios::binary);

  // Create vector to read the posterior draws of chol_Sigma_V
  std::vector<double> chol_Sigma_V(dm1*dm1*n_proposal);
  // Load all posterior draws of chol_Sigma_V
  ff_Sigma_V.read(reinterpret_cast<char*>(chol_Sigma_V.data()),
                  sizeof(double) * dm1 * dm1 * n_proposal);


  // Load all posterior draws of theta_ij and zeta_ij as well
  std::vector<double> theta(n_samples*d*n_proposal);
  std::vector<double> zeta(n_samples*d*n_proposal);
  ff_theta.read(reinterpret_cast<char*>(theta.data()),
                  sizeof(double) * n_samples * d * n_proposal);
  ff_zeta.read(reinterpret_cast<char*>(zeta.data()),
               sizeof(double) * n_samples * d * n_proposal);

  std::vector<double> theta_cur(d, 0.0);
  std::vector<double> zeta_cur(d, 0.0);



  // Create vector to allocate the counts for a given sample unit i
  std::vector<int> y(d, 0);
  int ntrial = 0;
  double progress = 0.0;

  // Iterate over the observations samples
  for (int i=0; i < n_samples; i++) {

    //int base_i = i * ndpost * p;

    // Get current values of Y_i and compute the N_i=\sum_j(y_{ij})
    ntrial = 0;
    for (int j = 0; j < d; j++) {
      y[j] = Yrm[i * d + j];
      ntrial += y[j];
    }

    // For each posterior draws compute the log-likelihood
    for (int k=0; k < n_proposal; k++) {

      // Read the current theta^{(k)}_ij for given posterior draw k
      for (int j=0; j < d; j++) {


      }


    }

  }





}

// Exposing a C++ class in R
RCPP_MODULE(inverse_posterior) {

  // Expose class InversePosterior as "InversePosterior" on the R side
  Rcpp::class_<InversePosterior>("InversePosterior")

  // Constructor
  .constructor<int, int, int, int, std::string, std::string>()

  // Methods
  .method("SamplerMLBARTeSS", &InversePosterior::SamplerMLBARTeSS)
  .method("SamplerZANIMBARTeSS", &InversePosterior::SamplerZANIMBARTeSS)
  .method("SamplerZANIMLNBARTeSS", &InversePosterior::SamplerZANIMLNBARTeSS)
  .method("lmlZANIM", &InversePosterior::lmlZANIM)

  ;

}
