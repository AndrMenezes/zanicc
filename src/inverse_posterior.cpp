#include "inverse_posterior.h"
#include "pmfs.h"
#include "utils.h"
#include "rng.h"
#include "write_read.h"

constexpr double PI_2 = 6.283185307179586231996;

// Constructor
InversePosterior::InversePosterior(int d, int ntrees_theta, int ntrees_zeta,
                                   std::string forests_dir) :
                                   d(d),
                                   ntrees_theta(ntrees_theta),
                                   ntrees_zeta(ntrees_zeta),
                                   forests_dir(forests_dir) {


  dm1 = d-1;
}

// Traverse the tree to get tree-specific the prediction
double InversePosterior::GetMu(Node *tree, std::vector<double> &x) {
  if (tree->is_leaf) return tree->mu[0];
  if (x[tree->predictor] <= tree->cutoff) return GetMu(tree->left, x);
  else return GetMu(tree->right, x);
}

// Compute the BART predictions
void InversePosterior::GetBARTPredictions(std::vector<double> &x,
                                          std::vector<double> &theta,
                                          std::vector<double> &zeta,
                                          const std::vector<std::vector<Node*>> &forest_theta,
                                          const std::vector<std::vector<Node*>> &forest_zeta) {

  std::fill(theta.begin(), theta.end(), 0.0);
  std::fill(zeta.begin(), zeta.end(), 0.0);

  double s_theta = 0.0;
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
    zeta[j] = R::pnorm5(zeta[j], 0.0, 1.0, 1.0, 0.0);
    theta[j] = exp(theta[j]);
    s_theta += theta[j];
  }
  // Normalise the theta's
  for (auto &u : theta) u /= s_theta;
}

// Compute effective sample size for importance sampling based methods
double InversePosterior::EffectiveSampleSize(std::vector<double> &probs) {
  double s = 0.0;
  for (size_t j=0; j < probs.size(); j++) s += probs[j]*probs[j];
  return 1.0 / s;
}

// Run multiple imputation with SIR approach to sample the inverse posterior
void InversePosterior::SIR(arma::umat Y, int n_proposal, int ndpost,
                           arma::mat B, std::string dir_posterior_fx) {

  Rcpp::RNGScope scope;
  // Dimension
  int n_samples = Y.n_rows, d = Y.n_cols, dm1 = d - 1;

  // Transform data into row-major vectors
  std::vector<double> Brm = mat_to_double_rowmajor(B);

  // Open file for the the posterior draws of chol(Sigma_V)
  std::ifstream ff_Sigma_V(forests_dir + "/chol_Sigma_V.bin", std::ios::binary);
  std::ifstream ff_theta(dir_posterior_fx + "/theta_ij.bin", std::ios::binary);
  std::ifstream ff_zeta(dir_posterior_fx + "/zeta_ij.bin", std::ios::binary);

  // To read the posterior draws of chol_Sigma_V
  std::vector<double> chol_Sigma_V(dm1*dm1, 0.0);
  // To read one posterior draw of theta_ij and zeta_ij
  std::vector<double> theta(n_proposal*d, 0.0), zeta(n_proposal*d, 0.0);
  // To copy the values for a given observation i
  std::vector<double> theta_cur(d, 0.0), zeta_cur(d, 0.0);
  // To copy the counts for a given sample unit i
  std::vector<int> y(d, 0);

  // The predictions are stored by posterior draws, so
  // [1 block][2 block]...[ndpost block], for k = ndpost
  // Each k block is (n_proposal × d) in column-major.
  std::vector<double> log_w(n_proposal, 0.0), probs(n_proposal, 0.0);
  // std::vector<int> sir_indices(ndpost*n_samples, 0);

  ess_sir.resize(ndpost*n_samples, 0.0);
  std::fill(ess_sir.begin(), ess_sir.end(), 0.0);
  indices_sir.resize(ndpost*n_samples, 0);
  std::fill(indices_sir.begin(), indices_sir.end(), 0);

  double progress = 0.0;

  // Iterate over posterior draws
  for (int k=0; k < ndpost; k++) {
    progress = (double) 100 * k / ndpost;
    Rprintf("%3.2f%% completed", progress);
    Rprintf("\r");
    // Read one block (n_proposal*d)
    ff_theta.read(reinterpret_cast<char*>(theta.data()),
                  sizeof(double) * n_proposal * d);
    ff_zeta.read(reinterpret_cast<char*>(zeta.data()),
                 sizeof(double) * n_proposal * d);
    // Read current posterior draw of chol(Sigma_V)
    ff_Sigma_V.read(reinterpret_cast<char*>(chol_Sigma_V.data()),
                    sizeof(double) * dm1 * dm1);
    // Iterate over samples
    for (int i=0; i < n_samples; i++) {
      // Copy current observation into y_cur
      for (int j=0; j < d; j++) y[j] = Y(i, j);
      // Iterate over proposal values and compute the log-likelihood contribution
      for (int l=0; l < n_proposal; l++) {
        // Copy the current theta_ij
        for (int j=0; j < d; j++) {
          theta_cur[j] = theta[j*n_proposal + l];
          zeta_cur[j] = zeta[j*n_proposal + l];
        }
        // Compute the log-likelihood
        log_w[l] = log_pmf_zanim_ln_conditional(y, theta_cur, zeta_cur, chol_Sigma_V, Brm);
      }
      // Normalise weights and resample
      probs = normalise_weights(log_w, n_proposal);
      ess_sir[i*ndpost + k] = EffectiveSampleSize(probs);
      indices_sir[i*ndpost + k] = sample_discrete(probs, n_proposal);
    }
  }
  // Close files
  ff_zeta.close();  ff_theta.close(); ff_Sigma_V.close();
}



// Run one update of ESS for ZANIM-LN-BART
std::vector<double> InversePosterior::UpdateESS(
    std::vector<double> &x_cur,
    std::vector<int> &y,
    std::vector<double> &chol_Sigma_V,
    std::vector<double> &B,
    std::vector<double> &theta, std::vector<double> &zeta,
    const std::vector<std::vector<Node*>> &forest_theta,
    const std::vector<std::vector<Node*>> &forest_zeta,
    int n_particles) {

  // Define objects
  std::vector<double> nu(p, 0.0), x_proposal(p, 0.0), x_tilde(p, 0.0);
  double lr, nu_angle, nu_max, nu_min;

  // Log-likelihood threshold
  // double ll_cur = log_pmf_zanim_ln(n_particles, y, theta, zeta, chol_Sigma_V, B);
  lr = log(R::unif_rand()) + log_pmf_zanim_ln(n_particles, y, theta, zeta, chol_Sigma_V, B);

  // Draw the angle
  rmvnorm_chol2(nu, chol_S_prior, p);

  nu_angle = R::unif_rand() * PI_2;
  nu_max = nu_angle;
  nu_min = nu_angle - PI_2;
  // Draw an proposal
  axpby(x_proposal.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
  // Correct for the prior mean
  for (int k=0; k < p; k++) x_tilde[k] = x_proposal[k] + mu_prior[k];
  // Compute the forests predictions for initial proposal
  GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
  int counter = 0;
  // Start slice
  do {
    // double ll_prop = log_pmf_zanim_ln(n_particles, y, theta, zeta, chol_Sigma_V, B);
    // std::cout << " ll_cur=" << ll_cur << " ll_prop=" << ll_prop << "\n";
    // std::cout << " theta1=" << theta[0] << " theta2=" << theta[1] << " theta3=" << theta[2]<< " theta4=" << theta[3]
    //           << " zeta1=" << zeta[0] << " zeta2=" << zeta[1] << " zeta3=" << zeta[2]<< " zeta4=" << zeta[3] <<"\n";
    if (log_pmf_zanim_ln(n_particles, y, theta, zeta, chol_Sigma_V, B) > lr) break;
    if (counter > 1000) break;
    // Update the angle
    if (nu_angle < 0) nu_min = nu_angle;
    else nu_max = nu_angle;
    nu_angle = nu_min + (nu_max - nu_min) * R::unif_rand();
    // Draw new proposal
    axpby(x_proposal.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
    // Correct for the prior mean
    for (int k=0; k < p; k++) x_tilde[k] = x_proposal[k] + mu_prior[k];
    // Compute BART predictions for the new proposal
    GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
    counter++;
  } while (true);
  return x_proposal;
}

// Elliptical slice sampling for ZANIM-LN-BART
std::vector<double> InversePosterior::ESS(arma::umat Y,
                                          arma::mat X_ini,
                                          int ndpost,
                                          int nburnin, int n_particles,
                                          std::vector<double> mean_prior,
                                          arma::mat S_prior,
                                          arma::mat B) {
  Rcpp::RNGScope scope;
  // To read the trees
  int np = 1;

  // Setting field
  p = X_ini.n_cols;
  n_samples = Y.n_rows;
  // Gaussian prior. Compute the (upper triangle) Cholesky of the prior and
  // transform it to row-major
  arma::mat chol_S = arma::chol(S_prior);
  chol_S_prior = mat_to_double_rowmajor(chol_S);
  mu_prior = mean_prior;

  // Transform data (y, X_ini) and B matrix into row-major vectors
  std::vector<int> Yrm = umat_to_int_rowmajor(Y);
  std::vector<double> Xrm = mat_to_double_rowmajor(X_ini);
  std::vector<double> Brm = mat_to_double_rowmajor(B);

  // Open files to read the forests
  std::vector<std::ifstream> files_theta, files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff1 = forests_dir + "/forests_theta_" + std::to_string(j) + ".bin";
    std::string ff2 = forests_dir + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_theta.emplace_back(ff1, std::ios::binary);
    files_zeta.emplace_back(ff2, std::ios::binary);
  }
  // Open file for the the posterior draws of chol(Sigma_V)
  std::ifstream ff_Sigma_V(forests_dir + "/chol_Sigma_V.bin", std::ios::binary);
  // Create placeholder vector for dynamic read the posterior draws
  std::vector<double> chol_Sigma_V(dm1*dm1, 0.0);

  // Vector to keep the posterior draws
  std::vector<double> draws(ndpost*p*n_samples, 0.0);
  // Vectors for the proposal and the BART predictions
  std::vector<double> x_cur(p, 0.0), x_tilde(p, 0.0), theta(d, 0.0), zeta(d, 0.0);

  // Vector to allocate the counts for a given observation i
  std::vector<int> y(d, 0);
  double progress = 0.0;

  // Iterate over posterior draws of forward model
  for (int t=0; t < ndpost; t++) {
    // std::cout << t << "\n";
    progress = (double) 100 * t / ndpost;
    Rprintf("%3.2f%% Sampling completed", progress);
    Rprintf("\r");

    // Load regression trees parameters
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

    // Loop over the observations
    for (int i=0; i < n_samples; i++) {
      int base_i = i * ndpost * p;

      // Get current values of Y_i and x_i
      for (int j = 0; j < d; j++) y[j] = Y(i, j); //[i * d + j];
      for (int k = 0; k < p; k++) {
        x_cur[k] = Xrm[i * p + k];
        x_tilde[k] = x_cur[k] + mean_prior[k];
      }
      // Compute the forests predictions for given observation.
      // Don't need to re-compute this inside the burn-in loop, because I am
      // passing by reference, so this {theta} and {zeta} are both already updated
      GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
      // Start inverse-sampling using ESS
      for (int k = 0; k < nburnin; k++) {
        x_cur = UpdateESS(x_cur, y, chol_Sigma_V, Brm, theta, zeta, forest_theta,
                          forest_zeta, n_particles);
      }
      // Update the "initial" value of x for the next iteration and save
      for (int k = 0; k < p; k++) {
        Xrm[i * p + k] = x_cur[k];
        draws[base_i + t * p + k] = x_cur[k] + mean_prior[k];
      }
    }
    // Delete the trees (to free the memory usage)
    for (int j = 0; j < d; ++j){
      for (auto *tree : forest_theta[j]) delete tree;
      for (auto *tree : forest_zeta[j]) delete tree;
    }
  }
  return draws;
}

// Run one update of constrained ESS for ZANIM-LN-BART for p>=2
std::vector<double> InversePosterior::UpdateCESS(
    std::vector<double> &x_cur,
    std::vector<int> &y,
    std::vector<double> &chol_Sigma_V,
    std::vector<double> &B,
    std::vector<double> &Amat,
    std::vector<double> &bvec,
    double &eta,
    std::vector<double> &theta, std::vector<double> &zeta,
    const std::vector<std::vector<Node*>> &forest_theta,
    const std::vector<std::vector<Node*>> &forest_zeta,
    int n_particles) {

  // Define objects
  std::vector<double> nu(p, 0.0), x_proposal(p, 0.0), x_tilde(p, 0.0);
  double lr, nu_angle, nu_max, nu_min;

  // Log-likelihood threshold
  // double ll_cur = log_pmf_zanim_ln(n_particles, y, theta, zeta, chol_Sigma_V, B);
  lr = log(R::unif_rand()) + log_pmf_zanim_ln(n_particles, y, theta, zeta, chol_Sigma_V, B);
  lr += log_I_lc(x_cur, mu_prior, Amat, bvec, eta);
  // lr += log_I_lc2(x_cur, Amat, bvec, eta);

  // Draw the angle
  rmvnorm_chol2(nu, chol_S_prior, p);

  nu_angle = R::unif_rand() * PI_2;
  nu_max = nu_angle;
  nu_min = nu_angle - PI_2;
  // Draw an proposal
  axpby(x_proposal.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
  // Correct for the prior mean
  for (int k=0; k < p; k++) x_tilde[k] = x_proposal[k] + mu_prior[k];
  // Compute the forests predictions for initial proposal
  GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
  int counter = 0;
  // Start slice
  do {
    double ll_prop = log_pmf_zanim_ln(n_particles, y, theta, zeta, chol_Sigma_V, B);
    ll_prop += log_I_lc(x_proposal, mu_prior, Amat, bvec, eta);
    // ll_prop += log_I_lc2(x_proposal, Amat, bvec, eta);
    if (ll_prop > lr) break;
    if (counter > 1000) break;
    // Update the angle
    if (nu_angle < 0) nu_min = nu_angle;
    else nu_max = nu_angle;
    nu_angle = nu_min + (nu_max - nu_min) * R::unif_rand();
    // Draw new proposal
    axpby(x_proposal.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
    // Correct for the prior mean
    for (int k=0; k < p; k++) x_tilde[k] = x_proposal[k] + mu_prior[k];
    // Compute BART predictions for the new proposal
    GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
    counter++;
  } while (true);
  return x_proposal;
}

// Constrained elliptical slice sampling for ZANIM-LN-BART for p>=2
std::vector<double> InversePosterior::CESS(arma::umat Y, arma::mat X_ini,
                                           int ndpost,
                                           int nburnin, int n_particles,
                                           std::vector<double> mean_prior,
                                           arma::mat S_prior, arma::mat B,
                                           arma::mat A, std::vector<double> bvec,
                                           double eta) {
  Rcpp::RNGScope scope;
  // To read the trees
  int np = 1;

  // Setting field
  p = X_ini.n_cols;
  n_samples = Y.n_rows;
  // Gaussian prior. Compute the (upper triangle) Cholesky of the prior and
  // transform it to row-major
  arma::mat chol_S = arma::chol(S_prior);
  chol_S_prior = mat_to_double_rowmajor(chol_S);
  mu_prior = mean_prior;

  // Transform data (y, X_ini) and B matrix into row-major vectors
  std::vector<int> Yrm = umat_to_int_rowmajor(Y);
  std::vector<double> Xrm = mat_to_double_rowmajor(X_ini);
  std::vector<double> Brm = mat_to_double_rowmajor(B);
  // Transform matrix A of the convex-hull constrained to row-major vector
  std::vector<double> Arm = mat_to_double_rowmajor(A);

  // Open files to read the forests
  std::vector<std::ifstream> files_theta, files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff1 = forests_dir + "/forests_theta_" + std::to_string(j) + ".bin";
    std::string ff2 = forests_dir + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_theta.emplace_back(ff1, std::ios::binary);
    files_zeta.emplace_back(ff2, std::ios::binary);
  }
  // Open file for the the posterior draws of chol(Sigma_V)
  std::ifstream ff_Sigma_V(forests_dir + "/chol_Sigma_V.bin", std::ios::binary);
  // Create placeholder vector for dynamic read the posterior draws
  std::vector<double> chol_Sigma_V(dm1*dm1, 0.0);

  // Vector to keep the posterior draws
  std::vector<double> draws(ndpost*p*n_samples, 0.0);
  // Vectors for the proposal and the BART predictions
  std::vector<double> x_cur(p, 0.0), x_tilde(p, 0.0), theta(d, 0.0), zeta(d, 0.0);

  // Vector to allocate the counts for a given observation i
  std::vector<int> y(d, 0);
  double progress = 0.0;

  // Iterate over posterior draws of forward model
  for (int t=0; t < ndpost; t++) {
    // std::cout << t << "\n";
    progress = (double) 100 * t / ndpost;
    Rprintf("%3.2f%% Sampling completed", progress);
    Rprintf("\r");

    // Load regression trees parameters
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

    // Loop over the observations
    for (int i=0; i < n_samples; i++) {
      int base_i = i * ndpost * p;

      // Get current values of Y_i and x_i
      for (int j = 0; j < d; j++) y[j] = Yrm[i * d + j];
      for (int k = 0; k < p; k++) {
        x_cur[k] = Xrm[i * p + k];
        x_tilde[k] = x_cur[k] + mean_prior[k];
      }

      // Compute the forests predictions for given observation.
      // Don't need to re-compute this inside the burn-in loop, because I am
      // passing by reference, so this {theta} and {zeta} are both already updated
      GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);

      // Start inverse-sampling using ESS
      for (int k = 0; k < nburnin; k++) {
        x_cur = UpdateCESS(x_cur, y, chol_Sigma_V, Brm, Arm, bvec, eta,
                           theta, zeta, forest_theta, forest_zeta,
                           n_particles);
      }
      // Update the "initial" value of x for the next iteration
      for (int k = 0; k < p; k++) {
        Xrm[i * p + k] = x_cur[k];
        draws[base_i + t * p + k] = x_cur[k] + mean_prior[k];
      }
    }
    // Delete the trees (to free the memory usage)
    for (int j = 0; j < d; ++j){
      for (auto *tree : forest_theta[j]) delete tree;
      for (auto *tree : forest_zeta[j]) delete tree;
    }
  }
  return draws;
}


// Run one update of ESS for ZANIM-LN-BART for p=1
double InversePosterior::UpdateESS1p(double &x_cur,
                                     std::vector<int> &y,
                                     double &mu_prior,
                                     double &sd_prior,
                                     std::vector<double> &chol_Sigma_V,
                                     std::vector<double> &B,
                                     std::vector<double> &theta,
                                     std::vector<double> &zeta,
                                     const std::vector<std::vector<Node*>> &forest_theta,
                                     const std::vector<std::vector<Node*>> &forest_zeta,
                                     int n_particles) {

  // Define objects
  // std::vector<double> nu(p, 0.0), x_proposal(p, 0.0), x_tilde(p, 0.0);
  std::vector<double> x_tilde(1, 0.0);
  double nu, x_proposal, lr, nu_angle, nu_max, nu_min;

  // Log-likelihood threshold
  lr = log(R::unif_rand()) + log_pmf_zanim_ln(n_particles, y, theta, zeta, chol_Sigma_V, B);

  // Draw the angle
  nu = R::norm_rand()*sd_prior + mu_prior;

  nu_angle = R::unif_rand() * PI_2;
  nu_max = nu_angle;
  nu_min = nu_angle - PI_2;
  // Draw an proposal
  x_proposal = x_cur * cos(nu_angle) + nu * sin(nu_angle);
  x_tilde[0] = x_proposal + mu_prior;
  // Compute the forests predictions for initial proposal
  GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
  int counter = 0;
  // Start slice
  do {
    if (log_pmf_zanim_ln(n_particles, y, theta, zeta, chol_Sigma_V, B) > lr) break;
    if (counter > 1000) break;
    // Update the angle
    if (nu_angle < 0) nu_min = nu_angle;
    else nu_max = nu_angle;
    nu_angle = nu_min + (nu_max - nu_min) * R::unif_rand();
    // Draw new proposal
    x_proposal = x_cur * cos(nu_angle) + nu * sin(nu_angle);
    x_tilde[0] = x_proposal + mu_prior;
    // Compute BART predictions for the new proposal
    GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
    counter++;
  } while (true);
  return x_proposal;
}



// Constrained elliptical slice sampling for ZANIM-LN-BART for p>=2
std::vector<double> InversePosterior::ESS1p(arma::umat Y, std::vector<double> X_ini,
                                            int ndpost,
                                            int nburnin, int n_particles,
                                            double mean_prior, double s_prior,
                                            arma::mat B) {
  Rcpp::RNGScope scope;
  // To read the trees
  int np = 1;

  // Settings
  n_samples = Y.n_rows;
  // mu_prior_1 = mean_prior;
  // sd_prior = s_prior;

  // Define sample-specific mean and sd priors
  std::vector<double> m_prior(n_samples, mean_prior), sd_prior(n_samples, s_prior);
  std::vector<double> m_adapt(n_samples, 0.0), s2_adapt(n_samples, 0.0), sd_adapt(n_samples, 0.0);


  // Transform data and B matrix into row-major vectors
  std::vector<int> Yrm = umat_to_int_rowmajor(Y);
  std::vector<double> Brm = mat_to_double_rowmajor(B);

  // Open files to read the forests
  std::vector<std::ifstream> files_theta, files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff1 = forests_dir + "/forests_theta_" + std::to_string(j) + ".bin";
    std::string ff2 = forests_dir + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_theta.emplace_back(ff1, std::ios::binary);
    files_zeta.emplace_back(ff2, std::ios::binary);
  }
  // Open file for the the posterior draws of chol(Sigma_V)
  std::ifstream ff_Sigma_V(forests_dir + "/chol_Sigma_V.bin", std::ios::binary);
  // Create placeholder vector for dynamic read the posterior draws
  std::vector<double> chol_Sigma_V(dm1*dm1, 0.0);

  // Vector to keep the posterior draws
  std::vector<double> draws(ndpost*n_samples, 0.0);
  // Vectors for the proposal and the BART predictions
  std::vector<double> x_tilde(1, 0.0), theta(d, 0.0), zeta(d, 0.0);
  double x_cur;

  // Vector to allocate the counts for a given observation i
  std::vector<int> y(d, 0);
  double progress = 0.0;

  // Iterate over posterior draws of forward model
  for (int t=0; t < ndpost; t++) {
    // std::cout << t << "\n";
    progress = (double) 100 * t / ndpost;
    Rprintf("%3.2f%% Sampling completed", progress);
    Rprintf("\r");

    // Load regression trees parameters
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

    // Loop over the observations
    for (int i=0; i < n_samples; i++) {
      // Get current values of Y_i and x_i
      for (int j = 0; j < d; j++) y[j] = Yrm[i * d + j];
      // Current observation
      x_cur = X_ini[i];
      x_tilde[0] = x_cur + m_prior[i];
      // Compute the forests predictions for given observation.
      GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
      // Start inverse-sampling using ESS
      for (int k = 0; k < nburnin; k++) {
        x_cur = UpdateESS1p(x_cur, y, m_prior[i], sd_prior[i], chol_Sigma_V, Brm,
                            theta, zeta, forest_theta, forest_zeta, n_particles);
      }
      // Update the "initial" value of x for the next iteration
      X_ini[i] = x_cur;
      draws[i * ndpost + t] = x_cur + m_prior[i];
    }
    // Delete the trees (to free the memory usage)
    for (int j = 0; j < d; ++j){
      for (auto *tree : forest_theta[j]) delete tree;
      for (auto *tree : forest_zeta[j]) delete tree;
    }
  }
  return draws;
}


// Run one update of constrained ESS for ZANIM-LN-BART for p=1
double InversePosterior::UpdateCESS1p(double &x_cur,
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
                                      int n_particles) {

  // Define objects
  // std::vector<double> nu(p, 0.0), x_proposal(p, 0.0), x_tilde(p, 0.0);
  std::vector<double> x_tilde(1, 0.0);
  double nu, x_proposal, lr, nu_angle, nu_max, nu_min;

  // Log-likelihood threshold
  lr = log(R::unif_rand()) + log_pmf_zanim_ln(n_particles, y, theta, zeta, chol_Sigma_V, B);
  x_tilde[0] = x_cur + mu_prior_1;
  lr += log_I_ab(x_tilde[0], lower, upper, eta);

  // Draw the angle
  nu = R::norm_rand()*sd_prior_1 + mu_prior_1;

  nu_angle = R::unif_rand() * PI_2;
  nu_max = nu_angle;
  nu_min = nu_angle - PI_2;
  // Draw an proposal
  x_proposal = x_cur * cos(nu_angle) + nu * sin(nu_angle);
  x_tilde[0] = x_proposal + mu_prior_1;
  // Compute the forests predictions for initial proposal
  GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
  int counter = 0;
  // Start slice
  do {
    double ll_prop = log_pmf_zanim_ln(n_particles, y, theta, zeta, chol_Sigma_V, B);
    ll_prop += log_I_ab(x_tilde[0], lower, upper, eta);
    if (ll_prop > lr) break;
    if (counter > 1000) break;
    // Update the angle
    if (nu_angle < 0) nu_min = nu_angle;
    else nu_max = nu_angle;
    nu_angle = nu_min + (nu_max - nu_min) * R::unif_rand();
    // Draw new proposal
    x_proposal = x_cur * cos(nu_angle) + nu * sin(nu_angle);
    x_tilde[0] = x_proposal + mu_prior_1;
    // Compute BART predictions for the new proposal
    GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
    counter++;
  } while (true);
  return x_proposal;
}


// Constrained elliptical slice sampling for ZANIM-LN-BART for p>=2
std::vector<double> InversePosterior::CESS1p(arma::umat Y, std::vector<double> X_ini,
                                             int ndpost,
                                             int nburnin, int n_particles,
                                             double mean_prior, double s_prior,
                                             arma::mat B,
                                             double lower, double upper, double eta) {
  Rcpp::RNGScope scope;
  // To read the trees
  int np = 1;

  // Settings
  n_samples = Y.n_rows;
  mu_prior_1 = mean_prior;
  sd_prior_1 = s_prior;

  // Transform data and B matrix into row-major vectors
  std::vector<int> Yrm = umat_to_int_rowmajor(Y);
  std::vector<double> Brm = mat_to_double_rowmajor(B);

  // Open files to read the forests
  std::vector<std::ifstream> files_theta, files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff1 = forests_dir + "/forests_theta_" + std::to_string(j) + ".bin";
    std::string ff2 = forests_dir + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_theta.emplace_back(ff1, std::ios::binary);
    files_zeta.emplace_back(ff2, std::ios::binary);
  }
  // Open file for the the posterior draws of chol(Sigma_V)
  std::ifstream ff_Sigma_V(forests_dir + "/chol_Sigma_V.bin", std::ios::binary);
  // Create placeholder vector for dynamic read the posterior draws
  std::vector<double> chol_Sigma_V(dm1*dm1, 0.0);

  // Vector to keep the posterior draws
  std::vector<double> draws(ndpost*n_samples, 0.0);
  // Vectors for the proposal and the BART predictions
  std::vector<double> x_tilde(1, 0.0), theta(d, 0.0), zeta(d, 0.0);
  double x_cur;

  // Vector to allocate the counts for a given observation i
  std::vector<int> y(d, 0);
  double progress = 0.0;

  // Iterate over posterior draws of forward model
  for (int t=0; t < ndpost; t++) {
    // std::cout << t << "\n";
    progress = (double) 100 * t / ndpost;
    Rprintf("%3.2f%% Sampling completed", progress);
    Rprintf("\r");

    // Load regression trees parameters
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

    // Loop over the observations
    for (int i=0; i < n_samples; i++) {
      // Get current values of Y_i and x_i
      for (int j = 0; j < d; j++) y[j] = Yrm[i * d + j];
      // Current observation
      x_cur = X_ini[i];
      x_tilde[0] = x_cur + mean_prior;
      // Compute the forests predictions for given observation.
      GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
      // Start inverse-sampling using ESS
      for (int k = 0; k < nburnin; k++) {
        x_cur = UpdateCESS1p(x_cur, y, chol_Sigma_V, Brm, lower, upper, eta,
                             theta, zeta, forest_theta, forest_zeta,
                             n_particles);
      }
      // Update the "initial" value of x for the next iteration
      X_ini[i] = x_cur;
      draws[i * ndpost + t] = x_cur + mu_prior_1;
    }
    // Delete the trees (to free the memory usage)
    for (int j = 0; j < d; ++j){
      for (auto *tree : forest_theta[j]) delete tree;
      for (auto *tree : forest_zeta[j]) delete tree;
    }
  }
  return draws;
}


// Exposing a C++ class in R
RCPP_MODULE(inverse_posterior) {

  // Expose class on the R side
  Rcpp::class_<InversePosterior>("InversePosterior")

  // Constructor
  .constructor<int, int, int, std::string>()

  // Methods for the ZANIM-LN-BART
  .method("SIR", &InversePosterior::SIR)
  .method("ESS", &InversePosterior::ESS)
  .method("ESS1p", &InversePosterior::ESS1p)
  .method("CESS", &InversePosterior::CESS)
  .method("CESS1p", &InversePosterior::CESS1p)


  // effective sample size of SIR and the indices resampled in the sir
  .field("ess_sir", &InversePosterior::ess_sir)
  .field("indices_sir", &InversePosterior::indices_sir)
  ;

}
