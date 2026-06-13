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

////////////////////////////////////////////////////////////////////////////////////
// Methods that work and have been and tested for the ZANIM-LN-BART model
////////////////////////////////////////////////////////////////////////////////////

// Run multiple imputation with SIR approach to sample the inverse posterior
void InversePosterior::SIR(arma::umat Y, int n_proposal, int ndpost,
                           arma::mat B, std::string draws_dir) {

  Rcpp::RNGScope scope;
  // Dimension
  int n_samples = Y.n_rows, d = Y.n_cols, dm1 = d - 1;

  // Transform data into row-major vectors
  std::vector<double> Brm = mat_to_double_rowmajor(B);

  // Open file for the the posterior draws of chol(Sigma_V)
  std::ifstream ff_Sigma_V(forests_dir + "/chol_Sigma_V.bin", std::ios::binary);
  std::ifstream ff_theta(draws_dir + "/theta_ij.bin", std::ios::binary);
  std::ifstream ff_zeta(draws_dir + "/zeta_ij.bin", std::ios::binary);

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

  // ess_sir.resize(ndpost*n_samples, 0.0);
  // std::fill(ess_sir.begin(), ess_sir.end(), 0.0);
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
      // ess_sir[i*ndpost + k] = ComputeEfSS(probs);
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
                                            int ndpost, int nadapt,
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

      // if (nadapt>1){
      //   // Adapt the mean and sd of observation "i"
      //   double x_t, mu_t_1;
      //   if (t < nadapt) {
      //     x_t = x_cur + m_prior[i];
      //     mu_t_1 = m_adapt[i];
      //     m_adapt[i] += (x_t - mu_t_1)/(t+1);
      //     s2_adapt[i] += (x_t - mu_t_1) * (x_t - m_adapt[i]);
      //     // std::cout << m_adapt[i]  << "\n";
      //   }
      //   if (t == nadapt) {
      //     m_prior[i] = m_adapt[i] ;
      //     sd_prior[i] = std::sqrt(s2_adapt[i] / (nadapt - 1));
      //     // std::cout << "\n\n" << m_adapt[i] << " " << std::sqrt(s2_adapt[i] / (nadapt - 1)) << "\n";
      //   }
      // }

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


////////////////////////////////////////////////////////////////////////////////////
// Here are some experimental methods for different models or different sampling
// schemes


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


// Elliptical slice sampling for ML-BART
std::vector<double> InversePosterior::SamplerMLBARTeSS(arma::umat Y,
                                                       arma::mat X_ini,
                                                       int ndpost,
                                                       std::vector<double> mean_prior,
                                                       arma::mat S_prior,
                                                       int nburnin) {

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
  std::vector<double> nu(p, 0.0), x_cur(p, 0.0), x_tilde(p, 0.0), x_star(p, 0.0);
  std::vector<double> theta(d, 0.0);
  double u_s, nu_angle, nu_max, nu_min;

  // Compute the Cholesky and transform it to row-major
  arma::mat chol_S = arma::chol(S_prior);
  std::vector<double> chol_Srm = mat_to_double_rowmajor(chol_S);
  // std::cout << chol_Srm[0] << "\n";

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

      // For a given MCMC draw of "f" run a nested MCMC
      for (int k=0; k < nburnin; k++) {
        // Draw from the prior
        // rmvnorm_chol(nu, mean_prior, chol_Srm, p);
        rmvnorm_chol2(nu, chol_Srm, p);
        // Get the predictions for theta and zeta given the x_cur
        std::fill(theta.begin(), theta.end(), 0.0);
        for (int l=0; l < p; l++) x_tilde[l] = x_cur[l] + mean_prior[l];
        GetPredictionMLBART(x_tilde, theta, forest_theta);
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
        for (int l=0; l < p; l++) x_tilde[l] = x_star[l] + mean_prior[l];
        // Start slice sampling
        do {
          std::fill(theta.begin(), theta.end(), 0.0);
          // Get the predictions for theta  given the x_star
          GetPredictionMLBART(x_tilde, theta, forest_theta);
          // double ll = log_pmf_mult(y, total, theta);
          // std::cout << counter << " " << ll << " " << u_s << " nu_angle " << nu_angle << " nu_min: " << nu_min << " nu_max: " << nu_max << "\n";
          if (log_pmf_mult(y, ntrial, theta) > u_s) break;
          if (nu_angle < 0) nu_min = nu_angle;
          else nu_max = nu_angle;
          // Update the angle and the proposal
          nu_angle = nu_min + (nu_max - nu_min) * R::unif_rand();
          // for (int j = 0; j < p;j++) x_star[j] = cos(nu_angle) * x_cur[j] + sin(nu_angle) * nu[j];
          axpby(x_star.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
          for (int l=0; l < p; l++) x_tilde[l] = x_star[l] + mean_prior[l];
        } while (true);
        // Update x_cur
        x_cur = x_star;
      }
      // Save the posterior draw
      for (int k = 0; k < p; k++) x_posterior[base_i + t * p + k] = x_cur[k] + mean_prior[k];
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
                                                          int nburnin, int conditional) {

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
  std::vector<double> nu(p, 0.0), x_cur(p, 0.0), x_tilde(p, 0.0), x_star(p, 0.0), theta(d, 0.0), zeta(d, 0.0);
  double u_s, nu_angle, nu_max, nu_min;

  // Compute the Cholesky and transform it to row-major
  arma::mat chol_S = arma::chol(S_prior);
  std::vector<double> chol_Srm = mat_to_double_rowmajor(chol_S);

  // Create vector to allocate the counts for a given i
  std::vector<int> y(d, 0);
  double progress = 0.0;

  // Iterate over the observations samples
  for (int i=0; i < n_samples; i++) {

    int base_i = i * ndpost * p;

    // Get current values of Y_i and compute the N_i=\sum_j(y_{ij})
    for (int j = 0; j < d; j++) y[j] = Yrm[i * d + j];
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


      // For a given MCMC draw of "f" run a nested MCMC
      for (int k=0; k < nburnin; k++) {
        // Draw from the prior
        // rmvnorm_chol(nu, mean_prior, chol_Srm, p);
        rmvnorm_chol2(nu, chol_Srm, p);
        // Get the predictions for theta and zeta given the x_cur
        for (int l=0; l < p; l++) x_tilde[l] = x_cur[l] + mean_prior[l];
        GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
        // Set a log-likelihood threshold
        u_s = log(R::unif_rand());
        // double ll_cur = log_pmf(y, theta, zeta);
        u_s += log_pmf(y, theta, zeta);
        // Draw an angle and the proposal
        nu_angle = R::unif_rand() * PI_2;
        nu_max = nu_angle;
        nu_min = nu_angle - PI_2;
        axpby(x_star.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
        for (int l=0; l < p; l++) x_tilde[l] = x_star[l] + mean_prior[l];
        // Start slice sampling
        do {
          // Get the predictions for theta  given the x_star
          GetBARTPredictions(x_tilde, theta, zeta, forest_theta, forest_zeta);
          // double ll = log_pmf(y, theta, zeta);
          //std::cout << "PROPOSAL and CURRENT log-likelihood " << ll << " " << ll_cur << " " << u_s << "\n";
          if (log_pmf(y, theta, zeta) > u_s) break;
          if (nu_angle < 0) nu_min = nu_angle;
          else nu_max = nu_angle;
          // Update the angle and the proposal
          nu_angle = nu_min + (nu_max - nu_min) * R::unif_rand();
          axpby(x_star.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
          for (int l=0; l < p; l++) x_tilde[l] = x_star[l] + mean_prior[l];
        } while (true);
        // Update x_cur
        x_cur = x_star;
      }
      // Save the posterior draw
      for (int k = 0; k < p; k++) x_posterior[base_i + t * p + k] = x_cur[k] + mean_prior[k];
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

// Run multiple imputation with SIR approach to sample the inverse posterior
std::vector<int> InversePosterior::ABCSIRZANIMLNBART(std::vector<int> y,
                                                     int n_proposal,
                                                     int ndpost,
                                                     arma::mat B,
                                                     std::string draws_dir,
                                                     int kernel,
                                                     double h,
                                                     int n_particles) {

  // Dimension
  int d = y.size(), dm1 = d - 1;

  // Transform data into row-major vectors
  std::vector<double> Brm = mat_to_double_rowmajor(B);

  // Open file for the the posterior draws of chol(Sigma_V)
  std::ifstream ff_Sigma_V(forests_dir + "/chol_Sigma_V.bin", std::ios::binary);
  std::ifstream ff_theta(draws_dir + "/theta_ij.bin", std::ios::binary);
  std::ifstream ff_zeta(draws_dir + "/zeta_ij.bin", std::ios::binary);

  // Create vector to read the posterior draws of chol_Sigma_V
  std::vector<double> chol_Sigma_V(dm1*dm1, 0.0);

  // Create vector to read one posterior draw of theta_ij and zeta_ij
  std::vector<double> theta(n_proposal*d, 0.0);
  std::vector<double> zeta(n_proposal*d, 0.0);
  // Another vector to copy the values for a given observation i
  std::vector<double> theta_cur(d, 0.0);
  std::vector<double> zeta_cur(d, 0.0);

  // Number of trials, and the summary statistics for a given observed count is
  // s(y_i) = (y_i1, ..., y_id) / n_trial
  int ntrial = std::accumulate(y.begin(), y.end(), 0.0);

  // Summary statistics for observed data
  std::vector<double> sy = clr(y, 0.5);
  // std::vector<double> sy(d, 0.0);
  // for(int j=0; j < d; j++) sy[j] = (double)y[j] / ntrial;

  std::vector<double> sy_prop(d, 0.0);

  // The predictions are stored by posterior draws, so
  // [1 block][2 block]...[ndpost block], for k = ndpost
  // Each k block is (n_proposal × d) in column-major.
  std::vector<double> log_w(n_proposal, 0.0), log_k(n_particles, 0.0), probs(n_proposal);
  std::vector<int> sir_indices(ndpost, 0);
  ess_sir.resize(ndpost);
  std::fill(ess_sir.begin(), ess_sir.end(), 0.0);

  double log_n_paricles = std::log(n_particles);
  double progress = 0.0;
  // Iterate over posterior draws
  for (int k=0; k < ndpost; k++) {
    progress = (double) 100 * k / ndpost;
    Rprintf("%3.2f%% completed for computing the log-weights", progress);
    Rprintf("\r");
    // Read one block (n_proposal*d)
    ff_theta.read(reinterpret_cast<char*>(theta.data()),
                  sizeof(double) * n_proposal * d);
    ff_zeta.read(reinterpret_cast<char*>(zeta.data()),
                 sizeof(double) * n_proposal * d);
    // Read current posterior draw of chol(Sigma_V)
    ff_Sigma_V.read(reinterpret_cast<char*>(chol_Sigma_V.data()),
                    sizeof(double) * dm1 * dm1);
    // Iterate over proposal values (samples) and compute the log-likelihood
    for (int i=0; i < n_proposal; i++) {
      // Copy the current theta_ij
      for (int j=0; j < d; j++) {
        theta_cur[j] = theta[j*n_proposal + i];
        zeta_cur[j] = zeta[j*n_proposal + i];
      }
      // Iterate over the particles
      for (int t=0; t < n_particles; t++) {
        // Simulate y_proposal | theta(x), zeta(x)
        std::vector<int> y_prop = rzanimln(ntrial, theta_cur, zeta_cur, chol_Sigma_V, Brm);
        // Compute summary statistics (y_prop / n_trial)
        // for (int j=0; j < d; j++) sy_prop[j] = (double) y_prop[j] / ntrial;
        sy_prop = clr(y_prop, 0.5);
        // Compute the log-Kernel
        if (kernel) {
          log_k[t] = log_kernel_exp(sy, sy_prop, h);
        } else {
          log_k[t] = log_kernel_gauss(sy, sy_prop, h);
        }
        log_w[i] = log_sum_exp(log_k) - log_n_paricles;
      }
    }
    // Normalise weights and resample (only one draw)
    probs = normalise_weights(log_w, n_proposal);
    // effective sample size
    for (int i=0; i < n_proposal; i++) ess_sir[k] += probs[i]*probs[i];
    sir_indices[k] = sample_discrete(probs, n_proposal);
  }
  // Close files
  ff_zeta.close();  ff_theta.close(); ff_Sigma_V.close();
  // return log_w;
  return sir_indices;
}

// Run multiple imputation with SIR approach to sample the inverse posterior
std::vector<int> InversePosterior::SIRZANIMLNBART(std::vector<int> y,
                                                  int n_proposal,
                                                  int ndpost,
                                                  arma::mat B,
                                                  std::string draws_dir,
                                                  int n_particles,
                                                  int mixture) {

  // Dimension
  int d = y.size(), dm1 = d - 1;

  // Transform data into row-major vectors
  std::vector<double> Brm = mat_to_double_rowmajor(B);

  // Open file for the the posterior draws of chol(Sigma_V)
  std::ifstream ff_Sigma_V(forests_dir + "/chol_Sigma_V.bin", std::ios::binary);
  std::ifstream ff_theta(draws_dir + "/theta_ij.bin", std::ios::binary);
  std::ifstream ff_zeta(draws_dir + "/zeta_ij.bin", std::ios::binary);

  // Create vector to read the posterior draws of chol_Sigma_V
  std::vector<double> chol_Sigma_V(dm1*dm1, 0.0);

  // Create vector to read one posterior draw of theta_ij and zeta_ij
  std::vector<double> theta(n_proposal*d, 0.0);
  std::vector<double> zeta(n_proposal*d, 0.0);
  // Another vector to copy the values for a given observation i
  std::vector<double> theta_cur(d, 0.0);
  std::vector<double> zeta_cur(d, 0.0);

  // Create vector to allocate the counts for a given sample unit i
  int ntrial = std::accumulate(y.begin(), y.end(), 0.0);
  // The predictions are stored by posterior draws, so
  // [1 block][2 block]...[ndpost block], for k = ndpost
  // Each k block is (n_proposal × d) in column-major.
  std::vector<double> log_w(n_proposal, 0.0), log_k(n_particles, 0.0), probs(n_proposal);
  std::vector<int> sir_indices(ndpost, 0);
  ess_sir.resize(ndpost, 0.0);
  std::fill(ess_sir.begin(), ess_sir.end(), 0.0);

  double log_n_particles = std::log(n_particles);
  double progress = 0.0;
  // Iterate over posterior draws
  for (int k=0; k < ndpost; k++) {
    progress = (double) 100 * k / ndpost;
    Rprintf("%3.2f%% completed for computing the log-weights", progress);
    Rprintf("\r");
    // Read one block (n_proposal*d)
    ff_theta.read(reinterpret_cast<char*>(theta.data()),
                  sizeof(double) * n_proposal * d);
    ff_zeta.read(reinterpret_cast<char*>(zeta.data()),
                 sizeof(double) * n_proposal * d);
    // Read current posterior draw of chol(Sigma_V)
    ff_Sigma_V.read(reinterpret_cast<char*>(chol_Sigma_V.data()),
                    sizeof(double) * dm1 * dm1);

    // Write another loop here to iterate over observations Y's

    // Iterate over proposal values (samples) and compute the log-likelihood
    for (int i=0; i < n_proposal; i++) {
      // Copy the current theta_ij
      for (int j=0; j < d; j++) {
        theta_cur[j] = theta[j*n_proposal + i];
        zeta_cur[j] = zeta[j*n_proposal + i];
      }

      // Iterate over the particles
      for (int t=0; t < n_particles; t++) {
        if (mixture) {
          log_k[t] = log_pmf_zanim_ln_conditional(y, theta_cur, zeta_cur, chol_Sigma_V, Brm);
        } else {
          log_k[t] = log_pmf_zanim_ln_conditional(y, theta_cur, zeta_cur, chol_Sigma_V, Brm);
        }
      }
      // Compute the log-likelihood of observation i and posterior draw k
      log_w[i] = log_sum_exp(log_k) - log_n_particles;
    }
    // Normalise weights and resample (only one draw)
    probs = normalise_weights(log_w, n_proposal);
    ess_sir[k] = ComputeEfSS(probs);
    sir_indices[k] = sample_discrete(probs, n_proposal);
  }
  // Close files
  ff_zeta.close();  ff_theta.close(); ff_Sigma_V.close();
  return sir_indices;
}

// MI with SIR for ML-BART
std::vector<int> InversePosterior::SIRZANIMBART(std::vector<int> y,
                                                int n_proposal,
                                                int ndpost,
                                                std::string draws_dir,
                                                int conditional) {


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

  // Dimension
  int d = y.size(), dm1 = d - 1;

  // Open file for the the posterior draws of chol(Sigma_V)
  std::ifstream ff_theta(draws_dir + "/theta_ij.bin", std::ios::binary);
  std::ifstream ff_zeta(draws_dir + "/zeta_ij.bin", std::ios::binary);

  // Create vector to read the posterior draws of chol_Sigma_V
  std::vector<double> chol_Sigma_V(dm1*dm1, 0.0);

  // Create vector to read one posterior draw of theta_ij and zeta_ij
  std::vector<double> theta(n_proposal*d, 0.0);
  std::vector<double> zeta(n_proposal*d, 0.0);
  // Another vector to copy the values for a given observation i
  std::vector<double> theta_cur(d, 0.0);
  std::vector<double> zeta_cur(d, 0.0);

  // Create vector to allocate the counts for a given sample unit i
  int ntrial = std::accumulate(y.begin(), y.end(), 0.0);
  // The predictions are stored by posterior draws, so
  // [1 block][2 block]...[ndpost block], for k = ndpost
  // Each k block is (n_proposal × d) in column-major.
  std::vector<double> log_w(n_proposal, 0.0), probs(n_proposal);
  std::vector<int> sir_indices(ndpost, 0);

  double m, ll, progress = 0.0;
  // Iterate over posterior draws
  for (int k=0; k < ndpost; k++) {
    progress = (double) 100 * k / ndpost;
    Rprintf("%3.2f%% completed for computing the log-weights", progress);
    Rprintf("\r");
    // Read one block (n_proposal*d)
    ff_theta.read(reinterpret_cast<char*>(theta.data()),
                  sizeof(double) * n_proposal * d);
    ff_zeta.read(reinterpret_cast<char*>(zeta.data()),
                 sizeof(double) * n_proposal * d);
    // Iterate over proposal values (samples) and compute the log-likelihood
    for (int i=0; i < n_proposal; i++) {
      // Copy the current theta_ij
      for (int j=0; j < d; j++) {
        theta_cur[j] = theta[j*n_proposal + i];
        zeta_cur[j] = zeta[j*n_proposal + i];
      }
      // Compute the log-likelihood of observation i and posterior draw k
      // log_w[i] = log_pmf_zanim_ln_conditional(y, theta_cur, zeta_cur, chol_Sigma_V, Brm);
      log_w[i] = log_pmf(y, theta_cur, zeta_cur);
      // log_w[i*ndpost + k] = ll ;
    }
    // Normalise weights and resample (only one draw)
    probs = normalise_weights(log_w, n_proposal);
    sir_indices[k] = sample_discrete(probs, n_proposal);
  }
  // Close files
  ff_zeta.close();  ff_theta.close();
  // return log_w;
  return sir_indices;
}

// MI with SIR for ML-BART
std::vector<int> InversePosterior::SIRMLBART(std::vector<int> y,
                                             int n_proposal, int ndpost,
                                             std::string draws_dir) {

  // Dimension
  int d = y.size(), dm1 = d - 1;

  // Open file for the the posterior draws of chol(Sigma_V)
  std::ifstream ff_theta(draws_dir + "/theta_ij.bin", std::ios::binary);

  // Create vector to read one posterior draw of theta_ij  and copy
  // the values for a given observation i
  std::vector<double> theta(n_proposal*d, 0.0);
  std::vector<double> theta_cur(d, 0.0);

  // Create vector to allocate the counts for a given sample unit i
  int ntrial = std::accumulate(y.begin(), y.end(), 0.0);
  // The predictions are stored by posterior draws, so
  // [1 block][2 block]...[ndpost block], for k = ndpost
  // Each k block is (n_proposal × d) in column-major.
  std::vector<double> log_w(n_proposal, 0.0), probs(n_proposal);
  std::vector<int> sir_indices(ndpost, 0);

  double m, ll, progress = 0.0;
  // Iterate over posterior draws
  for (int k=0; k < ndpost; k++) {
    progress = (double) 100 * k / ndpost;
    Rprintf("%3.2f%% completed for computing the log-weights", progress);
    Rprintf("\r");
    // Read one block (n_proposal*d)
    ff_theta.read(reinterpret_cast<char*>(theta.data()),
                  sizeof(double) * n_proposal * d);
    // Iterate over proposal values (samples) and compute the log-likelihood
    for (int i=0; i < n_proposal; i++) {
      // Copy the current theta_ij
      for (int j=0; j < d; j++) theta_cur[j] = theta[j*n_proposal + i];
      // Compute the log-likelihood of observation i and posterior draw k
      log_w[i] = log_pmf_mult(y, ntrial, theta_cur);
    }
    // Normalise weights and resample (only one draw)
    probs = normalise_weights(log_w, n_proposal);
    sir_indices[k] = sample_discrete(probs, n_proposal);
  }
  // Close files
  ff_theta.close();
  return sir_indices;
}


////////////////////////////////////////////////////////////////////////////////////
// Implement Adaptive IS (PMC)

double InversePosterior::ComputeEfSS(std::vector<double> &probs) {
  double s = 0.0;
  for (size_t j=0; j < probs.size(); j++) s += probs[j]*probs[j];
  return 1.0 / s;
}
void InversePosterior::WeightedMeanVar(double &mu, double &s2,
                                       std::vector<double> &x,
                                       std::vector<double> &probs) {
  mu = 0.0;
  s2 = 0.0;
  int n = x.size();
  for (int i=0; i < n; i++) mu += probs[i] * x[i];
  for (int i = 0; i < n; i++) s2 += probs[i] * std::pow(x[i] - mu, 2.0);
}

void InversePosterior::PopulationMC(std::vector<int> y,
                                    int ndpost,
                                    int n_particles_x, arma::mat B,
                                    std::vector<double> range_prior,
                                    double scale_prop,
                                    double prob_level,
                                    double ep) {
  std::vector<double> ps = {prob_level / 2.0, 1.0 - prob_level / 2.0};

  double lower = range_prior[0], upper = range_prior[1], delta = upper - lower;

  // To read the trees
  int np = 1;

  // Setting field
  p = 1;

  // Transform B matrix into row-major vectors
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

  // Vector to keep the posterior draws and the current "particles"
  std::vector<double> x_particles(n_particles_x,  0.0), old_particles(n_particles_x,  0.0), x_prop(p, 0.0);
  std::vector<double> theta(d, 0.0), zeta(d, 0.0);

  // Vector to keep the posterior
  // x_posterior.resize(ndpost*n_particles_x, 0.0);
  x_posterior.resize(ndpost, 0.0);
  std::fill(x_posterior.begin(), x_posterior.end(), 0.0);

  ess_sir.resize(ndpost);
  std::fill(ess_sir.begin(), ess_sir.end(), 0.0);

  // Vector to allocate the log-(un-normalised) weights
  std::vector<double> log_w(n_particles_x, 0.0), probs(n_particles_x, 0.0);

  double progress = 0.0, sd, mu, s2;

  // Initialise the particles, using first posterior draw of f's.

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

  // Log-prior
  double l_prior = std::log(range_prior[1] - range_prior[0]);
  double step = delta / (n_particles_x - 1);

  // Sample from uniform prior and compute the log-likelihood
  for (int j = 0; j < n_particles_x; j++) {
    x_prop[0] = lower + j * step; //R::runif(range_prior[0], range_prior[1]);
    x_particles[j] = x_prop[0];
    // Compute the forests predictions for given observation.
    GetBARTPredictions(x_prop, theta, zeta, forest_theta, forest_zeta);
    // Compute the log-likelihood
    log_w[j] = log_pmf_zanim_ln_conditional(y, theta, zeta, chol_Sigma_V, Brm);
  }
  probs = normalise_weights(log_w, n_particles_x);
  ess_sir[0] = ComputeEfSS(probs);

  // Computed weighted mean and variance to used for the new Gaussian proposal
  // WeightedMeanVar(mu, s2, x_particles, probs);
  // sd = std::sqrt(scale_prop*s2);

  // Resample
  old_particles = x_particles;
  for (int j=0; j < n_particles_x; j++) {
    x_particles[j] = old_particles[sample_discrete(probs, n_particles_x)];
  }
  x_posterior[0] = x_particles[0];//old_particles[sample_discrete(probs, n_particles_x)];

  // Compute the quantiles of resampled particles
  std::vector<double> qs = quantile(x_particles, ps);

  // Update the priors
  lower = qs[0] - ep * delta;
  upper = qs[1] - ep * delta;
  delta = upper - lower;


  // Delete the trees (to free the memory usage)
  for (int j = 0; j < d; ++j){
    for (auto *tree : forest_theta[j]) delete tree;
    for (auto *tree : forest_zeta[j]) delete tree;
  }

  // // Iterate over posterior draws of forward model
  for (int t=1; t < ndpost; t++) {
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

    // Simulate new particles and compute its weights
    for (int j = 0; j < n_particles_x; j++) {
      x_prop[0] = R::runif(qs[0], qs[1]);
      x_particles[j] = x_prop[0];
      GetBARTPredictions(x_prop, theta, zeta, forest_theta, forest_zeta);
      log_w[j] = log_pmf_zanim_ln_conditional(y, theta, zeta, chol_Sigma_V, Brm);

      // double x_prev = x_particles[j];
      // x_prop[0] = rtnorm_ab(x_prev, sd, range_prior[0], range_prior[1]);
      // x_particles[j] = x_prop[0];
      // GetBARTPredictions(x_prop, theta, zeta, forest_theta, forest_zeta);
      // log_w[j] = log_pmf_zanim_ln_conditional(y, theta, zeta, chol_Sigma_V, Brm)
      //   - l_prior - ldtrucnorm(x_particles[j], x_prev, sd, range_prior[0], range_prior[1]);

    }

    // Compute probabilities, and updated the sd
    probs = normalise_weights(log_w, n_particles_x);
    // WeightedMeanVar(mu, s2, x_particles, probs);
    // sd = std::sqrt(scale_prop*s2);

    // Resample
    old_particles = x_particles;
    for (int j=0; j < n_particles_x; j++) {
      x_particles[j] = old_particles[sample_discrete(probs, n_particles_x)];
      // x_posterior[t * n_particles_x + j] = x_particles[j];
    }
    x_posterior[t] = x_particles[0];
    ess_sir[t] = ComputeEfSS(probs);

    // Update the priors
    qs = quantile(x_particles, ps);
    lower = qs[0] - ep * delta;
    upper = qs[1] - ep * delta;
    delta = upper - lower;

    // Delete the trees (to free the memory usage)
    for (int j = 0; j < d; ++j){
      for (auto *tree : forest_theta[j]) delete tree;
      for (auto *tree : forest_zeta[j]) delete tree;
    }
  }

}

////////////////////////////////////////////////////////////////////////////////////

// Compute the log marginal likelihood of ZANIM-BART given y and x,
double InversePosterior::lmlZANIM(std::vector<int> &y, std::vector<double> &x,
                                  int n_particles) {

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
    GetBARTPredictions(x, theta, zeta, forest_theta, forest_zeta);

    // Compute the likelihood
    lml[k] = log_pmf_zanim_conditional(y, theta, zeta);

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



// Compute the Log-likelihood of ZANIM-LN-BART given y and x across the MCMC
// draws of f
double InversePosterior::LogLikelihoodZANIMLN(std::vector<int> &y,
                                              std::vector<double> &x,
                                              int ndpost,
                                              int chain_index,
                                              arma::mat B) {

  int np = 1, dm1 = d - 1;

  // Open files to read the forests
  std::vector<std::ifstream> files_theta, files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff1 = forests_dir + "/forests_theta_" + std::to_string(j) + ".bin";
    std::string ff2 = forests_dir + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_theta.emplace_back(ff1, std::ios::binary);
    files_zeta.emplace_back(ff2, std::ios::binary);
  }
  std::ifstream ff_Sigma_V(forests_dir + "/chol_Sigma_V.bin", std::ios::binary);
  std::vector<double> chol_Sigma_V(dm1*dm1, 0.0);
  std::vector<double> Brm = mat_to_double_rowmajor(B);

  std::vector<std::vector<Node*>> forest_theta(d);
  std::vector<std::vector<Node*>> forest_zeta(d);
  for (int j = 0; j < d; j++) {
    forest_theta[j].reserve(ntrees_theta);
    forest_zeta[j].reserve(ntrees_zeta);
  }
  chain_index -= 1;
  std::vector<double>  theta(d, 0.0), zeta(d, 0.0);
  // Iterate over the MCMC samples
  for (int k = 0; k < ndpost; k++) {
    // Load only the chosen tree
    for (int j = 0; j < d; j++) {
      forest_theta[j].clear();
      forest_zeta[j].clear();
      // Load only target tree
      for (int h = 0; h < ntrees_theta; h++) {
        forest_theta[j].push_back(deserialise_tree(files_theta[j], np));
      }
      for (int h = 0; h < ntrees_zeta; h++) {
        forest_zeta[j].push_back(deserialise_tree(files_zeta[j], np));
      }
    }
    ff_Sigma_V.read(reinterpret_cast<char*>(chol_Sigma_V.data()),
                    sizeof(double) * dm1 * dm1);
    if (k == chain_index) {
      // Compute the regression trees predictions for the
      // std::fill(theta.begin(), theta.end(), 0.0);
      // std::fill(zeta.begin(), zeta.end(), 0.0);
      GetBARTPredictions(x, theta, zeta, forest_theta, forest_zeta);

      // Compute the likelihood
      // double lml = log_pmf_zanim_conditional(y, theta, zeta);
      double lml = log_pmf_zanim_ln_conditional(y, theta, zeta, chol_Sigma_V, Brm);
      // Remove the trees (to free the memory usage)
      for (int j = 0; j < d; ++j){
        for (auto *tree : forest_theta[j]) delete tree;
        for (auto *tree : forest_zeta[j]) delete tree;
      }
      for (int j=0; j<d; j++) {
        files_theta[j].close(); files_zeta[j].close();
      }
      ff_Sigma_V.close();
      return lml;
    }
    for (int j = 0; j < d; ++j){
      for (auto *tree : forest_theta[j]) delete tree;
      for (auto *tree : forest_zeta[j]) delete tree;
    }
  }
  return 0.0;
}

std::vector<double> InversePosterior::LogLikelihoodZANIMLN_2(std::vector<int> &y,
                                                           std::vector<double> &x,
                                                           int ndpost,
                                                           arma::mat B) {

  int np = 1;
  int dm1 = d-1;
  // Open files to read the forests
  std::vector<std::ifstream> files_theta, files_zeta;
  for (int j=0; j < d; j++) {
    std::string ff1 = forests_dir + "/forests_theta_" + std::to_string(j) + ".bin";
    std::string ff2 = forests_dir + "/forests_zeta_" + std::to_string(j) + ".bin";
    files_theta.emplace_back(ff1, std::ios::binary);
    files_zeta.emplace_back(ff2, std::ios::binary);
  }
  std::ifstream ff_Sigma_V(forests_dir + "/chol_Sigma_V.bin", std::ios::binary);
  std::vector<double> chol_Sigma_V(dm1*dm1, 0.0);
  std::vector<double> Brm = mat_to_double_rowmajor(B);

  std::vector<std::vector<Node*>> forest_theta(d);
  std::vector<std::vector<Node*>> forest_zeta(d);
  for (int j = 0; j < d; j++) {
    forest_theta[j].reserve(ntrees_theta);
    forest_zeta[j].reserve(ntrees_zeta);
  }
  std::vector<double> lml(ndpost, 0.0), theta(d, 0.0), zeta(d, 0.0);
  // Iterate over the MCMC samples
  for (int k = 0; k < ndpost; k++) {
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
    ff_Sigma_V.read(reinterpret_cast<char*>(chol_Sigma_V.data()),
                    sizeof(double) * dm1 * dm1);

    // Compute the regression trees predictions for the
    std::fill(theta.begin(), theta.end(), 0.0);
    std::fill(zeta.begin(), zeta.end(), 0.0);

    // Compute the likelihood
    lml[k] = log_pmf_zanim_ln_conditional(y, theta, zeta, chol_Sigma_V, Brm);
    // Remove the trees (to free the memory usage)
    for (int j = 0; j < d; ++j){
      for (auto *tree : forest_theta[j]) delete tree;
      for (auto *tree : forest_zeta[j]) delete tree;
    }
  }
  for (int j=0; j<d; j++) {
    files_theta[j].close(); files_zeta[j].close();
  }
  return lml;
}

////////////////////////////////////////////////////////////////////////////////////
// --------------------------------------------------------------------------------
void InversePosterior::GetTreesPredictionsZANIMBART(std::vector<double> &x,
                                              std::vector<double> &lambda,
                                              std::vector<double> &zeta,
                                              const std::vector<std::vector<Node*>> &forest_theta,
                                              const std::vector<std::vector<Node*>> &forest_zeta) {
  // Iterate over categories
  for (int j = 0; j < d; j++) {
    // Iterate over trees
    for (int h = 0; h < ntrees_theta; h++) {
      // Do the predictions
      lambda[j] += GetMu(forest_theta[j][h], x);
    }
    for (int h = 0; h < ntrees_zeta; h++) {
      // Do the predictions
      zeta[j] += GetMu(forest_zeta[j][h], x);
    }
    lambda[j] = exp(lambda[j]);
    zeta[j] = R::pnorm5(zeta[j], 0.0, 1.0, 1.0, 0.0);
  }
}


// Log-target of u_i with marginalised phi_i
double InversePosterior::LogTargetU(std::vector<double> &u, std::vector<int> &y,
                                    std::vector<double> &z,
                                    std::vector<double> &lambda) {

  int idx = 0;
  int k=0;
  for (int j=0; j < d; j++) if (z[j] > 0) k++;
  // int k = std::count_if(z.begin(), z.end(), [](double x) {return x > 0;});
  std::vector<double> lterms(k, 0.0);
  double l = 0.0, n_trials = 0.0;
  for (int j=0; j < d; j++) {
    n_trials += y[j];
    l += y[j] * u[j];
    if (z[j] > 0) lterms[idx++] = std::log(lambda[j]) + u[j];
  }
  if (lterms.empty()) return l; // Need to return prob = 1, i.e. l=0.
  return l - n_trials * log_sum_exp(lterms);
}

// ESS for update v_i under the "full" covariance prior (also work for the factor model)
std::vector<double> InversePosterior::UpdateESSV(std::vector<double> &v,
                                                 std::vector<double> &chol_Sigma_V,
                                                 std::vector<double> &B,
                                                 std::vector<int> &y,
                                                 std::vector<double> &z,
                                                 std::vector<double> &lambda) {
  // Create vectors
  std::vector<double> u(d, 0.0), u_prop(d, 0.0), v_prop(dm1, 0.0), nu(dm1, 0.0);

  // log-likelihood threshold
  double logy = log(R::unif_rand());
  Bv(u, v, B, d, dm1);
  double ll_cur = LogTargetU(u, y, z, lambda);
  logy += ll_cur;
  // std::cout << logy << "\n";
  // Draw angle
  rmvnorm_chol2(nu, chol_Sigma_V, dm1);
  double theta = R::unif_rand() * PI_2;
  double theta_max = theta;
  double theta_min = theta - PI_2;
  // Draw proposal
  axpby(v_prop.data(), v.data(), nu.data(), cos(theta), sin(theta), dm1);
  Bv(u, v_prop, B, d, dm1);
  do {
    double ll = LogTargetU(u, y, z, lambda);
    if (ll > logy) break;
    // Shrink the angle
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    // Draw a new angle, then a new proposal
    theta = theta_min + (theta_max - theta_min) * R::unif_rand();
    axpby(v_prop.data(), v.data(), nu.data(), cos(theta), sin(theta), dm1);
    Bv(u, v_prop, B, d, dm1);
  } while (true);
  return v_prop;
}


// Run an update of ESS for ZANIM-LN-BART using Poisson-type likelihood
std::vector<double> InversePosterior::UpdateESSZANIMLNBART2(
    std::vector<double> &x_cur,
    std::vector<int> &y,
    std::vector<double> &z, std::vector<double> &u,
    double &phi,
    std::vector<double> &lambda, std::vector<double> &zeta,
    const std::vector<std::vector<Node*>> &forest_theta,
    const std::vector<std::vector<Node*>> &forest_zeta) {

  // Define objects
  std::vector<double> nu(p, 0.0), x_proposal(p, 0.0), x_tilde(p, 0.0);
  double lr, nu_angle, nu_max, nu_min;

  // Correct for the prior mean
  // for (int k=0; k < p; k++) x_tilde[k] = x_cur[k] + mean_prior[k];
  // Compute the forests predictions, f_j^{(c)} and f_j^{(0)}  for current observation
  // GetTreesPredictionsZANIMBART(x_tilde, lambda, zeta, forest_theta, forest_zeta);

  // Log-likelihood threshold
  lr = log(R::unif_rand()) + log_pmf_zanim_ln_augmented(y, z, zeta, lambda, u, phi);

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
  std::fill(lambda.begin(), lambda.end(), 0.0);
  std::fill(zeta.begin(), zeta.end(), 0.0);
  GetTreesPredictionsZANIMBART(x_tilde, lambda, zeta, forest_theta, forest_zeta);
  // Start slice
  do {
    if (log_pmf_zanim_ln_augmented(y, z, zeta, lambda, u, phi) > lr) break;
    // Update the angle
    if (nu_angle < 0) nu_min = nu_angle;
    else nu_max = nu_angle;
    nu_angle = nu_min + (nu_max - nu_min) * R::unif_rand();
    // Draw new proposal
    axpby(x_proposal.data(), x_cur.data(), nu.data(), cos(nu_angle), sin(nu_angle), p);
    // Correct for the prior mean
    for (int k=0; k < p; k++) x_tilde[k] = x_proposal[k] + mu_prior[k];
    // Compute BART predictions for the new proposal
    std::fill(lambda.begin(), lambda.end(), 0.0);
    std::fill(zeta.begin(), zeta.end(), 0.0);
    GetTreesPredictionsZANIMBART(x_tilde, lambda, zeta, forest_theta, forest_zeta);
  } while (true);
  return x_proposal;
}

// Elliptical slice sampling for ZANIM-LN-BART
std::vector<double> InversePosterior::ESSZANIMLNBART2(arma::umat Y,
                                                     arma::mat X_ini,
                                                     int ndpost,
                                                     int nburnin,
                                                     std::vector<double> mean_prior,
                                                     arma::mat S_prior,
                                                     arma::mat B) {
  // To read the trees
  int np = 1;

  // Setting field
  p = X_ini.n_cols;
  n_samples = Y.n_rows;
  // Gaussian prior
  // Compute the (upper triangle) Cholesky of the prior and transform it to
  // row-major
  arma::mat chol_S = arma::chol(S_prior);
  chol_S_prior = mat_to_double_rowmajor(chol_S);
  mu_prior = mean_prior;

  // Transform data and B matrix into row-major vectors
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

  // if (!ff_Sigma_V) std::cerr << "Read chol(Sigma_V) failed\n";
  // std::cout << "chol_Sigma_V_ALL " << chol_Sigma_V_ALL[0] << " " << chol_Sigma_V_ALL[1] << "\n";

  // Vector to keep the posterior draws
  std::vector<double> x_posterior(ndpost*p*n_samples, 0.0);

  // Vectors for the proposal and the BART predictions
  std::vector<double> x_cur(p, 0.0), x_tilde(p, 0.0), lambda(d, 0.0), zeta(d, 0.0);

  // Vector to allocate the counts for a given observation i
  std::vector<int> y(d, 0);
  double progress = 0.0;

  // Iterate over posterior draws of forward model
  for (int t=0; t < ndpost; t++) {
    progress = 0.0;
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
      int n_trial = 0;
      for (int j = 0; j < d; j++) {
        y[j] = Yrm[i * d + j];
        n_trial += y[j];
      }
      for (int k = 0; k < p; k++) {
        x_cur[k] = Xrm[i * p + k];
        x_tilde[k] = x_cur[k] + mean_prior[k];
      }

      // Compute the forests predictions for current observation
      std::fill(lambda.begin(), lambda.end(), 0.0);
      std::fill(zeta.begin(), zeta.end(), 0.0);
      GetTreesPredictionsZANIMBART(x_tilde, lambda, zeta, forest_theta, forest_zeta);

      // Initialise latent variables drawing from their priors
      std::vector<double> z(d, 1.0), u(d, 0.0), v(dm1, 0.0);
      rmvnorm_chol2(v, chol_Sigma_V, dm1);
      // Get u = Bv
      Bv(u, v, Brm, d, dm1);

      // Sample z and phi
      double rate = 0.0;
      for (int j = 0; j < d; j++) {
        if (y[j] == 0)
          z[j] = R::rbinom(1, 1.0 - zeta[j]);
        if (z[j] > 0.0)
          rate += lambda[j] * exp(u[j]) * z[j];
      }
      double phi = R::rgamma(n_trial, 1.0 / rate);
      // Start inverse-sampling using ESS
      for (int k = 0; k < nburnin; k++) {
        x_cur = UpdateESSZANIMLNBART2(x_cur, y, z, u, phi, lambda, zeta, forest_theta,
                                      forest_zeta);
        // Update v, then compute u = Bv
        v = UpdateESSV(v, chol_Sigma_V, Brm, y, z, lambda);
        Bv(u, v, Brm, d, dm1);
        // Update latent variables using full Gibbs
        rate = 0.0;
        for (int j = 0; j < d; j++) {
          if (y[j] == 0) {
            double prob = (1.0 - zeta[j]) * exp(-phi * u[j] * lambda[j]);
            prob /= (prob + zeta[j]);
            z[j] = R::rbinom(1, prob);
          }
          if (z[j] > 0.0) rate += lambda[j] * exp(u[j]);
        }
        phi = R::rgamma(n_trial, 1.0 / rate);
      }
      // Update the "initial" value of x for the next iteration
      for (int k = 0; k < p; k++) {
        Xrm[i * p + k] = x_cur[k];
      }
      // Save the posterior draw
      for (int k = 0; k < p; k++) x_posterior[base_i + t * p + k] = x_cur[k] + mean_prior[k];
    }
    // Delete the trees (to free the memory usage)
    for (int j = 0; j < d; ++j){
      for (auto *tree : forest_theta[j]) delete tree;
      for (auto *tree : forest_zeta[j]) delete tree;
    }
  }
  return x_posterior;
}


// Exposing a C++ class in R
RCPP_MODULE(inverse_posterior) {

  // Expose class on the R side
  Rcpp::class_<InversePosterior>("InversePosterior")

  // Constructor
  .constructor<int, int, int, std::string>()

  // Experimental methods...
  .method("SamplerMLBARTeSS", &InversePosterior::SamplerMLBARTeSS)
  .method("SamplerZANIMBARTeSS", &InversePosterior::SamplerZANIMBARTeSS)
  .method("SIRZANIMBART", &InversePosterior::SIRZANIMBART)
  .method("SIRMLBART", &InversePosterior::SIRMLBART)
  .method("lmlZANIM", &InversePosterior::lmlZANIM)
  .method("LogLikelihoodZANIMLN", &InversePosterior::LogLikelihoodZANIMLN)
  .method("LogLikelihoodZANIMLN_2", &InversePosterior::LogLikelihoodZANIMLN_2)
  .method("ABCSIRZANIMLNBART", &InversePosterior::ABCSIRZANIMLNBART)
  .method("PopulationMC", &InversePosterior::PopulationMC)
  .method("ESSZANIMLNBART2", &InversePosterior::ESSZANIMLNBART2)


  // Methods for the ZANIM-LN-BART
  .method("SIR", &InversePosterior::SIR)
  .method("ESS", &InversePosterior::ESS)
  .method("ESS1p", &InversePosterior::ESS1p)
  .method("CESS", &InversePosterior::CESS)
  .method("CESS1p", &InversePosterior::CESS1p)


  // effective sample size of SIR and the indices resampled in the sir
  .field("ess_sir", &InversePosterior::ess_sir)
  .field("indices_sir", &InversePosterior::indices_sir)
  .field("x_posterior", &InversePosterior::x_posterior)
  ;

}
