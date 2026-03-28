#include "dm_reg.h"
// #include "rng.h"

constexpr double PI_2 = 6.283185307179586231996;

DMLinearReg::DMLinearReg(const arma::umat &Y, const arma::mat &X) : Y(Y), X(X), rng() {
  d = Y.n_cols;
  n = Y.n_rows;
  p = X.n_cols;
  n_trials = arma::sum(Y, 1);
}

DMLinearReg::~DMLinearReg(){
  // delete rng;
};


void DMLinearReg::SetMCMC(arma::cube V_prior_betas_, int ndpost_, int nskip_,
                          int nthin_, int keep_draws_, int save_draws_,
                          std::string dir_draws_) {
  V_prior_betas = V_prior_betas_;
  ndpost = ndpost_;
  nskip = nskip_;
  nthin = nthin_;
  keep_draws = keep_draws_;
  save_draws = save_draws_;
  dir_draws = dir_draws_;

// Add this, if use arma::randn
  Rcpp::RNGScope scope;
  // rng = new RNG2();

  // Initialise the reg coefficients drawing from the prior
  betas = arma::zeros<arma::mat>(p, d);

  // Initialise the linear predictions coefficients
  alphas = arma::zeros<arma::mat>(n, d);
  lambdas = arma::zeros<arma::mat>(n, d);

  // Compute the cholesky decomposition of the covariance matrix of the full conditional
  // distribution of the probit-reg coefficients for all categories
  chol_V_prior = arma::zeros<arma::cube>(p,p,d);
  for (int j=0; j < d; j++) {
    chol_V_prior.slice(j) = arma::chol(V_prior_betas.slice(j));
  }
  ep = arma::zeros<arma::rowvec>(p);
  nu = arma::zeros<arma::vec>(p);

  // for (int k=0;k<10;k++) std::cout << rng.normal() << "\n";
  //
  // std::cout << rng.normal(2, 0.001) << "\n";
  // std::cout << rng.uniform() << "\n";
  // std::cout << rng.gamma(1.0, 1.0) << "\n";

  // chol_V_prior.print();

  // First draw for the latent variable phi_i
  phi = arma::zeros<arma::vec>(n);
  // for (int i = 0; i < n; i++) phi(i) = R::rgamma(n_trials(i), 1.0);
  for (int i = 0; i < n; i++) phi(i) = rng.gamma(n_trials(i), 1.0);
  // std::cout << "passou" << '\n';

  // First draw for the latent variable \lambda_{ij} ~ Gamma[\alpha_{ij} + y_{ij}, 1 + \phi_i]
  for (int j = 0; j < d; j++) {
    for (int i=0; i < n; i ++) {
      // lambdas(i, j) = R::rgamma(Y(i, j) + 1.0, 1.0);
      lambdas(i, j) = rng.gamma(Y(i, j) + 1.0, 1.0);
    }
  }
  std::cout << "DM-reg model set up! \n\n";
}

double DMLinearReg::LogTargetBetas(arma::vec &beta_cur, int &j) {

  arma::vec alphas = exp(X * beta_cur);
  double t1 = 0.0;
  for (int i = 0; i < n; i++) {
    t1 += alphas(i) * log(lambdas(i, j)) - std::lgamma(alphas(i));
  }
  return t1;
}

arma::vec DMLinearReg::UpdateBetasESS(int &j) {

  arma::vec beta_cur = betas.col(j);

  // Draw from the prior
  for (int k=0; k<p; k++) ep[k] = rng.normal();
  arma::vec nu = (ep * chol_V_prior.slice(j)).t();

  // Set a log-likelihood threshold
  double u_s = log(R::unif_rand());
  double ll_cur = LogTargetBetas(beta_cur, j);
  u_s += ll_cur; //LogTargetBetas(beta_cur, j);
  // Draw an angle
  // double theta = R::unif_rand() * PI_2;
  double theta = rng.uniform() * PI_2;
  double theta_max = theta;
  double theta_min = theta - PI_2;
  // Draw proposal
  arma::vec beta_star = beta_cur * cos(theta) + nu * sin(theta);
  // Start slicing
  do {
    // TODO: IF LL = -inf, then don't evaluate the loop
    double ll = LogTargetBetas(beta_star, j);
    // beta_star.print();
    // std::cout << "prop " << ll << " cur " << ll_cur << "\n";
    if (LogTargetBetas(beta_star, j) > u_s) break;
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    theta = theta_min + (theta_max - theta_min) * rng.uniform();//R::unif_rand();
    beta_star = beta_cur * cos(theta) + nu * sin(theta);
  } while (true);

  return beta_star;
}

void DMLinearReg::UpdateLatentVariables() {
  for (int j=0; j < d; j++) {
    // Compute the linear predictions
    alphas.col(j) = exp(X * betas.col(j));
    // Update (\lambda_{ij} | .)
    for (int i = 0; i < n; i++) {
      lambdas(i, j) = rng.gamma(Y(i, j) + alphas(i, j), 1.0) / (1.0 + phi(i));
      if (lambdas(i, j) < 1e-100) lambdas(i, j) = 1e-100;
    }
  }
  // Compute the \sum_{j=1}^d \lambda_{ij}
  rt = arma::sum(lambdas, 1);
  // Update \phi_i \sim Gamma[N_i, \sum_j \lambda_{ij}]
  for (int i=0; i < n; i++) phi(i) = rng.gamma(n_trials(i), 1.0) / rt(i);
}

void DMLinearReg::RunMCMC() {

  if (keep_draws) {
    // Initialise container for keep the draws
    draws_betas = arma::zeros<arma::cube>(p, d, ndpost);
    draws_varthetas = arma::zeros<arma::cube>(n, d, ndpost);
    draws_alphas = arma::zeros<arma::cube>(n, d, ndpost);
    //draws_phi = arma::zeros<arma::mat>(n, ndpost);
  }

  // Start MCMC
  std::cout << "Doing the warm-up (burn-in) of " << nskip << "\n\n";
  double progress = 0;
  for (int k=0; k < nskip; k++) {
    progress = (double) 100 * k / nskip;
    Rprintf("%3.2f%% Warm-up completed", progress);
    Rprintf("\r");
    // Update latent variables
    UpdateLatentVariables();
    // Update regression coefficients
    for (int j = 0; j < d; j++) {
      // Update \beta_{j} using ESS
      betas.col(j) = UpdateBetasESS(j);
    }
  }

  // Open file to write regression coefficient draws beta_{jk}
  std::ofstream ff(dir_draws + "/draws_betas.bin", std::ios::binary | std::ios::app);

  // Run the post-burn in iterations
  std::cout << "Starting post-burn-in iterations of " << ndpost << "\n\n";
  progress = 0;


  for (int k=0; k < ndpost; k++) {
    progress = (double) 100 * k / ndpost;
    Rprintf("%3.2f%% Posterior samples completed", progress);
    Rprintf("\r");

    // Update latent variables
    UpdateLatentVariables();

    // Iterate over the categories
    for (int j = 0; j < d; j++) {
      // Update \beta_{j} using ESS
      betas.col(j) = UpdateBetasESS(j);
    }

    // Save draws of the regression coefficients
    if (save_draws) {
      ff.write(reinterpret_cast<const char*>(betas.memptr()), sizeof(double)*p*d);
    }

    // Keep draws
    if (keep_draws) {
      draws_betas.slice(k) = betas;
      draws_varthetas.slice(k) = lambdas.each_col() / arma::sum(lambdas, 1);
      draws_alphas.slice(k) = alphas;
      //draws_phi.col(k) = phi;
    }
  }
  // Close file
  ff.close();
}

// Exposing a C++ class in R
//using namespace Rcpp;
RCPP_MODULE(dm_linear_reg) {

  // Expose class on the R side
  Rcpp::class_<DMLinearReg>("DMLinearReg")

  // Exposing constructor
  .constructor<arma::umat, arma::mat>()

  // Exposing member functions
  .method("SetMCMC", &DMLinearReg::SetMCMC)
  .method("RunMCMC", &DMLinearReg::RunMCMC)

  // Exposing some attributes
  .field("draws_betas", &DMLinearReg::draws_betas)
  .field("draws_phi", &DMLinearReg::draws_phi)
  .field("draws_abundance", &DMLinearReg::draws_varthetas)
  .field("draws_alphas", &DMLinearReg::draws_alphas)
  ;

}
