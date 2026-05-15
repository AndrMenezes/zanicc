#ifndef ZANIMPFM_H
#define ZANIMPFM_H

double log_pmf_zanim(std::vector<int> x, std::vector<double> prob,
                     std::vector<double> zeta);
double log_pmf_zanim_approx(std::vector<int> x, std::vector<double> prob,
                            std::vector<double> zeta, double scale, int mc,
                            int nskip);
double log_pmf_zanidm(std::vector<int> x, std::vector<double> alpha,
                      std::vector<double> zeta);
double log_pmf_mult(std::vector<int> &x, int &size,
                           std::vector<double> &prob);
double log_pmf_zanim_conditional(std::vector<int> x, std::vector<double> prob,
                                 std::vector<double> zeta);
double log_pmf_zanim_ln_conditional(std::vector<int> &x, std::vector<double> &prob,
                                    std::vector<double> &zeta,
                                    std::vector<double> &chol_Sigma_V,
                                    std::vector<double> &B);
double log_pmf_zanim_ln(int mc, std::vector<int> &x, std::vector<double> &prob,
                        std::vector<double> &zeta,
                        std::vector<double> &chol_Sigma_V,
                        std::vector<double> &B);
double log_pmf_zanim_ln_augmented(std::vector<int> &lamx,
                                  std::vector<double> &z,
                                  std::vector<double> &zeta,
                                  std::vector<double> &lambda,
                                  std::vector<double> &u,
                                  double &phi);
double log_I_lc(std::vector<double> &x, std::vector<double> &mu,
                std::vector<double> &A, std::vector<double> &b,
                double eta);
#endif
