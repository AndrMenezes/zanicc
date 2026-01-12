#ifndef ZANIMPFM_H
#define ZANIMPFM_H

double log_pmf_zanim(std::vector<int> x, std::vector<double> prob,
                     std::vector<double> zeta);
double log_pmf_zanim_approx(std::vector<int> x, std::vector<double> prob,
                            std::vector<double> zeta, double scale, int mc,
                            int nskip);
double log_pmf_zanidm(std::vector<int> x, std::vector<double> alpha,
                      std::vector<double> zeta);
#endif
