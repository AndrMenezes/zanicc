#include <RcppArmadillo.h>
#include "tree_mcmc.h"
#include "multinomial_bart.h"
#include "probit_bart.h"

double prob_tree_split(int depth, double a, double b) {
  return(a * pow(1.0 + depth, -b));
}

double trans_prob_grow_prune(size_t n_leaves, size_t b_parents, double logprob_grow,
                             double logprob_prune) {
  return(logprob_prune - log(1.0 + b_parents) - logprob_grow - log(n_leaves));
}

double trans_prob_prune_grow(size_t n_leaves, size_t b_parents, double logprob_grow,
                             double logprob_prune) {
  return(logprob_grow - log(n_leaves - 1.0) - logprob_prune - log(b_parents));
}

template<typename ModelType>
void Grow(Node *tree, ModelType &Model) {
  // Get the leaves
  std::vector<Node*> leaves;
  tree->GetLeaves(leaves);
  int b_parents = tree->NNoG();
  // Compute transition ratio
  double lr = trans_prob_grow_prune(leaves.size(), b_parents, Model.logprob_grow,
                                    Model.logprob_prune);
  // Sample a leave
  Node *leaf_to_split = leaves[sample_discrete(leaves.size())];
  arma::uvec old_ids = leaf_to_split->ids;
  // Sample a covariate
  int j = sample_discrete(Model.splitprobs, Model.p);
  arma::vec xj = Model.X.col(j);
  xj = xj(old_ids);
  // Sample from pre-defined cutpoints
  arma::vec xj_cuts = Model.x_breaks.col(j);
  int k = sample_discrete(Model.numcut);
  double cutoff = xj_cuts(k);
  // Split the observation ids
  arma::uvec go_left = (xj <= cutoff);
  int n_left = arma::accu(go_left);
  int n_right = leaf_to_split->nobs - n_left;
  arma::uvec ids_left = arma::zeros<arma::uvec>(n_left);
  arma::uvec ids_right = arma::zeros<arma::uvec>(n_right);
  int idx_left = 0, idx_right = 0;
  for (int k = 0; k < leaf_to_split->nobs; k++) {
    if (go_left[k]) ids_left[idx_left++] = old_ids[k];
    else ids_right[idx_right++] = old_ids[k];
  }
  // Exit the grow in case left OR right have less than 5 obs
  if (n_left < 5 || n_right < 5) return;
  // Compute the log-marginal likelihood and the log-tree-prior before growing
  double leaf_prior = prob_tree_split(leaf_to_split->depth, Model.base, Model.power);
  lr -= Model.lml(leaf_to_split) + log(1.0 - leaf_prior);
  // Add the leaves
  leaf_to_split->AddLeaves();
  leaf_to_split->predictor = j;
  leaf_to_split->cutoff = cutoff;
  // Don't erase the ids even if the leaf becomes a internal node.
  leaf_to_split->left->nobs = n_left;
  leaf_to_split->right->nobs = n_right;
  leaf_to_split->left->ids = ids_left;
  leaf_to_split->right->ids = ids_right;
  // Update the sufficient statistics for the two new leaves
  Model.UpdateSuffStats(leaf_to_split->left);
  Model.UpdateSuffStats(leaf_to_split->right);
  // Compute log-likelihood and log-prior after
  lr += (Model.lml(leaf_to_split->left) + Model.lml(leaf_to_split->right)
          + log(1.0 - leaf_prior)
          + log(1.0 - prob_tree_split(leaf_to_split->right->depth, Model.base, Model.power))
          + log(1.0 - prob_tree_split(leaf_to_split->right->depth, Model.base, Model.power)));
  //std::cout << "Grow lr: " << lr << "\n";
  // Perform the MH step
  Model.flag_grow = 1;
  if (log(R::unif_rand()) > lr) {
    //std::cout << "Grow rejected \n";
    Model.flag_grow = 0;
    leaf_to_split->DeleteLeaves();
  }
}

template<typename ModelType>
void Prune(Node *tree, ModelType &Model) {
  std::vector<Node*> nogs;
  tree->GetNoG(nogs);
  int n_leaves = tree->NLeaves();
  // Compute transition ratio
  double lr = trans_prob_prune_grow(n_leaves, nogs.size(), Model.logprob_grow,
                                    Model.logprob_prune);
  // Sample a nog
  Node *node_to_prune = nogs[sample_discrete(nogs.size())];
  arma::uvec old_ids = node_to_prune->ids;
  //Compute the likelihood and tree prior before prune the tree
  double internal_prior = prob_tree_split(node_to_prune->depth, Model.base, Model.power);
  double leaf_prior = prob_tree_split(node_to_prune->left->depth, Model.base, Model.power);
  lr -= (Model.lml(node_to_prune->left) + Model.lml(node_to_prune->right)
           + log(1.0 - leaf_prior) + log(1.0 - leaf_prior) + log(internal_prior));
  // Update node to prune
  Model.UpdateSuffStats(node_to_prune);
  // Compute log-likelihood and prior after movement
  lr += Model.lml(node_to_prune) + log(1.0 - internal_prior);
  //std::cout << "Prune lr: " << lr << "\n";
  Model.flag_prune = 0;
  if (log(R::unif_rand()) < lr) {
    //std::cout << "Prune accepted \n";
    node_to_prune->DeleteLeaves();
    Model.flag_prune = 1;
  }
}

template<typename ModelType>
void Change(Node *tree, ModelType &Model) {
  // Sample an available node to change the decision rule
  std::vector<Node*> nogs;
  tree->GetNoG(nogs);
  // Sample a leave
  Node* node_to_change = nogs[sample_discrete(nogs.size())];
  arma::uvec old_ids = node_to_change->ids;
  // Copy the old_ids for left and right;
  arma::uvec ids_left_old = node_to_change->left->ids;
  arma::uvec ids_right_old = node_to_change->right->ids;
  arma::vec ss1_left_old = node_to_change->left->ss1;
  arma::vec ss1_right_old = node_to_change->right->ss1;
  arma::vec ss2_left_old = node_to_change->left->ss2;
  arma::vec ss2_right_old = node_to_change->right->ss2;
  // Sample a covariate
  int j = sample_discrete(Model.splitprobs, Model.p);
  arma::vec xj = Model.X.col(j);
  xj = xj(old_ids);
  // Sample from pre-defined cutpoints
  arma::vec xj_cuts = Model.x_breaks.col(j);
  int k = sample_discrete(Model.numcut);
  double cutoff = xj_cuts(k);
  // Split the ids observation
  arma::uvec go_left = (xj <= cutoff);
  int n_left = arma::accu(go_left);
  int n_right = node_to_change->nobs - n_left;
  arma::uvec ids_left = arma::zeros<arma::uvec>(n_left);
  arma::uvec ids_right = arma::zeros<arma::uvec>(n_right);
  int idx_left=0, idx_right=0;
  for (int k = 0; k < node_to_change->nobs; k++) {
    if (go_left[k]) ids_left[idx_left++] = old_ids[k];
    else ids_right[idx_right++] = old_ids[k];
  }
  // Exit the change in case left OR right have less than 5 observations
  if (n_left < 5 || n_right < 5) return;
  double lr = -Model.lml(node_to_change->left) - Model.lml(node_to_change->right);
  // Update the sufficient statistics for the two new leaves
  node_to_change->left->nobs = n_left;
  node_to_change->right->nobs = n_right;
  node_to_change->left->ids = ids_left;
  node_to_change->right->ids = ids_right;
  Model.UpdateSuffStats(node_to_change->left);
  Model.UpdateSuffStats(node_to_change->right);
  // Compute log-likelihood and log-prior after
  lr += Model.lml(node_to_change->left) + Model.lml(node_to_change->right);
  //std::cout << "Change lr: " << lr << "\n";
  if (log(R::unif_rand()) < lr) {
    //std::cout << "Change accepted \n";
    // Update the decision rule
    Model.flag_change = 1;
    node_to_change->predictor = j;
    node_to_change->cutoff = cutoff;
    // Model.ar_change[tree->h]++;
  } else {
    Model.flag_change = 0;
    //std::cout << "Change rejected \n";
    // Get back to the old  left and right children
    //node_to_change->predictor = j_old;
    node_to_change->left->ids = ids_left_old;
    node_to_change->left->nobs = ids_left_old.size();
    node_to_change->left->ss1 = ss1_left_old;
    node_to_change->left->ss2 = ss2_left_old;
    node_to_change->right->ids = ids_right_old;
    node_to_change->right->nobs = ids_right_old.size();
    node_to_change->right->ss1 = ss1_right_old;
    node_to_change->right->ss2 = ss2_right_old;
  }

};

template void Grow<MultinomialBART>(Node*, MultinomialBART&);
template void Prune<MultinomialBART>(Node*, MultinomialBART&);
template void Change<MultinomialBART>(Node*, MultinomialBART&);

template void Grow<ProbitBART>(Node*, ProbitBART&);
template void Prune<ProbitBART>(Node*, ProbitBART&);
template void Change<ProbitBART>(Node*, ProbitBART&);
