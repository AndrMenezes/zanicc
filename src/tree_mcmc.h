#ifndef TREE_MCMC_H
#define TREE_MCMC_H
#include "rng.h"
#include "node.h"

template<typename ModelType>
void Grow(Node *tree, ModelType &Model);
template<typename ModelType>
void Prune(Node *tree, ModelType &Model);
template<typename ModelType>
void Change(Node *tree, ModelType &Model);

double prob_tree_split(int depth, double a, double b);
double trans_prob_grow_prune(size_t n_leaves, size_t b_parents, double logprob_grow,
                             double logprob_prune);
double trans_prob_prune_grow(size_t n_leaves, size_t b_parents, double logprob_grow,
                             double logprob_prune);

#endif
