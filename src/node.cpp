#include <RcppArmadillo.h>
#include <vector>
#include "node.h"

// Constructor
// template<size_t T>
Node::Node(int np) : np(np) {
  left = nullptr;
  right = nullptr;
  parent = nullptr;
  // Node information
  is_root = 0;
  is_leaf = 1;
  depth = 1;
  mu = arma::vec(np, arma::fill::zeros);
  ss1 = arma::vec(np, arma::fill::zeros);
  ss2 = arma::vec(np, arma::fill::zeros);
}


// Destructor
// template<size_t T>
Node::~Node() {
  if (left != NULL) {
    delete left; left = nullptr;
    delete right; right = nullptr;
    //delete parent; parent = NULL;
  }
}

// Method to add leaves
// template<size_t T>
void Node::AddLeaves() {
  left = new Node(np);
  right = new Node(np);
  is_leaf = 0;
  is_root = 0;
  // Keep the pointer address of the parent
  left->parent = this;
  right->parent = this;
  // Update depth
  left->depth = depth + 1L;
  right->depth = depth + 1L;
  // Tree identifier
  right->h = h;
  left->h = h;
}

// Method to delete a leaf
// template<size_t T>
void Node::DeleteLeaves() {
  delete left;
  delete right;
  left = nullptr;
  right = nullptr;
  is_leaf = 1;
  predictor = 0;
  cutoff = 0;
}

// Get the leaves (use for grow)
// template<size_t T>
void Node::GetLeaves(std::vector<Node*> &leaves) {
  if (left) {
    left->GetLeaves(leaves);
    right->GetLeaves(leaves);
  } else leaves.push_back(this);
}

// Get the no grandchildren nodes (use for prune)
// template<size_t T>
void Node::GetNoG(std::vector<Node*> &nogs) {
  // Is there a child?
  if (left) {
    // Check for grand-children
    if ((left->left) || (right->left)) {
      if (left->left) left->GetNoG(nogs);
      if (right->left) right->GetNoG(nogs);
    } else nogs.push_back(this);
  }
}

// template<size_t T>
int Node::NNoG() {
  if (!left) return(0L);
  if (left->left || right->left) return(left->NNoG() + right->NNoG());
  else return(1L);
}

// template<size_t T>
int Node::NLeaves() {
  if (!left) return(1L);
  else return(left->NLeaves() + right->NLeaves());
}

// template<size_t T>
int Node::GetDepth(Node *root) {
  if (root == nullptr) {
    return 0;  // Base case: empty tree has depth 0
  }
  int l_d = GetDepth(root->left);
  int r_d = GetDepth(root->right);
  return 1 + std::max(l_d, r_d);
}


// NodeShared for Shared tree approach!
// Constructor
// NodeShared::NodeShared(int np) : Node(), np(np) {
//   mu = arma::vec(np, arma::fill::zeros);
//   ss1 = arma::vec(np, arma::fill::zeros);
// }
//
// NodeShared::~NodeShared() {};
//
// void NodeShared::AddLeaves() {
//   left = new NodeShared(np);
//   right = new NodeShared(np);
//   is_leaf = 0;
//   is_root = 0;
//   left->parent = this;
//   right->parent = this;
//   left->depth = depth + 1;
//   right->depth = depth + 1;
//   left->h = h;
//   right->h = h;
// }
