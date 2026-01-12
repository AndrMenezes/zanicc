#ifndef NODE_H
#define NODE_H
#include <RcppArmadillo.h>
#include <vector>

// template<size_t T>
struct Node {
  // Constructor and destructor
  Node(int np);
  ~Node();

  int np;
  int is_root;
  int is_leaf;
  int depth;

  // Left and right Nodes
  Node *left;
  Node *right;
  // Parent Node
  Node *parent;

  // Tree identifier
  int h;

  // Partition Node information
  int predictor;
  double cutoff;

  // Number of observations and their indices in the current Node, (if terminal)
  int nobs;
  arma::uvec ids;

  // Methods
  void AddLeaves();
  void DeleteLeaves();

  // Get a vector of pointers to the bottom nodes
  void GetLeaves(std::vector<Node*> &leaves);

  // Get a vector of pointers to the nodes that has no grandchildren
  void GetNoG(std::vector<Node*> &nogs);

  int NNoG();
  int NLeaves();
  int GetDepth(Node* root);

  // Posterior draw
  arma::vec mu;
  // Sufficient statistics (ss1 = r and ss2 = s)
  arma::vec ss1;
  arma::vec ss2;
  // double ss2;

};

// Node structure for shared tree approach
// struct NodeShared : public Node {
//   int np;
//   // Constructor and destructor
//   NodeShared(int np);
//   ~NodeShared();
//   arma::vec mu;
//   arma::vec ss1;
//   void AddLeaves();
// };


#endif
