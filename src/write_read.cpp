#include "write_read.h"
#include "node.h"

// (de)-serialise tree object in binary format using the depth-first-search
// with pre-order traverse.

void serialise_tree(const Node *node, std::ostream &os, int &d) {
  bool valid = node != nullptr;
  os.write(reinterpret_cast<const char*>(&valid), sizeof(valid));
  if (!valid) return;
  os.write(reinterpret_cast<const char*>(&node->is_leaf), sizeof(node->is_leaf));
  if (node->is_leaf) {
    os.write(reinterpret_cast<const char*>(node->mu.memptr()), sizeof(double) * d);
  } else {
    os.write(reinterpret_cast<const char*>(&node->predictor), sizeof(node->predictor));
    os.write(reinterpret_cast<const char*>(&node->cutoff), sizeof(node->cutoff));
  }
  serialise_tree(node->left, os, d);
  serialise_tree(node->right, os, d);
}

Node* deserialise_tree(std::istream &is, int &d) {
  bool valid;
  is.read(reinterpret_cast<char*>(&valid), sizeof(valid));
  if (!valid) return nullptr;
  Node* node = new Node(d);
  is.read(reinterpret_cast<char*>(&node->is_leaf), sizeof(node->is_leaf));
  if (node->is_leaf) {
    is.read(reinterpret_cast<char*>(node->mu.memptr()), sizeof(double) * d);
  } else {
    is.read(reinterpret_cast<char*>(&node->predictor), sizeof(node->predictor));
    is.read(reinterpret_cast<char*>(&node->cutoff), sizeof(node->cutoff));
  }
  node->left = deserialise_tree(is, d);
  node->right = deserialise_tree(is, d);
  return node;
}

// Aux function to print the tree as an array
void print_tree(Node* node) {
  if (!node) {
    std::cout << "# ";
    return;
  }
  std::cout << node->is_leaf << " ";
  if (node->is_leaf) {
    std::cout << node->mu << " ";
  } else {
    std::cout << node->predictor << " " << node->cutoff << " ";
  }
  print_tree(node->left);
  print_tree(node->right);
}
