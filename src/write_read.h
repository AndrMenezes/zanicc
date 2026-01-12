#include "node.h"
#include <vector>
#include <iostream>
#include <fstream>

void serialise_tree(const Node *node, std::ostream &os, int &d);
Node* deserialise_tree(std::istream& is, int &d);
