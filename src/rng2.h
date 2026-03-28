#include <boost/random.hpp>
#include <random>
#include <cmath>

class RNG2 {
private:
  // the generator of pseudo-rng method: see Table 31.6. generators
  boost::random::mt19937 gen;
  // distributions
  boost::random::uniform_real_distribution<double> uniform_dist;
  boost::random::normal_distribution<double> normal_dist;

public:
  // constructor (as default seed uses system entropy (random_device()))
  RNG2(unsigned int seed = std::random_device{}()) :
  gen(seed), uniform_dist(0.0, 1.0), normal_dist(0.0, 1.0) {}
  ~RNG2(){};

  // methods
  double uniform() {return uniform_dist(gen);}
  double normal() {return normal_dist(gen);}
  double normal(double loc, double scale) {return loc + scale*normal_dist(gen);}
  double gamma(double shape, double scale) {
    boost::random::gamma_distribution<double> gamma_dist(shape, scale);
    return gamma_dist(gen);
  }

};
