#include <RcppArmadillo.h>

//
// arma::cube save_csv(int n, int d, int m) {
//   std::ofstream fout("output.csv", std::ios::app);
//   arma::cube out = arma::zeros<arma::cube>(n, d, m);
//   for (int i = 0; i < m; ++i) {
//     arma::mat matrix = arma::randu(n, d);
//     out.slice(i)  = matrix;
//     matrix.raw_print(fout, "");  // omit header
//   }
//   fout.close();
//   return out;
// }

// [[Rcpp::export]]
void save_csv(int n, int d, int m, std::string fname) {
  std::ofstream fout(fname, std::ios::app);
  std::stringstream buffer;
  arma::mat matrix = arma::zeros<arma::mat>(n, d);
  for (int i = 0; i < m; ++i) {
    matrix = arma::randu(n, d);
    for (size_t r = 0; r < n; r++) {
      for (size_t c = 0; c < d; c++) {
        fout << matrix(r, c) << " ";
      }
      fout << "\n";
    }
    fout << buffer.str();
  }
}


// [[Rcpp::export]]
arma::cube save_bin(int n, int d, int m, std::string fname) {
  //fname = fname + "_draws";
  // open in binary append mode
  std::ofstream fout(fname, std::ios::app | std::ios::binary);
  arma::mat matrix;
  arma::cube out = arma::zeros<arma::cube>(n, d, m);
  double size = sizeof(double) * n * d;
  for (int i = 0; i < m; i++) {
    matrix = arma::randu(n, d);
    out.slice(i) = matrix;
    // write to disk in binary format
    fout.write(reinterpret_cast<const char*>(matrix.memptr()), size);
  }
  fout.close();
  return out;
}
