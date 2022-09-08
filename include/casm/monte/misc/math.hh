#ifndef CASM_monte_misc_math
#define CASM_monte_misc_math

#include "casm/global/definitions.hh"
#include "casm/global/eigen.hh"

namespace CASM {
namespace monte {

inline double covariance(Eigen::VectorXd const &x, Eigen::VectorXd const &y) {
  Index n = x.size();
  double x_mean = x.mean();
  double y_mean = y.mean();
  double cov = 0.0;
  for (Index i = 0; i < n; ++i) {
    cov += (x(i) - x_mean) * (y(i) - y_mean);
  }
  return cov / n;
}

inline double variance(Eigen::VectorXd const &x) {
  Index n = x.size();
  double x_mean = x.mean();
  double cov = 0.0;
  for (Index i = 0; i < n; ++i) {
    double x_diff = x(i) - x_mean;
    cov += x_diff * x_diff;
  }
  return cov / n;
}

}  // namespace monte
}  // namespace CASM

#endif
