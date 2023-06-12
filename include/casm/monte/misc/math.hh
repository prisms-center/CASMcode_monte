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

inline double covariance(Eigen::VectorXd const &x, Eigen::VectorXd const &y,
                         double mean) {
  Index n = x.size();
  double cov = 0.0;
  for (Index i = 0; i < n; ++i) {
    cov += (x(i) - mean) * (y(i) - mean);
  }
  return cov / n;
}

inline double variance(Eigen::VectorXd const &x, double x_mean) {
  Index n = x.size();
  double cov = 0.0;
  for (Index i = 0; i < n; ++i) {
    double x_diff = x(i) - x_mean;
    cov += x_diff * x_diff;
  }
  return cov / n;
}

inline double variance(Eigen::VectorXd const &x) {
  return variance(x, x.mean());
}

inline double weighted_variance(Eigen::VectorXd const &x, double x_mean,
                                Eigen::VectorXd const &w, double w_sum) {
  Index n = x.size();
  double cov = 0.0;
  for (Index i = 0; i < n; ++i) {
    double d = x(i) - x_mean;
    cov += w(i) * d * d;
  }
  return cov / w_sum;
}

/// \brief Error function inverse
///
/// Notes:
/// - From "A handy approximation for the error function and its inverse" by
///   Sergei Winitzk
/// - Maximum relative error is about 0.00013.
inline double approx_erf_inv(double x) {
  const double one = 1.0;
  const double PI = 3.141592653589793238463;
  const double a = 0.147;

  double sgn = (x < 0.0) ? -one : one;
  double b = std::log((one - x) * (one + x));
  double c = 2.0 / (PI * a) + b * 0.5;
  double d = b / a;
  return sgn * std::sqrt(std::sqrt(c * c - d) - c);
}

}  // namespace monte
}  // namespace CASM

#endif
