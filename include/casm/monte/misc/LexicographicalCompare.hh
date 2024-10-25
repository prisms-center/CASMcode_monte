#ifndef CASM_monte_LexicographicalCompare
#define CASM_monte_LexicographicalCompare

#include <algorithm>

#include "casm/global/eigen.hh"
#include "casm/misc/CASM_Eigen_math.hh"

namespace CASM::monte {

/// \brief Lexicographically compare Eigen::VectorXi
struct LexicographicalCompare {
  bool operator()(Eigen::VectorXi const &A, Eigen::VectorXi const &B) const {
    return std::lexicographical_compare(A.data(), A.data() + A.size(), B.data(),
                                        B.data() + B.size());
  }

  bool operator()(Eigen::VectorXl const &A, Eigen::VectorXl const &B) const {
    return std::lexicographical_compare(A.data(), A.data() + A.size(), B.data(),
                                        B.data() + B.size());
  }
};

/// \brief Lexicographically compare Eigen::VectorXd
struct FloatLexicographicalCompare {
  double tol;

  FloatLexicographicalCompare(double tol) : tol(tol) {}

  bool operator()(Eigen::VectorXd const &A, Eigen::VectorXd const &B) const {
    return float_lexicographical_compare(A.data(), A.data() + A.size(),
                                         B.data(), B.data() + B.size(), tol);
  }
};

}  // namespace CASM::monte

#endif
