#ifndef CASM_monte_ValueMap
#define CASM_monte_ValueMap

#include <map>
#include <string>

#include "casm/global/eigen.hh"
#include "casm/monte/definitions.hh"

namespace CASM {
namespace monte {

/// Map of value name to scalar/vector/matrix value
struct ValueMap {
  std::map<std::string, bool> boolean_values;
  std::map<std::string, double> scalar_values;
  std::map<std::string, Eigen::VectorXd> vector_values;
  std::map<std::string, Eigen::MatrixXd> matrix_values;
};

/// \brief Return true if A and B do not have the same properties
bool is_mismatched(ValueMap const &A, ValueMap const &B);

/// \brief Return values + n_increment*increment
ValueMap make_incremented_values(ValueMap values, ValueMap const &increment,
                                 double n_increment);

// --- Implementation ---

/// \brief Return true if A and B do not have the same properties
inline bool is_mismatched(ValueMap const &A, ValueMap const &B) {
  for (auto const &pair : B.boolean_values) {
    if (!A.boolean_values.count(pair.first)) {
      return true;
    }
  }
  for (auto const &pair : B.scalar_values) {
    if (!A.scalar_values.count(pair.first)) {
      return true;
    }
  }
  for (auto const &pair : B.vector_values) {
    if (!A.vector_values.count(pair.first)) {
      return true;
    }
  }
  for (auto const &pair : B.matrix_values) {
    if (!A.matrix_values.count(pair.first)) {
      return true;
    }
  }
  return false;
}

/// \brief Return values + n_increment*increment
///
/// Note: does not change boolean_values
inline ValueMap make_incremented_values(ValueMap values,
                                        ValueMap const &increment,
                                        double n_increment) {
  for (auto const &pair : increment.scalar_values) {
    values.scalar_values.at(pair.first) += pair.second * n_increment;
  }
  for (auto const &pair : increment.vector_values) {
    values.vector_values.at(pair.first) += pair.second * n_increment;
  }
  for (auto const &pair : increment.matrix_values) {
    values.matrix_values.at(pair.first) += pair.second * n_increment;
  }
  return values;
}

}  // namespace monte
}  // namespace CASM

#endif
