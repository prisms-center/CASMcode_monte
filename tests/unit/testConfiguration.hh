#ifndef CASM_unittest_testConfiguration
#define CASM_unittest_testConfiguration
#include <map>
#include <string>

#include "casm/global/eigen.hh"

namespace test {

/// \brief Minimal configuration for testing purposes
///
/// Notes:
/// - currently occupation only
struct Configuration {
  Configuration(CASM::Index n_sublat,
                Eigen::Matrix3l const &transformation_matrix_to_super) {
    CASM::Index volume = transformation_matrix_to_super.determinant();
    occupation.resize(n_sublat * volume);
    occupation.setZero();
  }

  Eigen::VectorXi occupation;
};

}  // namespace test

#endif
