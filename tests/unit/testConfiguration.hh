#ifndef CASM_unittest_testConfiguration
#define CASM_unittest_testConfiguration
#include <map>
#include <string>

#include "casm/global/eigen.hh"
#include "casm/monte/Conversions.hh"

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

/// Set config to random occupation
template <typename GeneratorType>
void random_config(test::Configuration &config,
                   CASM::monte::Conversions &convert,
                   GeneratorType &random_number_generator) {
  config.occupation.setZero();
  for (CASM::Index l = 0; l < config.occupation.size(); ++l) {
    int Nocc = convert.occ_size(convert.l_to_asym(l));
    config.occupation[l] = random_number_generator.random_int(Nocc - 1);
  }
}

/// Set a single non-default occupant
template <typename GeneratorType>
void dilute_config(test::Configuration &config,
                   CASM::monte::Conversions &convert,
                   GeneratorType &random_number_generator) {
  config.occupation.setZero();
  for (CASM::Index i = 0; i < convert.species_size(); ++i) {
    for (CASM::Index l = 0; l < config.occupation.size(); ++l) {
      CASM::Index asym = convert.l_to_asym(l);
      if (config.occupation[l] == 0 && convert.species_allowed(asym, i)) {
        config.occupation[l] = convert.occ_index(asym, i);
        break;
      }
    }
  }
}

}  // namespace test

#endif
