#ifndef CASM_monte_methods_metropolis
#define CASM_monte_methods_metropolis

#include <cmath>

#include "casm/external/MersenneTwister/MersenneTwister.h"

namespace CASM {
namespace monte {

/// \brief Metropolis acceptance method
///
/// A proposed event is accepted if:
/// - delta_potential_energy < 0.0,
/// - or rand in [0,1) < exp(-delta_potential_energy * beta)
///
/// \param delta_potential_energy The total (per_supercell) change in potential
///     energy due to the proposed event
/// \param beta Thermodynamic beta, equals 1.0 / (CASM::KB * temperature)
/// \param random_number_generator Random number generator
///
/// \returns true, if event should be accepted; false, if the event should be
///     rejected
///
template <typename GeneratorType>
bool metropolis_acceptance(double delta_potential_energy, double beta,
                           GeneratorType &random_number_generator) {
  if (delta_potential_energy < 0.0) {
    return true;
  }

  double rand = random_number_generator.random_real(1.0);
  double prob = std::exp(-delta_potential_energy * beta);
  return rand < prob;
}

}  // namespace monte
}  // namespace CASM

#endif
