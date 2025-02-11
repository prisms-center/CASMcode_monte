#ifndef CASM_monte_RandomNumberGenerator
#define CASM_monte_RandomNumberGenerator

#include <random>
#include <vector>

#include "casm/monte/MTRandEngine.hh"
#include "casm/monte/definitions.hh"

namespace CASM {
namespace monte {

/// \brief Random number generator with interface used for CASM::monte
template <typename EngineType = default_engine_type>
struct RandomNumberGenerator {
  std::shared_ptr<EngineType> engine;

  /// Constructor, automatically construct and seed from random device if engine
  /// is empty
  RandomNumberGenerator(
      std::shared_ptr<EngineType> _engine = std::shared_ptr<EngineType>())
      : engine(_engine) {
    if (this->engine == nullptr) {
      this->engine = std::make_shared<EngineType>();
      std::random_device device;
      engine->seed(device());
    }
  }

  /// \brief Return uniformly distributed integer in [0, maximum_value]
  template <typename IntType>
  IntType random_int(IntType maximum_value) {
    return std::uniform_int_distribution<IntType>(0, maximum_value)(*engine);
  }

  /// \brief Return uniformly distributed floating point value in [0,
  /// maximum_value)
  template <typename RealType>
  RealType random_real(RealType maximum_value) {
    return std::uniform_real_distribution<RealType>(0, maximum_value)(*engine);
  }
};

}  // namespace monte
}  // namespace CASM

#endif
