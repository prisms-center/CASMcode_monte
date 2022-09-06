#ifndef CASM_monte_RandomNumberGenerator
#define CASM_monte_RandomNumberGenerator

#include <random>
#include <vector>

#include "casm/external/MersenneTwister/MersenneTwister.h"

namespace CASM {
namespace monte {

/// \brief Random number generator with interface used for CASM::monte
template <typename EngineType = std::mt19937_64>
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

/// \brief A compatible random number engine using the original MTRand
struct MTRandEngine {
  typedef MTRand::uint32 result_type;
  MTRand mtrand;

  /// \brief Default constructor
  explicit MTRandEngine() {}

  /// \brief Construct with MTRand object
  explicit MTRandEngine(MTRand const &_mtrand) : mtrand(_mtrand) {}

  /// \brief Construct and seed with one integer
  MTRandEngine(result_type const one_seed) : mtrand(one_seed) {}

  /// \brief Construct and seed with an array of integers
  MTRandEngine(result_type *const big_seed,
               const result_type seed_length = MTRand::N)
      : mtrand(big_seed, seed_length) {}

  /// \brief Construct and seed with a SeedSequence type
  template <typename Sseq>
  MTRandEngine(Sseq &seq) {
    this->seed(seq);
  }

  static constexpr result_type min() { return result_type(0); }

  static constexpr result_type max() { return result_type(4294967295); }

  result_type operator()() { return mtrand.randInt(); }

  /// \brief Seed with the default
  void seed() { mtrand.seed(); }

  /// \brief Seed with one integer
  void seed(result_type const one_seed) { mtrand.seed(one_seed); }

  /// \brief Seed with an array of integers
  void seed(result_type *const big_seed,
            const result_type seed_length = MTRand::N) {
    mtrand.seed(big_seed, seed_length);
  }

  /// \brief Seed with a SeedSequence type
  template <typename Sseq>
  void seed(Sseq &seq) {
    std::vector<result_type> seeds(seq.size());
    seq.generate(seeds.begin(), seeds.end());
    mtrand.seed(seeds.data(), seeds.size());
  }
};

}  // namespace monte
}  // namespace CASM

#endif
