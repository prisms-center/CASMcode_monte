#ifndef CASM_monte_MTRandEngine
#define CASM_monte_MTRandEngine

#include <random>
#include <vector>

#include "casm/external/MersenneTwister/MersenneTwister.h"

namespace CASM {
namespace monte {

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

  // stream insertion operator:
  friend std::ostream &operator<<(std::ostream &os, MTRandEngine const &e) {
    os << e.mtrand;
    return os;
  }

  // stream extraction operator:
  friend std::istream &operator>>(std::istream &is, MTRandEngine &e) {
    is >> e.mtrand;
    return is;
  }
};

}  // namespace monte
}  // namespace CASM

#endif
