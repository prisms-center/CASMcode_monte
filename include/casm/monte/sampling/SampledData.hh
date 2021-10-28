#ifndef CASM_monte_SampledData
#define CASM_monte_SampledData

#include <vector>

#include "casm/global/eigen.hh"
#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

// Sampled data from a single run (constant conditions)
struct SampledData {
  // TODO: use Eigen::MatrixXd here?

  /// Map of <sampler name>:<sampler>
  std::map<std::string, std::shared_ptr<Sampler>> samplers;

  /// Vector of counts (could be pass or step) when a sample occurred
  std::vector<CountType> count;

  /// Vector of times when a sample occurred
  std::vector<TimeType> time;
};

/// \brief Last sampled_data.count value (else 0)
std::optional<CountType> get_count(SampledData const &sampled_data);

/// \brief Last sampled_data.time value (else 0)
std::optional<TimeType> get_time(SampledData const &sampled_data);

/// \brief Get Sampler::n_samples() value (assumes same for all)
/// (else 0)
CountType get_n_samples(SampledData const &sampled_data);

}  // namespace monte
}  // namespace CASM

// --- Inline implementations ---

namespace CASM {
namespace monte {

inline std::optional<CountType> get_count(SampledData const &sampled_data) {
  if (sampled_data.count.size()) {
    return sampled_data.count.back();
  }
  return std::nullopt;
}

inline std::optional<TimeType> get_time(SampledData const &sampled_data) {
  if (sampled_data.time.size()) {
    return sampled_data.time.back();
  }
  return std::nullopt;
}

inline CountType get_n_samples(SampledData const &sampled_data) {
  return get_n_samples(sampled_data.samplers);
}

}  // namespace monte
}  // namespace CASM

#endif
