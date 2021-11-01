#ifndef CASM_monte_SamplingParams
#define CASM_monte_SamplingParams

#include <vector>

#include "casm/monte/definitions.hh"

namespace CASM {
namespace monte {

/// How often to sample runs
enum class SAMPLE_MODE { BY_STEP, BY_PASS, BY_TIME };

/// How to sample by time
enum class SAMPLE_METHOD { LINEAR, LOG };

/// What to sample and how
struct SamplingParams {
  /// Default constructor
  SamplingParams();

  /// \brief Sample by step, pass, or time
  ///
  /// Default=SAMPLE_MODE::BY_PASS
  SAMPLE_MODE sample_mode;

  /// \brief Sample linearly or logarithmically
  ///
  /// Default=SAMPLE_METHOD::LINEAR
  ///
  /// Parameters for determining when samples are taken by count
  ///
  /// For SAMPLE_METHOD::LINEAR, take the n-th sample when:
  ///
  ///    sample/pass = ceil( begin + (period / samples_per_period) * n )
  ///           time = begin + (period / samples_per_period) * n
  ///
  /// For SAMPLE_METHOD::LOG, take the n-th sample when:
  ///
  ///    sample/pass = ceil( begin + period ^ ( (n + shift) /
  ///                      samples_per_period ) )
  ///           time = begin + period ^ ( (n + shift) / samples_per_period )
  ///
  SAMPLE_METHOD sample_method;

  // --- Parameters for determining when samples are taken by count ---

  double begin;
  double period;
  double samples_per_period;
  double shift;

  /// \brief What to sample
  ///
  /// Default={}
  std::vector<std::string> sampler_names;

  /// \brief Whether to sample the complete configuration
  ///
  /// Default=false
  bool sample_trajectory;
};

/// The pass/step/time when a particular sample is due
double sample_due(SamplingParams const &sampling_params,
                  CountType sample_index);

}  // namespace monte
}  // namespace CASM

// --- Inline implementations ---

namespace CASM {
namespace monte {

/// Default constructor
///
/// Default values are:
/// - sample_mode=SAMPLE_MODE::BY_PASS
/// - sample_method=SAMPLE_METHOD::LINEAR
/// - begin=0.0
/// - period=1.0
/// - samples_per_period=1.0
/// - shift=0.0
/// - sampler_names={}
/// - sample_trajectory=false
inline SamplingParams::SamplingParams()
    : sample_mode(SAMPLE_MODE::BY_PASS),
      sample_method(SAMPLE_METHOD::LINEAR),
      begin(0.0),
      period(1.0),
      samples_per_period(1.0),
      shift(0.0),
      sampler_names({}),
      sample_trajectory(false) {}

/// The pass/step/time when a particular sample is due
inline double sample_due(SamplingParams const &sampling_params,
                         CountType sample_index) {
  SamplingParams const &s = sampling_params;
  double n = static_cast<double>(sample_index);
  double value;
  if (s.sample_method == SAMPLE_METHOD::LINEAR) {
    value = s.begin + (s.period / s.samples_per_period) * n;
  } else /* sample_method == SAMPLE_METHOD::LOG */ {
    value = s.begin + std::pow(s.period, (n + s.shift) / s.samples_per_period);
  }
  return value;
}

}  // namespace monte
}  // namespace CASM

#endif
