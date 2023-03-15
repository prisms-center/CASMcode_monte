#ifndef CASM_monte_SamplingParams
#define CASM_monte_SamplingParams

#include <vector>

#include "casm/monte/definitions.hh"

namespace CASM {
namespace monte {

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
  /// For SAMPLE_METHOD::LINEAR, take the n-th sample when:
  ///
  ///    sample/pass = round( begin + (period / samples_per_period) * n )
  ///           time = begin + (period / samples_per_period) * n
  ///
  /// For SAMPLE_METHOD::LOG, take the n-th sample when:
  ///
  ///    sample/pass = round( begin + period ^ ( (n + shift) /
  ///                      samples_per_period ) )
  ///           time = begin + period ^ ( (n + shift) / samples_per_period )
  ///
  SAMPLE_METHOD sample_method;

  // --- Parameters for determining when samples are taken ---

  /// \brief See `sample_method`
  double begin;

  /// \brief See `sample_method`
  double period;

  /// \brief See `sample_method`
  double samples_per_period;

  /// \brief See `sample_method`
  double shift;

  /// \brief What quantities to sample
  ///
  /// These name must match StateSamplingFunction names.
  ///
  /// Default={}
  std::vector<std::string> sampler_names;

  /// \brief If true, save the configuration when a sample is taken
  ///
  /// Default=false
  bool do_sample_trajectory;

  /// \brief If true, save current time when taking a sample
  ///
  /// Default=false
  bool do_sample_time;
};

// /// The pass/step/time when a particular sample should be taken
// double sample_at(SamplingParams const &sampling_params, CountType
// sample_index);
//
// /// \brief Returns true if samples should be taken - count based sampling
// bool sample_is_due(SamplingParams const &sampling_params,
//                    CountType sample_index, CountType count);
//
// /// \brief Returns true if samples should be taken - time based sampling
// bool sample_is_due(SamplingParams const &sampling_params,
//                    CountType sample_index, TimeType time);

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
/// - do_sample_trajectory=false
/// - do_sample_time=false
inline SamplingParams::SamplingParams()
    : sample_mode(SAMPLE_MODE::BY_PASS),
      sample_method(SAMPLE_METHOD::LINEAR),
      begin(0.0),
      period(1.0),
      samples_per_period(1.0),
      shift(0.0),
      sampler_names({}),
      do_sample_trajectory(false),
      do_sample_time(false) {}

// /// The pass/step/time when a particular sample should be taken
// inline double sample_at(SamplingParams const &sampling_params,
//                         CountType sample_index) {
//   SamplingParams const &s = sampling_params;
//   double n = static_cast<double>(sample_index);
//   double value;
//   if (s.sample_method == SAMPLE_METHOD::LINEAR) {
//     value = s.begin + (s.period / s.samples_per_period) * n;
//   } else /* sample_method == SAMPLE_METHOD::LOG */ {
//     value = s.begin + std::pow(s.period, (n + s.shift) /
//     s.samples_per_period);
//   }
//   return value;
// }
//
// /// \brief Returns true if samples should be taken - count based sampling
// inline bool sample_is_due(SamplingParams const &sampling_params,
//                           CountType sample_index, CountType count) {
//   return count == static_cast<CountType>(
//                       std::round(sample_at(sampling_params, sample_index)));
// }
//
// /// \brief Returns true if samples should be taken - time based sampling
// inline bool sample_is_due(SamplingParams const &sampling_params,
//                           CountType sample_index, TimeType time) {
//   return time >= sample_at(sampling_params, sample_index);
// }
//
//
// /// The pass/step/time when a particular sample should be taken
// inline double sample_at(SAMPLE_METHOD sample_method,
//                         double begin,
//                         double period,
//                         double samples_per_period,
//                         double shift,
//                         CountType sample_index) {
//   SamplingParams const &s = sampling_params;
//   double n = static_cast<double>(sample_index);
//   double value;
//   if (sample_method == SAMPLE_METHOD::LINEAR) {
//     value = begin + (period / samples_per_period) * n;
//   } else /* sample_method == SAMPLE_METHOD::LOG */ {
//     value = begin + std::pow(period, (n + shift) / samples_per_period);
//   }
//   return value;
// }
//
// /// \brief Returns true if samples should be taken - count based sampling
// inline bool sample_is_due(SamplingParams const &sampling_params,
//                           CountType sample_index, CountType count) {
//   return count == static_cast<CountType>(
//                       std::round(sample_at(sampling_params, sample_index)));
// }
//
// /// \brief Returns true if samples should be taken - time based sampling
// inline bool sample_is_due(SamplingParams const &sampling_params,
//                           CountType sample_index, TimeType time) {
//   return time >= sample_at(sampling_params, sample_index);
// }

}  // namespace monte
}  // namespace CASM

#endif
