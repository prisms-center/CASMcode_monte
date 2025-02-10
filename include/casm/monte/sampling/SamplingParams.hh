#ifndef CASM_monte_SamplingParams
#define CASM_monte_SamplingParams

#include <utility>
#include <vector>

#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/definitions.hh"

namespace CASM {
namespace monte {

/// What to sample and how
struct SamplingParams {
  /// Default constructor
  SamplingParams();

  /// \brief What quantities to sample
  ///
  /// These name must match StateSamplingFunction names.
  ///
  /// Default={}
  std::vector<std::string> sampler_names;

  /// \brief What quantities to sample as JSON output
  ///
  /// These name must match jsonStateSamplingFunction names.
  ///
  /// Default={}
  std::vector<std::string> json_sampler_names;

  /// \brief Sample by step, pass, or time
  ///
  /// Default=SAMPLE_MODE::BY_PASS
  SAMPLE_MODE sample_mode;

  /// \brief Sample linearly or logarithmically
  ///
  /// Default=SAMPLE_METHOD::LINEAR
  ///
  /// For SAMPLE_METHOD::LINEAR, take the n-th sample (n=0, 1, 2, ...) when:
  ///
  ///    sample/pass = round( begin + period * n )
  ///           time = begin + period * n
  ///
  /// For SAMPLE_METHOD::LOG, take the n-th sample when:
  ///
  ///    sample/pass = round( begin + base ^ ( n + shift )
  ///           time = begin + base ^ ( n + shift )
  ///
  /// For SAMPLE_METHOD::CUSTOM, take the n-th sample when:
  ///
  ///    sample/pass = round( custom_sample_at(n) )
  ///           time = custom_sample_at(n)
  ///
  /// If stochastic_sample_period == true, then instead of setting the sample
  /// time / count deterministically, use the sampling period to determine the
  /// sampling rate and determine the next sample time / count stochastically.
  ///
  SAMPLE_METHOD sample_method;

  // --- Parameters for determining when samples are taken ---

  /// \brief See `sample_method`
  double period;

  /// \brief See `sample_method`
  double begin;

  /// \brief See `sample_method`
  double base;

  /// \brief See `sample_method`
  double shift;

  /// \brief Custom sample spacing function
  std::function<double(CountType)> custom_sample_at;

  /// \brief See `sample_method`
  bool stochastic_sample_period;

  /// \brief If true, save the configuration when a sample is taken
  ///
  /// Default=false
  bool do_sample_trajectory;

  /// \brief If true, save current time when taking a sample
  ///
  /// Default=false
  bool do_sample_time;
};

/// \brief Return the count / time when the sample_index-th sample should be
///     taken
double sample_at(CountType sample_index, SamplingParams const &sampling_params);

/// \brief Stochastically determine how many steps or passes
///     until the next sample
template <typename EngineType>
CountType stochastic_count_step(
    double sample_rate,
    monte::RandomNumberGenerator<EngineType> &random_number_generator);

/// \brief Stochastically determine much time
///     until the next sample
template <typename EngineType>
TimeType stochastic_time_step(
    TimeType sample_rate,
    monte::RandomNumberGenerator<EngineType> &random_number_generator);

/// \brief Return the count / time when the sample_index-th sample should be
///     taken
template <typename EngineType>
double stochastic_sample_at(
    CountType sample_index, SamplingParams const &sampling_params,
    monte::RandomNumberGenerator<EngineType> &random_number_generator,
    std::vector<CountType> const &sample_count,
    std::vector<TimeType> const &sample_time);

}  // namespace monte
}  // namespace CASM

// --- Inline implementations ---

namespace CASM {
namespace monte {

/// Default constructor
///
/// Default values are:
/// - sampler_names={}
/// - json_sampler_names={}
/// - sample_mode=SAMPLE_MODE::BY_PASS
/// - sample_method=SAMPLE_METHOD::LINEAR
/// - begin=1.0
/// - period=1.0
/// - base=std::pow(10.0,1.0/10.0)
/// - shift=10.0
/// - stochastic_sample_period=false
/// - do_sample_trajectory=false
/// - do_sample_time=false
inline SamplingParams::SamplingParams()
    : sampler_names({}),
      json_sampler_names({}),
      sample_mode(SAMPLE_MODE::BY_PASS),
      sample_method(SAMPLE_METHOD::LINEAR),
      begin(1.0),
      period(1.0),
      base(std::pow(10.0, 1.0 / 10.0)),
      shift(10.0),
      stochastic_sample_period(false),
      do_sample_trajectory(false),
      do_sample_time(false) {}

/// \brief Return the count / time when the sample_index-th sample should be
///     taken
///
/// \param sample_index Index for sample (0, 1, 2, ...)
/// \param sampling_params Sampling method parameters
/// \return The time or count at which the sample_index-th sample should be
///     taken, as a double
///
inline double sample_at(CountType sample_index,
                        SamplingParams const &sampling_params) {
  SamplingParams const &s = sampling_params;

  double n = static_cast<double>(sample_index);
  if (s.sample_method == SAMPLE_METHOD::LINEAR) {
    return s.begin + s.period * n;
  } else if (s.sample_method == SAMPLE_METHOD::LOG) {
    return s.begin + std::pow(s.base, (n + s.shift));
  } else {
    if (!s.custom_sample_at) {
      throw std::runtime_error(
          "Error in sample_at: sample_method==SAMPLE_METHOD::CUSTOM and "
          "!custom_sample_at");
    }
    return s.custom_sample_at(n);
  }
}

/// \brief Stochastically determine how many steps or passes
///     until the next sample
///
/// \tparam EngineType Random number engine
/// \param sample_rate Mean sample rate, in samples per count. Valid range is
///     less than 1.0. If 1.0 or greater, a sample is taken every step or pass.
///     If less than 1.0, a sample is taken with probability equal to the
///     sample rate.
/// \param random_number_generator Random number generator.
/// \return Steps or pass until the next sample should be taken
template <typename EngineType>
CountType stochastic_count_step(
    double sample_rate,
    monte::RandomNumberGenerator<EngineType> &random_number_generator) {
  CountType dn = 1;
  double max = 1.0;
  while (true) {
    if (random_number_generator.random_real(max) < sample_rate) {
      return dn;
    }
    ++dn;
  }
}

/// \brief Stochastically determine how much time until the next sample
///
/// \tparam EngineType Random number engine
/// \param sample_rate Mean sample rate, in samples per time.
/// \param random_number_generator Random number generator.
/// \return Time until the next sample should be taken. Returns
///     -ln(R)/sample_rate, where R is a random number in [0, 1.0).
template <typename EngineType>
TimeType stochastic_time_step(
    TimeType sample_rate,
    monte::RandomNumberGenerator<EngineType> &random_number_generator) {
  TimeType max = 1.0;
  return -std::log(random_number_generator.random_real(max)) / sample_rate;
}

/// \brief Return the count / time when the sample_index-th sample should be
///     taken
///
/// \param sample_index Index for sample (0, 1, 2, ...)
/// \param sampling_params Sampling method parameters
/// \param random_number_generator Random number generator, used for stochastic
///     sampling only.
/// \param sample_count Vector of counts (could be pass or step) when a sample
///     occurred. Used for stochastic sampling by count only.
/// \param sample_time Vector of times when a sample occurred. Used for
///     stochastic sampling by time only.
/// \return The time or count at which the sample-index-th sample should be
///     taken, as a double
//
/// Notes:
/// - If stochastic_sample_period == true, then the next sample is chosen at
///   a count or time using the input sampling parameters to determine a rate
/// - If stochastic_sample_period == true, then sample_index must equal
///   the current sample_count or sample_time size
///
template <typename EngineType>
double stochastic_sample_at(
    CountType sample_index, SamplingParams const &sampling_params,
    monte::RandomNumberGenerator<EngineType> &random_number_generator,
    std::vector<CountType> const &sample_count,
    std::vector<TimeType> const &sample_time) {
  SamplingParams const &s = sampling_params;
  if (sample_index == 0) {
    return s.begin;
  }
  double n = static_cast<double>(sample_index);
  double rate;
  if (s.sample_method == SAMPLE_METHOD::LINEAR) {
    rate = 1.0 / s.period;
  } else if (s.sample_method == SAMPLE_METHOD::LOG) {
    rate = 1.0 / (std::log(s.base) * std::pow(s.base, (n + s.shift)));
  } else if (s.sample_method == SAMPLE_METHOD::LOG) {
    if (!s.custom_sample_at) {
      throw std::runtime_error(
          "Error in stochastic_sample_at: "
          "sample_method==SAMPLE_METHOD::CUSTOM and "
          "!custom_sample_at");
    }
    rate = 1.0 / (s.custom_sample_at(n + 1) - s.custom_sample_at(n));
  }
  if (s.sample_mode == SAMPLE_MODE::BY_TIME) {
    return sample_time.back() +
           stochastic_time_step(rate, random_number_generator);
  } else {
    return sample_count.back() +
           stochastic_count_step(rate, random_number_generator);
  }
}

}  // namespace monte
}  // namespace CASM

#endif
