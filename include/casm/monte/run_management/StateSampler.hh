#ifndef CASM_monte_StateSampler
#define CASM_monte_StateSampler

#include "casm/casm_io/Log.hh"
#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/definitions.hh"
#include "casm/monte/run_management/State.hh"
#include "casm/monte/sampling/Sampler.hh"
#include "casm/monte/sampling/SamplingParams.hh"
#include "casm/monte/sampling/StateSamplingFunction.hh"

namespace CASM {
namespace monte {

/// \brief A data structure to help encapsulate typical Monte Carlo sampling
///
/// - Holds information describing what to sample, and when
/// - Holds the functions that take the samples
/// - Holds `step`, `pass`, and `time` counters
/// - Holds the data that is sampled, and when it was sampled
/// - Includes methods for incrementing the step/pass/time, and checking if a
///   sample is due and taking the sample
template <typename _ConfigType, typename _EngineType>
struct StateSampler {
  typedef _ConfigType ConfigType;
  typedef _EngineType EngineType;
  typedef _ConfigType config_type;
  typedef _EngineType engine_type;

  // --- Parameters for determining when samples are taken, what is sampled ---

  /// Random number generator
  monte::RandomNumberGenerator<engine_type> random_number_generator;

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
  /// If stochastic_sample_period == true, then instead of setting the sample
  /// time / count deterministally, use the sampling period to determine the
  /// sampling rate and determine the next sample time / count stochastically.
  ///
  SAMPLE_METHOD sample_method;

  /// \brief See `sample_method`
  double begin;

  /// \brief See `sample_method`
  double period;

  /// \brief See `sample_method`
  double samples_per_period;

  /// \brief See `sample_method`
  double shift;

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

  /// \brief State sampling functions to be used when taking a sample
  ///
  /// Each function returns an Eigen::VectorXd
  std::vector<StateSamplingFunction> functions;

  /// --- Step / pass / time tracking ---

  /// \brief Tracks the number of Monte Carlo steps
  CountType step;

  /// \brief Tracks the number of Monte Carlo passes
  CountType pass;

  /// \brief The number of steps per pass
  ///
  /// Typically the number of steps per pass is set equal to the number of
  /// mutating sites
  CountType steps_per_pass;

  /// \brief Equal to either the number of steps or passes, depending on
  ///     sampling mode.
  CountType count;

  /// \brief Monte Carlo time, if applicable
  TimeType time;

  /// \brief Number of steps with an accepted event
  long long n_accept;

  /// \brief Number of steps with a rejected event
  long long n_reject;

  /// \brief Next count at which to take a sample, if applicable
  CountType next_sample_count;

  /// \brief Next time at which to take a sample, if applicable
  TimeType next_sample_time;

  // --- Sampled data ---

  /// \brief Map of <quantity name>:<sampler>
  ///
  /// A `Sampler` stores Eigen::MatrixXd with the raw sampled data. Rows of the
  /// matrix corresponds to individual samples of VectorXd. The matrices are
  /// constructed with extra rows and encapsulated in a class so that
  /// resizing can be done intelligently as needed. Sampler provides
  /// accessors so that the data can be efficiently accessed by index or by
  /// component name for equilibration and convergence checking of
  /// individual components.
  std::map<std::string, std::shared_ptr<Sampler>> samplers;

  /// \brief The count (either step or pass) when a sample was taken
  std::vector<CountType> sample_count;

  /// \brief The time when a sample was taken, if applicable
  std::vector<TimeType> sample_time;

  /// \brief The weight to give a sample, if applicable
  Sampler sample_weight;

  /// \brief The clocktime when a sample was taken, if applicable
  std::vector<TimeType> sample_clocktime;

  /// \brief The configuration when a sample was taken
  ///
  /// The trajectory is sampled if `sample_trajectory==true`
  std::vector<ConfigType> sample_trajectory;

  /// \brief Constructor
  ///
  /// Note: Call `reset(double _steps_per_pass)` before sampling begins.
  StateSampler(std::shared_ptr<EngineType> _engine,
               SamplingParams const &_sampling_params,
               StateSamplingFunctionMap const &sampling_functions)
      : StateSampler(
            _engine, _sampling_params.sample_mode,
            {},  // functions populated in constructor body
            _sampling_params.sample_method, _sampling_params.begin,
            _sampling_params.period, _sampling_params.samples_per_period,
            _sampling_params.shift, _sampling_params.stochastic_sample_period,
            _sampling_params.do_sample_trajectory,
            _sampling_params.do_sample_time) {
    // populate functions, samplers
    for (std::string name : _sampling_params.sampler_names) {
      auto const &function = sampling_functions.at(name);
      auto shared_sampler =
          std::make_shared<Sampler>(function.shape, function.component_names);
      functions.push_back(function);
      samplers.emplace(name, shared_sampler);
    }
  }

  /// \brief Constructor
  ///
  /// \param _sample_mode Sample by step, pass, or time
  /// \param _functions State sampling functions to be used when taking a
  ///     sample. Each function returns an `Eigen::VectorXd`.
  /// \param _sample_method Whether to take linearly spaced or logarithmically
  ///     spaced samples.
  /// \param _sample_begin When the first sample is taken. See `sample_method`.
  /// \param _sampling_period A number of counts, or period of time. Used to
  ///     specify sampling spacing. See `sample_method`.
  /// \param _samples_per_period How many samples to take per the specified
  ///     period. See `sample_method`.
  /// \param _log_sampling_shift Controls logarithmically sampling spacing. See
  ///     `sample_method`.
  /// \param _stochastic_sample_period If true, then instead of setting the
  ///     sample time / count deterministally, use the sampling period to
  ///     determine the sampling rate and determine the next sample
  ///     time / count stochastically.
  /// \param _do_sample_trajectory If true, save the configuration when a sample
  ///     is taken
  ///
  /// Note: Call `reset(double _steps_per_pass)` before sampling begins.
  StateSampler(std::shared_ptr<EngineType> _engine, SAMPLE_MODE _sample_mode,
               std::vector<StateSamplingFunction> const &_functions,
               SAMPLE_METHOD _sample_method = SAMPLE_METHOD::LINEAR,
               double _sample_begin = 0.0, double _sampling_period = 1.0,
               double _samples_per_period = 1.0,
               double _log_sampling_shift = 0.0,
               bool _stochastic_sample_period = false,
               bool _do_sample_trajectory = false, bool _do_sample_time = false)
      : random_number_generator(_engine),
        sample_mode(_sample_mode),
        sample_method(_sample_method),
        begin(_sample_begin),
        period(_sampling_period),
        samples_per_period(_samples_per_period),
        shift(_log_sampling_shift),
        stochastic_sample_period(_stochastic_sample_period),
        do_sample_trajectory(_do_sample_trajectory),
        do_sample_time(_do_sample_time),
        functions(_functions),
        sample_weight({}) {
    reset(1.0);
  }

  /// \brief Reset sampler to be ready for sampling
  ///
  /// Reset does the following:
  /// - Set step / pass / count / time to zero
  /// - Set steps_per_pass
  /// - Clear all sampled data containers
  void reset(double _steps_per_pass) {
    steps_per_pass = _steps_per_pass;
    step = 0;
    pass = 0;
    count = 0;
    time = 0.0;
    n_accept = 0;
    n_reject = 0;
    samplers.clear();
    for (auto const &function : functions) {
      auto shared_sampler =
          std::make_shared<Sampler>(function.shape, function.component_names);
      samplers.emplace(function.name, shared_sampler);
    }
    sample_count.clear();
    sample_time.clear();
    sample_weight.clear();
    sample_clocktime.clear();
    sample_trajectory.clear();

    if (sample_mode == SAMPLE_MODE::BY_TIME) {
      next_sample_count = 0;
      next_sample_time = sample_at(sample_time.size());
      if (next_sample_time < 0.0) {
        throw std::runtime_error(
            "Error: state sampling period parameter error, next_sample_time < "
            "0.0");
      }
    } else {
      next_sample_time = 0.0;
      next_sample_count =
          static_cast<CountType>(std::round(sample_at(sample_count.size())));
      if (next_sample_count < 0) {
        throw std::runtime_error(
            "Error: state sampling period parameter error, next_sample_count < "
            "0");
      }
    }
  }

  /// \brief Stochastically determine how many steps or passes
  ///     until the next sample
  CountType stochastic_count_step(double sample_rate) {
    CountType dn = 1;
    double max = 1.0;
    while (true) {
      if (random_number_generator.random_real(max) < sample_rate) {
        return dn;
      }
      ++dn;
    }
  }

  /// \brief Stochastically determine much time
  ///     until the next sample
  TimeType stochastic_time_step(TimeType sample_rate) {
    TimeType max = 1.0;
    return -std::log(random_number_generator.random_real(max)) / sample_rate;
  }

  /// \brief Return the count / time when the sample_index-th sample should be
  ///     taken
  ///
  /// Notes:
  /// - If stochastic_sample_period == true, then the next sample is chosen at
  ///   a count or time using the input sampling parameters to determine a rate
  /// - If stochastic_sample_period == true, then sample_index must equal
  ///   the current sample_count or sample_time size
  double sample_at(CountType sample_index) {
    if (stochastic_sample_period) {
      if (sample_index == 0) {
        return begin;
      }
      double n = static_cast<double>(sample_index);
      double rate;
      if (sample_method == SAMPLE_METHOD::LINEAR) {
        rate = 1.0 / (period / samples_per_period);
      } else /* sample_method == SAMPLE_METHOD::LOG */ {
        rate = 1.0 / (std::log(period) *
                      std::pow(period, (n + shift) / samples_per_period) /
                      samples_per_period);
      }
      if (sample_mode == SAMPLE_MODE::BY_TIME) {
        return sample_time.back() + stochastic_time_step(rate);
      } else {
        return sample_count.back() + stochastic_count_step(rate);
      }
    } else {
      double n = static_cast<double>(sample_index);
      if (sample_method == SAMPLE_METHOD::LINEAR) {
        return begin + (period / samples_per_period) * n;
      } else /* sample_method == SAMPLE_METHOD::LOG */ {
        return begin + std::pow(period, (n + shift) / samples_per_period);
      }
    }
  }

  // /// \brief Return true if sample is due (count based sampling)
  // ///
  // /// \returns True if `count == count when sample is due`
  // bool sample_is_due() const {
  //   double value = sample_at(sample_count.size());
  //   return count == static_cast<CountType>(std::round(value));
  // }

  // \brief Set weight given to next sample
  void push_back_sample_weight(double weight) {
    sample_weight.push_back(weight);
  }

  /// \brief Sample data, if due (count based sampling)
  ///
  /// \param state, The state to sample
  /// \param log, A Log, from which the clocktime is obtained when a
  ///     sample is taken
  ///
  /// Note:
  /// - Call `reset(double _steps_per_pass)` before sampling begins.
  /// - Apply chosen event before this
  /// - Call `increment_step()` before this
  void sample_data(monte::State<ConfigType> const &state, Log &log) {
    // - Record count
    sample_count.push_back(count);

    // - Record simulated time
    if (do_sample_time) {
      sample_time.push_back(time);
    }

    // - Record clocktime
    sample_clocktime.push_back(log.time_s());

    // - Record configuration
    if (do_sample_trajectory) {
      sample_trajectory.push_back(state.configuration);
    }

    // - Evaluate functions and record data
    for (auto const &function : functions) {
      samplers.at(function.name)->push_back(function());
    }

    // - Set next sample count
    if (sample_mode == SAMPLE_MODE::BY_TIME) {
      next_sample_time = sample_at(sample_time.size());
      if (next_sample_time <= time) {
        throw std::runtime_error(
            "Error: state sampling period parameter error, next_sample_time <= "
            "current time");
      }
    } else {
      next_sample_count =
          static_cast<CountType>(std::round(sample_at(sample_count.size())));
      if (next_sample_count <= count) {
        throw std::runtime_error(
            "Error: state sampling period parameter error, next_sample_count "
            "<= current count");
      }
    }
  }

  void sample_data_by_count_if_due(monte::State<ConfigType> const &state,
                                   Log &log) {
    if (sample_mode != SAMPLE_MODE::BY_TIME && count == next_sample_count) {
      sample_data(state, log);
    }
  }

  // Note: Not sure if this is useful in practice
  void sample_data_by_time_if_due(monte::State<ConfigType> const &state,
                                  double event_time, Log &log) {
    if (sample_mode != SAMPLE_MODE::BY_TIME && event_time >= next_sample_time) {
      sample_data(state, log);
    }
  }

  /// \brief Increment by one acceptance
  void increment_n_accept() { ++n_accept; }

  /// \brief Increment by one rejection
  void increment_n_reject() { ++n_reject; }

  /// \brief Increment by one step (updating pass, count as appropriate)
  void increment_step() {
    ++step;
    if (sample_mode == SAMPLE_MODE::BY_STEP) {
      ++count;
    }
    if (step == steps_per_pass) {
      ++pass;
      if (sample_mode != SAMPLE_MODE::BY_STEP) {
        ++count;
      }
      step = 0;
    }

    // // If sampling by step, set count to step. Otherwise, set count to pass.
    // count = (sample_mode == SAMPLE_MODE::BY_STEP) ? step : pass;
  }

  /// \brief Set time
  void set_time(double event_time) { time = event_time; }
};

}  // namespace monte
}  // namespace CASM

#endif
