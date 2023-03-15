#ifndef CASM_monte_StateSampler
#define CASM_monte_StateSampler

#include "casm/casm_io/Log.hh"
#include "casm/monte/definitions.hh"
#include "casm/monte/sampling/Sampler.hh"
#include "casm/monte/sampling/SamplingParams.hh"
#include "casm/monte/state/State.hh"

namespace CASM {
namespace monte {

/// \brief A function to be evaluated when taking a sample of a Monte Carlo
///     calculation state
///
/// - Each StateSamplingFunction takes a State<ConfigType> and returns an
///   Eigen::VectorXd
/// - A StateSamplingFunction has additional information (name, description,
///   component_names) to enable specifying convergence criteria, allow input
///   and output descriptions, help and error messages, etc.
/// - Use `reshaped` (in casm/monte/sampling/Sampler.hh) to output scalars or
///   matrices as vectors.
///
template <typename _ConfigType>
struct StateSamplingFunction {
  typedef _ConfigType ConfigType;

  /// \brief Constructor - default component names
  StateSamplingFunction(std::string _name, std::string _description,
                        std::vector<Index> _shape,
                        std::function<Eigen::VectorXd()> _function);

  /// \brief Constructor - custom component names
  StateSamplingFunction(std::string _name, std::string _description,
                        std::vector<std::string> const &_component_names,
                        std::vector<Index> _shape,
                        std::function<Eigen::VectorXd()> _function);

  /// \brief Function name (and quantity to be sampled)
  std::string name;

  /// \brief Description of the function
  std::string description;

  /// \brief Shape of quantity, with column-major unrolling
  ///
  /// Scalar: [], Vector: [n], Matrix: [m, n], etc.
  std::vector<Index> shape;

  /// \brief A name for each component of the resulting Eigen::VectorXd
  ///
  /// Can be string representing an index (i.e "0", "1", "2", etc.) or can
  /// be a descriptive string (i.e. "Mg", "Va", "O", etc.)
  std::vector<std::string> component_names;

  /// \brief The function to be evaluated
  std::function<Eigen::VectorXd()> function;

  /// \brief Evaluates `function`
  Eigen::VectorXd operator()() const;
};

template <typename ConfigType, typename ValueType>
void set_value(std::map<SamplerComponent, ValueType> &component_map,
               StateSamplingFunctionMap<ConfigType> const &sampling_functions,
               std::string const &sampler_name, ValueType const &value);

template <typename ConfigType, typename ValueType>
void set_value_by_component_index(
    std::map<SamplerComponent, double> &component_map,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions,
    std::string const &sampler_name, Index component_index,
    ValueType const &value);

template <typename ConfigType, typename ValueType>
void set_value_by_component_name(
    std::map<SamplerComponent, double> &component_map,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions,
    std::string const &sampler_name, std::string const &component_name,
    ValueType const &value);

/// \brief A data structure to help encapsulate typical Monte Carlo sampling
///
/// - Holds information describing what to sample, and when
/// - Holds the functions that take the samples
/// - Holds `step`, `pass`, and `time` counters
/// - Holds the data that is sampled, and when it was sampled
/// - Includes methods for incrementing the step/pass/time, and checking if a
///   sample is due and taking the sample
template <typename _ConfigType>
struct StateSampler {
  typedef _ConfigType ConfigType;

  // --- Parameters for determining when samples are taken, what is sampled ---

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

  /// \brief See `sample_method`
  double begin;

  /// \brief See `sample_method`
  double period;

  /// \brief See `sample_method`
  double samples_per_period;

  /// \brief See `sample_method`
  double shift;

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
  /// Each function takes a State<ConfigType> and returns an Eigen::VectorXd
  std::vector<StateSamplingFunction<ConfigType>> functions;

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
  std::vector<TimeType> sample_weight;

  /// \brief The clocktime when a sample was taken, if applicable
  std::vector<TimeType> sample_clocktime;

  /// \brief The configuration when a sample was taken
  ///
  /// The trajectory is sampled if `sample_trajectory==true`
  std::vector<ConfigType> sample_trajectory;

  /// \brief Constructor
  ///
  /// Note: Call `reset(double _steps_per_pass)` before sampling begins.
  StateSampler(SamplingParams const &_sampling_params,
               StateSamplingFunctionMap<ConfigType> const &sampling_functions)
      : StateSampler(
            _sampling_params.sample_mode,
            {},  // functions populated in constructor body
            _sampling_params.sample_method, _sampling_params.begin,
            _sampling_params.period, _sampling_params.samples_per_period,
            _sampling_params.shift, _sampling_params.do_sample_trajectory,
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
  ///     sample. Each function takes a `State<ConfigType>` and returns an
  ///     `Eigen::VectorXd`.
  /// \param _sample_method Whether to take linearly spaced or logarithmically
  ///     spaced samples.
  /// \param _sample_begin When the first sample is taken. See `sample_method`.
  /// \param _sampling_period A number of counts, or period of time. Used to
  ///     specify sampling spacing. See `sample_method`.
  /// \param _samples_per_period How many samples to take per the specified
  ///     period. See `sample_method`.
  /// \param _log_sampling_shift Controls logarithmically sampling spacing. See
  ///     `sample_method`.
  /// \param _do_sample_trajectory If true, save the configuration when a sample
  ///     is taken
  ///
  /// Note: Call `reset(double _steps_per_pass)` before sampling begins.
  StateSampler(SAMPLE_MODE _sample_mode,
               std::vector<StateSamplingFunction<ConfigType>> const &_functions,
               SAMPLE_METHOD _sample_method = SAMPLE_METHOD::LINEAR,
               double _sample_begin = 0.0, double _sampling_period = 1.0,
               double _samples_per_period = 1.0,
               double _log_sampling_shift = 0.0,
               bool _do_sample_trajectory = false, bool _do_sample_time = false)
      : sample_mode(_sample_mode),
        sample_method(_sample_method),
        begin(_sample_begin),
        period(_sampling_period),
        samples_per_period(_samples_per_period),
        shift(_log_sampling_shift),
        do_sample_trajectory(_do_sample_trajectory),
        do_sample_time(_do_sample_time),
        functions(_functions) {
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
    } else {
      next_sample_time = 0.0;
      next_sample_count =
          static_cast<CountType>(std::round(sample_at(sample_count.size())));
    }
  }

  /// \brief Return the count / time when the sample_index-th sample should be
  ///     taken
  double sample_at(CountType sample_index) const {
    double n = static_cast<double>(sample_index);
    if (sample_method == SAMPLE_METHOD::LINEAR) {
      return begin + (period / samples_per_period) * n;
    } else /* sample_method == SAMPLE_METHOD::LOG */ {
      return begin + std::pow(period, (n + shift) / samples_per_period);
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
    } else {
      next_sample_count =
          static_cast<CountType>(std::round(sample_at(sample_count.size())));
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
      if (sample_mode == SAMPLE_MODE::BY_PASS) {
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

/// \brief Get component names for a particular function, else use defaults
template <typename ConfigType>
std::vector<std::string> get_scalar_component_names(
    std::string const &function_name, double const &value,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions);

/// \brief Get component names for a particular function, else use defaults
template <typename ConfigType>
std::vector<std::string> get_vector_component_names(
    std::string const &function_name, Eigen::VectorXd const &value,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions);

/// \brief Get component names for a particular function, else use defaults
template <typename ConfigType>
std::vector<std::string> get_matrix_component_names(
    std::string const &function_name, Eigen::MatrixXd const &value,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions);

}  // namespace monte
}  // namespace CASM

// --- Inline implementations ---

namespace CASM {
namespace monte {

/// \brief Constructor - custom component names
template <typename _ConfigType>
StateSamplingFunction<_ConfigType>::StateSamplingFunction(
    std::string _name, std::string _description, std::vector<Index> _shape,
    std::function<Eigen::VectorXd()> _function)
    : name(_name),
      description(_description),
      shape(_shape),
      component_names(default_component_names(shape)),
      function(_function) {}

/// \brief Constructor - custom component names
template <typename _ConfigType>
StateSamplingFunction<_ConfigType>::StateSamplingFunction(
    std::string _name, std::string _description,
    std::vector<std::string> const &_component_names, std::vector<Index> _shape,
    std::function<Eigen::VectorXd()> _function)
    : name(_name),
      description(_description),
      shape(_shape),
      component_names(_component_names),
      function(_function) {}

/// \brief Take a sample
template <typename _ConfigType>
Eigen::VectorXd StateSamplingFunction<_ConfigType>::operator()() const {
  return function();
}

/// \brief Adds values in a map of SamplerComponent -> ValueType
///
/// \param component_map Map to add a std::pair<SamplerComponent, ValueType>
/// into \param sampling_functions Container of sampling functions referred to
/// \param sampler_name The name of a StateSamplingFunction in
/// `sampling_functions` \param value The value added to `component_map` for
/// each SamplerComponent
///     of the StateSamplingFunction specified by `sampler_name`.
///
/// \throws std::runtime_error if `sampler_name` cannot be found.
template <typename ConfigType, typename ValueType>
void set_value(std::map<SamplerComponent, ValueType> &component_map,
               StateSamplingFunctionMap<ConfigType> const &sampling_functions,
               std::string const &sampler_name, ValueType const &value) {
  auto it = sampling_functions.find(sampler_name);
  if (it == sampling_functions.end()) {
    std::stringstream ss;
    ss << "Error: no sampling function with name '" << sampler_name << "'";
    throw std::runtime_error(ss.str());
  }
  Index component_index = 0;
  for (std::string const &component_name : it->second.component_names) {
    component_map.emplace(
        SamplerComponent(sampler_name, component_index, component_name), value);
    ++component_index;
  }
}

/// \brief Adds a value in a map of SamplerComponent -> ValueType
///
/// \param component_map Map to add a std::pair<SamplerComponent, ValueType>
/// into \param sampling_functions Container of sampling functions referred to
/// \param sampler_name The name of a StateSamplingFunction in
/// `sampling_functions` \param component_index An index into components of the
/// StateSamplingFunction
///     specified by `sample_name`.
/// \param value The value added to `component_map` for the SamplerComponent
///     specified by `sampler_name` and `component_name`.
///
/// \throws std::runtime_error if either `sampler_name` cannot be found or
///     `component_index` is out of range.
template <typename ConfigType, typename ValueType>
void set_value_by_component_index(
    std::map<SamplerComponent, double> &component_map,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions,
    std::string const &sampler_name, Index component_index,
    ValueType const &value) {
  auto it = sampling_functions.find(sampler_name);
  if (it == sampling_functions.end()) {
    std::stringstream ss;
    ss << "Error: no sampling function with name '" << sampler_name << "'";
    throw std::runtime_error(ss.str());
  }
  if (component_index >= it->second.component_names.size()) {
    std::stringstream ss;
    ss << "Error: component index " << component_index
       << " is out of range for sampling function '" << sampler_name << "'";
    throw std::runtime_error(ss.str());
  }
  component_map.emplace(
      SamplerComponent(sampler_name, component_index,
                       it->second.component_names[component_index]),
      value);
}

/// \brief Adds a value in a map of SamplerComponent -> ValueType
///
/// \param component_map Map to add a std::pair<SamplerComponent, ValueType>
/// into \param sampling_functions Container of sampling functions referred to
/// \param sampler_name The name of a StateSamplingFunction in
/// `sampling_functions` \param component_name A name in `component_names` of
/// the StateSamplingFunction
///     specified by `sample_name`.
/// \param value The value added to `component_map` for the SamplerComponent
///     specified by `sampler_name` and `component_name`.
///
/// \throws std::runtime_error if either `sampler_name` or `component_name`
/// cannot
///     be found.
template <typename ConfigType, typename ValueType>
void set_value_by_component_name(
    std::map<SamplerComponent, double> &component_map,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions,
    std::string const &sampler_name, std::string const &component_name,
    ValueType const &value) {
  auto it = sampling_functions.find(sampler_name);
  if (it == sampling_functions.end()) {
    std::stringstream ss;
    ss << "Error: no sampling function with name '" << sampler_name << "'";
    throw std::runtime_error(ss.str());
  }
  Index component_index = 0;
  for (auto const &name : it->second.component_names) {
    if (name == component_name) {
      component_map.emplace(
          SamplerComponent(sampler_name, component_index, name), value);
      return;
    }
    ++component_index;
  }

  // Error if component_name not found
  std::stringstream ss;
  ss << "Error: component name '" << component_name
     << "' is not found for sampling function '" << sampler_name << "'";
  throw std::runtime_error(ss.str());
}

/// \brief Get component names for a particular function, else use defaults
///
/// Notes:
/// - Used for naming conditions vector components using a sampling function
///   of the same name.
/// - If function not found, returns default component names ("0")
/// - Throws if function found, but component_names dimension does not match
template <typename ConfigType>
std::vector<std::string> get_scalar_component_names(
    std::string const &function_name, double const &value,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions) {
  std::vector<Index> shape({});
  auto function_it = sampling_functions.find(function_name);
  if (function_it == sampling_functions.end()) {
    return default_component_names(shape);
  } else {
    if (function_it->second.component_names.size() != 1) {
      std::stringstream msg;
      msg << "Error in get_scalar_component_names: Dimension of \""
          << function_name << "\" (" << 1
          << ") does not match the corresponding sampling function.";
      throw std::runtime_error(msg.str());
    }
    return function_it->second.component_names;
  }
}

/// \brief Get component names for a particular function, else use defaults
///
/// Notes:
/// - Used for naming conditions vector components using a sampling function
///   of the same name.
/// - If function not found, returns default component names ("0", "1", "2",
///   ...)
/// - Throws if function found, but component_names dimension does not match
///   value.size().
template <typename ConfigType>
std::vector<std::string> get_vector_component_names(
    std::string const &function_name, Eigen::VectorXd const &value,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions) {
  std::vector<Index> shape({value.size()});
  auto function_it = sampling_functions.find(function_name);
  if (function_it == sampling_functions.end()) {
    return default_component_names(shape);
  } else {
    if (function_it->second.component_names.size() != value.size()) {
      std::stringstream msg;
      msg << "Error in get_vector_component_names: Dimension of \""
          << function_name << "\" (" << value.size()
          << ") does not match the corresponding sampling function.";
      throw std::runtime_error(msg.str());
    }
    return function_it->second.component_names;
  }
}

/// \brief Get component names for a particular function, else use defaults
///
/// Notes:
/// - Used for naming conditions vector components using a sampling function
///   of the same name.
/// - If function not found, returns default component names ("0", "1", "2",
///   ...)
/// - Throws if function found, but component_names dimension does not match
///   value.size().
template <typename ConfigType>
std::vector<std::string> get_matrix_component_names(
    std::string const &function_name, Eigen::MatrixXd const &value,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions) {
  std::vector<Index> shape({value.rows(), value.cols()});
  auto function_it = sampling_functions.find(function_name);
  if (function_it == sampling_functions.end()) {
    return default_component_names(shape);
  } else {
    if (function_it->second.component_names.size() != value.size()) {
      std::stringstream msg;
      msg << "Error in get_matrix_component_names: Dimension of \""
          << function_name << "\" (" << value.size()
          << ") does not match the corresponding sampling function.";
      throw std::runtime_error(msg.str());
    }
    return function_it->second.component_names;
  }
}

}  // namespace monte
}  // namespace CASM

#endif
