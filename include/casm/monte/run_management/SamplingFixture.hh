#ifndef CASM_monte_SamplingFixture
#define CASM_monte_SamplingFixture

#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/checks/ConvergenceCheck.hh"
#include "casm/monte/checks/CutoffCheck.hh"
#include "casm/monte/checks/EquilibrationCheck.hh"
#include "casm/monte/definitions.hh"
#include "casm/monte/misc/memory_used.hh"
#include "casm/monte/run_management/Results.hh"
#include "casm/monte/run_management/ResultsAnalysisFunction.hh"
#include "casm/monte/run_management/io/ResultsIO.hh"
#include "casm/monte/sampling/Sampler.hh"
#include "casm/monte/sampling/SamplingParams.hh"
#include "casm/monte/sampling/SelectedEventFunctions.hh"
#include "casm/monte/sampling/StateSamplingFunction.hh"

// logging
#include "casm/casm_io/Log.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/MethodLog.hh"
#include "casm/monte/checks/io/json/CompletionCheck_json_io.hh"

namespace CASM {
namespace monte {

template <typename ConfigType, typename StatisticsType>
struct SamplingFixtureParams {
  typedef ConfigType config_type;
  typedef StatisticsType stats_type;
  typedef ::CASM::monte::Results<config_type, stats_type> results_type;
  typedef ResultsIO<results_type> results_io_type;

  SamplingFixtureParams(
      std::string _label, StateSamplingFunctionMap _sampling_functions,
      jsonStateSamplingFunctionMap _json_sampling_functions,
      ResultsAnalysisFunctionMap<ConfigType, StatisticsType>
          _analysis_functions,
      monte::SamplingParams _sampling_params,
      monte::CompletionCheckParams<StatisticsType> _completion_check_params,
      std::vector<std::string> _analysis_names,
      std::unique_ptr<results_io_type> _results_io = nullptr,
      monte::MethodLog _method_log = monte::MethodLog())
      : label(_label),
        sampling_functions(_sampling_functions),
        json_sampling_functions(_json_sampling_functions),
        analysis_functions(_analysis_functions),
        sampling_params(_sampling_params),
        completion_check_params(_completion_check_params),
        analysis_names(_analysis_names),
        results_io(std::move(_results_io)),
        method_log(_method_log) {
    for (auto const &name : sampling_params.sampler_names) {
      if (!sampling_functions.count(name)) {
        std::stringstream ss;
        ss << "SamplingFixtureParams constructor error: No sampling function "
              "for '"
           << name << "'";
        throw std::runtime_error(ss.str());
      }
    }
    for (auto const &name : sampling_params.json_sampler_names) {
      if (!json_sampling_functions.count(name)) {
        std::stringstream ss;
        ss << "SamplingFixtureParams constructor error: No sampling function "
              "for '"
           << name << "'";
        throw std::runtime_error(ss.str());
      }
    }
  }

  /// Label, to distinguish multiple sampling fixtures
  std::string label;

  /// State sampling functions
  StateSamplingFunctionMap sampling_functions;

  /// State sampling functions
  jsonStateSamplingFunctionMap json_sampling_functions;

  /// Results analysis functions
  ResultsAnalysisFunctionMap<ConfigType, StatisticsType> analysis_functions;

  /// Sampling parameters
  monte::SamplingParams sampling_params;

  /// Completion check params
  monte::CompletionCheckParams<StatisticsType> completion_check_params;

  /// Analysis functions to evaluate
  std::vector<std::string> analysis_names;

  /// Results I/O implementation -- May be empty
  notstd::cloneable_ptr<results_io_type> results_io;

  /// Logging
  monte::MethodLog method_log;
};

/// \brief Step / pass / time tracking ---
struct MonteCounter {
  MonteCounter() { reset(SAMPLE_MODE::BY_PASS, 1); }

  /// \brief Sample by step, pass, or time
  ///
  /// Default=SAMPLE_MODE::BY_PASS
  SAMPLE_MODE sample_mode;

  /// \brief The number of steps per pass
  ///
  /// Typically the number of steps per pass is set equal to the number of
  /// mutating sites
  CountType steps_per_pass;

  /// \brief Tracks the number of Monte Carlo steps
  CountType step;

  /// \brief Tracks the number of Monte Carlo passes
  CountType pass;

  /// \brief Equal to either the number of steps or passes, depending on
  ///     sampling mode.
  CountType count;

  /// \brief Monte Carlo time, if applicable
  TimeType time;

  /// \brief Number of steps with an accepted event
  BigCountType n_accept;

  /// \brief Number of steps with a rejected event
  BigCountType n_reject;

  /// \brief Reset counters to zero
  void reset(SAMPLE_MODE _sample_mode, CountType _steps_per_pass) {
    sample_mode = _sample_mode;
    steps_per_pass = _steps_per_pass;
    step = 0;
    pass = 0;
    count = 0;
    time = 0.0;
    n_accept = 0;
    n_reject = 0;
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

template <typename _ConfigType, typename _StatisticsType, typename _EngineType>
class SamplingFixture {
 public:
  typedef _ConfigType config_type;
  typedef _StatisticsType stats_type;
  typedef _EngineType engine_type;
  typedef State<config_type> state_type;

  SamplingFixture(SamplingFixtureParams<config_type, stats_type> const &_params,
                  std::shared_ptr<engine_type> _engine)
      : m_params(_params),
        m_random_number_generator(_engine),
        m_n_samples(0),
        m_count(0),
        m_is_complete(false),
        m_completion_check(m_params.completion_check_params),
        m_results(
            m_params.sampling_params.sampler_names, m_params.sampling_functions,
            m_params.sampling_params.json_sampler_names,
            m_params.json_sampling_functions, m_params.analysis_functions) {}

  /// \brief Label, to distinguish multiple sampling fixtures
  std::string label() const { return m_params.label; }

  /// \brief Access sampling fixture parameters
  SamplingFixtureParams<config_type, stats_type> const &params() const {
    return m_params;
  }

  /// \brief Access current step / pass / time
  MonteCounter const &counter() const { return m_counter; }

  /// \brief Access results
  Results<config_type, stats_type> const &results() const { return m_results; }

  void initialize(Index steps_per_pass) {
    m_n_samples = 0;
    m_count = 0;
    m_is_complete = false;
    m_counter.reset(m_params.sampling_params.sample_mode, steps_per_pass);
    m_completion_check.reset();
    m_results.reset();
    m_results.initial_memory_used_MiB = memory_used_MiB(true);

    if (m_params.sampling_params.sample_mode == SAMPLE_MODE::BY_TIME) {
      m_next_sample_count = 0;
      m_next_sample_time = this->sample_at(m_results.sample_time.size());
      if (m_next_sample_time < 0.0) {
        throw std::runtime_error(
            "Error: sampling period parameter error, next_sample_time < "
            "0.0");
      }
    } else {
      m_next_sample_time = 0.0;
      m_next_sample_count = static_cast<CountType>(
          std::round(this->sample_at(m_results.sample_count.size())));
      if (m_next_sample_count < 0) {
        throw std::runtime_error(
            "Error: sampling period parameter error, next_sample_count < "
            "0");
      }
    }

    this->sampling_functions.clear();
    for (auto const &name : m_params.sampling_params.sampler_names) {
      if (!m_params.sampling_functions.count(name)) {
        std::stringstream ss;
        ss << "Sampling parameters error: No sampling function for '" << name
           << "'";
        throw std::runtime_error(ss.str());
      }
      this->sampling_functions.emplace(name,
                                       m_params.sampling_functions.at(name));
    }

    this->json_sampling_functions.clear();
    for (auto const &name : m_params.sampling_params.json_sampler_names) {
      if (!m_params.json_sampling_functions.count(name)) {
        std::stringstream ss;
        ss << "Sampling parameters error: No json sampling function for '"
           << name << "'";
        throw std::runtime_error(ss.str());
      }
      this->json_sampling_functions.emplace(
          name, m_params.json_sampling_functions.at(name));
    }

    Log &log = m_params.method_log.log;
    log.restart_clock();
    log.begin_lap();
  }

  bool is_complete() {
    if (m_is_complete) {
      return true;
    }
    Log &log = m_params.method_log.log;
    if (m_params.sampling_params.do_sample_time) {
      m_is_complete = m_completion_check.is_complete(
          m_results.samplers, m_results.sample_weight, m_counter.count,
          m_counter.time, log);

    } else {
      m_is_complete = m_completion_check.is_complete(
          m_results.samplers, m_results.sample_weight, m_counter.count, log);
    }
    return m_is_complete;
  }

  // Return completion check results
  CompletionCheckResults<stats_type> const &completion_check_results() const {
    return m_completion_check.results();
  }

  void write_status(Index run_index) {
    if (m_params.method_log.logfile_path.empty()) {
      return;
    }
    Log &log = m_params.method_log.log;
    m_params.method_log.reset();
    jsonParser json;
    json["run_index"] = run_index;
    json["time"] = log.time_s();
    to_json(m_completion_check.results(), json["completion_check_results"]);
    log << json << std::endl;
    log.begin_lap();
  }

  void write_status_if_due(Index run_index) {
    // Log method status - for efficiency, do not check clocktime every step
    // unless sampling by step
    std::optional<double> &log_frequency = m_params.method_log.log_frequency;
    if (!log_frequency.has_value()) {
      return;
    }
    if (m_n_samples != get_n_samples(m_results.samplers) ||
        m_count != m_counter.count) {
      m_n_samples = get_n_samples(m_results.samplers);
      m_count = m_counter.count;

      Log &log = m_params.method_log.log;
      if (log_frequency.has_value() && log.lap_time() > *log_frequency) {
        write_status(run_index);
      }
    }
  }

  void increment_n_accept() { m_counter.increment_n_accept(); }

  void increment_n_reject() { m_counter.increment_n_reject(); }

  void increment_step() { m_counter.increment_step(); }

  void set_time(double event_time) { m_counter.set_time(event_time); }

  void push_back_sample_weight(double weight) {
    m_results.sample_weight.push_back(weight);
  }

  /// \brief Next count at which to take a sample, if applicable
  CountType next_sample_count() const { return m_next_sample_count; }

  /// \brief Next time at which to take a sample, if applicable
  TimeType next_sample_time() const { return m_next_sample_time; }

  // Save event statistics
  void set_selected_event_data(SelectedEventData &&selected_event_data) {
    m_results.selected_event_data = std::move(selected_event_data);
  }

  template <bool DebugMode = false>
  void sample_data(state_type const &state) {
    // Debug log
    if constexpr (DebugMode) {
      Log &log = CASM::log();
      log.custom("Sample data");
      log.indent() << "- fixture: " << m_params.label << std::endl;

      // - Set next sample count
      if (m_params.sampling_params.sample_mode == SAMPLE_MODE::BY_TIME) {
        log.indent() << "- sample_by: TIME" << std::endl;
        log.indent() << "- next_sample_time: " << m_next_sample_time
                     << std::endl;
      } else {
        log.indent() << "- sample_by: COUNT" << std::endl;
        log.indent() << "- next_sample_count: " << m_next_sample_count
                     << std::endl;
      }

      log.indent() << "- step: " << m_counter.step << std::endl;
      log.indent() << "- pass: " << m_counter.pass << std::endl;
      log.indent() << "- count: " << m_counter.count << std::endl;
      log.indent() << "- time: " << m_counter.time << std::endl;
      log << std::endl;
      log.increase_indent();
      log.end_section();
    }

    // - Record count
    m_results.sample_count.push_back(m_counter.count);

    // - Record simulated time
    if (m_params.sampling_params.do_sample_time) {
      m_results.sample_time.push_back(m_counter.time);
    }

    // - Record clocktime
    m_results.sample_clocktime.push_back(m_params.method_log.log.time_s());

    // - Record configuration
    if (m_params.sampling_params.do_sample_trajectory) {
      m_results.sample_trajectory.push_back(state.configuration);
    }

    // - Evaluate functions and record data
    // Debug log
    if constexpr (DebugMode) {
      Log &log = CASM::log();
      log.custom("Evaluate sampling functions");
    }
    for (auto const &name : m_params.sampling_params.sampler_names) {
      if (this->sampling_functions.find(name) ==
          this->sampling_functions.end()) {
        std::stringstream ss;
        ss << "Error in SamplingFixture::sample_data: did not find sampling "
              "function '"
           << name << "'";
        throw std::runtime_error(ss.str());
      }

      // Debug log - function name
      if constexpr (DebugMode) {
        Log &log = CASM::log();
        log.indent() << "- function: " << name << std::endl;
      }

      // Sample data
      auto const &function = this->sampling_functions.at(name);
      m_results.samplers.at(name)->push_back(function());

      // Debug log - function value
      if constexpr (DebugMode) {
        Log &log = CASM::log();
        log.increase_indent();

        auto const &sampler = *m_results.samplers.at(name);
        int r = sampler.n_samples() - 1;
        jsonParser tjson;
        tjson["shape"] = sampler.shape();
        tjson["component_names"] = sampler.component_names();
        to_json(sampler.values().row(r), tjson["value"],
                jsonParser::as_array());
        log.indent() << "- shape: " << tjson["shape"] << std::endl;
        log.indent() << "- component_names: " << tjson["component_names"]
                     << std::endl;
        log.indent() << "- value: " << tjson["value"] << std::endl;
        log.decrease_indent();
      }
    }
    if constexpr (DebugMode) {
      if (m_params.sampling_params.sampler_names.size() == 0) {
        Log &log = CASM::log();
        log.indent() << "- no sampling functions" << std::endl;
      }
      Log &log = CASM::log();
      log << std::endl;
      log.end_section();
    }

    // Evaluate JSON sampling functions
    // Debug log
    if constexpr (DebugMode) {
      Log &log = CASM::log();
      log.custom("Evaluate JSON sampling functions");
    }
    for (auto const &name : m_params.sampling_params.json_sampler_names) {
      if (this->json_sampling_functions.find(name) ==
          this->json_sampling_functions.end()) {
        std::stringstream ss;
        ss << "Error in SamplingFixture::sample_data: did not find json "
              "sampling function'"
           << name << "'";
        throw std::runtime_error(ss.str());
      }

      // Debug log - function name
      if constexpr (DebugMode) {
        Log &log = CASM::log();
        log.indent() << "- JSON function: " << name << std::endl;
      }

      // Sample JSON data
      auto const &function = this->json_sampling_functions.at(name);
      m_results.json_samplers.at(name)->values.push_back(function());

      // Debug log - function name
      if constexpr (DebugMode) {
        Log &log = CASM::log();
        log.increase_indent();

        auto const &sampler = *m_results.json_samplers.at(name);
        log.indent() << "- value: " << sampler.values.back() << std::endl;
        log.decrease_indent();
      }
    }
    if constexpr (DebugMode) {
      if (m_params.sampling_params.json_sampler_names.size() == 0) {
        Log &log = CASM::log();
        log.indent() << "- no JSON sampling functions" << std::endl;
      }
      Log &log = CASM::log();
      log << std::endl << std::endl;
      log.end_section();
    }

    if constexpr (DebugMode) {
      Log &log = CASM::log();
      log.custom("JSON Summary");

      jsonParser json;
      json["sampling_functions"] = jsonParser::object();
      for (auto const &name : m_params.sampling_params.sampler_names) {
        auto const &sampler = *m_results.samplers.at(name);
        int r = sampler.n_samples() - 1;
        jsonParser &tjson = json["sampling_functions"][name];
        tjson["shape"] = sampler.shape();
        tjson["component_names"] = sampler.component_names();
        to_json(sampler.values().row(r), tjson["value"],
                jsonParser::as_array());
      }

      json["json_sampling_functions"] = jsonParser::object();
      for (auto const &name : m_params.sampling_params.json_sampler_names) {
        auto const &sampler = *m_results.json_samplers.at(name);
        json["json_sampling_functions"][name] = sampler.values.back();
      }

      log.indent() << json << std::endl << std::endl;
      log.end_section();
    }

    // - Set next sample count
    if (m_params.sampling_params.sample_mode == SAMPLE_MODE::BY_TIME) {
      m_next_sample_time = this->sample_at(m_results.sample_time.size());
      if (m_next_sample_time <= m_counter.time) {
        throw std::runtime_error(
            "Error: state sampling period parameter error, next_sample_time <= "
            "current time");
      }
      if constexpr (DebugMode) {
        Log &log = CASM::log();
        log.custom("Set next sample time");
        log.indent() << "- next_sample_time: " << m_next_sample_time
                     << std::endl
                     << std::endl;
        log.end_section();
      }
    } else {
      m_next_sample_count = static_cast<CountType>(
          std::round(this->sample_at(m_results.sample_count.size())));
      if (m_next_sample_count <= m_counter.count) {
        throw std::runtime_error(
            "Error: state sampling period parameter error, next_sample_count "
            "<= current count");
      }
      if constexpr (DebugMode) {
        Log &log = CASM::log();
        log.custom("Set next sample count");
        log.indent() << "- next_sample_count: " << m_next_sample_count
                     << std::endl
                     << std::endl;
        log.end_section();
      }
    }

    if constexpr (DebugMode) {
      Log &log = CASM::log();
      log.decrease_indent();
    }
  }

  template <bool DebugMode = false>
  void sample_data_by_count_if_due(state_type const &state) {
    if (m_params.sampling_params.sample_mode != SAMPLE_MODE::BY_TIME &&
        m_counter.count == m_next_sample_count) {
      sample_data<DebugMode>(state);
    }
  }

  template <bool DebugMode = false>
  void sample_data_by_time_if_due(state_type const &state, double event_time) {
    if (m_params.sampling_params.sample_mode != SAMPLE_MODE::BY_TIME &&
        event_time >= m_next_sample_time) {
      sample_data<DebugMode>(state);
    }
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
    if (m_params.sampling_params.stochastic_sample_period) {
      return stochastic_sample_at(
          sample_index, m_params.sampling_params, m_random_number_generator,
          m_results.sample_count, m_results.sample_time);
    } else {
      return monte::sample_at(sample_index, m_params.sampling_params);
    }
  }

  /// \brief Write results and final status
  void finalize(state_type const &state, Index run_index) {
    Log &log = m_params.method_log.log;
    m_results.final_memory_used_MiB = memory_used_MiB(true);
    m_results.elapsed_clocktime = log.time_s();
    m_results.completion_check_results = m_completion_check.results();
    m_results.analysis = make_analysis(m_results, m_params.analysis_functions,
                                       m_params.analysis_names);
    m_results.n_accept = m_counter.n_accept;
    m_results.n_reject = m_counter.n_reject;

    if (m_params.results_io) {
      m_params.results_io->write(m_results, state.conditions, run_index);
    }

    write_status(run_index);
  }

 private:
  /// \brief Parameters controlling what is sampled and when
  SamplingFixtureParams<config_type, stats_type> m_params;

  /// State sampling functions (copied made during `initialize`)
  StateSamplingFunctionMap sampling_functions;

  /// JSON State sampling functions (copied made during `initialize`)
  jsonStateSamplingFunctionMap json_sampling_functions;

  /// \brief Random number generator
  monte::RandomNumberGenerator<engine_type> m_random_number_generator;

  /// \brief This is for write_status_if_due only
  Index m_n_samples = 0;

  /// \brief This is for write_status_if_due only
  Index m_count = 0;

  /// \brief Store whether this fixture has completed
  bool m_is_complete = false;

  /// \brief Count steps / passes / time
  MonteCounter m_counter;

  /// \brief Next count at which to take a sample, if applicable
  CountType m_next_sample_count;

  /// \brief Next time at which to take a sample, if applicable
  TimeType m_next_sample_time;

  /// \brief Completion checker
  CompletionCheck<stats_type> m_completion_check;

  /// \brief Holds sampled data
  Results<config_type, stats_type> m_results;
};

}  // namespace monte
}  // namespace CASM

#endif
