#ifndef CASM_monte_SamplingFixture
#define CASM_monte_SamplingFixture

#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/checks/ConvergenceCheck.hh"
#include "casm/monte/checks/CutoffCheck.hh"
#include "casm/monte/checks/EquilibrationCheck.hh"
#include "casm/monte/definitions.hh"
#include "casm/monte/results/Results.hh"
#include "casm/monte/results/io/ResultsIO.hh"
#include "casm/monte/sampling/Sampler.hh"
#include "casm/monte/sampling/SamplingParams.hh"
#include "casm/monte/state/RunData.hh"

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
      std::string _label,
      StateSamplingFunctionMap<ConfigType> _sampling_functions,
      ResultsAnalysisFunctionMap<ConfigType, StatisticsType>
          _analysis_functions,
      monte::SamplingParams _sampling_params,
      monte::CompletionCheckParams<StatisticsType> _completion_check_params,
      std::unique_ptr<results_io_type> _results_io,
      monte::MethodLog _method_log = monte::MethodLog())
      : label(_label),
        sampling_functions(_sampling_functions),
        analysis_functions(_analysis_functions),
        sampling_params(_sampling_params),
        completion_check_params(_completion_check_params),
        results_io(std::move(_results_io)),
        method_log(_method_log) {}

  /// Label, to distinguish multiple sampling fixtures
  std::string label;

  /// State sampling functions
  StateSamplingFunctionMap<ConfigType> sampling_functions;

  /// Results analysis functions
  ResultsAnalysisFunctionMap<ConfigType, StatisticsType> analysis_functions;

  /// Sampling parameters
  monte::SamplingParams sampling_params;

  /// Completion check params
  monte::CompletionCheckParams<StatisticsType> completion_check_params;

  /// Results I/O implementation -- May be empty
  notstd::cloneable_ptr<results_io_type> results_io;

  /// Logging
  monte::MethodLog method_log;
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
        m_engine(_engine),
        m_n_samples(0),
        m_count(0),
        m_is_complete(false),
        m_state_sampler(m_engine, m_params.sampling_params,
                        m_params.sampling_functions),
        m_completion_check(m_params.completion_check_params) {}

  /// \brief Label, to distinguish multiple sampling fixtures
  std::string label() const { return m_params.label; }

  /// \brief Sampling fixture parameters
  SamplingFixtureParams<config_type, stats_type> const &params() const {
    return m_params;
  }

  /// \brief State sampler
  StateSampler<config_type, engine_type> const &state_sampler() const {
    return m_state_sampler;
  }

  void initialize(state_type const &state, Index steps_per_pass) {
    m_n_samples = 0;
    m_count = 0;
    m_is_complete = false;
    m_state_sampler.reset(steps_per_pass);
    m_completion_check.reset();

    Log &log = m_params.method_log.log;
    log.restart_clock();
    log.begin_lap();
  }

  bool is_complete() {
    if (m_is_complete) {
      return true;
    }
    Log &log = m_params.method_log.log;
    if (m_state_sampler.do_sample_time) {
      m_is_complete = m_completion_check.is_complete(
          m_state_sampler.samplers, m_state_sampler.sample_weight,
          m_state_sampler.count, m_state_sampler.time, log);

    } else {
      m_is_complete = m_completion_check.is_complete(
          m_state_sampler.samplers, m_state_sampler.sample_weight,
          m_state_sampler.count, log);
    }
    return m_is_complete;
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
    if (m_n_samples != get_n_samples(m_state_sampler.samplers) ||
        m_count != m_state_sampler.count) {
      m_n_samples = get_n_samples(m_state_sampler.samplers);
      m_count = m_state_sampler.count;

      Log &log = m_params.method_log.log;
      if (log_frequency.has_value() && log.lap_time() > *log_frequency) {
        write_status(run_index);
      }
    }
  }

  void increment_n_accept() { m_state_sampler.increment_n_accept(); }

  void increment_n_reject() { m_state_sampler.increment_n_reject(); }

  void increment_step() { m_state_sampler.increment_step(); }

  void set_time(double event_time) { m_state_sampler.set_time(event_time); }

  void push_back_sample_weight(double weight) {
    m_state_sampler.push_back_sample_weight(weight);
  }

  void sample_data(state_type const &state) {
    Log &log = m_params.method_log.log;
    m_state_sampler.sample_data(state, log);
  }

  void sample_data_by_count_if_due(state_type const &state) {
    Log &log = m_params.method_log.log;
    m_state_sampler.sample_data_by_count_if_due(state, log);
  }

  // Note: Not sure if this is useful in practice
  void sample_data_by_time_if_due(state_type const &state, double event_time) {
    Log &log = m_params.method_log.log;
    m_state_sampler.sample_data_by_time_if_due(state, event_time, log);
  }

  void finalize(state_type const &state, Index run_index,
                RunData<config_type> const &run_data) {
    Log &log = m_params.method_log.log;
    m_results.elapsed_clocktime = log.time_s();
    m_results.samplers = std::move(m_state_sampler.samplers);
    m_results.sample_count = std::move(m_state_sampler.sample_count);
    m_results.sample_time = std::move(m_state_sampler.sample_time);
    m_results.sample_weight = std::move(m_state_sampler.sample_weight);
    m_results.sample_clocktime = std::move(m_state_sampler.sample_clocktime);
    m_results.sample_trajectory = std::move(m_state_sampler.sample_trajectory);
    m_results.completion_check_results = m_completion_check.results();
    m_results.analysis =
        make_analysis(run_data, m_results, m_params.analysis_functions);
    m_results.n_accept = m_state_sampler.n_accept;
    m_results.n_reject = m_state_sampler.n_reject;

    if (m_params.results_io) {
      m_params.results_io->write(m_results, run_data.conditions, run_index);
    }

    write_status(run_index);
  }

 private:
  SamplingFixtureParams<config_type, stats_type> m_params;

  /// Random number generator engine
  std::shared_ptr<engine_type> m_engine;

  /// \brief This is for write_status_if_due only
  Index m_n_samples = 0;

  /// \brief This is for write_status_if_due only
  Index m_count = 0;

  bool m_is_complete = false;

  StateSampler<config_type, engine_type> m_state_sampler;

  CompletionCheck<stats_type> m_completion_check;

  Results<config_type, stats_type> m_results;
};

}  // namespace monte
}  // namespace CASM

#endif
