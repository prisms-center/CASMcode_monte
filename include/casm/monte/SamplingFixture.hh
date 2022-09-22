#ifndef CASM_monte_SamplingFixture
#define CASM_monte_SamplingFixture

#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/checks/ConvergenceCheck.hh"
#include "casm/monte/checks/CutoffCheck.hh"
#include "casm/monte/checks/EquilibrationCheck.hh"
#include "casm/monte/definitions.hh"
#include "casm/monte/results/io/ResultsIO.hh"
#include "casm/monte/sampling/Sampler.hh"
#include "casm/monte/sampling/SamplingParams.hh"

// logging
#include "casm/casm_io/Log.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/MethodLog.hh"
#include "casm/monte/checks/io/json/CompletionCheck_json_io.hh"

namespace CASM {
namespace monte {

template <typename ConfigType>
struct SamplingFixtureParams {
  SamplingFixtureParams(
      std::string _label,
      StateSamplingFunctionMap<ConfigType> _sampling_functions,
      ResultsAnalysisFunctionMap<ConfigType> _analysis_functions,
      monte::SamplingParams _sampling_params,
      monte::CompletionCheckParams _completion_check_params,
      std::unique_ptr<ResultsIO<ConfigType>> _results_io,
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
  ResultsAnalysisFunctionMap<ConfigType> analysis_functions;

  /// Sampling parameters
  monte::SamplingParams sampling_params;

  /// Completion check params
  monte::CompletionCheckParams completion_check_params;

  /// Results I/O implementation -- May be empty
  notstd::cloneable_ptr<ResultsIO<ConfigType>> results_io;

  /// Logging
  monte::MethodLog method_log;
};

template <typename _ConfigType>
class SamplingFixture {
 public:
  typedef _ConfigType config_type;
  typedef State<config_type> state_type;

  SamplingFixture(SamplingFixtureParams<config_type> const &_params)
      : m_params(_params),
        m_n_samples(0),
        m_is_complete(false),
        m_state_sampler(m_params.sampling_params, m_params.sampling_functions),
        m_completion_check(m_params.completion_check_params) {}

  SamplingFixtureParams<config_type> const &params() const { return m_params; }

  void initialize(state_type const &state, Index steps_per_pass) {
    m_n_samples = 0;
    m_is_complete = false;
    m_state_sampler.reset(steps_per_pass);
    m_completion_check.reset();

    m_results.initial_state = state;
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
      m_is_complete = m_completion_check.is_complete(m_state_sampler.samplers,
                                                     m_state_sampler.count,
                                                     m_state_sampler.time, log);

    } else {
      m_is_complete = m_completion_check.is_complete(
          m_state_sampler.samplers, m_state_sampler.count, log);
    }
    return m_is_complete;
  }

  void write_status_if_due() {
    // Log method status - for efficiency, only check after a new sample is
    // taken
    if (m_n_samples != get_n_samples(m_state_sampler.samplers)) {
      m_n_samples = get_n_samples(m_state_sampler.samplers);
      Log &log = m_params.method_log.log;
      std::optional<double> &log_frequency = m_params.method_log.log_frequency;
      if (log_frequency.has_value() && log.lap_time() > *log_frequency) {
        m_params.method_log.reset();
        jsonParser json;
        json["status"] = "incomplete";
        json["time"] = log.time_s();
        to_json(m_completion_check.results(),
                json["convergence_check_results"]);
        log << json << std::endl;
        log.begin_lap();
      }
    }
  }

  void increment_step() { m_state_sampler.increment_step(); }

  void increment_time(double time_increment) {
    m_state_sampler.increment_time(time_increment);
  }

  void sample_data_by_count_if_due(state_type const &state) {
    Log &log = m_params.method_log.log;
    m_state_sampler.sample_data_by_count_if_due(state, log);
  }

  void sample_data_by_time_if_due(state_type const &state,
                                  double time_increment) {
    Log &log = m_params.method_log.log;
    m_state_sampler.sample_data_by_time_if_due(state, time_increment, log);
  }

  void finalize(state_type const &state, Index run_index) {
    Log &log = m_params.method_log.log;
    m_results.final_state = state;
    m_results.elapsed_clocktime = log.time_s();
    m_results.samplers = std::move(m_state_sampler.samplers);
    m_results.sample_count = std::move(m_state_sampler.sample_count);
    m_results.sample_time = std::move(m_state_sampler.sample_time);
    m_results.sample_clocktime = std::move(m_state_sampler.sample_clocktime);
    m_results.sample_trajectory = std::move(m_state_sampler.sample_trajectory);
    m_results.completion_check_results = m_completion_check.results();
    m_results.analysis = make_analysis(m_results, m_params.analysis_functions);

    if (m_params.results_io) {
      m_params.results_io->write(m_results, run_index);
    }
  }

 private:
  SamplingFixtureParams<config_type> m_params;

  Index m_n_samples = 0;

  bool m_is_complete = false;

  StateSampler<config_type> m_state_sampler;

  CompletionCheck m_completion_check;

  Results<config_type> m_results;
};

}  // namespace monte
}  // namespace CASM

#endif
