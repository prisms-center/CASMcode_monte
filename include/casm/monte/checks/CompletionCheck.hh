#ifndef CASM_monte_CompletionCheck
#define CASM_monte_CompletionCheck

#include <optional>

#include "casm/casm_io/Log.hh"
#include "casm/monte/checks/ConvergenceCheck.hh"
#include "casm/monte/checks/CutoffCheck.hh"
#include "casm/monte/checks/EquilibrationCheck.hh"
#include "casm/monte/definitions.hh"
#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

// --- Completion checking (cutoff & convergence) ---

/// \brief Parameters that determine if a calculation is complete
template <typename StatisticsType>
struct CompletionCheckParams {
  /// \brief Default constructor
  CompletionCheckParams();

  /// \brief Completion check parameters that don't depend on the sampled values
  CutoffCheckParams cutoff_params;

  /// \brief Function that performs equilibration checking
  ///
  /// Defaults to monte::default_equilibration_check
  EquilibrationCheckFunction equilibration_check_f;

  /// \brief Function to calculate statistics
  ///
  /// Defaults to monte::default_statistics_calculator<StatisticsType>()
  CalcStatisticsFunction<StatisticsType> calc_statistics_f;

  /// \brief Sampler components that must be checked for convergence, and the
  ///     estimated precision to which the mean must be converged
  std::map<SamplerComponent, RequestedPrecision> requested_precision;

  //  For "linear" spacing, the n-th check (n=0,1,2,...) will be taken when:
  //
  //      sample_check_linear(n) = round( check_begin + check_period * n )
  //
  //  For "log" spacing, the n-th check will be taken when:
  //
  //      sample_check_log(n) = round( check_begin + check_base ^ (n +
  //      check_shift) )
  //
  //  However, if sample(n) - sample(n-1) > check_period_max; then subsequent
  //  samples are taken every `check_period_max` samples.

  /// Logarithmic checking or linear check spacing
  bool log_spacing = false;

  // Check spacing parameters
  CountType check_begin = 100;
  CountType check_period = 100;
  double check_base = 10.0;
  double check_shift = 2.0;
  CountType check_period_max = 10000;

  /// \brief Sample at which to perform the n-th completion check, for
  /// linear-checking
  CountType sample_check_linear(Index n) const {
    return check_begin + check_period * n;
  }

  /// \brief Sample at which to perform the n-th completion check, for
  /// log-checking
  CountType sample_check_log(Index n) const {
    return check_begin + static_cast<CountType>(std::round(
                             std::pow(check_base, (n + check_shift))));
  }

  /// \brief For log-checking with a maximum check period, find when to switch
  /// to linear checking
  Index find_n_begin_linear() const {
    Index n_begin_linear = 0;
    auto check_delta = [&](Index n) {
      return sample_check_log(n) - sample_check_log(n - 1);
    };
    while (check_delta(n_begin_linear + 1) <= check_period_max) {
      n_begin_linear += 1;
    }
    return n_begin_linear;
  }

  /// \brief Sample at which to perform the n-th completion check, for
  /// log-checking
  ///     with a maximum check period
  CountType sample_check_log(Index n, Index n_begin_linear) const {
    if (n <= n_begin_linear) {
      return sample_check_log(n);
    } else {
      return sample_check_log(n_begin_linear) +
             check_period_max * (n - n_begin_linear);
    }
  }
};

/// \brief Stores completion check results
template <typename StatisticsType>
struct CompletionCheckResults {
  /// Parameters used for the completion check
  CompletionCheckParams<StatisticsType> params;

  /// Current count (if given)
  std::optional<CountType> count;

  /// Current time (if given)
  std::optional<TimeType> time;

  /// Elapsed clocktime
  TimeType clocktime;

  /// Current number of samples
  CountType n_samples = 0;

  /// Minimums cutoff check results
  bool has_all_minimums_met = false;

  /// Maximums cutoff check results
  bool has_any_maximum_met = false;

  /// Number of samples when equilibration and convergence checks performed
  std::optional<CountType> n_samples_at_convergence_check;

  /// Equilibration check results, when last performed
  EquilibrationCheckResults equilibration_check_results;

  /// Convergence check results, when last performed
  ConvergenceCheckResults<StatisticsType> convergence_check_results;

  /// True if calculation is complete, either due to convergence or cutoff
  bool is_complete = false;

  /// \brief Reset for step by step updates
  ///
  /// Reset most values, but not:
  /// - params
  /// - n_samples_at_convergence_check
  /// - equilibration_check_results
  /// - convergence_check_results
  void partial_reset(std::optional<CountType> _count = std::nullopt,
                     std::optional<TimeType> _time = std::nullopt,
                     TimeType _clocktime = 0.0, CountType _n_samples = 0) {
    // params: do not reset
    count = _count;
    time = _time;
    clocktime = _clocktime;
    n_samples = _n_samples;
    has_all_minimums_met = false;
    has_any_maximum_met = false;
    is_complete = false;
  }

  /// \brief Reset for next run
  ///
  /// Reset all values, except:
  /// - params
  void full_reset(std::optional<CountType> _count = std::nullopt,
                  std::optional<TimeType> _time = std::nullopt,
                  TimeType _clocktime = 0.0, CountType _n_samples = 0) {
    // params: do not reset
    partial_reset(_count, _time, _clocktime, _n_samples);
    n_samples_at_convergence_check = std::nullopt;
    equilibration_check_results = EquilibrationCheckResults();
    convergence_check_results = ConvergenceCheckResults<StatisticsType>();
  }
};

/// \brief Checks if a cutoff or convergence criteria are met
template <typename StatisticsType>
class CompletionCheck {
 public:
  CompletionCheck(CompletionCheckParams<StatisticsType> params);

  CompletionCheckParams<StatisticsType> const &params() const {
    return m_params;
  }

  void reset();

  bool is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      Sampler const &sample_weight, Log &log);

  bool is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      Sampler const &sample_weight, CountType count, Log &log);

  bool is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      Sampler const &sample_weight, TimeType time, Log &log);

  bool is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      Sampler const &sample_weight, CountType count, TimeType time, Log &log);

  CompletionCheckResults<StatisticsType> const &results() const {
    return m_results;
  }

 private:
  bool _is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      Sampler const &sample_weight, std::optional<CountType> count,
      std::optional<TimeType> time, Log &log);

  void _check_convergence(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      Sampler const &sample_weight, CountType n_samples);

  CompletionCheckParams<StatisticsType> m_params;

  CompletionCheckResults<StatisticsType> m_results;

  Index m_n_checks = 0;
  Index m_n_begin_linear = 0;

  Index m_last_n_samples = 0;

  double m_last_clocktime = 0.0;
};

// --- Inline definitions ---

/// \brief Default constructor
template <typename StatisticsType>
CompletionCheckParams<StatisticsType>::CompletionCheckParams()
    : cutoff_params(),
      equilibration_check_f(default_equilibration_check),
      calc_statistics_f(default_statistics_calculator<StatisticsType>()) {}

template <typename StatisticsType>
void CompletionCheck<StatisticsType>::reset() {
  m_results.full_reset();
  m_n_checks = 0;
  m_last_n_samples = 0;
  m_last_clocktime = 0.0;
}

template <typename StatisticsType>
bool CompletionCheck<StatisticsType>::is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    Sampler const &sample_weight, Log &log) {
  return _is_complete(samplers, sample_weight, std::nullopt, std::nullopt, log);
}

template <typename StatisticsType>
bool CompletionCheck<StatisticsType>::is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    Sampler const &sample_weight, CountType count, Log &log) {
  return _is_complete(samplers, sample_weight, count, std::nullopt, log);
}

template <typename StatisticsType>
bool CompletionCheck<StatisticsType>::is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    Sampler const &sample_weight, TimeType time, Log &log) {
  return _is_complete(samplers, sample_weight, std::nullopt, time, log);
}

template <typename StatisticsType>
bool CompletionCheck<StatisticsType>::is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    Sampler const &sample_weight, CountType count, TimeType time, Log &log) {
  return _is_complete(samplers, sample_weight, count, time, log);
}

template <typename StatisticsType>
bool CompletionCheck<StatisticsType>::_is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    Sampler const &sample_weight, std::optional<CountType> count,
    std::optional<TimeType> time, Log &log) {
  CountType n_samples = get_n_samples(samplers);

  // for efficiency, only update clocktime after a new sample is taken
  TimeType clocktime = m_last_clocktime;
  if (n_samples != m_last_n_samples) {
    clocktime = log.time_s();
    m_last_n_samples = n_samples;
    m_last_clocktime = clocktime;
  }

  m_results.partial_reset(count, time, clocktime, n_samples);

  m_results.has_all_minimums_met = all_minimums_met(
      m_params.cutoff_params, count, time, n_samples, clocktime);

  // if all minimums not met, continue, otherwise can stop
  if (!m_results.has_all_minimums_met) {
    return false;
  }

  // if any maximum met, stop even if not converged
  m_results.has_any_maximum_met = any_maximum_met(m_params.cutoff_params, count,
                                                  time, n_samples, clocktime);

  if (m_results.has_any_maximum_met) {
    m_results.is_complete = true;
    // force convergence check
    if (n_samples != m_results.n_samples_at_convergence_check) {
      _check_convergence(samplers, sample_weight, n_samples);
    }
    return true;
  }

  // if maximums not met, check equilibration and convergence if due
  Index check_at;
  if (m_params.log_spacing) {
    check_at = m_params.sample_check_log(m_n_checks, m_n_begin_linear);
  } else {
    check_at = m_params.sample_check_linear(m_n_checks);
  }
  if (n_samples >= check_at) {
    m_n_checks += 1;
    _check_convergence(samplers, sample_weight, n_samples);
  }

  // if all requested to converge are converged, then complete
  if (m_results.convergence_check_results.all_converged) {
    m_results.is_complete = true;
  }

  return m_results.is_complete;
}

template <typename StatisticsType>
CompletionCheck<StatisticsType>::CompletionCheck(
    CompletionCheckParams<StatisticsType> params)
    : m_params(params), m_n_begin_linear(m_params.find_n_begin_linear()) {
  m_results.params = m_params;
  m_results.is_complete = false;

  if (m_params.equilibration_check_f == nullptr) {
    throw std::runtime_error(
        "Error constructing CompletionCheck: params.equilibration_check_f == "
        "nullptr");
  }
  if (m_params.calc_statistics_f == nullptr) {
    throw std::runtime_error(
        "Error constructing CompletionCheck: params.calc_statistics_f == "
        "nullptr");
  }
}

/// \brief Check for equilibration and convergence, then set m_results
template <typename StatisticsType>
void CompletionCheck<StatisticsType>::_check_convergence(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    Sampler const &sample_weight, CountType n_samples) {
  // if auto convergence mode:
  if (m_params.requested_precision.size()) {
    m_results.n_samples_at_convergence_check = n_samples;

    // check for equilibration
    bool check_all = false;
    m_results.equilibration_check_results = equilibration_check(
        m_params.equilibration_check_f, m_params.requested_precision, samplers,
        sample_weight, check_all);

    // if all requested to converge are equilibrated, then check convergence
    if (m_results.equilibration_check_results.all_equilibrated) {
      m_results.convergence_check_results = convergence_check(
          samplers, sample_weight, m_params.requested_precision,
          m_results.equilibration_check_results
              .N_samples_for_all_to_equilibrate,
          m_params.calc_statistics_f);
    } else {
      m_results.convergence_check_results =
          ConvergenceCheckResults<StatisticsType>();
    }
  }
}

}  // namespace monte
}  // namespace CASM

#endif
