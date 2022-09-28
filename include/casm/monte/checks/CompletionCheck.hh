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
struct CompletionCheckParams {
  /// \brief Completion check parameters that don't depend on the sampled values
  CutoffCheckParams cutoff_params;

  /// \brief Sampler components that must be checked for convergence, and the
  ///     estimated precision to which the mean must be converged
  std::map<SamplerComponent, double> requested_precision;

  /// \brief Confidence level for calculated precision of mean
  double confidence = 0.95;

  //  For "linear" spacing, the n-th check will be taken when:
  //
  //      sample = round( check_begin + (check_period / checks_per_period) * n )
  //
  //  For "log" spacing, the n-th check will be taken when:
  //
  //      sample = round( check_begin + check_period ^ ( (n + check_shift) /
  //                      checks_per_period ) )

  /// Logirithmic checking or linear check spacing
  bool log_spacing = true;

  // Check spacing parameters
  double check_begin = 0.0;
  double check_period = 10.0;
  double checks_per_period = 1.0;
  double check_shift = 1.0;
};

/// \brief Stores completion check results
struct CompletionCheckResults {
  /// Parameters used for the completion check
  CompletionCheckParams params;

  /// \brief Confidence level used for calculated precision of mean
  double confidence = 0.95;

  /// Current count (if given)
  std::optional<CountType> count;

  /// Current time (if given)
  std::optional<TimeType> time;

  /// Elapsed clocktime (if given)
  std::optional<TimeType> clocktime;

  /// Current number of samples
  CountType n_samples = 0;

  /// Minimums cutoff check results
  bool has_all_minimums_met = false;

  /// Maximums cutoff check results
  bool has_any_maximum_met = false;

  /// Equilibration and convergence checks are performed if:
  /// - n_samples >= check_begin && n_samples % check_frequency == 0, and
  /// - requested_precision.size() > 0
  bool convergence_check_performed = false;

  EquilibrationCheckResults equilibration_check_results;

  ConvergenceCheckResults convergence_check_results;

  /// True if calculation is complete, either due to convergence or cutoff
  bool is_complete = false;

  void reset(std::optional<CountType> _count = std::nullopt,
             std::optional<TimeType> _time = std::nullopt,
             CountType _n_samples = 0) {
    // params: do not reset
    // confidence: do not reset
    count = _count;
    time = _time;
    clocktime = std::nullopt;
    n_samples = _n_samples;
    has_all_minimums_met = false;
    has_any_maximum_met = false;
    if (convergence_check_performed) {
      convergence_check_performed = false;
      equilibration_check_results = EquilibrationCheckResults();
      convergence_check_results = ConvergenceCheckResults();
    }
    is_complete = false;
  }
};

/// \brief Checks if a cutoff or convergence criteria are met
class CompletionCheck {
 public:
  CompletionCheck(CompletionCheckParams params);

  void reset();

  bool is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      Log &log);

  bool is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      CountType count, Log &log);

  bool is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      TimeType time, Log &log);

  bool is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      CountType count, TimeType time, Log &log);

  CompletionCheckResults const &results() const { return m_results; }

 private:
  bool _is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      std::optional<CountType> count, std::optional<TimeType> time, Log &log);

  void _check(std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
              std::optional<CountType> count, std::optional<TimeType> time,
              CountType n_samples);

  CompletionCheckParams m_params;

  CompletionCheckResults m_results;

  double m_n_checks = 0.0;

  Index m_last_n_samples = 0.0;

  Index m_last_clocktime = 0.0;
};

// --- Inline definitions ---

inline void CompletionCheck::reset() {
  m_results.reset();
  m_n_checks = 0.0;
  m_last_n_samples = 0.0;
  m_last_clocktime = 0.0;
}

inline bool CompletionCheck::is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers, Log &log) {
  return _is_complete(samplers, std::nullopt, std::nullopt, log);
}

inline bool CompletionCheck::is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    CountType count, Log &log) {
  return _is_complete(samplers, count, std::nullopt, log);
}

inline bool CompletionCheck::is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    TimeType time, Log &log) {
  return _is_complete(samplers, std::nullopt, time, log);
}

inline bool CompletionCheck::is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    CountType count, TimeType time, Log &log) {
  return _is_complete(samplers, count, time, log);
}

inline bool CompletionCheck::_is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    std::optional<CountType> count, std::optional<TimeType> time, Log &log) {
  CountType n_samples = get_n_samples(samplers);

  // for efficiency, only update clocktime after a new sample is taken
  TimeType clocktime = m_last_clocktime;
  if (n_samples != m_last_n_samples) {
    clocktime = log.time_s();
    m_last_n_samples = n_samples;
    m_last_clocktime = clocktime;
  }

  m_results.reset(count, time, n_samples);

  m_results.has_all_minimums_met = all_minimums_met(
      m_params.cutoff_params, count, time, n_samples, clocktime);

  // if all minimums not met, continue, otherwise can stop
  if (!m_results.has_all_minimums_met) {
    return false;
  }

  // check equilibration and convergence
  double check_at;
  if (m_params.log_spacing) {
    check_at =
        m_params.check_begin +
        std::pow(m_params.check_period, (m_n_checks + m_params.check_shift) /
                                            m_params.checks_per_period);
  } else {
    check_at =
        m_params.check_begin +
        (m_params.check_period / m_params.checks_per_period) * m_n_checks;
  }
  if (n_samples >= static_cast<CountType>(std::round(check_at))) {
    m_n_checks += 1.0;
    _check(samplers, count, time, n_samples);
  }

  // if any maximum met, stop even if not converged
  m_results.has_any_maximum_met = any_maximum_met(m_params.cutoff_params, count,
                                                  time, n_samples, clocktime);

  if (m_results.has_any_maximum_met) {
    m_results.is_complete = true;
  }

  return m_results.is_complete;
}

}  // namespace monte
}  // namespace CASM

#endif
