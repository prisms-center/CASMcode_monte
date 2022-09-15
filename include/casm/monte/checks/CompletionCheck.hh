#ifndef CASM_monte_CompletionCheck
#define CASM_monte_CompletionCheck

#include <optional>

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

  /// \brief Minimum number of samples before checking for completion
  CountType check_begin = 10;

  /// \brief How often to check for completion
  ///
  /// Check for completion performed if:
  /// - n_samples % check_frequency == 0 && n_samples >= check_begin
  CountType check_frequency = 1;
};

/// \brief Stores completion check results
struct CompletionCheckResults {
  /// Minimums cutoff check results
  bool has_all_minimums_met = false;

  /// Maximums cutoff check results
  bool has_any_maximum_met = false;

  /// Current count (if given)
  std::optional<CountType> count;

  /// Current time (if given)
  std::optional<TimeType> time;

  /// Elapsed clocktime
  TimeType clocktime;

  /// Current number of samples
  CountType n_samples = 0;

  /// Equilibration and convergence checks are performed if:
  /// - n_samples >= check_begin && n_samples % check_frequency == 0, and
  /// - requested_precision.size() > 0
  bool convergence_check_performed = false;

  /// True if calculation is complete, either due to convergence or cutoff
  bool is_complete = false;

  EquilibrationCheckResults equilibration_check_results;

  /// \brief Confidence level used for calculated precision of mean
  double confidence = 0.95;

  ConvergenceCheckResults convergence_check_results;
};

/// \brief Checks if a cutoff or convergence criteria are met
class CompletionCheck {
 public:
  CompletionCheck(CompletionCheckParams params);

  bool is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      TimeType clocktime);

  bool is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      CountType count, TimeType clocktime);

  bool is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      TimeType time, TimeType clocktime);

  bool is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      CountType count, TimeType time, TimeType clocktime);

  CompletionCheckResults const &results() const { return m_results; }

 private:
  bool _is_complete(
      std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
      std::optional<CountType> count, std::optional<TimeType> time,
      TimeType clocktime);

  void _check(std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
              std::optional<CountType> count, std::optional<TimeType> time,
              CountType n_samples);

  CompletionCheckParams m_params;

  CompletionCheckResults m_results;
};

// --- Inline definitions ---

inline bool CompletionCheck::is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    TimeType clocktime) {
  return _is_complete(samplers, std::nullopt, std::nullopt, clocktime);
}

inline bool CompletionCheck::is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    CountType count, TimeType clocktime) {
  return _is_complete(samplers, count, std::nullopt, clocktime);
}

inline bool CompletionCheck::is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    TimeType time, TimeType clocktime) {
  return _is_complete(samplers, std::nullopt, time, clocktime);
}

inline bool CompletionCheck::is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    CountType count, TimeType time, TimeType clocktime) {
  return _is_complete(samplers, count, time, clocktime);
}

inline bool CompletionCheck::_is_complete(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    std::optional<CountType> count, std::optional<TimeType> time,
    TimeType clocktime) {
  CountType n_samples = get_n_samples(samplers);

  m_results = CompletionCheckResults();
  m_results.confidence = m_params.confidence;
  m_results.count = count;
  m_results.time = time;
  m_results.clocktime = clocktime;
  m_results.n_samples = n_samples;
  m_results.has_all_minimums_met = all_minimums_met(
      m_params.cutoff_params, count, time, n_samples, clocktime);

  // if all minimums not met, continue, otherwise can stop
  if (!m_results.has_all_minimums_met) {
    return false;
  }

  // check equilibration and convergence
  if (n_samples >= m_params.check_begin &&
      n_samples % m_params.check_frequency == 0) {
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
