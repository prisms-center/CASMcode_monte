#ifndef CASM_monte_ConvergenceCheck
#define CASM_monte_ConvergenceCheck

#include <cmath>
#include <iostream>

#include "casm/monte/definitions.hh"
#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

/// \brief Convergence check results data structure (individual component)
template <typename StatisticsType>
struct IndividualConvergenceCheckResult {
  /// \brief True if mean (<X>) is converged to requested precision
  bool is_converged;

  /// \brief Requested precision in <X>
  RequestedPrecision requested_precision;

  StatisticsType stats;
};

/// \brief Check convergence of a range of observations
template <typename StatisticsType>
IndividualConvergenceCheckResult<StatisticsType> convergence_check(
    StatisticsType const &stats, RequestedPrecision const &requested_precision);

/// \brief Check convergence of a range of observations
template <typename StatisticsType>
IndividualConvergenceCheckResult<StatisticsType> component_convergence_check(
    Sampler const &sampler, Sampler const &sample_weight,
    SamplerComponent const &key, RequestedPrecision const &requested_precision,
    CountType N_samples_for_statistics,
    CalcStatisticsFunction<StatisticsType> calc_statistics_f);

/// \brief Convergence check results data structure (all requested components)
template <typename StatisticsType>
struct ConvergenceCheckResults {
  /// \brief True if all required properties are converged to required precision
  ///
  /// Notes:
  /// - True if completion check finds all required properties are converged to
  /// the requested precision
  /// - False otherwise, including if no convergence checks were requested
  bool all_converged = false;

  /// \brief How many samples were used to get statistics
  ///
  /// Notes:
  /// - Set to the total number of samples if no convergence checks were
  /// requested
  CountType N_samples_for_statistics = 0;

  /// \brief Results from checking convergence criteria
  std::map<SamplerComponent, IndividualConvergenceCheckResult<StatisticsType>>
      individual_results;
};

/// \brief Check convergence of all requested properties
template <typename StatisticsType>
ConvergenceCheckResults<StatisticsType> convergence_check(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    Sampler const &sample_weight,
    std::map<SamplerComponent, RequestedPrecision> const &requested_precision,
    CountType N_samples_for_equilibration,
    CalcStatisticsFunction<StatisticsType> calc_statistics_f);

/// --- template implementations ---

/// \brief Check convergence of a range of observations
template <typename StatisticsType>
IndividualConvergenceCheckResult<StatisticsType> convergence_check(
    StatisticsType const &stats,
    RequestedPrecision const &requested_precision) {
  IndividualConvergenceCheckResult<StatisticsType> result;
  result.stats = stats;
  result.requested_precision = requested_precision;
  result.is_converged = true;
  if (result.requested_precision.abs_convergence_is_required) {
    result.is_converged &= get_calculated_precision(stats) <
                           result.requested_precision.abs_precision;
  }
  if (result.requested_precision.rel_convergence_is_required) {
    result.is_converged &= get_calculated_relative_precision(stats) <
                           result.requested_precision.rel_precision;
  }
  return result;
}

/// \brief Check convergence of a range of observations
template <typename StatisticsType>
IndividualConvergenceCheckResult<StatisticsType> component_convergence_check(
    Sampler const &sampler, Sampler const &sample_weight,
    SamplerComponent const &key, RequestedPrecision const &requested_precision,
    CountType N_samples_for_statistics,
    CalcStatisticsFunction<StatisticsType> calc_statistics_f) {
  if (calc_statistics_f == nullptr) {
    throw std::runtime_error(
        "Error in component_convergence_check: calc_statistics_f == nullptr");
  }

  if (sample_weight.n_samples() != 0) {
    return convergence_check(
        calc_statistics_f(
            sampler.component(key.component_index)
                .tail(N_samples_for_statistics),
            sample_weight.component(0).tail(N_samples_for_statistics)),
        requested_precision);
  } else {
    static Eigen::VectorXd empty_sample_weight;
    return convergence_check(
        calc_statistics_f(sampler.component(key.component_index)
                              .tail(N_samples_for_statistics),
                          empty_sample_weight),
        requested_precision);
  }
}

/// \brief Check convergence of all requested properties
///
/// \param samplers All samplers
/// \param sample_weight Weight to give to each observation, if applicable.
/// \param requested_precision Sampler components to check, with requested
///     precision
/// \param N_samples_for_equilibration Number of samples to exclude from
///     statistics because the system is out of equilibrium
/// \param sample_weight If size != 0, weight to give to each observation.
///     Weights are normalized to sum to N, the number of observations,
///     then applied to the properties.
///
/// \returns A ConvergenceCheckResults instance. Note that
///     N_samples_for_statistics is set to the total number of samples
///     if no convergence checks are requested (when
///     `requested_precision.size() == 0`), otherwise it will be equal to
///     `get_n_samples(samplers) - N_samples_for_equilibration`.
template <typename StatisticsType>
ConvergenceCheckResults<StatisticsType> convergence_check(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    Sampler const &sample_weight,
    std::map<SamplerComponent, RequestedPrecision> const &requested_precision,
    CountType N_samples_for_equilibration,
    CalcStatisticsFunction<StatisticsType> calc_statistics_f) {
  ConvergenceCheckResults<StatisticsType> results;
  CountType N_samples = get_n_samples(samplers);

  // if nothing to check, return
  if (!requested_precision.size()) {
    results.N_samples_for_statistics = N_samples;
    return results;
  }

  // if no samples after equilibration, return
  if (N_samples_for_equilibration >= N_samples) {
    return results;
  }

  // set number of samples used for statistics
  results.N_samples_for_statistics = N_samples - N_samples_for_equilibration;

  // will set to false if any requested sampler components are not converged
  results.all_converged = true;

  // check requested sampler components for equilibration
  for (auto const &p : requested_precision) {
    SamplerComponent const &key = p.first;
    RequestedPrecision const &component_requested_precision = p.second;

    // find and validate sampler name && component index
    Sampler const &sampler = *find_or_throw(samplers, key)->second;

    // do convergence check
    IndividualConvergenceCheckResult<StatisticsType> current =
        component_convergence_check(
            sampler, sample_weight, key, component_requested_precision,
            results.N_samples_for_statistics, calc_statistics_f);

    // combine results
    results.all_converged &= current.is_converged;
    results.individual_results.emplace(key, current);
  }
  return results;
}

}  // namespace monte
}  // namespace CASM

#endif
