#include "casm/monte/checks/EquilibrationCheck.hh"

#include <iostream>

namespace CASM {
namespace monte {

/// \brief Check if a range of observations have equilibrated
///
/// This uses the following algorithm, based on:
///  Van de Walle and Asta, Modelling Simul. Mater. Sci. Eng. 10 (2002) 521–538.
///
/// Partition observations into three ranges:
///
///   - equilibriation stage:  [0, start1)
///   - first partition:  [start1, start2)
///   - second partition: [start2, N)
///
/// where N is observations.size(), start1 and start 2 are indices into
/// observations such 0 <= start1 < start2 <= N, the number of elements in
/// the first and second partition are the same (within 1).
///
/// The calculation is considered equilibrated at start1 if the mean of the
/// elements in the first and second partition are approximately equal to the
/// desired precsion: (std::abs(mean1 - mean2) < prec).
///
/// Additionally, the value start1 is incremented as much as needed to ensure
/// that the equilibriation stage has observations on either side of the total
/// mean.
///
/// If all observations are approximately equal, then:
/// - is_equilibrated = true
/// - N_samples_for_equilibration = 0
///
/// If the equilibration conditions are met, the result contains:
/// - is_equilibrated = true
/// - N_samples_for_equilibration = start1
///
/// If the equilibration conditions are not met, the result contains:
/// - is_equilibrated = false
/// - N_samples_for_equilibration = <undefined>
///
/// \param observations An Eigen::VectorXd of observations
/// \param prec Desired absolute precision (<X> +/- prec)
/// \param check_all If true, return results for all requested sampler
/// components. If false, return when the first component that is not
/// equilibrated is encountered.
/// \returns An IndividualEquilibrationCheckResult instance
///
IndividualEquilibrationCheckResult _default_equilibration_check(
    Eigen::VectorXd const &observations, double prec) {
  if (observations.size() == 0) {
    throw std::runtime_error(
        "Error in equilibration_check: observations.size()==0");
  }
  CountType start1, start2, N;
  IndividualEquilibrationCheckResult result;
  double sum1, sum2;
  double eps =
      (observations(0) == 0.0) ? 1e-8 : std::abs(observations(0)) * 1e-8;

  N = observations.size();
  bool is_even = ((N % 2) == 0);

  // -------------------------------------------------
  // if the values are all the same to observations(0)*1e-8, set
  // m_is_equilibrated = true; m_equil_samples = 0;
  bool all_same = true;
  for (CountType i = 0; i < N; i++)
    if (std::abs(observations(i) - observations(0)) > eps) {
      all_same = false;
      break;
    }
  if (all_same) {
    result.is_equilibrated = true;
    result.N_samples_for_equilibration = 0;
    return result;
  }

  // find partitions
  start1 = 0;
  start2 = (is_even) ? N / 2 : (N / 2) + 1;

  // find sums for each partition
  sum1 = observations.head(start2).sum();
  sum2 = observations.segment(start2, N - start2).sum();

  // increment start1 (and update start2, sum1, and sum2)
  // until abs(mean1 - mean2) < prec
  while (std::abs((sum1 / (start2 - start1)) - (sum2 / (N - start2))) > prec &&
         start1 < N - 2) {
    if (is_even) {
      sum1 -= observations(start1);
      sum1 += observations(start2);
      sum2 -= observations(start2);
      start2++;
    } else {
      sum1 -= observations(start1);
    }

    start1++;
    is_even = !is_even;
  }

  // ensure that the equilibration stage
  // has values on either side of the total mean
  double mean_tot = (sum1 + sum2) / (N - start1);
  if (observations(start1) < mean_tot) {
    while (observations(start1) < mean_tot && start1 < N - 1) start1++;
  } else {
    while (observations(start1) > mean_tot && start1 < N - 1) start1++;
  }

  result.is_equilibrated = (start1 < N - 1);
  result.N_samples_for_equilibration = start1;
  return result;
}

IndividualEquilibrationCheckResult default_equilibration_check(
    Eigen::VectorXd const &observations, Eigen::VectorXd const &sample_weight,
    RequestedPrecision requested_precision) {
  double prec;
  if (requested_precision.abs_convergence_is_required) {
    prec = requested_precision.abs_precision;
  } else if (requested_precision.rel_convergence_is_required) {
    prec = std::abs(observations.mean() * requested_precision.rel_precision);
  } else {
    IndividualEquilibrationCheckResult result;
    result.is_equilibrated = true;
    result.N_samples_for_equilibration = 0;
    return result;
  }

  if (sample_weight.size() == 0) {
    return _default_equilibration_check(observations, prec);
  } else {
    // weighted observations
    if (sample_weight.size() != observations.size()) {
      throw std::runtime_error(
          "Error in equilibration_check: sample_weight.size() != "
          "observations.size()");
    }

    // if weighting, use weighted_observation(i) = sample_weight[i] *
    // observation(i) * N / W where W = sum_i sample_weight[i]; same
    // weight_factor N/W applies for all properties
    double weight_factor;
    Index N = sample_weight.size();
    double W = 0.0;
    for (Index i = 0; i < sample_weight.size(); ++i) {
      W += sample_weight[i];
    }
    weight_factor = N / W;

    Eigen::VectorXd weighted_observations = observations;
    for (Index i = 0; i < weighted_observations.size(); ++i) {
      weighted_observations(i) *= weight_factor * sample_weight[i];
    }

    return _default_equilibration_check(weighted_observations, prec);
  }
}

/// \brief Check convergence of all requested properties
///
/// \param requested_precision Sampler components to check, with requested
///     precision
/// \param samplers All samplers
/// \param sample_weight If size != 0, weight to give to each observation.
///     Weights are normalized to sum to N, the number of observations,
///     then applied to the properties.
/// \param check_all If true, check convergence of all requested properties.
///     Otherwise, break if one is found to not be equilibrated.
///
/// \returns A ConvergenceCheckResults instance. Note that
///     N_samples_for_statistics is set to the total number of samples
///     if no convergence checks are requested (when
///     `convergence_check_params.size() == 0`), otherwise it will be equal to
///     `get_n_samples(samplers) - N_samples_for_equilibration`.
EquilibrationCheckResults equilibration_check(
    EquilibrationCheckFunction equilibration_check_f,
    std::map<SamplerComponent, RequestedPrecision> const &requested_precision,
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    Sampler const &sample_weight, bool check_all) {
  if (equilibration_check_f == nullptr) {
    throw std::runtime_error(
        "Error in equilibration_check: equilibration_check_f == nullptr");
  }

  EquilibrationCheckResults results;

  if (!requested_precision.size()) {
    return results;
  }

  // will set to false if any requested sampler components are not equilibrated
  results.all_equilibrated = true;

  // check requested sampler components for equilibration
  for (auto const &p : requested_precision) {
    SamplerComponent const &key = p.first;
    RequestedPrecision const &component_requested_precision = p.second;

    // find and validate sampler name && component index
    Sampler const &sampler = *find_or_throw(samplers, key)->second;

    // do equilibration check
    IndividualEquilibrationCheckResult current = equilibration_check_f(
        sampler.component(key.component_index),  // observations
        sample_weight.component(0), component_requested_precision);

    // combine results
    results.N_samples_for_all_to_equilibrate =
        std::max(results.N_samples_for_all_to_equilibrate,
                 current.N_samples_for_equilibration);
    results.all_equilibrated &= current.is_equilibrated;
    results.individual_results.emplace(key, current);

    // break if possible
    if (!check_all && !results.all_equilibrated) {
      break;
    }
  }
  return results;
}

}  // namespace monte
}  // namespace CASM
