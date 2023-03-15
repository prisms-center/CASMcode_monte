#ifndef CASM_monte_Results
#define CASM_monte_Results

#include <vector>

#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/state/State.hh"

namespace CASM {
namespace monte {

/// \brief Standard Monte Carlo calculation results data structure
///
/// This data structure stores results for a Monte Carlo calculation.
template <typename ConfigType, typename StatisticsType>
struct Results {
  typedef ConfigType config_type;
  typedef StatisticsType stats_type;

  /// Elapsed clocktime
  std::optional<TimeType> elapsed_clocktime;

  /// Map of <sampler name>:<sampler>
  /// - `Sampler` stores a Eigen::MatrixXd with sampled data. Rows of the matrix
  ///   corresponds to individual VectorXd samples. The matrices are
  ///   constructed with extra rows and encapsulated in a class so that
  ///   resizing can be done intelligently as needed. Sampler provides
  ///   accessors so that the data can be efficiently accessed by index or by
  ///   component name for equilibration and convergence checking of
  ///   individual components.
  std::map<std::string, std::shared_ptr<Sampler>> samplers;

  /// Map of <analysis name>:<value>
  std::map<std::string, Eigen::VectorXd> analysis;

  /// Vector of counts (could be pass or step) when a sample occurred
  std::vector<CountType> sample_count;

  /// Vector of times when a sample occurred
  std::vector<TimeType> sample_time;

  /// Vector of weights given to sample (not normalized)
  std::vector<double> sample_weight;

  /// Vector of clocktimes when a sample occurred
  std::vector<TimeType> sample_clocktime;

  /// Vector of the configuration when a sample occurred
  std::vector<ConfigType> sample_trajectory;

  /// Completion check results
  CompletionCheckResults<StatisticsType> completion_check_results;

  /// Number of acceptances
  long long n_accept;

  /// Number of rejections
  long long n_reject;
};

template <typename ConfigType, typename StatisticsType>
double confidence(Results<ConfigType, StatisticsType> const &results) {
  return results.completion_check_results.confidence;
}

template <typename ConfigType, typename StatisticsType>
CalcStatisticsFunction<StatisticsType> calc_statistics_f(
    Results<ConfigType, StatisticsType> const &results) {
  return results.completion_check_results.params.calc_statistics_f;
}

template <typename ConfigType, typename StatisticsType>
bool is_auto_converge_mode(Results<ConfigType, StatisticsType> const &results) {
  return results.completion_check_results.params.requested_precision.size() !=
         0;
}

template <typename ConfigType, typename StatisticsType>
bool is_requested_to_converge(
    SamplerComponent const &key,
    Results<ConfigType, StatisticsType> const &results) {
  auto const &requested_precision =
      results.completion_check_results.params.requested_precision;
  return requested_precision.find(key) != requested_precision.end();
}

template <typename ConfigType, typename StatisticsType>
double requested_precision(SamplerComponent const &key,
                           Results<ConfigType, StatisticsType> const &results) {
  auto const &requested_precision =
      results.completion_check_results.params.requested_precision;
  return requested_precision[key];
}

template <typename ConfigType, typename StatisticsType>
Index N_samples_for_all_to_equilibrate(
    Results<ConfigType, StatisticsType> const &results) {
  return results.completion_check_results.equilibriation_check_results
      .N_samples_for_all_to_equilibrate;
}

template <typename ConfigType, typename StatisticsType>
Index N_samples_for_statistics(
    Results<ConfigType, StatisticsType> const &results) {
  return results.completion_check_results.convergence_check_results
      .N_samples_for_statistics;
}

template <typename ConfigType, typename StatisticsType>
Index N_samples(Results<ConfigType, StatisticsType> const &results) {
  return get_n_samples(results.samplers);
}

template <typename ConfigType, typename StatisticsType>
bool all_equilibrated(Results<ConfigType, StatisticsType> const &results) {
  if (!results.completion_check_results.equilibration_check_results
           .all_equilibrated) {
    return false;
  }
  if (N_samples_for_statistics(results) == 0) {
    return false;
  }
  return true;
}

template <typename ConfigType, typename StatisticsType>
bool all_converged(Results<ConfigType, StatisticsType> const &results) {
  return results.completion_check_results.convergence_check_results
      .all_converged;
}

template <typename ConfigType, typename StatisticsType>
double acceptance_rate(Results<ConfigType, StatisticsType> const &results) {
  return static_cast<double>(results.n_accept) /
         static_cast<double>(results.n_accept + results.n_reject);
}

template <typename ConfigType, typename StatisticsType>
double elapsed_clocktime(Results<ConfigType, StatisticsType> const &results) {
  return results.elapsed_clocktime;
}

template <typename ConfigType, typename StatisticsType>
IndividualEquilibrationCheckResult equilibration_result(
    SamplerComponent const &key,
    Results<ConfigType, StatisticsType> const &results) {
  return results.completion_check_results.equilibration_check_results
      .individual_results[key];
}

template <typename ConfigType, typename StatisticsType>
IndividualConvergenceCheckResult<StatisticsType> convergence_result(
    SamplerComponent const &key,
    Results<ConfigType, StatisticsType> const &results) {
  return results.completion_check_results.convergence_check_results
      .individual_results[key];
}

template <typename ConfigType, typename StatisticsType>
struct QuantityStats {
  QuantityStats(std::string quantity_name, Sampler const &sampler,
                Results<ConfigType, StatisticsType> const &results)
      : shape(sampler.shape()),
        is_scalar((shape.size() == 0)),
        component_names(sampler.component_names()) {
    auto stats_f = calc_statistics_f(results);

    Index i = 0;
    for (auto const &component_name : component_names) {
      SamplerComponent key(quantity_name, i, component_name);

      if (is_auto_converge_mode(results)) {
        if (!all_equilibrated(results)) {
          if (is_requested_to_converge(key, results)) {
            is_converged.push_back(false);
            component_stats.push_back(std::nullopt);
          } else {
            is_converged.push_back(std::nullopt);
            component_stats.push_back(std::nullopt);
          }
        } else {
          if (is_requested_to_converge(key, results)) {
            auto const &convergence_r = convergence_result(key, results);
            is_converged.push_back(convergence_r.is_converged);
            component_stats.push_back(convergence_r.stats);
          } else {
            is_converged.push_back(std::nullopt);
            component_stats.push_back(stats_f(
                sampler.component(i).tail(N_samples_for_statistics(results)),
                confidence(results)));
          }
        }
      } else {
        is_converged.push_back(std::nullopt);
        component_stats.push_back(
            stats_f(sampler.component(i).tail(sampler.n_samples()),
                    confidence(results)));
      }
    }
  }

  std::vector<Index> shape;
  bool is_scalar;
  std::vector<std::string> component_names;

  /// \brief No value if not auto convergence mode; otherwise equilibration
  /// check result
  std::optional<bool> did_not_equilibrate;

  /// \brief No value if component not requested to converge; otherwise
  /// convergence check result
  std::vector<std::optional<bool>> is_converged;

  /// \brief No value if auto convergence mode and did not equilibrate;
  /// otherwise statistics
  std::vector<std::optional<StatisticsType>> component_stats;
};

}  // namespace monte
}  // namespace CASM

#endif
