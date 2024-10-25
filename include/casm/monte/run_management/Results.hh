#ifndef CASM_monte_Results
#define CASM_monte_Results

#include <vector>

#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/run_management/ResultsAnalysisFunction.hh"
#include "casm/monte/run_management/State.hh"
#include "casm/monte/sampling/SelectedEventFunctions.hh"
#include "casm/monte/sampling/StateSamplingFunction.hh"

namespace CASM {
namespace monte {

/// \brief Standard Monte Carlo calculation results data structure
///
/// This data structure stores results for a Monte Carlo calculation.
template <typename ConfigType, typename StatisticsType>
struct Results {
  typedef ConfigType config_type;
  typedef StatisticsType stats_type;

  Results(std::vector<std::string> _sampler_names,
          StateSamplingFunctionMap const &_sampling_functions,
          std::vector<std::string> _json_sampler_names,
          jsonStateSamplingFunctionMap const &_json_sampling_functions,
          ResultsAnalysisFunctionMap<ConfigType, StatisticsType> const
              &_analysis_functions)
      : sampler_names(_sampler_names),
        sampling_functions(_sampling_functions),
        json_sampler_names(_json_sampler_names),
        json_sampling_functions(_json_sampling_functions),
        analysis_functions(_analysis_functions),
        sample_weight({}) {}

  /// Quantities to sample
  std::vector<std::string> sampler_names;

  /// Sampling functions
  StateSamplingFunctionMap sampling_functions;

  /// JSON Quantities to sample
  std::vector<std::string> json_sampler_names;

  /// JSON sampling functions
  jsonStateSamplingFunctionMap json_sampling_functions;

  /// Results analysis functions
  ResultsAnalysisFunctionMap<ConfigType, StatisticsType> analysis_functions;

  /// Memory used at initialization
  std::optional<double> initial_memory_used_MiB;

  /// Memory used at initialization
  std::optional<double> final_memory_used_MiB;

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

  /// Map of <sampler name>:<json sampler>
  std::map<std::string, std::shared_ptr<jsonSampler>> json_samplers;

  /// Optional SelectedEventData - collected with each event occurance
  std::optional<SelectedEventData> selected_event_data;

  /// Map of <analysis name>:<value>
  std::map<std::string, Eigen::VectorXd> analysis;

  /// Vector of counts (could be pass or step) when a sample occurred
  std::vector<CountType> sample_count;

  /// Vector of times when a sample occurred
  std::vector<TimeType> sample_time;

  /// Vector of weights given to sample (not normalized)
  Sampler sample_weight;

  /// Vector of clocktimes when a sample occurred
  std::vector<TimeType> sample_clocktime;

  /// Vector of the configuration when a sample occurred
  std::vector<ConfigType> sample_trajectory;

  /// Completion check results
  CompletionCheckResults<StatisticsType> completion_check_results;

  /// Number of acceptances
  BigCountType n_accept;

  /// Number of rejections
  BigCountType n_reject;

  void reset() {
    initial_memory_used_MiB.reset();
    final_memory_used_MiB.reset();
    elapsed_clocktime.reset();
    samplers.clear();
    json_samplers.clear();
    selected_event_data.reset();
    analysis.clear();
    sample_count.clear();
    sample_time.clear();
    sample_weight.clear();
    sample_clocktime.clear();
    sample_trajectory.clear();
    completion_check_results.full_reset();
    n_accept = 0;
    n_reject = 0;

    for (auto const &sampler_name : sampler_names) {
      auto it = sampling_functions.find(sampler_name);
      if (it == sampling_functions.end()) {
        std::stringstream ss;
        ss << "Results::reset error." << std::endl;
        ss << "Failed to find sampling function '" << sampler_name << "'."
           << std::endl;
        ss << "Options are: " << std::endl;
        for (auto const &pair : json_sampling_functions) {
          ss << pair.first << std::endl;
        }
        ss << std::endl;
        throw std::runtime_error(ss.str());
      }
      auto const &function = sampling_functions.at(sampler_name);
      auto shared_sampler =
          std::make_shared<Sampler>(function.shape, function.component_names);
      samplers.emplace(function.name, shared_sampler);
    }

    for (auto const &name : json_sampler_names) {
      auto it = json_sampling_functions.find(name);
      if (it == json_sampling_functions.end()) {
        std::stringstream ss;
        ss << "Results::reset error." << std::endl;
        ss << "Failed to find json sampling function '" << name << "'."
           << std::endl;
        ss << "Options are: " << std::endl;
        for (auto const &pair : json_sampling_functions) {
          ss << pair.first << std::endl;
        }
        ss << std::endl;
        throw std::runtime_error(ss.str());
      }
      auto const &function = json_sampling_functions.at(name);
      auto shared_sampler = std::make_shared<jsonSampler>();
      json_samplers.emplace(function.name, shared_sampler);
    }
  }
};

template <typename ConfigType, typename StatisticsType>
double confidence(Results<ConfigType, StatisticsType> const &results) {
  return results.completion_check_results.params.confidence;
}

template <typename ConfigType, typename StatisticsType>
CalcStatisticsFunction<StatisticsType> get_calc_statistics_f(
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
  return results.completion_check_results.equilibration_check_results
      .N_samples_for_all_to_equilibrate;
}

template <typename ConfigType, typename StatisticsType>
Index N_samples(Results<ConfigType, StatisticsType> const &results) {
  return get_n_samples(results.samplers);
}

template <typename ConfigType, typename StatisticsType>
Index N_samples_for_statistics(
    Results<ConfigType, StatisticsType> const &results) {
  if (is_auto_converge_mode(results)) {
    return results.completion_check_results.convergence_check_results
        .N_samples_for_statistics;
  } else {
    return N_samples(results);
  }
}

template <typename ConfigType, typename StatisticsType>
bool all_equilibrated(Results<ConfigType, StatisticsType> const &results) {
  return results.completion_check_results.equilibration_check_results
      .all_equilibrated;
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
std::optional<double> elapsed_clocktime(
    Results<ConfigType, StatisticsType> const &results) {
  return results.elapsed_clocktime;
}

template <typename ConfigType, typename StatisticsType>
IndividualEquilibrationCheckResult equilibration_result(
    SamplerComponent const &key,
    Results<ConfigType, StatisticsType> const &results) {
  return results.completion_check_results.equilibration_check_results
      .individual_results.find(key)
      ->second;
}

template <typename ConfigType, typename StatisticsType>
IndividualConvergenceCheckResult<StatisticsType> convergence_result(
    SamplerComponent const &key,
    Results<ConfigType, StatisticsType> const &results) {
  return results.completion_check_results.convergence_check_results
      .individual_results.find(key)
      ->second;
}

template <typename ConfigType, typename StatisticsType>
struct QuantityStats {
  QuantityStats(std::string quantity_name, Sampler const &sampler,
                Results<ConfigType, StatisticsType> const &results)
      : shape(sampler.shape()),
        is_scalar((shape.size() == 0)),
        component_names(sampler.component_names()) {
    auto calc_statistics_f = get_calc_statistics_f(results);
    if (calc_statistics_f == nullptr) {
      throw std::runtime_error(
          "Error in QuantityStats: calc_statistics_f == nullptr");
    }

    Index i = 0;
    for (auto const &component_name : component_names) {
      SamplerComponent key(quantity_name, i, component_name);

      if (is_auto_converge_mode(results)) {
        if (N_samples_for_statistics(results) == 0) {
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
            Index N_stats = N_samples_for_statistics(results);
            if (results.sample_weight.n_samples() == 0) {
              static Eigen::VectorXd empty_sample_weight;
              component_stats.push_back(calc_statistics_f(
                  sampler.component(key.component_index).tail(N_stats),
                  empty_sample_weight));
            } else {
              component_stats.push_back(calc_statistics_f(
                  sampler.component(key.component_index).tail(N_stats),
                  results.sample_weight.component(0).tail(N_stats)));
            }
          }
        }
      } else {
        is_converged.push_back(std::nullopt);
        Index N_stats = sampler.n_samples();
        if (results.sample_weight.n_samples() == 0) {
          static Eigen::VectorXd empty_sample_weight;
          component_stats.push_back(calc_statistics_f(
              sampler.component(key.component_index).tail(N_stats),
              empty_sample_weight));
        } else {
          component_stats.push_back(calc_statistics_f(
              sampler.component(key.component_index).tail(N_stats),
              results.sample_weight.component(0).tail(N_stats)));
        }
      }

      ++i;
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

template <typename ResultsType>
QuantityStats<typename ResultsType::config_type,
              typename ResultsType::stats_type>
make_quantity_stats(std::string quantity_name, Sampler const &sampler,
                    ResultsType const &results) {
  return QuantityStats<typename ResultsType::config_type,
                       typename ResultsType::stats_type>(quantity_name, sampler,
                                                         results);
}

}  // namespace monte
}  // namespace CASM

#endif
