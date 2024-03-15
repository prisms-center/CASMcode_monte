#ifndef CASM_monte_RequestedPrecisionConstructor
#define CASM_monte_RequestedPrecisionConstructor

#include <set>

#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/sampling/StateSamplingFunction.hh"

namespace CASM {
namespace monte {

/// \brief Helper for compact construction of requested precision
///
/// Allows setting absolute or relative precision to the specified level for
/// the specified quantities. By default, all components are converged to
/// the same level. If `component_name` or `component_index` are specified,
/// then only the specified components are requested to converge to that level.
///
/// As a shortcut, this class is constructed using the `converge` method.
/// For example:
/// \code
/// converge(sampling_functions, completion_check_params)
///         .set_abs_precision("formation_energy", 0.001)
///         .set_rel_precision("parametric_composition", 0.001, {"a"})
///         .set_abs_and_rel_precision("corr", 0.001, 0.001 {0,1,2,3})
/// \endcode
///
template <typename StatisticsType>
struct RequestedPrecisionConstructor {
  /// \brief Constructor
  RequestedPrecisionConstructor(
      StateSamplingFunctionMap const &_sampling_functions,
      CompletionCheckParams<StatisticsType> &_completion_check_params)
      : sampling_functions(_sampling_functions),
        completion_check_params(_completion_check_params) {}

  RequestedPrecisionConstructor &set_abs_precision(std::string sampler_name,
                                                   double abs_precision) {
    return set_abs_precision(sampler_name, abs_precision,
                             all_component_index(sampler_name));
  }

  RequestedPrecisionConstructor &set_rel_precision(std::string sampler_name,
                                                   double rel_precision) {
    return set_rel_precision(sampler_name, rel_precision,
                             all_component_index(sampler_name));
  }

  RequestedPrecisionConstructor &set_abs_and_rel_precision(
      std::string sampler_name, double abs_precision, double rel_precision) {
    return set_abs_and_rel_precision(sampler_name, abs_precision, rel_precision,
                                     all_component_index(sampler_name));
  }

  RequestedPrecisionConstructor &set_abs_precision(
      std::string sampler_name, double abs_precision,
      std::set<int> component_index) {
    auto const &f = sampling_functions.at(sampler_name);
    for (int i : component_index) {
      SamplerComponent key(sampler_name, i, f.component_names[i]);
      completion_check_params.requested_precision[key] =
          RequestedPrecision::abs(abs_precision);
    }
    return *this;
  }

  RequestedPrecisionConstructor &set_rel_precision(
      std::string sampler_name, double rel_precision,
      std::set<int> component_index) {
    auto const &f = sampling_functions.at(sampler_name);
    for (int i : component_index) {
      SamplerComponent key(sampler_name, i, f.component_names[i]);
      completion_check_params.requested_precision[key] =
          RequestedPrecision::rel(rel_precision);
    }
    return *this;
  }

  RequestedPrecisionConstructor &set_abs_and_rel_precision(
      std::string sampler_name, double abs_precision, double rel_precision,
      std::set<int> component_index) {
    auto const &f = sampling_functions.at(sampler_name);
    for (int i : component_index) {
      SamplerComponent key(sampler_name, i, f.component_names[i]);
      completion_check_params.requested_precision[key] =
          RequestedPrecision::abs_and_rel(abs_precision, rel_precision);
    }
    return *this;
  }

  RequestedPrecisionConstructor &set_abs_precision(
      std::string sampler_name, double abs_precision,
      std::set<std::string> component_name) {
    return set_abs_precision(
        sampler_name, abs_precision,
        component_index_from_names(sampler_name, component_name));
  }

  RequestedPrecisionConstructor &set_rel_precision(
      std::string sampler_name, double rel_precision,
      std::set<std::string> component_name) {
    return set_rel_precision(
        sampler_name, rel_precision,
        component_index_from_names(sampler_name, component_name));
  }

  RequestedPrecisionConstructor &set_abs_and_rel_precision(
      std::string sampler_name, double abs_precision, double rel_precision,
      std::set<std::string> component_name) {
    return set_abs_and_rel_precision(
        sampler_name, abs_precision, rel_precision,
        component_index_from_names(sampler_name, component_name));
  }

  /// \brief Conversion operator
  operator std::map<SamplerComponent, RequestedPrecision> const &() const {
    return completion_check_params.requested_precision;
  }

  StateSamplingFunctionMap const &sampling_functions;
  CompletionCheckParams<StatisticsType> &completion_check_params;

  std::set<int> all_component_index(std::string sampler_name) {
    std::set<int> component_index;
    for (int i = 0;
         i < sampling_functions.at(sampler_name).component_names.size(); ++i) {
      component_index.insert(i);
    }
    return component_index;
  }

  std::set<int> component_index_from_names(
      std::string sampler_name, std::set<std::string> component_name) {
    std::set<int> component_index;
    int i = 0;
    for (std::string _name :
         sampling_functions.at(sampler_name).component_names) {
      if (component_name.count(_name)) {
        component_index.insert(i);
      }
      ++i;
    }
    return component_index;
  }
};

/// \brief Helper for setting completion_check_params.requested_precision
///
/// Example usage:
/// \code
/// converge(sampling_functions, completion_check_params)
///         .set_abs_precision("formation_energy", 0.001)
///         .set_rel_precision("parametric_composition", 0.001, {"a"})
///         .set_abs_and_rel_precision("corr", 0.001, 0.001 {0,1,2,3})
/// \endcode
///
/// Allows setting absolute or relative precision to the specified level for
/// the specified quantities. By default, all components are converged to
/// the same level. If `component_name` or `component_index` are specified,
/// then only the specified components are requested to converge to that level.
///
/// \param sampling_functions State sampling function map
/// \param completion_check_param Completion check parameters to set
///     requested_precision
/// \return rpc, A RequestedPrecisionConstructor
template <typename StatisticsType>
RequestedPrecisionConstructor<StatisticsType> converge(
    StateSamplingFunctionMap const &sampling_functions,
    CompletionCheckParams<StatisticsType> &completion_check_params) {
  return RequestedPrecisionConstructor<StatisticsType>(sampling_functions,
                                                       completion_check_params);
}

}  // namespace monte
}  // namespace CASM

#endif
