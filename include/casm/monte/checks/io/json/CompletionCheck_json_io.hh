#ifndef CASM_monte_checks_CompletionCheck_json_io
#define CASM_monte_checks_CompletionCheck_json_io

#include "casm/casm_io/json/InputParser_impl.hh"
#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/state/StateSampler.hh"

namespace CASM {
namespace monte {
struct CompletionCheckParams;

/// \brief Construct CompletionCheckParams from JSON
template <typename ConfigType>
void parse(InputParser<CompletionCheckParams> &parser,
           StateSamplingFunctionMap<ConfigType> const &sampling_functions);

// --- Inline definitions ---

namespace CompletionCheck_json_io_impl {

/// \brief If successfully parsed, result is not empty
template <typename ConfigType>
std::unique_ptr<StateSamplingFunction<ConfigType>> _parse_quantity(
    InputParser<CompletionCheckParams> &parser, fs::path const &option) {
  std::unique_ptr<std::string> quantity =
      parser.require<std::string>(option / "quantity");
  if (quantity == nullptr) {
    return std::unique_ptr<StateSamplingFunction<ConfigType>>();
  }
  auto function_it = sampling_functions.find(*quantity);
  if (function_it == sampling_functions.end()) {
    std::stringstream msg;
    msg << "Error: \"" << *quantity << "\" is not a sampling option.";
    parser.insert_error(option / *quantity, msg.str());
    return std::unique_ptr<StateSamplingFunction<ConfigType>>();
  }
  return std::make_unique<StateSamplingFunction<ConfigType>>(*function_it);
}

/// \brief If successfully parsed, adds elements to convergence_check_params
template <typename ConfigType>
void _parse_component_index(
    InputParser<CompletionCheckParams> &parser, fs::path const &option,
    StateSamplingFunction<ConfigType> const &function,
    SamplerConvergenceParams const &sampler_convergence_params,
    std::map<SamplerComponent, SamplerConvergenceParams>
        &convergence_check_params) {
  // converge components specified by index
  std::vector<Index> component_index;
  parser.optional(component_index, option / "component_index");

  for (Index index : component_index) {
    auto size = function.component_names.size();
    if (index < 0 || index >= size) {
      std::stringstream msg;
      msg << "Error: For \"" << *quantity << "\", component index " << index
          << " is out of range. Valid range is [0," << size << ").";
      parser.insert_error(option / "component_index", msg.str());
      continue;
    }
    convergence_check_params.emplace(
        SamplerComponent(function.name, index, function.component_names[index]),
        sampler_convergence_params);
  }
}

/// \brief If successfully parsed, adds elements to convergence_check_params
template <typename ConfigType>
void _parse_component_name(
    InputParser<CompletionCheckParams> &parser, fs::path const &option,
    StateSamplingFunction<ConfigType> const &function,
    SamplerConvergenceParams const &sampler_convergence_params,
    std::map<SamplerComponent, SamplerConvergenceParams>
        &convergence_check_params) {
  // converge components specified by name
  std::vector<std::string> component_name;
  parser.optional(component_name, option / "component_name");

  for (Index name : component_name) {
    auto begin = function.component_names.begin();
    auto end = function.component_names.end();
    auto it = std::find(begin, end, name);
    if (it == end) {
      std::stringstream msg;
      msg << "Error: For \"" << function.name << "\", component name " << name
          << " is not valid.";
      parser.insert_error(option / "component_name", msg.str());
      continue;
    }
    Index index = std::distance(begin, it);
    SamplerComponent first(*quantity, index, name);
    SamplerConvergenceParams second(precision);
    convergence_check_params.emplace(first, second);
  }
}

/// \brief If successfully parsed, adds elements to convergence_check_params
template <typename ConfigType>
void _parse_components(
    InputParser<CompletionCheckParams> &parser, fs::path const &option,
    StateSamplingFunction<ConfigType> const &function,
    SamplerConvergenceParams const &sampler_convergence_params,
    std::map<SamplerComponent, SamplerConvergenceParams>
        &convergence_check_params) {
  bool has_index =
      (parser.self.find_at(option / "component_index") != parser.self.end());
  bool has_name =
      (parser.self.find_at(option / "component_name") != parser.self.end());
  if (has_index && has_name) {
    parser.insert_error(option,
                        "Error: cannot specify both \"component_index\" and "
                        "\"component_name\"");
  } else if (has_index) {
    _parse_component_index(parser, option, function, sampler_convergence_params,
                           convergence_check_params);
  } else if (has_name) {
    _parse_component_name(parser, option, function, sampler_convergence_params,
                          convergence_check_params);
  } else {
    // else, converge all components
    for (Index index = 0; index < function.component_names.size(); ++index) {
      convergence_check_params.emplace(
          SamplerComponent(function.name, index,
                           function.component_names[index]),
          sampler_convergence_params);
    }
  }
}

/// \brief If successfully parsed, adds elements to convergence_check_params
template <typename ConfigType>
void _parse_convergence_criteria(
    InputParser<CompletionCheckParams> &parser,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions,
    std::map<SamplerComponent, SamplerConvergenceParams>
        &convergence_check_params) {
  auto it = parser.self.find("convergence");
  if (it == parser.self.end()) {
    return;
  }
  if (!it->is_array()) {
    parser.insert_error("convergence",
                        "Error: \"convergence\" must be an array");
    return;
  }

  for (Index i = 0; i < it->size(); ++i) {
    fs::path option = fs::path("convergence") / std::to_string(i);

    // parse "quantity"
    std::unique_ptr<StateSamplingFunction<ConfigType>> function =
        _parse_quantity(parser, option);
    if (function == nullptr) {
      continue;
    }

    // parse "precision"
    double precision;
    parser.require(precision, option / "precision");
    SamplerConvergenceParams sampler_convergence_params(precision);

    // parse "component_index", "component_name",
    //   or default (neither given, converges all components)
    _parse_components(parser, option, *function, sampler_convergence_params,
                      convergence_check_params);
  }
}

}  // namespace CompletionCheck_json_io_impl

/// \brief Construct CompletionCheckParams from JSON
///
/// Expected:
///   cutoff: dict (optional, default={})
///     Hard limits that prevent the calculation from stopping too soon, or
///     force it to stop. May include:
///
///       count: dict (optional, default={})
///         Sets a minimum and maximum for how many steps or passes the
///         calculation runs. If sampling by pass, then the count refers to the
///         number of passes, else the count refers to the number of steps. May
///         include:
///
///           min: int (optional, default=null)
///             Applies a minimum count, if not null.
///
///           max: int (optional, default=null)
///             Applies a maximum count, if not null.
///
///       sample: dict (optional, default={})
///         Sets a minimum and maximum for how many samples are taken. Options
///         are `min` and `max`, the same as for `count`.
///
///       time: dict (optional, default={})
///         If a time-based calculation, sets minimum and maximum cuttoffs for
///         time. Options are `min` and `max`, the same as for `count`.
///
///   begin: int (optional, default=10)
///     Begin checking convergence of sampled quantities for completion, after
///     this many samples have been taken.
///
///   frequency: int (optional, default=10)
///     How many samples in between checking convergence of sampled quantities.
///
///   confidence: number (optional, default=0.95)
///     Confidence level, in range (0, 1.0), used for calculated precision of
///     the mean.
///
///   convergence: array of dict (optional)
///     Specify which components of which sampled quantities should be checked
///     for convergence. When all specified are converged to the requested
///     precision, the calculation will finish. It consists of an array of dict,
///     each dict having the following format. If neither `component_index`,
///     nor `component_name` is provided, then all components of the specified
///     quantity will be converged to the specified precision.
///
///       quantity: string (required)
///         Name of sampled quantity
///
///       confidence: number (optional, default=0.95)
///         Confidence level, in range (0, 1.0), to check convergence.
///
///       precision: number (required)
///         The required (absolute) precision in the average of the quantity
///         for the calculation to be considered converged.
///
///       component_index: array of int (optional)
///         Array of indices of the selected sampled quantity to converge to
///         the specified precision. Example:
///
///           {
///             "quantity": "comp_n", "component_index": [1, 2]
///           }
///
///       component_name: array of string (optional)
///         Array of names of the components of the selected sampled quantity
///         to converge to the specified precision. Example:
///
///           {
///             "quantity": "comp_n", "component_name": ["Va", "O"]
///           }
///
template <typename ConfigType>
void parse(InputParser<CompletionCheckParams> &parser,
           StateSamplingFunctionMap<ConfigType> const &sampling_functions) {
  using namespace CompletionCheck_json_io_impl;
  CompletionCheckParams completion_check_params;

  // parse "cutoff"
  auto cutoff_params_subparser = parser.optional<CutoffCheckParams>("cutoff");
  if (cutoff_params_subparser->value != nullptr) {
    completion_check_params.cutoff_params = *cutoff_params_subparser->value;
  }

  // parse "convergence"
  _parse_convergence_criteria(parser, sampling_functions,
                              completion_check_params.convergence_check_params);

  parser.optional_else(completion_check_params.confidence, "confidence", 0.95);
  parser.optional_else(completion_check_params.check_begin, "begin", 10);
  parser.optional_else(completion_check_params.check_frequency, "frequency", 1);

  if (parser.valid()) {
    parser.value = std::make_unique(completion_check_params);
  }
}

}  // namespace monte
}  // namespace CASM

#endif
