#ifndef CASM_monte_checks_CompletionCheck_json_io
#define CASM_monte_checks_CompletionCheck_json_io

#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/InputParser_impl.hh"
#include "casm/casm_io/json/optional.hh"
#include "casm/monte/BasicStatistics.hh"
#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/checks/io/json/ConvergenceCheck_json_io.hh"
#include "casm/monte/checks/io/json/CutoffCheck_json_io.hh"
#include "casm/monte/checks/io/json/EquilibrationCheck_json_io.hh"
#include "casm/monte/sampling/StateSamplingFunction.hh"

namespace CASM {
namespace monte {

template <typename StatisticsType>
struct CompletionCheckParams;

/// \brief Construct CompletionCheckParams<BasicStatistics> from JSON
void parse(InputParser<CompletionCheckParams<BasicStatistics>> &parser,
           StateSamplingFunctionMap const &sampling_functions);

/// \brief Convert CompletionCheckParams<BasicStatistics> to JSON
jsonParser &to_json(
    CompletionCheckParams<BasicStatistics> const &completion_check_params,
    jsonParser &json);

/// \brief CompletionCheckResults to JSON
template <typename StatisticsType>
jsonParser &to_json(CompletionCheckResults<StatisticsType> const &value,
                    jsonParser &json);

// --- Inline definitions ---

namespace CompletionCheck_json_io_impl {

/// \brief If successfully parsed, result is not empty
template <typename StatisticsType>
std::unique_ptr<StateSamplingFunction> _parse_quantity(
    InputParser<CompletionCheckParams<StatisticsType>> &parser,
    StateSamplingFunctionMap const &sampling_functions,
    fs::path const &option) {
  std::unique_ptr<std::string> quantity =
      parser.template require<std::string>(option / "quantity");
  if (quantity == nullptr) {
    return std::unique_ptr<StateSamplingFunction>();
  }
  auto function_it = sampling_functions.find(*quantity);
  if (function_it == sampling_functions.end()) {
    std::stringstream msg;
    msg << "Error: \"" << *quantity << "\" is not a sampling option.";
    parser.insert_error(option / *quantity, msg.str());
    return std::unique_ptr<StateSamplingFunction>();
  }
  return std::make_unique<StateSamplingFunction>(function_it->second);
}

/// \brief If successfully parsed, adds elements to requested_precision
template <typename StatisticsType>
void _parse_component_index(
    InputParser<CompletionCheckParams<StatisticsType>> &parser,
    fs::path const &option, StateSamplingFunction const &function,
    RequestedPrecision const &precision,
    std::map<SamplerComponent, RequestedPrecision> &requested_precision) {
  // converge components specified by index
  std::vector<Index> component_index;
  parser.optional(component_index, option / "component_index");

  for (Index index : component_index) {
    auto size = function.component_names.size();
    if (index < 0 || index >= size) {
      std::stringstream msg;
      msg << "Error: For \"" << function.name << "\", component index " << index
          << " is out of range. Valid range is [0," << size << ").";
      parser.insert_error(option / "component_index", msg.str());
      continue;
    }
    requested_precision.emplace(
        SamplerComponent(function.name, index, function.component_names[index]),
        precision);
  }
}

/// \brief If successfully parsed, adds elements to requested_precision
template <typename StatisticsType>
void _parse_component_name(
    InputParser<CompletionCheckParams<StatisticsType>> &parser,
    fs::path const &option, StateSamplingFunction const &function,
    RequestedPrecision const &precision,
    std::map<SamplerComponent, RequestedPrecision> &requested_precision) {
  // converge components specified by name
  std::vector<std::string> component_name;
  parser.optional(component_name, option / "component_name");

  for (std::string const &name : component_name) {
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
    requested_precision.emplace(SamplerComponent(function.name, index, name),
                                precision);
  }
}

/// \brief If successfully parsed, adds elements to requested_precision
template <typename StatisticsType>
void _parse_components(
    InputParser<CompletionCheckParams<StatisticsType>> &parser,
    fs::path const &option, StateSamplingFunction const &function,
    RequestedPrecision const &precision,
    std::map<SamplerComponent, RequestedPrecision> &requested_precision) {
  bool has_index =
      (parser.self.find_at(option / "component_index") != parser.self.end());
  bool has_name =
      (parser.self.find_at(option / "component_name") != parser.self.end());
  if (has_index) {
    _parse_component_index(parser, option, function, precision,
                           requested_precision);
  }
  if (has_name) {
    _parse_component_name(parser, option, function, precision,
                          requested_precision);
  }
  if (!has_index && !has_name) {
    // else, converge all components
    for (Index index = 0; index < function.component_names.size(); ++index) {
      requested_precision.emplace(
          SamplerComponent(function.name, index,
                           function.component_names[index]),
          precision);
    }
  }
}

/// \brief If successfully parsed, adds elements to requested_precision
template <typename StatisticsType>
void _parse_convergence_criteria(
    InputParser<CompletionCheckParams<StatisticsType>> &parser,
    StateSamplingFunctionMap const &sampling_functions,
    std::map<SamplerComponent, RequestedPrecision> &requested_precision) {
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
    std::unique_ptr<StateSamplingFunction> function =
        _parse_quantity(parser, sampling_functions, option);
    if (function == nullptr) {
      continue;
    }

    // parse "abs_precision", "rel_precision", "precision" (deprecated)
    RequestedPrecision precision;
    from_json(precision, (*it)[i]);

    // parse "component_index", "component_name",
    //   or default (neither given, converges all components)
    _parse_components(parser, option, *function, precision,
                      requested_precision);
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
///       clocktime: dict (optional, default={})
///         Sets minimum and maximum cuttoffs for elapsed calculation time in
///         seconds. Options are `min` and `max`, the same as for `count`.
///
///   spacing: string (optional, default="linear")
///     The spacing of convergence checks. One of "linear" or "log".
///
///     For "linear" spacing, the n-th check will be taken when:
///
///         sample(n) = begin + period * n
///
///     For "log" spacing, the n-th check will be taken when:
///
///         sample(n) = begin + round( base ^ (n + shift) )
///
///     However, if sample(n) - sample(n-1) > period_max, then subsequent
///     samples are taken every `period_max` samples.
///
///   begin: int (optional)
///     The earliest number of samples at which to begin convergence checking.
///     Default is 100 for linear checking spacing, 0 for log checking spacing.
///
///   period: int (optional, default=100)
///     Number of samples per check for linear check spacing.
///
///   base: number (optional, default=10.0)
///     Base for log check spacing.
///
///   period_max: int (optional, default=10000.0)
///     Maximum check spacing for log check spacing.
///
///   shift: number (optional, default=2.0)
///     Offset for the log check spacing exponent.
///
///   confidence: number (optional, default=0.95)
///     Confidence level, in range (0, 1.0), used for calculated precision of
///     the mean.
///
///   weighted_observations_method: int (optional, default=1)
///     Method used to estimate precision in the sample mean when observations
///     are weighted (N-fold way method). Option are:
///
///     1) Calculate weighted sample variance directly from weighted samples
///         and only autocorrelation factor (1+rho)/(1-rho) from resampled
///         observations
///     2) Calculate all statistics from resampled observations
///
///   n_resamples: int (optional, default=10000)
///     Number of resampled observations to make for autocovariance estimation
///     when observations are weighted.
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
///       precision: number (required)
///         The required (absolute) precision in the average of the quantity
///         for the calculation to be considered converged.
///
///       component_index: array of int (optional)
///         Array of indices of the selected sampled quantity to converge to
///         the specified precision. Example:
///
///           {
///             "quantity": "comp_n",
///             "precision": 0.001,
///             "component_index": [1, 2]
///           }
///
///       component_name: array of string (optional)
///         Array of names of the components of the selected sampled quantity
///         to converge to the specified precision. Example:
///
///           {
///             "quantity": "comp_n",
///             "precision": 0.001,
///             "component_name": ["Va", "O"]
///           }
///
inline void parse(InputParser<CompletionCheckParams<BasicStatistics>> &parser,
                  StateSamplingFunctionMap const &sampling_functions) {
  using namespace CompletionCheck_json_io_impl;

  double confidence = 0.95;
  parser.optional(confidence, "confidence");

  Index weighted_observations_method = 1;
  parser.optional(weighted_observations_method, "weighted_observations_method");

  Index n_resamples = 10000;
  parser.optional(n_resamples, "n_resamples");

  CompletionCheckParams<BasicStatistics> completion_check_params;
  completion_check_params.equilibration_check_f =
      monte::default_equilibration_check,
  completion_check_params.calc_statistics_f = BasicStatisticsCalculator(
      confidence, weighted_observations_method, n_resamples);

  // parse "cutoff"
  auto cutoff_params_subparser =
      parser.template subparse_else<CutoffCheckParams>("cutoff",
                                                       CutoffCheckParams());
  if (cutoff_params_subparser->valid()) {
    completion_check_params.cutoff_params =
        std::move(*cutoff_params_subparser->value);
  }

  // parse "convergence"
  _parse_convergence_criteria(parser, sampling_functions,
                              completion_check_params.requested_precision);

  // "spacing"
  std::string spacing = "linear";
  parser.optional(spacing, "spacing");
  if (spacing == "linear") {
    completion_check_params.log_spacing = false;
    completion_check_params.check_begin = 100;
    completion_check_params.check_period = 100;
  } else if (spacing == "log") {
    completion_check_params.log_spacing = true;
    completion_check_params.check_begin = 0;
    completion_check_params.check_base = 10.0;
    completion_check_params.check_shift = 2.0;
    completion_check_params.check_period_max = 10000;
  } else {
    parser.insert_error(
        "spacing", "Error: \"spacing\" must be one of \"linear\", \"log\".");
  }

  // "begin"
  parser.optional(completion_check_params.check_begin, "begin");

  // linear check spacing:

  // "period"
  parser.optional(completion_check_params.check_period, "period");
  if (completion_check_params.check_period <= 1) {
    parser.insert_error("period", "Error: \"period\" must > 0.");
  }

  // log check spacing:

  parser.optional(completion_check_params.check_base, "base");
  parser.optional(completion_check_params.check_shift, "shift");
  parser.optional(completion_check_params.check_period_max, "period_max");
  if (completion_check_params.check_base <= 1.0) {
    parser.insert_error("base", "Error: \"base\" must > 1.0");
  }

  if (parser.valid()) {
    parser.value = std::make_unique<CompletionCheckParams<BasicStatistics>>(
        completion_check_params);
  }
}

/// \brief Convert CompletionCheckParams<BasicStatistics> to JSON
inline jsonParser &to_json(
    CompletionCheckParams<BasicStatistics> const &completion_check_params,
    jsonParser &json) {
  // TODO: write out calc_statistics_f parameters
  //  to_json["calc_statistics_f_confidence"] = 0.95;
  //  to_json["calc_statistics_f_weighted_observations_method"] = 1;
  //  to_json["calc_statistics_f_n_resamples"] = 10000;

  json["cutoff"] = completion_check_params.cutoff_params;

  json["convergence"] = jsonParser::array();
  for (auto const &pair : completion_check_params.requested_precision) {
    auto const &key = pair.first;
    auto const &req_prec = pair.second;

    jsonParser tmp;
    tmp["quantity"] = key.sampler_name;

    std::vector<int> component_index;
    component_index.push_back(key.component_index);
    tmp["component_index"] = component_index;

    std::vector<std::string> component_name;
    component_name.push_back(key.component_name);
    tmp["component_name"] = component_name;

    to_json(req_prec, tmp);

    json["convergence"].push_back(tmp);
  }

  // "spacing"
  if (completion_check_params.log_spacing == false) {
    json["spacing"] = "linear";
    json["begin"] = completion_check_params.check_begin;
    json["period"] = completion_check_params.check_period;
  } else {
    json["spacing"] = "log";
    json["begin"] = completion_check_params.check_begin;
    json["base"] = completion_check_params.check_base;
    json["shift"] = completion_check_params.check_shift;
    json["period_max"] = completion_check_params.check_period_max;
  }

  return json;
}

/// \brief CompletionCheckResults to JSON
template <typename StatisticsType>
jsonParser &to_json(CompletionCheckResults<StatisticsType> const &value,
                    jsonParser &json) {
  json.put_obj();
  json["has_all_minimums_met"] = value.has_all_minimums_met;
  json["has_any_maximum_met"] = value.has_any_maximum_met;
  json["count"] = value.count;
  json["time"] = value.time;
  json["clocktime"] = value.clocktime;
  json["n_samples"] = value.n_samples;
  json["is_complete"] = value.is_complete;
  if (value.n_samples_at_convergence_check.has_value()) {
    json["n_samples_at_convergence_check"] =
        value.n_samples_at_convergence_check;
    json["equilibration_check_results"] = value.equilibration_check_results;
    json["convergence_check_results"] = value.convergence_check_results;
  }
  return json;
}

}  // namespace monte
}  // namespace CASM

#endif
