#ifndef CASM_monte_run_management_SamplingFixtureParams_json_io
#define CASM_monte_run_management_SamplingFixtureParams_json_io

#include "casm/monte/definitions.hh"
#include "casm/monte/misc/polymorphic_method_json_io.hh"
#include "casm/monte/run_management/ResultsAnalysisFunction.hh"
#include "casm/monte/run_management/SamplingFixture.hh"
#include "casm/monte/run_management/io/ResultsIO.hh"
#include "casm/monte/run_management/io/json/ResultsIO_json_io.hh"
#include "casm/monte/sampling/StateSamplingFunction.hh"
#include "casm/monte/sampling/io/json/SamplingParams_json_io.hh"

namespace CASM {

template <typename T>
class InputParser;

namespace monte {

template <typename ConfigType, typename StatisticsType>
void parse(
    InputParser<SamplingFixtureParams<ConfigType, StatisticsType>> &parser,
    std::string label, StateSamplingFunctionMap const &sampling_functions,
    jsonStateSamplingFunctionMap const &json_sampling_functions,
    ResultsAnalysisFunctionMap<ConfigType, StatisticsType> const
        &analysis_functions,
    MethodParserMap<ResultsIO<Results<ConfigType, StatisticsType>>> const
        &results_io_methods,
    bool time_sampling_allowed);

template <typename ConfigType, typename StatisticsType>
jsonParser &to_json(SamplingFixtureParams<ConfigType, StatisticsType> params,
                    jsonParser &json);

// ~~~ Definition ~~~

/// \brief Construct sampling_fixture_params_type from JSON
///
///
/// \code
/// {
///     "sampling": <monte::SamplingParams>
///         Options controlling which quantities are sampled and how often
///         sampling is performed.
///     "completion_check": <monte::CompletionCheck>
///         Controls when a sampling fixture is complete. Options include
///         convergence of sampled quantiies, min/max number of samples, min/
///         max number of passes, etc.
///     "analysis":
///         "functions": array of str (default=[])
///             Names of analysis functions to use to evaluate results with.
///             Standard options include the following (others may be included):
///
///             - "heat_capacity": Heat capacity
///             - "mol_susc": Chemical susceptibility (mol_composition)
///             - "param_susc": Chemical susceptibility (param_composition)
///             - "mol_thermocalc_susc": Thermo-chemical susceptibility
///               (mol_composition)
///             - "param_thermocalc_susc": Thermo-Chemical susceptibility
///               (param_composition)
///
///             Unless otherwise noted, assume per unitcell properties.
///     "results_io": <monte::ResultsIO> = null
///         Options controlling sampling fixture results output.
///     "log": (optional)
///         "file": str (default="status.json")
///             Provide the path where a log file should be written.
///         "frequency_in_s": number (default=600.0)
///             How often the log file should be written, in seconds.
/// }
/// \endcode
template <typename ConfigType, typename StatisticsType>
void parse(
    InputParser<SamplingFixtureParams<ConfigType, StatisticsType>> &parser,
    std::string label, StateSamplingFunctionMap const &sampling_functions,
    jsonStateSamplingFunctionMap const &json_sampling_functions,
    ResultsAnalysisFunctionMap<ConfigType, StatisticsType> const
        &analysis_functions,
    MethodParserMap<ResultsIO<Results<ConfigType, StatisticsType>>> const
        &results_io_methods,
    bool time_sampling_allowed) {
  // Read sampling params
  std::set<std::string> sampling_function_names;
  for (auto const &element : sampling_functions) {
    sampling_function_names.insert(element.first);
  }
  std::set<std::string> json_sampling_function_names;
  for (auto const &element : json_sampling_functions) {
    json_sampling_function_names.insert(element.first);
  }
  auto sampling_params_subparser =
      parser.template subparse<monte::SamplingParams>(
          "sampling", sampling_function_names, json_sampling_function_names,
          time_sampling_allowed);
  if (!parser.valid()) {
    return;
  }
  monte::SamplingParams const &sampling_params =
      *sampling_params_subparser->value;

  // Read completion check params
  auto completion_check_params_subparser =
      parser.template subparse<monte::CompletionCheckParams<StatisticsType>>(
          "completion_check", sampling_functions);

  // Read analysis functions
  std::vector<std::string> analysis_names;
  fs::path functions_path = fs::path("analysis") / "functions";
  parser.optional(analysis_names, functions_path);

  for (auto const &name : analysis_names) {
    auto it = analysis_functions.find(name);
    if (it == analysis_functions.end()) {
      std::stringstream msg;
      msg << "Error: function '" << name << "' not recognized";
      parser.insert_error(functions_path, msg.str());
    }
  }

  // Construct results I/O instance
  auto results_io_subparser =
      parser
          .template subparse_if<ResultsIO<Results<ConfigType, StatisticsType>>>(
              "results_io", results_io_methods);

  // Method log
  monte::MethodLog method_log;
  if (parser.self.contains("log")) {
    std::string log_file = "status.json";
    parser.optional(log_file, fs::path("log") / "file");
    double log_frequency = 600.0;
    parser.optional(log_frequency, fs::path("log") / "frequency_in_s");

    method_log.log_frequency = log_frequency;
    method_log.logfile_path = fs::path(log_file);
    method_log.reset();
  }

  if (parser.valid()) {
    parser.value =
        std::make_unique<SamplingFixtureParams<ConfigType, StatisticsType>>(
            label, sampling_functions, json_sampling_functions,
            analysis_functions, sampling_params,
            *completion_check_params_subparser->value, analysis_names,
            std::move(results_io_subparser->value), method_log);
  }
}

template <typename ConfigType, typename StatisticsType>
jsonParser &to_json(SamplingFixtureParams<ConfigType, StatisticsType> params,
                    jsonParser &json) {
  json.put_obj();

  to_json(params.sampling_params, json["sampling"]);
  to_json(params.completion_check_params, json["completion_check"]);

  json["analysis"] = jsonParser::object();
  json["analysis"]["functions"] = jsonParser::array();
  for (std::string name : params.analysis_names) {
    json["analysis"]["functions"].push_back(name);
  }

  if (params.results_io) {
    json["results_io"] = params.results_io->to_json();
  } else {
    json["results_io"].put_null();
  }

  if (!params.method_log.logfile_path.empty()) {
    json["log"] = jsonParser::object();
    json["log"]["file"] = params.method_log.logfile_path.string();
    if (params.method_log.log_frequency.has_value()) {
      json["log"]["frequency_in_s"] = params.method_log.log_frequency.value();
    }
  }
  return json;
}

}  // namespace monte
}  // namespace CASM

#endif
