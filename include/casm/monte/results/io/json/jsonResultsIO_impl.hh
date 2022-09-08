#ifndef CASM_monte_results_io_jsonResultsIO_impl
#define CASM_monte_results_io_jsonResultsIO_impl

#include "casm/casm_io/SafeOfstream.hh"
#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/results/Results.hh"
#include "casm/monte/results/io/json/jsonResultsIO.hh"

namespace CASM {
namespace monte {

namespace jsonResultsIO_impl {

/// \brief Make sure `json[key]` is an object, for each key in `keys`
inline jsonParser &ensure_initialized_objects(jsonParser &json,
                                              std::set<std::string> keys) {
  for (auto key : keys) {
    if (!json.contains(key)) {
      json[key].put_obj();
    } else if (!json[key].is_obj()) {
      std::stringstream msg;
      msg << "JSON Error: \"" << key << "\" is expected to be an object.";
      throw std::runtime_error(msg.str());
    }
  }
  return json;
}

/// \brief Make sure `json[key]` is an array, for each key in `keys`
inline jsonParser &ensure_initialized_arrays(jsonParser &json,
                                             std::set<std::string> keys) {
  for (auto key : keys) {
    if (!json.contains(key)) {
      json[key].put_array();
    } else if (!json[key].is_array()) {
      std::stringstream msg;
      msg << "JSON Error: \"" << key << "\" is expected to be an array.";
      throw std::runtime_error(msg.str());
    }
  }
  return json;
}

/// \brief Append condition value to results summary JSON
///
/// \code
/// <condition_name>: {
///   "shape": [5], // uses empty array [] for scalar
///   "component_names": ["0", "1", "2", "3", ...],
///   <component_name>: [...]  <-- appends to
/// }
/// \endcode
///
/// For matrix-valued conditions, uses on column-major unrolling:
/// \code
/// <condition_name>: {
///   "shape": [rows, cols],
///   "component_names": ["0,0", "1,0", "2,0", "0,1", ...],
///   "0,0": [...],  <-- appends to
///   "1,0": [...],  <-- appends to
///   ...
/// }
/// \endcode
///
///
inline jsonParser &append_condition_to_json(
    std::string const &name, Eigen::VectorXd const &value,
    std::vector<Index> const &shape,
    std::vector<std::string> const &component_names, jsonParser &json) {
  ensure_initialized_objects(json, {name});
  auto &j = json[name];

  // write shape
  j["shape"] = shape;

  // write component names
  j["component_names"] = component_names;

  // write value for each component separately
  Index i = 0;
  for (auto const &component_name : component_names) {
    if (!j.contains(component_name)) {
      j[component_name].put_array();
    }
    j[component_name].push_back(value(i));
    ++i;
  }
  return json;
}

template <typename ConfigType>
jsonParser &append_scalar_condition_to_json(
    std::pair<std::string, double> const &condition, jsonParser &json,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions) {
  return append_condition_to_json(
      condition.first, reshaped(condition.second), std::vector<Index>(),
      get_scalar_component_names(condition.first, condition.second,
                                 sampling_functions),
      json);
}

template <typename ConfigType>
jsonParser &append_vector_condition_to_json(
    std::pair<std::string, Eigen::VectorXd> const &condition, jsonParser &json,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions) {
  return append_condition_to_json(
      condition.first, reshaped(condition.second),
      std::vector<Index>({condition.second.size()}),
      get_vector_component_names(condition.first, condition.second,
                                 sampling_functions),
      json);
}

template <typename ConfigType>
jsonParser &append_matrix_condition_to_json(
    std::pair<std::string, Eigen::MatrixXd> const &condition, jsonParser &json,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions) {
  return append_condition_to_json(
      condition.first, reshaped(condition.second),
      std::vector<Index>({condition.second.rows(), condition.second.cols()}),
      get_matrix_component_names(condition.first, condition.second,
                                 sampling_functions),
      json);
}

/// \brief Append sampled data quantity to summary JSON
///
/// \code
/// <quantity>: {
///   "shape": [...],   // Scalar: [], Vector: [rows], Matrix: [rows, cols]
///   "component_names": ["0", "1", "2", "3", ...],
///   <component_name>: {
///     "mean": [...], <-- appends to
///     "calculated_precision": [...]  <-- appends to
///     "is_converged": [...] <-- appends to, only if requested to converge
///   }
/// }
/// \endcode
template <typename ConfigType>
jsonParser &append_sampled_data_to_json(
    std::pair<std::string, std::shared_ptr<Sampler>> quantity, jsonParser &json,
    Results<ConfigType> const &results,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions) {
  std::string const &quantity_name = quantity.first;
  Sampler const &sampler = *quantity.second;
  ensure_initialized_objects(json, {quantity_name});
  auto &quantity_json = json[quantity_name];

  auto function_it = sampling_functions.find(quantity_name);
  if (function_it == sampling_functions.end()) {
    std::stringstream msg;
    msg << "Error in append_sampled_data_to_json: No matching sampling "
           "function found for '"
        << quantity_name << "'.";
    throw std::runtime_error(msg.str());
  }
  std::vector<std::string> component_names =
      function_it->second.component_names;

  // write shape
  quantity_json["shape"] = function_it->second.shape;

  // write component names
  quantity_json["component_names"] = component_names;

  CompletionCheckResults const &completion_r = results.completion_check_results;
  ConvergenceCheckResults const &convergence_r =
      completion_r.convergence_check_results;

  // for each component calculate or get existing convergence check results
  Index i = 0;
  for (auto const &component_name : component_names) {
    SamplerComponent key(quantity_name, i, component_name);

    auto result_it = convergence_r.individual_results.find(key);
    bool is_requested_to_converge =
        (result_it != convergence_r.individual_results.end());
    IndividualConvergenceCheckResult result;

    if (is_requested_to_converge) {
      // if is a quantity specifically asked to be converged,
      //     use existing results
      result = result_it->second;
    } else if (convergence_r.N_samples_for_statistics != 0) {
      // if not a quantity specifically asked to be converged,
      //     do convergence check
      double required_precision = 0.0;
      result = convergence_check(
          sampler.component(i).tail(convergence_r.N_samples_for_statistics),
          required_precision, completion_r.confidence);
    }

    ensure_initialized_objects(quantity_json, {component_name});
    auto &component_json = quantity_json[component_name];
    ensure_initialized_arrays(component_json, {"mean", "calculated_precision"});
    if (is_requested_to_converge) {
      ensure_initialized_arrays(component_json, {"is_converged"});
    }

    component_json["mean"].push_back(result.mean);
    component_json["calculated_precision"].push_back(
        result.calculated_precision);
    if (is_requested_to_converge) {
      component_json["is_converged"].push_back(result.is_converged);
    }
    ++i;
  }
  return json;
}

/// \brief Append completion check results to summary JSON
///
/// \code
/// {
///   "all_equilibrated": [...], <-- appends to
///   "N_samples_for_all_to_equilibrate": [...], <-- appends to
///   "all_converged": [...], <-- appends to
///   "N_samples_for_statistics": [...], <-- appends to
/// }
/// \endcode
template <typename ConfigType>
jsonParser &append_completion_check_results_to_json(
    Results<ConfigType> const &results, jsonParser &json) {
  auto const &completion_r = results.completion_check_results;
  auto const &equilibration_r = completion_r.equilibration_check_results;
  auto const &convergence_r = completion_r.convergence_check_results;

  ensure_initialized_arrays(
      json, {"all_equilibrated", "N_samples_for_all_to_equilibrate",
             "all_converged", "N_samples_for_statistics"});

  json["all_equilibrated"].push_back(equilibration_r.all_equilibrated);

  json["N_samples_for_all_to_equilibrate"].push_back(
      equilibration_r.N_samples_for_all_to_equilibrate);

  json["all_converged"].push_back(convergence_r.all_converged);

  json["N_samples_for_statistics"].push_back(
      convergence_r.N_samples_for_statistics);

  return json;
}

/// \brief Append results analysis values to summary JSON
///
/// \code
/// <name>: {
///   "shape": [...],   // Scalar: [], Vector: [rows], Matrix: [rows, cols]
///   "component_names": ["0", "1", "2", "3", ...],
///   <component_name>: [...] <-- appends to
/// }
/// \endcode
template <typename ConfigType>
jsonParser &append_results_analysis_to_json(
    Results<ConfigType> const &results, jsonParser &json,
    ResultsAnalysisFunctionMap<ConfigType> const &analysis_functions) {
  // for each analysis value
  for (auto const &pair : results.analysis) {
    std::string const &name = pair.first;
    Eigen::VectorXd const &value = pair.second;
    jsonParser value_json = json[name];

    ensure_initialized_objects(json, {name});
    auto function_it = analysis_functions.find(name);
    if (function_it == analysis_functions.end()) {
      std::stringstream msg;
      msg << "Error in append_results_analysis_to_json: No matching analysis "
             "function found for '"
          << name << "'.";
      throw std::runtime_error(msg.str());
    }
    std::vector<std::string> component_names =
        function_it->second.component_names;

    // write shape
    value_json["shape"] = function_it->second.shape;

    // write component names
    value_json["component_names"] = component_names;

    // for each component, store result in array
    Index i = 0;
    for (auto const &component_name : component_names) {
      ensure_initialized_arrays(value_json, {component_name});
      value_json[component_name].push_back(value(i));
      ++i;
    }
  }
  return json;
}

/// \brief Append completion check results to summary JSON
///
/// \code
/// {
///   "initial_states": [...], <-- appends to
///   "final_states": [...] <-- appends to
/// }
/// \endcode
template <typename ConfigType>
jsonParser &append_trajectory_results_to_json(
    Results<ConfigType> const &results, jsonParser &json) {
  ensure_initialized_arrays(json, {"initial_states", "final_states"});
  if (results.initial_state.has_value()) {
    json["initial_states"].push_back(*results.initial_state);
  } else {
    json["initial_states"].push_back(jsonParser::null());
  }

  if (results.final_state.has_value()) {
    json["final_states"].push_back(*results.final_state);
  } else {
    json["final_states"].push_back(jsonParser::null());
  }
  return json;
}

}  // namespace jsonResultsIO_impl

template <typename _ConfigType>
jsonResultsIO<_ConfigType>::jsonResultsIO(
    fs::path _output_dir,
    StateSamplingFunctionMap<config_type> _sampling_functions,
    ResultsAnalysisFunctionMap<config_type> _analysis_functions,
    bool _write_trajectory, bool _write_observations)
    : m_output_dir(_output_dir),
      m_sampling_functions(_sampling_functions),
      m_analysis_functions(_analysis_functions),
      m_write_trajectory(_write_trajectory),
      m_write_observations(_write_observations) {}

/// \brief Read a vector of final states of completed runs
template <typename _ConfigType>
std::vector<typename jsonResultsIO<_ConfigType>::state_type>
jsonResultsIO<_ConfigType>::read_final_states() {
  jsonParser json = this->read_summary();
  return json["trajectory"]["final_states"].get<std::vector<state_type>>();
}

/// \brief Write results
///
/// Notes:
/// - See `write_summary` for summary.json output format
///   - Always written. Appends with each completed run.
/// - See `write_trajectory` for run.<index>/trajectory.json output format
///   - Only written if constructed with `write_trajectory == true`
/// - See `write_observations` for run.<index>/observations.json output format
///   - Only written if constructed with `write_observations == true`
template <typename _ConfigType>
void jsonResultsIO<_ConfigType>::write(results_type const &results,
                                       Index run_index) {
  this->write_summary(results);
  if (m_write_trajectory) {
    this->write_trajectory(results, run_index);
  }
  if (m_write_observations) {
    this->write_observations(results, run_index);
  }
}

/// \brief Write summary.json with results from each individual run
///
/// The summary format appends each new run result to form arrays of values for
/// each component of conditions, sampled data, analyzed data, etc. The index
/// into the arrays is the run index in the series of Monte Carlo calculation
/// performed.
///
/// Output format:
/// \code
/// {
///   "conditions": {
///     <condition_name> {
///       "shape": [...],
///       "component_names": ["0", "1", "2", "3", ...],
///       <component_name>: [...]
///   },
///   "sampled_data": {
///     <quantity>: {
///       "shape": [...],
///       "component_names": ["0", "1", "2", "3", ...],
///       <component_name>: {
///         "mean": [...],
///         "calculated_precision": [...],
///         "is_converged": [...] <-- only if requested to converge
///       }
///     },
///   },
///   "analyzed_data": {
///     <name>: [...],
///   },
///   "completion_check_results": {
///     "all_equilibrated": [...],
///     "N_samples_for_all_to_equilibrate": [...],
///     "all_converged": [...],
///     "N_samples_for_statistics": [...],
///   },
///   "trajectory": {
///     "initial_states": [...],
///     "final_states": [...]
///   }
/// }
/// \endcode
///
template <typename _ConfigType>
void jsonResultsIO<_ConfigType>::write_summary(results_type const &results) {
  using namespace jsonResultsIO_impl;

  StateSamplingFunctionMap<_ConfigType> const &sampling_functions =
      m_sampling_functions;
  ResultsAnalysisFunctionMap<_ConfigType> const &analysis_functions =
      m_analysis_functions;
  fs::path const &output_dir = m_output_dir;

  // read existing summary file (create if not existing)
  jsonParser json = this->read_summary();

  ensure_initialized_objects(
      json, {"conditions", "sampled_data", "completion_check_results",
             "analysis", "trajectory"});

  for (auto const &condition :
       results.initial_state->conditions.scalar_values) {
    append_scalar_condition_to_json(condition, json["conditions"],
                                    sampling_functions);
  }
  for (auto const &condition :
       results.initial_state->conditions.vector_values) {
    append_vector_condition_to_json(condition, json["conditions"],
                                    sampling_functions);
  }
  for (auto const &condition :
       results.initial_state->conditions.matrix_values) {
    append_matrix_condition_to_json(condition, json["conditions"],
                                    sampling_functions);
  }

  for (auto const &quantity : results.samplers) {
    append_sampled_data_to_json(quantity, json["sampled_data"], results,
                                sampling_functions);
  }

  // append completion check results
  append_completion_check_results_to_json(results,
                                          json["completion_check_results"]);

  // append results analysis
  append_results_analysis_to_json(results, json["analysis"],
                                  analysis_functions);

  // append trajectory results (initial and final states)
  append_trajectory_results_to_json(results, json["trajectory"]);

  // write summary file
  fs::path summary_path = output_dir / "summary.json";
  SafeOfstream file;
  file.open(summary_path);
  json.print(file.ofstream(), -1);
  file.close();
}

/// \brief Write run.<index>/trajectory.json
///
/// Output file is a JSON array of the configuration at the time a sample was
/// taken.
template <typename _ConfigType>
void jsonResultsIO<_ConfigType>::write_trajectory(results_type const &results,
                                                  Index run_index) {
  jsonParser json(results.sample_trajectory);
  json.write(this->run_dir(run_index) / "trajectory.json", -1);
}

/// \brief Write run.<index>/observations.json
///
/// Output format:
/// \code
/// {
///   "count": [...], // count (i.e. pass/step) at the time sample was taken
///   "time": [...], // time when sampled was taken (if exists)
///   <quantity>: {
///     "component_names": ["0", "1", ...],
///     "value": [[<matrix>]] // rows are samples, columns are components of
///                           // the sampled quantity
///   }
/// }
/// \endcode
template <typename _ConfigType>
void jsonResultsIO<_ConfigType>::write_observations(results_type const &results,
                                                    Index run_index) {
  jsonParser json = jsonParser::object();
  if (results.sample_count.size()) {
    json["count"] = results.sample_count;
  }
  if (results.sample_time.size()) {
    json["time"] = results.sample_time;
  }
  for (auto const &pair : results.samplers) {
    json[pair.first]["component_names"] = pair.second->component_names();
    json[pair.first]["value"] = pair.second->values();
  }
  json.write(run_dir(run_index) / "observations.json", -1);
}

/// \brief Read existing summary.json file, if exists, else provided default
template <typename _ConfigType>
jsonParser jsonResultsIO<_ConfigType>::read_summary() {
  fs::path summary_path = m_output_dir / "summary.json";
  if (!fs::exists(summary_path)) {
    jsonParser json;
    json["conditions"].put_obj();
    json["sampled_data"].put_obj();
    json["completion_check_results"].put_obj();
    json["trajectory"].put_obj();
    return json;
  }
  return jsonParser(summary_path);
}

template <typename _ConfigType>
fs::path jsonResultsIO<_ConfigType>::run_dir(Index run_index) {
  std::string _run_dir = "run." + std::to_string(run_index);
  fs::path result = m_output_dir / _run_dir;
  if (!fs::exists(result)) {
    fs::create_directories(result);
  }
  return result;
}

}  // namespace monte
}  // namespace CASM

#endif
