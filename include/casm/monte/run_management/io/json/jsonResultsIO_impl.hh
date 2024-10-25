#ifndef CASM_monte_results_io_jsonResultsIO_impl
#define CASM_monte_results_io_jsonResultsIO_impl

#include "casm/casm_io/SafeOfstream.hh"
#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/casm_io/json/optional.hh"
#include "casm/monte/run_management/Results.hh"
#include "casm/monte/run_management/io/json/jsonResultsIO.hh"
#include "casm/monte/sampling/io/json/SelectedEventFunctions_json_io.hh"

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

  if (shape.size() == 0) {
    // scalar
    if (!j.contains("value")) {
      j["value"].put_array();
    }
    j["value"].push_back(value(0));

  } else {
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
  }
  return json;
}

inline jsonParser &append_scalar_condition_to_json(
    std::pair<std::string, double> const &condition, jsonParser &json,
    StateSamplingFunctionMap const &sampling_functions) {
  return append_condition_to_json(
      condition.first, reshaped(condition.second), std::vector<Index>(),
      get_scalar_component_names(condition.first, condition.second,
                                 sampling_functions),
      json);
}

inline jsonParser &append_vector_condition_to_json(
    std::pair<std::string, Eigen::VectorXd> const &condition, jsonParser &json,
    StateSamplingFunctionMap const &sampling_functions) {
  return append_condition_to_json(
      condition.first, reshaped(condition.second),
      std::vector<Index>({condition.second.size()}),
      get_vector_component_names(condition.first, condition.second,
                                 sampling_functions),
      json);
}

inline jsonParser &append_matrix_condition_to_json(
    std::pair<std::string, Eigen::MatrixXd> const &condition, jsonParser &json,
    StateSamplingFunctionMap const &sampling_functions) {
  return append_condition_to_json(
      condition.first, reshaped(condition.second),
      std::vector<Index>({condition.second.rows(), condition.second.cols()}),
      get_matrix_component_names(condition.first, condition.second,
                                 sampling_functions),
      json);
}

/// \brief Append sampled data quantity to summary JSON
///
/// For non-scalar values:
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
///
/// For scalar values:
/// \code
/// <quantity>: {
///   "shape": [...],   // Scalar: [], Vector: [rows], Matrix: [rows, cols]
///   "value": {
///     "mean": [...], <-- appends to
///     "calculated_precision": [...]  <-- appends to
///     "is_converged": [...] <-- appends to, only if requested to converge
///   }
/// }
/// \endcode
template <typename ResultsType>
jsonParser &append_statistics_to_json(
    std::pair<std::string, std::shared_ptr<Sampler>> quantity, jsonParser &json,
    ResultsType const &results) {
  std::string const &quantity_name = quantity.first;
  auto qstats = make_quantity_stats(quantity_name, *quantity.second, results);

  ensure_initialized_objects(json, {quantity_name});
  auto &quantity_json = json[quantity_name];

  // write shape
  quantity_json["shape"] = qstats.shape;

  // append statistics
  auto append = [&](jsonParser &tjson, Index i) {
    append_statistics_to_json_arrays(qstats.component_stats[i], tjson);
    if (qstats.is_converged[i].has_value()) {
      ensure_initialized_arrays(tjson, {"is_converged"});
      tjson["is_converged"].push_back(qstats.is_converged[i].value());
    }
  };

  if (qstats.is_scalar) {
    ensure_initialized_objects(quantity_json, {"value"});
    append(quantity_json["value"], 0);
  } else {
    // write component names - if not scalar
    quantity_json["component_names"] = qstats.component_names;
    Index i = 0;
    for (auto const &component_name : qstats.component_names) {
      ensure_initialized_objects(quantity_json, {component_name});
      append(quantity_json[component_name], i);
      ++i;
    }
  }
  return json;
}

/// \brief Append completion check results to summary JSON
///
/// \code
/// {
///   "initial_memory_used_MiB": [...], <-- appends to
///   "final_memory_used_MiB": [...], <-- appends to
///   "elapsed_clocktime": [...], <-- appends to
///   "all_equilibrated": [...], <-- appends to
///   "N_samples_for_all_to_equilibrate": [...], <-- appends to
///   "all_converged": [...], <-- appends to
///   "N_samples_for_statistics": [...], <-- appends to
///   "N_samples": [...], <-- appends to
/// }
/// \endcode
template <typename ConfigType, typename StatisticsType>
jsonParser &append_completion_check_results_to_json(
    Results<ConfigType, StatisticsType> const &results, jsonParser &json) {
  bool auto_converge_mode = is_auto_converge_mode(results);

  if (auto_converge_mode) {
    ensure_initialized_arrays(json, {"all_equilibrated", "all_converged",
                                     "N_samples_for_all_to_equilibrate"});

    json["all_equilibrated"].push_back(all_equilibrated(results));

    json["all_converged"].push_back(all_converged(results));

    if (all_equilibrated(results)) {
      json["N_samples_for_all_to_equilibrate"].push_back(
          N_samples_for_all_to_equilibrate(results));

    } else {
      json["N_samples_for_all_to_equilibrate"].push_back("did_not_equilibrate");
    }
  }

  ensure_initialized_arrays(
      json, {"N_samples", "N_samples_for_statistics", "acceptance_rate",
             "initial_memory_used_MiB", "final_memory_used_MiB",
             "elapsed_clocktime", "count"});

  json["N_samples"].push_back(N_samples(results));

  json["N_samples_for_statistics"].push_back(N_samples_for_statistics(results));

  json["acceptance_rate"].push_back(acceptance_rate(results));

  json["initial_memory_used_MiB"].push_back(results.initial_memory_used_MiB);

  json["final_memory_used_MiB"].push_back(results.final_memory_used_MiB);

  json["elapsed_clocktime"].push_back(elapsed_clocktime(results));

  json["count"].push_back(results.sample_count.back());

  if (results.sample_time.size()) {
    ensure_initialized_arrays(json, {"time"});
    json["time"].push_back(results.sample_time.back());
  }

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
template <typename ConfigType, typename StatisticsType>
jsonParser &append_results_analysis_to_json(
    Results<ConfigType, StatisticsType> const &results, jsonParser &json) {
  auto const &analysis_functions = results.analysis_functions;
  // for each analysis value
  for (auto const &pair : results.analysis) {
    std::string const &name = pair.first;
    Eigen::VectorXd const &value = pair.second;
    jsonParser &value_json = json[name];

    ensure_initialized_objects(json, {name});
    auto function_it = analysis_functions.find(name);
    if (function_it == analysis_functions.end()) {
      std::stringstream msg;
      msg << "Error in append_results_analysis_to_json: No matching analysis "
             "function found for '"
          << name << "'.";
      throw std::runtime_error(msg.str());
    }

    // write shape
    value_json["shape"] = function_it->second.shape;

    bool is_scalar = (function_it->second.shape.size() == 0);

    if (is_scalar) {
      ensure_initialized_arrays(value_json, {"value"});
      if (is_auto_converge_mode(results) && !all_equilibrated(results)) {
        value_json["value"].push_back("did_not_equilibrate");
      } else {
        value_json["value"].push_back(value(0));
      }

    } else {
      std::vector<std::string> component_names =
          function_it->second.component_names;

      // write component names
      value_json["component_names"] = component_names;

      // for each component, store result in array
      Index i = 0;
      for (auto const &component_name : component_names) {
        ensure_initialized_arrays(value_json, {component_name});
        if (is_auto_converge_mode(results) && !all_equilibrated(results)) {
          value_json[component_name].push_back("did_not_equilibrate");
        } else {
          value_json[component_name].push_back(value(i));
        }
        ++i;
      }
    }
  }
  return json;
}

}  // namespace jsonResultsIO_impl

template <typename _ResultsType>
jsonResultsIO<_ResultsType>::jsonResultsIO(fs::path _output_dir,
                                           bool _write_trajectory,
                                           bool _write_observations)
    : m_output_dir(_output_dir),
      m_write_trajectory(_write_trajectory),
      m_write_observations(_write_observations) {}

/// \brief Write results
///
/// Notes:
/// - See `write_summary` for summary.json output format
///   - Always written. Appends with each completed run.
/// - See `write_trajectory` for run.<index>/trajectory.json output format
///   - Only written if constructed with `write_trajectory == true`
/// - See `write_observations` for run.<index>/observations.json output format
///   - Only written if constructed with `write_observations == true`
template <typename _ResultsType>
void jsonResultsIO<_ResultsType>::write(results_type const &results,
                                        ValueMap const &conditions,
                                        Index run_index) {
  this->write_summary(results, conditions);
  if (m_write_trajectory) {
    this->write_trajectory(results, run_index);
  }
  if (m_write_observations) {
    this->write_observations(results, run_index);
  }
}

/// \brief Write input parameters to JSON
template <typename _ResultsType>
jsonParser jsonResultsIO<_ResultsType>::to_json() {
  jsonParser json;
  json["method"] = "json";
  json["kwargs"] = jsonParser::object();
  json["kwargs"]["output_dir"] = m_output_dir.string();
  json["kwargs"]["write_trajectory"] = m_write_trajectory;
  json["kwargs"]["write_observations"] = m_write_observations;
  return json;
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
///   "statistics": {
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
///     "shape": [...],
///     "component_names": [...],
///     <name>: [...],
///   },
///   "completion_check_results": {
///     "all_equilibrated": [...],
///     "N_samples_for_all_to_equilibrate": [...],
///     "all_converged": [...],
///     "N_samples_for_statistics": [...],
///     "N_samples": [...],
///   }
/// }
/// \endcode
///
template <typename _ResultsType>
void jsonResultsIO<_ResultsType>::write_summary(results_type const &results,
                                                ValueMap const &conditions) {
  using namespace jsonResultsIO_impl;

  StateSamplingFunctionMap const &sampling_functions =
      results.sampling_functions;
  ResultsAnalysisFunctionMap<config_type, stats_type> const
      &analysis_functions = results.analysis_functions;
  fs::path const &output_dir = m_output_dir;

  // read existing summary file (create if not existing)
  jsonParser json = this->read_summary();

  ensure_initialized_objects(json, {"conditions", "statistics",
                                    "completion_check_results", "analysis"});

  for (auto const &condition : conditions.scalar_values) {
    append_scalar_condition_to_json(condition, json["conditions"],
                                    sampling_functions);
  }
  for (auto const &condition : conditions.vector_values) {
    append_vector_condition_to_json(condition, json["conditions"],
                                    sampling_functions);
  }
  for (auto const &condition : conditions.matrix_values) {
    append_matrix_condition_to_json(condition, json["conditions"],
                                    sampling_functions);
  }

  for (auto const &quantity : results.samplers) {
    append_statistics_to_json(quantity, json["statistics"], results);
  }

  // append completion check results
  append_completion_check_results_to_json(results,
                                          json["completion_check_results"]);

  // append results analysis
  append_results_analysis_to_json(results, json["analysis"]);

  // write summary file
  fs::path summary_path = output_dir / "summary.json";
  fs::create_directories(output_dir);
  SafeOfstream file;
  file.open(summary_path);
  json.print(file.ofstream(), -1);
  file.close();
}

/// \brief Write run.<index>/trajectory.json
///
/// Output file is a JSON array of the configuration at the time a sample was
/// taken.
template <typename _ResultsType>
void jsonResultsIO<_ResultsType>::write_trajectory(results_type const &results,
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
///   "weight": [...], // weight given to sample (if exists, not normalized)
///   "clocktime": [...], // clocktime when sampled was taken (if exists)
///   <quantity>: {
///     "shape": [...], // Scalar: [], Vector: [rows], Matrix: [rows, cols]
///     "component_names": ["0", "1", ...],
///     // scalar quantities are written as a vector, one element for each
///     sample "value": [scalar0, sample1, sample2, ...],
///     // vector and matrix quantities are written as an array of vectors,
///     // one vector for each sample, representing the quantitiy unrolled
///     // in column-major order
///     "value": [ [<vector0>], [<vector2>], [<vector3>], ...]
///   }
/// }
/// \endcode
template <typename _ResultsType>
void jsonResultsIO<_ResultsType>::write_observations(
    results_type const &results, Index run_index) {
  jsonParser json = jsonParser::object();
  if (results.sample_count.size()) {
    json["count"] = results.sample_count;
  }
  if (results.sample_time.size()) {
    json["time"] = results.sample_time;
  }
  if (results.sample_weight.component(0).size()) {
    json["weight"] = results.sample_weight.component(0);
  }
  if (results.sample_clocktime.size()) {
    json["clocktime"] = results.sample_clocktime;
  }
  for (auto const &pair : results.samplers) {
    json[pair.first]["shape"] = pair.second->shape();
    bool is_scalar = (pair.second->shape().size() == 0);
    if (is_scalar) {
      CASM::to_json(pair.second->values().col(0), json[pair.first]["value"],
                    jsonParser::as_array());
    } else {
      json[pair.first]["component_names"] = pair.second->component_names();
      json[pair.first]["value"] = pair.second->values();
    }
  }
  for (auto const &pair : results.json_samplers) {
    json[pair.first]["value"] = pair.second->values;
  }
  json.write(run_dir(run_index) / "observations.json", -1);
}

/// \brief Read existing summary.json file, if exists, else provided default
template <typename _ResultsType>
jsonParser jsonResultsIO<_ResultsType>::read_summary() {
  fs::path summary_path = m_output_dir / "summary.json";
  if (!fs::exists(summary_path)) {
    jsonParser json;
    json["conditions"].put_obj();
    json["statistics"].put_obj();
    json["completion_check_results"].put_obj();
    return json;
  }
  return jsonParser(summary_path);
}

template <typename _ResultsType>
fs::path jsonResultsIO<_ResultsType>::run_dir(Index run_index) {
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
