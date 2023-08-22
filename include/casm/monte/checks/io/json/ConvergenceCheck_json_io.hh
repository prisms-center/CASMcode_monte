#ifndef CASM_monte_checks_ConvergenceCheck_json_io
#define CASM_monte_checks_ConvergenceCheck_json_io

#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/checks/ConvergenceCheck.hh"
#include "casm/monte/sampling/io/json/Sampler_json_io.hh"

namespace CASM {

class jsonParser;

namespace monte {

template <typename StatisticsType>
struct IndividualConvergenceCheckResult;

template <typename StatisticsType>
struct ConvergenceCheckResults;

/// \brief IndividualConvergenceCheckResult to JSON
template <typename StatisticsType>
jsonParser &to_json(
    IndividualConvergenceCheckResult<StatisticsType> const &value,
    jsonParser &json);

/// \brief ConvergenceCheckResults to JSON
template <typename StatisticsType>
jsonParser &to_json(ConvergenceCheckResults<StatisticsType> const &value,
                    jsonParser &json);

/// --- template implementations ---

/// \brief IndividualConvergenceCheckResult to JSON
template <typename StatisticsType>
jsonParser &to_json(
    IndividualConvergenceCheckResult<StatisticsType> const &value,
    jsonParser &json) {
  json.put_obj();
  json["is_converged"] = value.is_converged;
  json["requested_precision"] = value.requested_precision;
  to_json(value.stats, json["stats"]);
  return json;
}

/// \brief ConvergenceCheckResults to JSON
template <typename StatisticsType>
jsonParser &to_json(ConvergenceCheckResults<StatisticsType> const &value,
                    jsonParser &json) {
  json.put_obj();
  json["all_converged"] = value.all_converged;
  json["N_samples_for_statistics"] = value.N_samples_for_statistics;
  json["individual_results"].put_array();
  for (auto const &pair : value.individual_results) {
    jsonParser tjson;
    to_json(pair.second, tjson);
    tjson["sampler_name"] = pair.first.sampler_name;
    tjson["component_name"] = pair.first.component_name;
    tjson["component_index"] = pair.first.component_index;
    json["individual_results"].push_back(tjson);
  }
  return json;
}

}  // namespace monte
}  // namespace CASM

#endif
