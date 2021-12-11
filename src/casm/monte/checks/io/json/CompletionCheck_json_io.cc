#include "casm/monte/checks/io/json/CompletionCheck_json_io.hh"

#include "casm/casm_io/json/jsonParser.hh"
#include "casm/casm_io/json/optional.hh"
#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/checks/io/json/ConvergenceCheck_json_io.hh"
#include "casm/monte/checks/io/json/EquilibrationCheck_json_io.hh"

namespace CASM {
namespace monte {

/// \brief CompletionCheckResults to JSON
jsonParser &to_json(CompletionCheckResults const &value, jsonParser &json) {
  json.put_obj();
  json["has_all_minimums_met"] = value.has_all_minimums_met;
  json["has_any_maximum_met"] = value.has_any_maximum_met;
  json["count"] = value.count;
  json["time"] = value.time;
  json["n_samples"] = value.n_samples;
  json["convergence_check_performed"] = value.convergence_check_performed;
  json["is_complete"] = value.is_complete;
  json["confidence"] = value.confidence;
  json["equilibration_check_results"] = value.equilibration_check_results;
  json["convergence_check_results"] = value.convergence_check_results;
  return json;
}

}  // namespace monte
}  // namespace CASM
