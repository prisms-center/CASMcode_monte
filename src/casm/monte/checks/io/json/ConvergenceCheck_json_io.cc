#include "casm/monte/checks/io/json/ConvergenceCheck_json_io.hh"

#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/checks/ConvergenceCheck.hh"

namespace CASM {
namespace monte {

/// \brief IndividualConvergenceCheckResult to JSON
jsonParser &to_json(IndividualConvergenceCheckResult const &value,
                    jsonParser &json) {
  json.put_obj();
  json["is_converged"] = value.is_converged;
  json["mean"] = value.mean;
  json["squared_norm"] = value.squared_norm;
  json["calculated_precision"] = value.calculated_precision;
  json["requested_precision"] = value.requested_precision;
  return json;
}

/// \brief ConvergenceCheckResults to JSON
jsonParser &to_json(ConvergenceCheckResults const &value, jsonParser &json) {
  json.put_obj();
  json["all_converged"] = value.all_converged;
  json["N_samples_for_statistics"] = value.N_samples_for_statistics;
  json["individual_results"];
  for (auto const &pair : value.individual_results) {
    std::string name =
        pair.first.sampler_name + "(" + pair.first.component_name + ")";
    json["individual_results"][name] = pair.second;
  }
  return json;
}

}  // namespace monte
}  // namespace CASM
