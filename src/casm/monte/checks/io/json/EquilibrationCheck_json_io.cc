#include "casm/monte/checks/io/json/EquilibrationCheck_json_io.hh"

#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/checks/EquilibrationCheck.hh"

namespace CASM {

class jsonParser;

namespace monte {
struct IndividualEquilibrationCheckResult;
struct EquilibrationCheckResults;

/// \brief IndividualEquilibrationCheckResult to JSON
jsonParser &to_json(IndividualEquilibrationCheckResult const &value,
                    jsonParser &json) {
  json.put_obj();
  json["is_equilibrated"] = value.is_equilibrated;
  json["N_samples_for_equilibration"] = value.N_samples_for_equilibration;
  return json;
}

/// \brief EquilibrationCheckResults to JSON
jsonParser &to_json(EquilibrationCheckResults const &value, jsonParser &json) {
  json.put_obj();
  json["all_equilibrated"] = value.all_equilibrated;
  json["N_samples_for_all_to_equilibrate"] =
      value.N_samples_for_all_to_equilibrate;
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
