#ifndef CASM_monte_RunData_json_io
#define CASM_monte_RunData_json_io

#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/optional.hh"
#include "casm/monte/run_management/RunData.hh"

namespace CASM {

template <typename ConfigType>
jsonParser &to_json(monte::RunData<ConfigType> const &run_data,
                    jsonParser &json, bool do_write_initial_states,
                    bool do_write_final_states) {
  if (do_write_initial_states) {
    json["initial_state"] = run_data.initial_state;
  }
  if (do_write_final_states) {
    json["final_state"] = run_data.final_state;
  }
  json["conditions"] = run_data.conditions;
  json["transformation_matrix_to_supercell"] =
      run_data.transformation_matrix_to_super;
  json["n_unitcells"] = run_data.n_unitcells;
  return json;
}

template <typename ConfigType>
void from_json(monte::RunData<ConfigType> &run_data, jsonParser const &json) {
  run_data.initial_state.reset();
  if (json.contains("initial_state")) {
    from_json(run_data.initial_state, json["initial_state"]);
  }

  run_data.final_state.reset();
  if (json.contains("final_state")) {
    from_json(run_data.final_state, json["final_state"]);
  }

  run_data.conditions = monte::ValueMap();
  if (json.contains("conditions")) {
    from_json(run_data.conditions, json["conditions"]);
  }

  from_json(run_data.transformation_matrix_to_super,
            json["transformation_matrix_to_supercell"]);
  from_json(run_data.n_unitcells, json["n_unitcells"]);
}

}  // namespace CASM

#endif
