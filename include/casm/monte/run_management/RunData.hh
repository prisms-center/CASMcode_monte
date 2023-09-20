#ifndef CASM_monte_RunData
#define CASM_monte_RunData

#include <optional>

#include "casm/monte/ValueMap.hh"
#include "casm/monte/run_management/State.hh"

namespace CASM {
namespace monte {

template <typename _ConfigType>
struct RunData {
  typedef _ConfigType config_type;
  typedef State<config_type> state_type;

  std::optional<state_type> initial_state;
  std::optional<state_type> final_state;
  ValueMap conditions;
  Eigen::Matrix3l transformation_matrix_to_super;
  Index n_unitcells;
};

}  // namespace monte
}  // namespace CASM

#endif
