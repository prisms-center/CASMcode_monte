#ifndef CASM_monte_Results
#define CASM_monte_Results

#include <vector>

#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/sampling/SampledData.hh"
#include "casm/monte/state/State.hh"

namespace CASM {
namespace monte {

/// \brief Standard Monte Carlo calculation results data structure
template <typename ConfigType>
struct Results {
  Results(State<ConfigType> const &_initial_state)
      : conditions(_initial_state.conditions),
        trajectory(1, _initial_state.configuration) {}

  VectorValueMap conditions;
  SampledData sampled_data;
  std::vector<ConfigType> trajectory;
  CompletionCheckResults completion_check_results;
};

}  // namespace monte
}  // namespace CASM

#endif
