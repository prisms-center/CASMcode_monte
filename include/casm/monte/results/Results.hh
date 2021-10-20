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
      : trajectory(1, _initial_state) {}

  SampledData sampled_data;
  std::vector<State<ConfigType>> trajectory;
  CompletionCheckResults completion_check_results;
};

}  // namespace monte
}  // namespace CASM

#endif
