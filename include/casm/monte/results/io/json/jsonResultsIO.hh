#ifndef CASM_monte_results_io_jsonResultsIO
#define CASM_monte_results_io_jsonResultsIO

#include "casm/monte/results/io/ResultsIO.hh"
#include "casm/monte/state/State.hh"

namespace CASM {
namespace monte {

template <typename _ConfigType>
class jsonResultsIO : public ResultsIO<_ConfigType> {
 public:
  typedef _ConfigType config_type;
  typedef monte::State<config_type> state_type;

  jsonResultsIO() {}

  std::vector<state_type> read_final_states() override {
    return std::vector<state_type>();
  }

  void write_trajectory(std::vector<state_type> const &trajectory,
                        Index run_index) override {}

  void write_observations(monte::SampledData const &sampled_data,
                          Index run_index) override {}

  void write_initial_state(state_type const &initial_state,
                           Index run_index) override {}

  void write_final_state(state_type const &final_state,
                         Index run_index) override {}

  void write_completion_check_results(
      monte::CompletionCheckResults const &completion_check_results,
      Index run_index) override {}

  void write_summary(
      monte::CompletionCheckResults const &completion_check_results,
      Index run_index) override {}
};

}  // namespace monte
}  // namespace CASM

#endif
