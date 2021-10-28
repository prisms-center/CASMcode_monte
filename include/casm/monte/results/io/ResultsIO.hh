#ifndef CASM_monte_results_io_ResultsIO
#define CASM_monte_results_io_ResultsIO

#include <vector>

#include "casm/global/definitions.hh"

namespace CASM {
namespace monte {

struct CompletionCheckResults;

struct SampledData;

template <typename _ConfigType>
struct State;

template <typename _ConfigType>
class ResultsIO {
 public:
  typedef _ConfigType config_type;
  typedef monte::State<config_type> state_type;

  virtual ~ResultsIO() {}

  virtual std::vector<state_type> read_final_states() = 0;
  virtual void write_trajectory(std::vector<state_type> const &trajectory,
                                Index run_index) = 0;
  virtual void write_observations(monte::SampledData const &sampled_data,
                                  Index run_index) = 0;
  virtual void write_initial_state(state_type const &initial_state,
                                   Index run_index) = 0;
  virtual void write_final_state(state_type const &final_state,
                                 Index run_index) = 0;
  virtual void write_completion_check_results(
      monte::CompletionCheckResults const &completion_check_results,
      Index run_index) = 0;
  virtual void write_summary(
      monte::CompletionCheckResults const &completion_check_results,
      Index run_index) = 0;
};

}  // namespace monte
}  // namespace CASM

#endif
