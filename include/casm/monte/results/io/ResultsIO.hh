#ifndef CASM_monte_results_io_ResultsIO
#define CASM_monte_results_io_ResultsIO

#include <vector>

#include "casm/global/definitions.hh"

namespace CASM {
namespace monte {

struct CompletionCheckResults;

template <typename _ConfigType>
struct Results;

struct SampledData;

template <typename _ConfigType>
struct State;

template <typename _ConfigType>
class ResultsIO {
 public:
  typedef _ConfigType config_type;
  typedef monte::State<config_type> state_type;
  typedef monte::Results<config_type> results_type;

  virtual ~ResultsIO() {}

  virtual std::vector<state_type> read_final_states() = 0;

  virtual void write(results_type const &results, Index run_index) = 0;
};

}  // namespace monte
}  // namespace CASM

#endif
