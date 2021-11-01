#ifndef CASM_monte_results_io_jsonResultsIO
#define CASM_monte_results_io_jsonResultsIO

#include "casm/monte/results/io/ResultsIO.hh"
#include "casm/monte/state/State.hh"

namespace CASM {
namespace monte {

/// \brief Write Monte Carlo results to JSON output files
///
///
template <typename _ConfigType>
class jsonResultsIO : public ResultsIO<_ConfigType> {
 public:
  typedef _ConfigType config_type;
  typedef monte::State<config_type> state_type;
  typedef monte::Results<config_type> results_type;

  jsonResultsIO(fs::path _output_dir,
                StateSamplingFunctionMap<config_type> _sampling_functions,
                bool _write_trajectory, bool _write_observations);

  /// \brief Read a vector of final states of completed runs
  std::vector<state_type> read_final_states() override;

  /// \brief Write results
  void write(results_type const &results, Index run_index) override;

  /// \brief Write summary.json with results from each individual run
  void write_summary(
      results_type const &results,
      StateSamplingFunctionMap<config_type> const &sampling_functions);

  /// \brief Write run.<index>/trajectory.json
  void write_trajectory(std::vector<state_type> const &trajectory,
                        Index run_index);

  /// \brief Write run.<index>/observations.json
  void write_observations(monte::SampledData const &sampled_data,
                          Index run_index);

  jsonParser read_summary();

  fs::path run_dir(Index run_index);

 private:
  fs::path m_output_dir;
  StateSamplingFunctionMap<config_type> m_sampling_functions;
  bool m_write_trajectory;
  bool m_write_observations;
};

}  // namespace monte
}  // namespace CASM

#endif
