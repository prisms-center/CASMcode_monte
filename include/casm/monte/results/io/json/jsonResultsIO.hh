#ifndef CASM_monte_results_io_jsonResultsIO
#define CASM_monte_results_io_jsonResultsIO

#include "casm/misc/cloneable_ptr.hh"
#include "casm/monte/results/ResultsAnalysisFunction.hh"
#include "casm/monte/results/io/ResultsIO.hh"
#include "casm/monte/state/State.hh"

namespace CASM {
namespace monte {

/// \brief Write Monte Carlo results to JSON output files
///
///
template <typename _ResultsType>
class jsonResultsIO : public ResultsIO<_ResultsType> {
  CLONEABLE(jsonResultsIO)
 public:
  typedef _ResultsType results_type;
  typedef typename results_type::config_type config_type;
  typedef typename results_type::stats_type stats_type;

  jsonResultsIO(
      fs::path _output_dir,
      StateSamplingFunctionMap<config_type> _sampling_functions,
      ResultsAnalysisFunctionMap<config_type, stats_type> _analysis_functions,
      bool _write_trajectory, bool _write_observations);

  /// \brief Write results
  void write(results_type const &results, ValueMap const &conditions,
             Index run_index) override;

 protected:
  /// \brief Write summary.json with results from each individual run
  void write_summary(results_type const &results, ValueMap const &conditions);

  /// \brief Write run.<index>/trajectory.json
  void write_trajectory(results_type const &results, Index run_index);

  /// \brief Write run.<index>/observations.json
  void write_observations(results_type const &results, Index run_index);

  jsonParser read_summary();

  fs::path run_dir(Index run_index);

 private:
  fs::path m_output_dir;
  StateSamplingFunctionMap<config_type> m_sampling_functions;
  ResultsAnalysisFunctionMap<config_type, stats_type> m_analysis_functions;
  bool m_write_trajectory;
  bool m_write_observations;
  bool m_write_conditions;
  bool m_write_initial_states;
  bool m_write_final_states;
};

}  // namespace monte
}  // namespace CASM

#endif
