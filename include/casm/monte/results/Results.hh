#ifndef CASM_monte_Results
#define CASM_monte_Results

#include <vector>

#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/state/State.hh"

namespace CASM {
namespace monte {

/// \brief Standard Monte Carlo calculation results data structure
///
/// This data structure stores results for Monte Carlo calculations, assuming
/// constant conditions.
template <typename ConfigType>
struct Results {
  /// Initial state
  std::optional<State<ConfigType>> initial_state;

  /// Final state
  std::optional<State<ConfigType>> final_state;

  /// Map of <sampler name>:<sampler>
  /// - `Sampler` stores a Eigen::MatrixXd with sampled data. Rows of the matrix
  ///   corresponds to individual VectorXd samples. The matrices are
  ///   constructed with extra rows and encapsulated in a class so that
  ///   resizing can be done intelligently as needed. Sampler provides
  ///   accessors so that the data can be efficiently accessed by index or by
  ///   component name for equilibration and convergence checking of
  ///   individual components.
  std::map<std::string, std::shared_ptr<Sampler>> samplers;

  /// Map of <analysis name>:<value>
  std::map<std::string, Eigen::VectorXd> analysis;

  /// Vector of counts (could be pass or step) when a sample occurred
  std::vector<CountType> sample_count;

  /// Vector of times when a sample occurred
  std::vector<TimeType> sample_time;

  /// Vector of the configuration when a sample occurred
  std::vector<ConfigType> sample_trajectory;

  /// Completion check results
  CompletionCheckResults completion_check_results;
};

}  // namespace monte
}  // namespace CASM

#endif
