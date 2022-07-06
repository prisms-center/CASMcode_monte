#ifndef CASM_monte_ConvergenceCheck
#define CASM_monte_ConvergenceCheck

#include "casm/monte/definitions.hh"
#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

/// \brief Convergence check results data structure (individual component)
struct IndividualConvergenceCheckResult {
  /// \brief True if mean (<X>) is converged to requested precision
  bool is_converged;

  /// \brief Mean of property (<X>)
  double mean;

  /// \brief Squared norm of property (\sum_i X_i*X_i)
  double squared_norm;

  /// \brief Calculated absolute precision in <X>
  ///
  /// Notes:
  /// - See `convergence_check` function for calculation details
  double calculated_precision;

  /// \brief Requested absolute precision in <X>
  double requested_precision;
};

/// \brief Check convergence of a range of observations
IndividualConvergenceCheckResult convergence_check(
    Eigen::VectorXd const &observations, double precision, double confidence);

/// \brief Convergence check results data structure (all requested components)
struct ConvergenceCheckResults {
  /// \brief True if all required properties are converged to required precision
  ///
  /// Notes:
  /// - True if completion check finds all required properties are converged to
  /// the requested precision
  /// - False otherwise, including if no convergence checks were requested
  bool all_converged = false;

  /// \brief How many samples were used to get statistics
  ///
  /// Notes:
  /// - Set to the total number of samples if no convergence checks were
  /// requested
  CountType N_samples_for_statistics = 0;

  /// \brief Results from checking convergence criteria
  std::map<SamplerComponent, IndividualConvergenceCheckResult>
      individual_results;
};

/// \brief Check convergence of all requested properties
ConvergenceCheckResults convergence_check(
    std::map<SamplerComponent, double> const &requested_precision,
    double confidence, CountType N_samples_for_equilibration,
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers);

}  // namespace monte
}  // namespace CASM

#endif
