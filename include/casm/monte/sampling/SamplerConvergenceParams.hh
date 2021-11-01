#ifndef CASM_monte_SamplerConvergenceParams
#define CASM_monte_SamplerConvergenceParams

#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

/// \brief Holds convergence checking parameters
struct SamplerConvergenceParams {
  SamplerConvergenceParams(double _precision);

  double precision;
};

/// \brief Helper for compact construction of sampler convergence params
///
/// Usage:
/// - This class is intended as a temporary intermediate constructed by the
/// `converge` function. See `converge` documentation for intended usage.
struct SamplerConvergenceParamsConstructor {
  /// \brief Constructor
  SamplerConvergenceParamsConstructor(std::string _sampler_name,
                                      Sampler const &_sampler);

  /// \brief Select only the specified component - by index
  SamplerConvergenceParamsConstructor &component(Index component_index);

  /// \brief Select only the specified component - by name
  SamplerConvergenceParamsConstructor &component(std::string component_name);

  /// \brief Set the requested convergence precision for selected components
  SamplerConvergenceParamsConstructor &precision(double _precision);

  /// \brief Conversion operator
  operator std::map<SamplerComponent, SamplerConvergenceParams> const &() const;

  std::string sampler_name;
  Sampler const &sampler;
  std::map<SamplerComponent, SamplerConvergenceParams> values;
};

/// \brief Helps to specify SamplerConvergenceParams
///
/// Example code:
/// \code
/// // Given some samplers:
/// std::map<std::string, std::shared_ptr<Sampler>> &samplers;
///
/// // Construct map of SamplerComponent -> SamplerConvergenceParams:
/// std::map<SamplerComponent, SamplerConvergenceParams> convergence_params =
///     merge(
///         converge(samplers, "formation_energy").precision(0.001),
///         converge(samplers, "comp_n").component("O").precision(0.001),
///         converge(samplers, "comp_n").component("Va").precision(0.01),
///         converge(samplers, "corr").component_index(1).precision(0.001),
///         converge(samplers, "corr").component_index(2).precision(0.001),
///         converge(samplers, "comp").precision(0.001));
/// \endcode
///
/// Notes:
/// - Example converging all components to the same precision:
///   \code
///   converge(samplers, "corr").precision(0.001)
///   \endcode
/// - Example converging a particular component (by index) to particular
///   precision:
///   \code
///   converge(samplers, "corr").component(1).precision(0.001)
///   \endcode
/// - Example converging a particular component (by name) to particular
///   precision:
///   \code
///   converge(samplers, "comp_n").component("Mg").precision(0.001)
///   \endcode
///
/// - Default precision if `.precision()` is left off is
///   `std::numeric_limits<double>::infinity` (essentially no convergence
///   requested)
///
SamplerConvergenceParamsConstructor converge(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    std::string sampler_name);

/// \brief Merge
///
/// See `converge` for example usage
inline SamplerConvergenceParamsConstructor &merge(
    SamplerConvergenceParamsConstructor &A,
    SamplerConvergenceParamsConstructor const &B) {
  for (auto const &pair : B.values) {
    A.values.insert(pair);
  }
  return A;
}

/// \brief Merge
///
/// See `converge` for example usage
template <typename T, typename... Args>
SamplerConvergenceParamsConstructor &merge(
    SamplerConvergenceParamsConstructor &A,
    SamplerConvergenceParamsConstructor const &B, Args &&...args) {
  for (auto const &pair : B.values) {
    A.values.insert(pair);
  }
  return merge(A, std::forward<Args>(args)...);
}

}  // namespace monte
}  // namespace CASM

#endif
