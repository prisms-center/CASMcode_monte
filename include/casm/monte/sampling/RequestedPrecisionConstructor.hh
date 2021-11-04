#ifndef CASM_monte_RequestedPrecisionConstructor
#define CASM_monte_RequestedPrecisionConstructor

#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

struct RequestedPrecisionConstructor;

/// \brief Helps to specify requested precision
///
/// Example code:
/// \code
/// // Given some samplers:
/// std::map<std::string, std::shared_ptr<Sampler>> &samplers;
///
/// // Construct map of SamplerComponent -> double precision:
/// std::map<SamplerComponent, double> requested_precision =
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
RequestedPrecisionConstructor converge(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    std::string sampler_name);

/// \brief Helper for compact construction of requested precision
///
/// Usage:
/// - This class is intended as a temporary intermediate constructed by the
/// `converge` function. See `converge` documentation for intended usage.
struct RequestedPrecisionConstructor {
  /// \brief Constructor
  RequestedPrecisionConstructor(std::string _sampler_name,
                                Sampler const &_sampler);

  /// \brief Select only the specified component - by index
  RequestedPrecisionConstructor &component(Index component_index);

  /// \brief Select only the specified component - by name
  RequestedPrecisionConstructor &component(std::string component_name);

  /// \brief Set the requested convergence precision for selected components
  RequestedPrecisionConstructor &precision(double _precision);

  /// \brief Conversion operator
  operator std::map<SamplerComponent, double> const &() const;

  std::string sampler_name;
  Sampler const &sampler;
  std::map<SamplerComponent, double> requested_precision;
};

/// \brief Merge
///
/// See `converge` for example usage
inline RequestedPrecisionConstructor &merge(
    RequestedPrecisionConstructor &A, RequestedPrecisionConstructor const &B) {
  for (auto const &pair : B.requested_precision) {
    A.requested_precision.insert(pair);
  }
  return A;
}

/// \brief Merge
///
/// See `converge` for example usage
template <typename T, typename... Args>
RequestedPrecisionConstructor &merge(RequestedPrecisionConstructor &A,
                                     RequestedPrecisionConstructor const &B,
                                     Args &&...args) {
  for (auto const &pair : B.requested_precision) {
    A.requested_precision.insert(pair);
  }
  return merge(A, std::forward<Args>(args)...);
}

}  // namespace monte
}  // namespace CASM

#endif
