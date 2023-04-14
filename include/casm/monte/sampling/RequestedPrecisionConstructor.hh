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
/// // Construct map of SamplerComponent -> RequestedPrecision precision:
/// std::map<SamplerComponent, RequestedPrecision> requested_precision =
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
/// - Example converging all components to the same absolute precision:
///   \code
///   converge(samplers, "corr").precision(0.001)
///   \endcode
/// - Example converging a particular component (by index) to particular
///   absolute precision:
///   \code
///   converge(samplers, "corr").component(1).precision(0.001)
///   \endcode
/// - Example converging a particular component (by name) to particular
///   absolute precision:
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

  /// \brief Set the requested convergence absolute precision for selected
  /// components
  RequestedPrecisionConstructor &precision(double _abs_precision);

  /// \brief Set the requested convergence absolute precision for selected
  /// components
  RequestedPrecisionConstructor &abs_precision(double _value);

  /// \brief Set the requested convergence relative precision for selected
  /// components
  RequestedPrecisionConstructor &rel_precision(double _value);

  /// \brief Set the requested convergence absolute and relative precision for
  /// selected components
  RequestedPrecisionConstructor &abs_and_rel_precision(double _abs_value,
                                                       double _rel_value);

  /// \brief Conversion operator
  operator std::map<SamplerComponent, RequestedPrecision> const &() const;

  std::string sampler_name;
  Sampler const &sampler;
  std::map<SamplerComponent, RequestedPrecision> requested_precision;
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
