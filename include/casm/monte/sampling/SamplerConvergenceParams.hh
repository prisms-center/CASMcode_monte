#ifndef CASM_monte_SamplerConvergenceParams
#define CASM_monte_SamplerConvergenceParams

#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

/// \brief Holds convergence checking parameters
struct SamplerConvergenceParams {
  SamplerConvergenceParams(double _precision, double _confidence = 0.95);

  double precision;

  double confidence;
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
  SamplerConvergenceParamsConstructor &component(Index component_name);

  /// \brief Set the requested convergence precision for selected components
  SamplerConvergenceParamsConstructor &precision(double _precision);

  /// \brief Set the requested convergence confidence level for selected
  ///     components
  SamplerConvergenceParamsConstructor &confidence(double _confidence);

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
///         converge(samplers, "formation_energy")
///             .precision(0.001).confidence(0.95),
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
/// - Default confidence level if `.confidence()` is left off is 0.95
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
  A.values.merge(B.values);
  return A;
}

/// \brief Merge
///
/// See `converge` for example usage
template <typename T, typename... Args>
SamplerConvergenceParamsConstructor &merge(
    SamplerConvergenceParamsConstructor &A,
    SamplerConvergenceParamsConstructor const &B, Args &&...args) {
  A.values.merge(B.values);
  return merge(A, std::forward<Args>(args)...);
}

}  // namespace monte
}  // namespace CASM

// --- Inline implementations ---

namespace CASM {
namespace monte {

SamplerConvergenceParams::SamplerConvergenceParams(double _precision,
                                                   double _confidence = 0.95)
    : precision(_precision), confidence(_confidence) {}

/// \brief Constructor
///
/// \param _sampler_name Sampler name
/// \param _sampler Sampler reference
///
/// Note:
/// - Constructs `values` to include convergence parameters for all
///   components of the specified sampler, with initial values precision =
///   `std::numeric_limits<double>::infinity()`, confidence = `0.95`.
SamplerConvergenceParamsConstructor::SamplerConvergenceParamsConstructor(
    std::string _sampler_name, Sampler const &_sampler)
    : sampler_name(_sampler_name), sampler(_sampler) {
  for (Index i = 0; i < sampler.n_components(); ++i) {
    values.emplace(
        SamplerComponent(sampler_name, i),
        SamplerConvergenceParams(std::numeric_limits<double>::infinity()));
  }
}

/// \brief Select only the specified component - by index
SamplerConvergenceParamsConstructor::SamplerConvergenceParamsConstructor &
component(Index component_index) {
  if (component_index >= sampler.component_names().size()) {
    std::stringstream msg;
    msg << "Error constructing sampler convergence parameters: Component "
           "index '"
        << component_name << "' out of range for sampler '" << sampler_name
        << "'";
    throw std::runtime_error(msg.str());
  }
  SamplerComponent component(sampler_name, component_index);
  SamplerConvergenceParams chosen = values.at(component);
  values.clear();
  values.emplace(component, chosen);
  return *this;
}

/// \brief Select only the specified component - by name
SamplerConvergenceParamsConstructor::SamplerConvergenceParamsConstructor &
component(Index component_name) {
  auto begin = sampler.component_names().begin();
  auto end = sampler.component_names().end();
  auto it = std::find(begin, end, component_name);
  if (it == end) {
    std::stringstream msg;
    msg << "Error constructing sampler convergence parameters: Cannot find "
           "component '"
        << component_name << "' for sampler '" << sampler_name << "'";
    throw std::runtime_error(msg.str());
  }
  Index component_index = std::distance(begin, it);
  SamplerComponent component(sampler_name, component_index);
  SamplerConvergenceParams chosen = values.at(component);
  values.clear();
  values.emplace(component, chosen);
  return *this;
}

/// \brief Set the requested convergence precision for selected components
SamplerConvergenceParamsConstructor::SamplerConvergenceParamsConstructor &
precision(double _precision) {
  for (auto &value : values) {
    value.second.precision = _precision;
  }
  return *this;
}

/// \brief Set the requested convergence confidence level for selected
///     components
SamplerConvergenceParamsConstructor::SamplerConvergenceParamsConstructor &
confidence(double _confidence) {
  for (auto &value : values) {
    value.second.confidence = _confidence;
  }
  return *this;
}

/// \brief Conversion operator
operator std::map<SamplerComponent, SamplerConvergenceParams> const &() const {
  return values;
}

SamplerConvergenceParamsConstructor converge(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    std::string sampler_name) {
  auto it = samplers.find(sampler_name);
  if (it == samplers.end()) {
    std::stringstream msg;
    msg << "Error constructing sampler convergence parameters: "
        << "Did not find a sampler named '" << sampler_name << "'";
    throw std::runtime_error(msg.str());
  }
  return SamplerConvergenceParamsConstructor(sampler_name, *it->second);
}

}  // namespace monte
}  // namespace CASM

#endif
