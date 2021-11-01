#include "casm/monte/sampling/SamplerConvergenceParams.hh"

#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

SamplerConvergenceParams::SamplerConvergenceParams(double _precision)
    : precision(_precision) {}

/// \brief Constructor
///
/// \param _sampler_name Sampler name
/// \param _sampler Sampler reference
///
/// Note:
/// - Constructs `values` to include convergence parameters for all
///   components of the specified sampler, with initial values precision =
///   `std::numeric_limits<double>::infinity()`.
SamplerConvergenceParamsConstructor::SamplerConvergenceParamsConstructor(
    std::string _sampler_name, Sampler const &_sampler)
    : sampler_name(_sampler_name), sampler(_sampler) {
  Index i = 0;
  for (std::string const &component_name : sampler.component_names()) {
    values.emplace(
        SamplerComponent(sampler_name, i, component_name),
        SamplerConvergenceParams(std::numeric_limits<double>::infinity()));
    ++i;
  }
}

/// \brief Select only the specified component - by index
SamplerConvergenceParamsConstructor &
SamplerConvergenceParamsConstructor::component(Index component_index) {
  if (component_index >= sampler.component_names().size()) {
    std::stringstream msg;
    msg << "Error constructing sampler convergence parameters: Component "
           "index '"
        << component_index << "' out of range for sampler '" << sampler_name
        << "'";
    throw std::runtime_error(msg.str());
  }
  SamplerComponent component(sampler_name, component_index,
                             sampler.component_names()[component_index]);
  SamplerConvergenceParams chosen = values.at(component);
  values.clear();
  values.emplace(component, chosen);
  return *this;
}

/// \brief Select only the specified component - by name
SamplerConvergenceParamsConstructor &
SamplerConvergenceParamsConstructor::component(std::string component_name) {
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
  SamplerComponent component(sampler_name, component_index, component_name);
  SamplerConvergenceParams chosen = values.at(component);
  values.clear();
  values.emplace(component, chosen);
  return *this;
}

/// \brief Set the requested convergence precision for selected components
SamplerConvergenceParamsConstructor &
SamplerConvergenceParamsConstructor::precision(double _precision) {
  for (auto &value : values) {
    value.second.precision = _precision;
  }
  return *this;
}

/// \brief Conversion operator
SamplerConvergenceParamsConstructor::operator std::map<
    SamplerComponent, SamplerConvergenceParams> const &() const {
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
