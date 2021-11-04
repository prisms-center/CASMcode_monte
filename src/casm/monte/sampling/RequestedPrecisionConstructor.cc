#include "casm/monte/sampling/RequestedPrecisionConstructor.hh"

#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

/// \brief Constructor
///
/// \param _sampler_name Sampler name
/// \param _sampler Sampler reference
///
/// Note:
/// - Constructs `requested_precision` to include convergence parameters for all
///   components of the specified sampler, with initial values precision =
///   `std::numeric_limits<double>::infinity()`.
RequestedPrecisionConstructor::RequestedPrecisionConstructor(
    std::string _sampler_name, Sampler const &_sampler)
    : sampler_name(_sampler_name), sampler(_sampler) {
  Index i = 0;
  for (std::string const &component_name : sampler.component_names()) {
    requested_precision.emplace(
        SamplerComponent(sampler_name, i, component_name),
        std::numeric_limits<double>::infinity());
    ++i;
  }
}

/// \brief Select only the specified component - by index
RequestedPrecisionConstructor &RequestedPrecisionConstructor::component(
    Index component_index) {
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
  double chosen = requested_precision.at(component);
  requested_precision.clear();
  requested_precision.emplace(component, chosen);
  return *this;
}

/// \brief Select only the specified component - by name
RequestedPrecisionConstructor &RequestedPrecisionConstructor::component(
    std::string component_name) {
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
  double chosen = requested_precision.at(component);
  requested_precision.clear();
  requested_precision.emplace(component, chosen);
  return *this;
}

/// \brief Set the requested convergence precision for selected components
RequestedPrecisionConstructor &RequestedPrecisionConstructor::precision(
    double _precision) {
  for (auto &value : requested_precision) {
    value.second = _precision;
  }
  return *this;
}

/// \brief Conversion operator
RequestedPrecisionConstructor::operator std::map<SamplerComponent,
                                                 double> const &() const {
  return requested_precision;
}

RequestedPrecisionConstructor converge(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    std::string sampler_name) {
  auto it = samplers.find(sampler_name);
  if (it == samplers.end()) {
    std::stringstream msg;
    msg << "Error constructing sampler convergence parameters: "
        << "Did not find a sampler named '" << sampler_name << "'";
    throw std::runtime_error(msg.str());
  }
  return RequestedPrecisionConstructor(sampler_name, *it->second);
}

}  // namespace monte
}  // namespace CASM
