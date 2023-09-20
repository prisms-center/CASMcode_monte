#ifndef CASM_monte_State
#define CASM_monte_State

#include <map>
#include <string>

#include "casm/global/eigen.hh"
#include "casm/monte/ValueMap.hh"
#include "casm/monte/definitions.hh"

namespace CASM {
namespace monte {

/// A state of a Monte Carlo calculation
template <typename _ConfigType>
struct State {
  typedef _ConfigType ConfigType;

  explicit State(ConfigType const &_configuration,
                 ValueMap _conditions = ValueMap(),
                 ValueMap _properties = ValueMap())
      : configuration(_configuration),
        conditions(_conditions),
        properties(_properties) {}

  /// Current configuration
  ConfigType configuration;

  /// Conditions of the state
  ///
  /// Thermodynamic conditions or calculation constraints, such as temperature,
  /// chemical potential (for grand canonical Monte Carlo), composition (for
  /// canonical Monte Carlo), etc., depending on the type of Monte Carlo
  /// calculation
  ValueMap conditions;

  /// Properties of the state
  ///
  /// Properties of the state could be formation_energy, potential_energy,
  /// comp_n, etc., depending on the type of Monte Carlo calculation.
  ValueMap properties;
};

}  // namespace monte
}  // namespace CASM

#endif
