#ifndef CASM_monte_IncrementalConditionsStateGenerator
#define CASM_monte_IncrementalConditionsStateGenerator

#include <map>
#include <string>

#include "casm/monte/definitions.hh"
#include "casm/monte/state/ConfigGenerator.hh"
#include "casm/monte/state/State.hh"
#include "casm/monte/state/StateGenerator.hh"

namespace CASM {
namespace monte {

/// \brief Generates a series of states by constant conditions increments
///
/// The run information needed to check completion and generate subsequent
/// states is the vector of final states from the previous runs.
///
/// This method generates states using the following steps:
/// 1) Set indepedently determined conditions, using:
///    \code
///    ValueMap conditions = make_incremented_values(
///        initial_conditions, conditions_increment, final_states.size());
///    \endcode
/// 2) Generate an initial configuration, using:
///    \code
///    ConfigType configuration =
///        (dependent_runs && final_states.size()) ?
///            ? final_states.back().configuration
///            : config_generator(conditions, final_states);
///    \endcode
/// 3) Make a state, using:
///    \code
///    State<ConfigType> state(configuration, conditions)
///    \endcode
/// 4) Set dependent conditions, using:
///    \code
///    for (auto const &pair : m_dependent_conditions) {
///      state.conditions.vector_values[pair.first] = pair.second(state);
///    }
///    \endcode
/// 5) Return `state`.
template <typename _ConfigType>
class IncrementalConditionsStateGenerator
    : public StateGenerator<_ConfigType, State<_ConfigType>> {
 public:
  typedef _ConfigType ConfigType;
  typedef State<ConfigType> RunInfoType;
  typedef ConfigGenerator<ConfigType, RunInfoType> ConfigGeneratorType;

  /// \brief Constructor
  ///
  /// \param _config_generator Function to generate configurations for the
  ///     initial state from the indepedently determined conditions and the
  ///     final states of previous runs.
  /// \param _initial_conditions The "indepedently determined conditions" for
  ///     the initial state.
  /// \param _conditions_increment The conditions to be changed between states,
  ///     and the amount to change them. A key in `_conditions_increment`
  ///     must also be a key of `_initial_conditions`.
  /// \param _n_states The total number of states to generate. Includes the
  ///     initial state.
  /// \param _dependent_runs If true, use the last configuration as the starting
  ///     point for the next state. If false, always use the configuration of
  ///     the initial state.
  /// \param _dependent_conditions State sampling functions
  ///     which are used to set conditions using values calculated from the
  ///     generated configuration. For example, a composition sampling function
  ///     could be provided to set the composition for a canonical Monte Carlo
  ///     calculation directly from the initial configuration proposed by
  ///     `_config_generator`. Currently restricted to setting vector valued
  ///     conditions.
  IncrementalConditionsStateGenerator(
      std::unique_ptr<ConfigGeneratorType> _config_generator,
      ValueMap const &_initial_conditions,
      ValueMap const &_conditions_increment, Index _n_states,
      bool _dependent_runs,
      StateSamplingFunctionMap<ConfigType> const &_dependent_conditions = {})
      : m_config_generator(std::move(_config_generator)),
        m_initial_conditions(_initial_conditions),
        m_conditions_increment(_conditions_increment),
        m_n_states(_n_states),
        m_dependent_runs(_dependent_runs),
        m_dependent_conditions(_dependent_conditions) {
    std::stringstream msg;
    msg << "Error constructing IncrementalConditionsStateGenerator: "
        << "Mismatch between initial conditions and conditions increment.";
    if (is_mismatched(m_initial_conditions, m_conditions_increment)) {
      throw std::runtime_error(msg.str());
    }
  }

  /// \brief Check if all requested states have been run
  bool is_complete(
      std::vector<State<ConfigType>> const &final_states) override {
    return final_states.size() >= m_n_states;
  }

  /// \brief Return the next state
  State<ConfigType> next_state(
      std::vector<State<ConfigType>> const &final_states) override {
    // Make conditions
    ValueMap conditions = make_incremented_values(
        m_initial_conditions, m_conditions_increment, final_states.size());

    // Make configuration
    ConfigType configuration =
        (m_dependent_runs && final_states.size())
            ? final_states.back().configuration
            : (*m_config_generator)(conditions, final_states);

    // Make state
    State<ConfigType> state(configuration, conditions);

    // Set dependent conditions
    for (auto const &pair : m_dependent_conditions) {
      state.conditions.vector_values[pair.first] = pair.second(state);
    }

    // Finished
    return state;
  }

 private:
  std::unique_ptr<ConfigGeneratorType> m_config_generator;
  ValueMap m_initial_conditions;
  ValueMap m_conditions_increment;
  Index m_n_states;
  bool m_dependent_runs;
  StateSamplingFunctionMap<ConfigType> m_dependent_conditions;
};

}  // namespace monte
}  // namespace CASM

#endif
