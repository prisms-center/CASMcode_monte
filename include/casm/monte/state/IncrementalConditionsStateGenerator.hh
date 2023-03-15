#ifndef CASM_monte_IncrementalConditionsStateGenerator
#define CASM_monte_IncrementalConditionsStateGenerator

#include <map>
#include <string>

#include "casm/monte/definitions.hh"
#include "casm/monte/state/ConfigGenerator.hh"
#include "casm/monte/state/RunData.hh"
#include "casm/monte/state/State.hh"
#include "casm/monte/state/StateGenerator.hh"
#include "casm/monte/state/StateModifyingFunction.hh"

namespace CASM {
namespace monte {

/// \brief Generates a series of states by constant conditions increments
///
/// The run information needed to check completion and generate subsequent
/// states is the vector of data from the previous runs.
///
/// This method generates states using the following steps:
/// 1) Set indepedently determined conditions, using:
///    \code
///    ValueMap conditions = make_incremented_values(
///        initial_conditions, conditions_increment, completed_runs.size());
///    \endcode
/// 2) Generate an initial configuration, using:
///    \code
///    ConfigType configuration =
///        (dependent_runs && completed_runs.size()) ?
///            ? completed_runs.back().final_state->configuration
///            : config_generator(conditions, completed_runs);
///    \endcode
/// 3) Make a state, using:
///    \code
///    State<ConfigType> state(configuration, conditions)
///    \endcode
/// 4) Apply custom state modifiers, using:
///    \code
///    for (auto const &f : m_modifiers) {
///      f(state);
///    }
///    \endcode
/// 5) Return `state`.
template <typename _ConfigType>
class IncrementalConditionsStateGenerator
    : public StateGenerator<_ConfigType, RunData<_ConfigType>> {
 public:
  typedef _ConfigType ConfigType;
  typedef RunData<ConfigType> RunInfoType;
  typedef ConfigGenerator<ConfigType, RunInfoType> ConfigGeneratorType;

  /// \brief Constructor
  ///
  /// \param _config_generator Function to generate configurations for the
  ///     initial state from the indepedently determined conditions and the
  ///     data from previous runs.
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
  /// \param _modifiers Functions that modify the generated state,
  ///     for instance to set the composition condition for canonical
  ///     calculations based on the composition of the generated or input
  ///     configuration so that it doesn't have to be pre-determined by
  ///     the user.
  IncrementalConditionsStateGenerator(
      std::unique_ptr<ConfigGeneratorType> _config_generator,
      ValueMap const &_initial_conditions,
      ValueMap const &_conditions_increment, Index _n_states,
      bool _dependent_runs,
      std::vector<StateModifyingFunction<ConfigType>> const &_modifiers = {})
      : m_config_generator(std::move(_config_generator)),
        m_initial_conditions(_initial_conditions),
        m_conditions_increment(_conditions_increment),
        m_n_states(_n_states),
        m_dependent_runs(_dependent_runs),
        m_modifiers(_modifiers) {
    std::stringstream msg;
    msg << "Error constructing IncrementalConditionsStateGenerator: "
        << "Mismatch between initial conditions and conditions increment.";
    if (is_mismatched(m_initial_conditions, m_conditions_increment)) {
      throw std::runtime_error(msg.str());
    }
  }

  /// \brief Check if all requested runs have been completed
  bool is_complete(
      std::vector<RunData<ConfigType>> const &completed_runs) override {
    return completed_runs.size() >= m_n_states;
  }

  /// \brief Return the next state
  State<ConfigType> next_state(
      std::vector<RunData<ConfigType>> const &completed_runs) override {
    if (m_dependent_runs && completed_runs.size() &&
        !completed_runs.back().final_state.has_value()) {
      throw std::runtime_error(
          "Error in IncrementalConditionsStateGenerator: when "
          "dependent_runs==true, must save the final state of the last "
          "completed run");
    }

    // Make conditions
    ValueMap conditions = make_incremented_values(
        m_initial_conditions, m_conditions_increment, completed_runs.size());

    // Make configuration
    ConfigType configuration =
        (m_dependent_runs && completed_runs.size())
            ? completed_runs.back().final_state->configuration
            : (*m_config_generator)(conditions, completed_runs);

    // Make state
    State<ConfigType> state(configuration, conditions);

    // Apply custom modifiers
    for (auto const &f : m_modifiers) {
      f(state);
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
  std::vector<StateModifyingFunction<ConfigType>> m_modifiers;
};

}  // namespace monte
}  // namespace CASM

#endif
