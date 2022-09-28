#ifndef CASM_monte_StateModifyingFunction
#define CASM_monte_StateModifyingFunction

#include "casm/monte/definitions.hh"
#include "casm/monte/state/State.hh"

namespace CASM {
namespace monte {

template <typename ConfigType>
struct StateModifyingFunction {
  /// \brief Constructor - default component names
  StateModifyingFunction(std::string _name, std::string _description,
                         std::function<void(State<ConfigType> &)> _function)
      : name(_name), description(_description), function(_function) {}

  std::string name;

  std::string description;

  std::function<void(State<ConfigType> &)> function;

  /// \brief Evaluates `function`
  void operator()(State<ConfigType> &state) const { function(state); }
};

}  // namespace monte
}  // namespace CASM

#endif
