#ifndef CASM_monte_methods_kinetic_monte_carlo
#define CASM_monte_methods_kinetic_monte_carlo

// logging
#include "casm/casm_io/Log.hh"
#include "casm/casm_io/container/stream_io.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/MethodLog.hh"
#include "casm/monte/checks/io/json/CompletionCheck_json_io.hh"

namespace CASM {
namespace monte {

template <typename EventIDType, typename ConfigType, typename EventSelectorType,
          typename GetEventType>
void kinetic_monte_carlo(State<ConfigType> &state, OccLocation &occ_location,
                         EventSelectorType &event_selector,
                         GetEventType get_event_f,
                         RunManager<ConfigType> &run_manager);

// --- Implementation ---

/// \brief Run a kinetic Monte Carlo calculation
///
/// \param state The state. Consists of both the initial
///     configuration and conditions. Conditions must include `temperature`
///     and any others required by `potential`.
/// \param occ_location An occupant location tracker, which enables efficient
///     event proposal. It must already be initialized with the input state.
/// \param potential A potential calculating method. Should match the interface
///     described below and already be set to calculate the potential for the
///     input state.
/// \param event_selector A method that selects events and returns an
///     std::pair<EventIDType, TimeIncrementType>.
/// \param get_event_f A method that gives an `OccEvent const &` corresponding
///     to the selected EventID.
/// \param run_manager Contains sampling fixtures and after completion holds
///     final results
///
/// \returns A Results<ConfigType> instance with run results.
///
/// Required interface for `State<ConfigType>`:
/// - `Eigen::VectorXi &get_occupation(State<ConfigType> const &state)`
/// - `Eigen::Matrix3l const &get_transformation_matrix_to_supercell(
///        State<ConfigType> const &state)`
///
/// State properties that are set:
/// - None
///
template <typename EventIDType, typename ConfigType, typename EventSelectorType,
          typename GetEventType>
void kinetic_monte_carlo(State<ConfigType> &state, OccLocation &occ_location,
                         EventSelectorType &event_selector,
                         GetEventType get_event_f,
                         RunManager<ConfigType> &run_manager) {
  // Used within the main loop:
  double time_increment = 0.0;
  clexmonte::EventID selected_event_id;

  // Main loop
  run_manager.initialize(state, occ_location.mol_size());
  run_manager.sample_data_by_time_if_due(state, 0.0);
  run_manager.sample_data_by_count_if_due(state);
  while (!run_manager.is_complete()) {
    run_manager.write_status_if_due();

    // Select an event
    std::tie(selected_event_id, time_increment) = event_selector.select_event();

    // Sample data, if a sample is due by time
    // (due if current time + time_interval >= next sample time)
    run_manager.sample_data_by_time_if_due(state, time_increment);

    // Apply event
    occ_location.apply(get_event_f(selected_event_id), get_occupation(state));

    // Increment count & time
    run_manager.increment_step();
    run_manager.increment_time(time_increment);

    // Sample data, if a sample is due by count
    run_manager.sample_data_by_count_if_due(state);
  }

  run_manager.finalize(state);
}

}  // namespace monte
}  // namespace CASM

#endif
