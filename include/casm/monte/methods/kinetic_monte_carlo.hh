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
void kinetic_monte_carlo(
    State<ConfigType> &state, OccLocation &occ_location,
    std::string &sampling_fixture_label, double &time,
    Eigen::MatrixXd &atom_positions_cart,
    std::map<std::string, double> &prev_time,
    std::map<std::string, Eigen::MatrixXd> &prev_atom_positions_cart,
    EventSelectorType &event_selector, GetEventType get_event_f,
    RunManager<ConfigType> &run_manager);

// --- Implementation ---

/// \brief Run a kinetic Monte Carlo calculation
///
/// TODO: clean up the way data is made available to samplers, especiallly
/// for storing and sharing data taken at the previous sample time.
///
/// \param state The state. Consists of both the initial
///     configuration and conditions. Conditions must include `temperature`
///     and any others required by `potential`.
/// \param occ_location An occupant location tracker, which enables efficient
///     event proposal. It must already be initialized with the input state.
/// \param sampling_fixture_label This will be set to the current sampling
///     fixture label before sampling data.
/// \param state_sampler This will be set to point to the current state
///     sampler sampling data.
/// \param time This will be set to the current time before sampling data.
/// \param atom_positions_cart This will be set to store positions since
///     occ_location was initialized. Before a sample is taken, this will be
///     updated to contain the current atom positions in Cartesian
///     coordinates, with shape=(3, n_atoms). Sampling functions can use this
///     to calculate displacements since the begininning of the calculation
///     or since the last sample time.
/// \param prev_time This will be set to store the time when the last sample
///     was taken, with key equal to sampling fixture label.
/// \param prev_atom_positions_cart This will be set to store positions since
///     occ_location was initialized. The keys are sampling fixture label,
///     and the values will be set to contain the atom positions, in Cartesian
///     coordinates, with shape=(3, n_atoms), at the previous sample time.
///     Sampling functions can use this to calculate displacements since the
///     begininning of the calculation or since the last sample time.
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
void kinetic_monte_carlo(
    State<ConfigType> &state, OccLocation &occ_location,
    std::string &sampling_fixture_label,
    monte::StateSampler<ConfigType> const *&state_sampler, double &time,
    Eigen::MatrixXd &atom_positions_cart,
    std::map<std::string, double> &prev_time,
    std::map<std::string, Eigen::MatrixXd> &prev_atom_positions_cart,
    EventSelectorType &event_selector, GetEventType get_event_f,
    RunManager<ConfigType> &run_manager) {
  // Used within the main loop:
  double event_time;
  double time_increment;
  clexmonte::EventID selected_event_id;

  // Initialize atom positions & time
  time = 0.0;
  atom_positions_cart = occ_location.atom_positions_cart();
  prev_atom_positions_cart.clear();
  for (auto &fixture : run_manager.sampling_fixtures) {
    prev_time.emplace(fixture.label(), time);
    prev_atom_positions_cart.emplace(fixture.label(), atom_positions_cart);
  }

  // Main loop
  run_manager.initialize(state, occ_location.mol_size());
  run_manager.update_next_sampling_fixture();
  run_manager.sample_data_by_count_if_due(state);
  while (!run_manager.is_complete()) {
    run_manager.write_status_if_due();

    // Select an event
    std::tie(selected_event_id, time_increment) = event_selector.select_event();
    event_time = time + time_increment;

    // Sample data, if a sample is due by time
    while (run_manager.next_sampling_fixture != nullptr &&
           event_time >= run_manager.next_sample_time) {
      auto &fixture = *run_manager.next_sampling_fixture;

      // TODO: if acceleration, update groups to sample time
      sampling_fixture_label = fixture.label();
      state_sampler = &fixture.state_sampler();
      time = run_manager.next_sample_time;
      atom_positions_cart = occ_location.atom_positions_cart();

      fixture.set_time(time);
      fixture.sample_data(state);

      prev_time[fixture.label()] = time;
      prev_atom_positions_cart[fixture.label()] = atom_positions_cart;

      run_manager.update_next_sampling_fixture();
    }

    // Apply event
    occ_location.apply(get_event_f(selected_event_id), get_occupation(state));
    time = event_time;

    // Set time -- for all fixtures
    run_manager.set_time(event_time);

    // Increment count -- for all fixtures
    run_manager.increment_step();

    // Sample data, if a sample is due by count
    for (auto &fixture : run_manager.sampling_fixtures) {
      auto const &ss = fixture.state_sampler();
      if (ss.sample_mode != SAMPLE_MODE::BY_TIME) {
        if (ss.count == ss.next_sample_count) {
          sampling_fixture_label = fixture.label();
          state_sampler = &fixture.state_sampler();
          atom_positions_cart = occ_location.atom_positions_cart();

          fixture.sample_data(state);
          prev_time[fixture.label()] = time;
          prev_atom_positions_cart[fixture.label()] = atom_positions_cart;
        }
      }
    }
  }

  run_manager.finalize(state);
}

}  // namespace monte
}  // namespace CASM

#endif
