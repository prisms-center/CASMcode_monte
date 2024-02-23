/// An implementation of an N-Fold Way occupation Monte Carlo
/// main loop that makes use of the RunManager provided by
/// casm/monte/run_management to implement sampling
/// fixtures and results data structures and input/output
/// methods.

#ifndef CASM_monte_methods_nfold
#define CASM_monte_methods_nfold

// logging
#include "casm/casm_io/Log.hh"
#include "casm/casm_io/container/stream_io.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/MethodLog.hh"
#include "casm/monte/checks/io/json/CompletionCheck_json_io.hh"
#include "casm/monte/events/OccLocation.hh"

namespace CASM {
namespace monte {

/// \brief Data that can be used by sampling functions
///
/// Notes:
/// - Allows sampling `expected_acceptance_rate`
template <typename ConfigType, typename StatisticsType, typename EngineType>
struct NfoldData {
  /// \brief This will be set to the current sampling
  ///     fixture label before sampling data.
  std::string sampling_fixture_label;

  /// \brief This will be set to point to the current sampling fixture
  monte::SamplingFixture<ConfigType, StatisticsType, EngineType> const
      *sampling_fixture;

  /// \brief Total number of events that could be selected at any time
  ///     This is set before running and remains constant.
  double n_events_possible;

  /// \brief This will be set to the expected metropolis algorithm acceptance
  ///     rate given the current event acceptance probabilities using
  ///     total_rate / n_events_possible.
  double expected_acceptance_rate;
};

template <typename EventIDType, typename ConfigType, typename EventSelectorType,
          typename GetEventType, typename StatisticsType, typename EngineType>
void nfold(State<ConfigType> &state, OccLocation &occ_location,
           NfoldData<ConfigType, StatisticsType, EngineType> &nfold_data,
           EventSelectorType &event_selector, GetEventType get_event_f,
           RunManager<ConfigType, StatisticsType, EngineType> &run_manager);

// --- Implementation ---

/// \brief Run a kinetic Monte Carlo calculation
///
/// \param state The state. Consists of both the initial
///     configuration and conditions. Conditions must include `temperature`
///     and any others required by `potential`.
/// \param occ_location An occupant location tracker, which enables efficient
///     event proposal. It must already be initialized with the input state.
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
/// - `Eigen::Matrix3l const &get_transformation_matrix_to_super(
///        State<ConfigType> const &state)`
///
/// State properties that are set:
/// - None
///
template <typename EventIDType, typename ConfigType, typename EventSelectorType,
          typename GetEventType, typename StatisticsType, typename EngineType>
void nfold(State<ConfigType> &state, OccLocation &occ_location,
           NfoldData<ConfigType, StatisticsType, EngineType> &nfold_data,
           EventSelectorType &event_selector, GetEventType get_event_f,
           RunManager<ConfigType, StatisticsType, EngineType> &run_manager) {
  // Used within the main loop:
  double total_rate;
  double time;
  double event_time;
  double time_increment;
  EventIDType selected_event_id;

  // Initialize time
  time = 0.0;

  // notes: it is important this uses
  // - the total_rate obtained before event selection
  // - the time_increment from event selection
  auto pre_sample_action =
      [&](SamplingFixture<ConfigType, StatisticsType, EngineType> &fixture,
          State<ConfigType> const &state) {
        nfold_data.sampling_fixture_label = fixture.label();
        nfold_data.sampling_fixture = &fixture;
        nfold_data.expected_acceptance_rate =
            total_rate / nfold_data.n_events_possible;
        if (fixture.params().sampling_params.sample_mode ==
            SAMPLE_MODE::BY_TIME) {
          fixture.push_back_sample_weight(1.0);
          time = fixture.next_sample_time();
        } else {
          fixture.push_back_sample_weight(time_increment);
        }
      };
  typename RunManager<ConfigType, StatisticsType, EngineType>::NullAction
      post_sample_action;

  // Main loop
  run_manager.initialize(occ_location.mol_size());
  run_manager.update_next_sampling_fixture();
  while (!run_manager.is_complete()) {
    run_manager.write_status_if_due();

    // Select an event
    total_rate = event_selector.total_rate();
    std::tie(selected_event_id, time_increment) = event_selector.select_event();
    event_time = time + time_increment;

    // Sample data, if a sample is due by count
    run_manager.sample_data_by_count_if_due(state, pre_sample_action,
                                            post_sample_action);

    // Sample data, if a sample is due by time
    run_manager.sample_data_by_time_if_due(event_time, state, pre_sample_action,
                                           post_sample_action);

    // Apply event
    run_manager.increment_n_accept();
    occ_location.apply(get_event_f(selected_event_id), get_occupation(state));
    time = event_time;

    // Set time -- for all fixtures
    run_manager.set_time(event_time);

    // Increment count -- for all fixtures
    run_manager.increment_step();
  }

  run_manager.finalize(state);
}

}  // namespace monte
}  // namespace CASM

#endif
