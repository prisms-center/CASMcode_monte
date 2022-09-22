#ifndef CASM_monte_RunManager
#define CASM_monte_RunManager

#include "casm/monte/SamplingFixture.hh"

namespace CASM {
namespace monte {

/// \brief Holds sampling fixtures and checks for completion
///
/// Notes:
/// - Currently, all sampling fixtures keep sampling, even if
///   completion criteria are completed, until all are completed.
///   Reading final states, and using as input to state
///   generator is more complicated otherwise.
///
template <typename _ConfigType>
struct RunManager {
  typedef _ConfigType config_type;
  typedef State<config_type> state_type;

  /// Sampling fixtures
  std::vector<SamplingFixture<config_type>> sampling_fixtures;

  /// Final states
  std::vector<state_type> final_states;

  RunManager(std::vector<SamplingFixtureParams<config_type>> const
                 &sampling_fixture_params) {
    for (auto const &params : sampling_fixture_params) {
      sampling_fixtures.emplace_back(params);
    }
  }

  /// \brief Return final states from existing results
  ///
  /// Notes:
  /// - Currently reads from the first sampling fixture
  ///   that has a results_io object
  void read_final_states() {
    final_states.clear();
    if (sampling_fixtures.size() == 0) {
      return;
    }
    for (auto const &fixture : sampling_fixtures) {
      if (fixture.params().results_io) {
        final_states = fixture.params().results_io->read_final_states();
        return;
      }
    }
    return;
  }

  void initialize(state_type const &state, Index steps_per_pass) {
    for (auto &fixture : sampling_fixtures) {
      fixture.initialize(state, steps_per_pass);
    }
  }

  bool is_complete() {
    for (auto &fixture : sampling_fixtures) {
      if (!fixture.is_complete()) {
        return false;
      }
    }
    return true;
  }

  void write_status_if_due() {
    for (auto &fixture : sampling_fixtures) {
      fixture.write_status_if_due();
    }
  }

  void increment_step() {
    for (auto &fixture : sampling_fixtures) {
      fixture.increment_step();
    }
  }

  void increment_time(double time_increment) {
    for (auto &fixture : sampling_fixtures) {
      fixture.increment_time(time_increment);
    }
  }

  void sample_data_by_count_if_due(state_type const &state) {
    for (auto &fixture : sampling_fixtures) {
      fixture.sample_data_by_count_if_due(state);
    }
  }

  void sample_data_by_time_if_due(state_type const &state,
                                  double time_increment) {
    for (auto &fixture : sampling_fixtures) {
      fixture.sample_data_by_time_if_due(state, time_increment);
    }
  }

  void finalize(state_type const &final_state) {
    final_states.push_back(final_state);
    for (auto &fixture : sampling_fixtures) {
      fixture.finalize(final_state, final_states.size());
    }
  }
};

}  // namespace monte
}  // namespace CASM

#endif
