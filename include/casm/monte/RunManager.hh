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

  /// Next time-based sampling fixture, or nullptr if none
  SamplingFixture<config_type> *next_sampling_fixture;

  /// Next time-based sampling sample time
  double next_sample_time;

  RunManager(std::vector<SamplingFixtureParams<config_type>> const
                 &sampling_fixture_params)
      : next_sampling_fixture(nullptr), next_sample_time(0.0) {
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
    // do not quit early so that status files can be printed with
    // the latest completion check results
    bool result = true;
    for (auto &fixture : sampling_fixtures) {
      result &= fixture.is_complete();
    }
    return result;
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

  void set_time(double event_time) {
    for (auto &fixture : sampling_fixtures) {
      fixture.set_time(event_time);
    }
  }

  void sample_data_by_count_if_due(state_type const &state) {
    for (auto &fixture : sampling_fixtures) {
      auto const &ss = fixture.state_sampler();
      if (ss.sample_mode != SAMPLE_MODE::BY_TIME) {
        if (ss.count == ss.next_sample_count) {
          fixture.sample_data(state);
        }
      }
    }
  }

  void update_next_sampling_fixture() {
    // update next_sample_time and next_sampling_fixture
    next_sampling_fixture = nullptr;
    for (auto &fixture : sampling_fixtures) {
      auto const &ss = fixture.state_sampler();
      if (ss.sample_mode == SAMPLE_MODE::BY_TIME) {
        if (next_sampling_fixture == nullptr ||
            ss.next_sample_time < next_sample_time) {
          next_sample_time = ss.next_sample_time;
          next_sampling_fixture = &fixture;
        }
      }
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
