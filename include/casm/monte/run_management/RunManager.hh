#ifndef CASM_monte_RunManager
#define CASM_monte_RunManager

#include "casm/monte/run_management/SamplingFixture.hh"

namespace CASM {
namespace monte {

struct RunCounter {};

/// \brief Holds sampling fixtures and checks for completion
///
/// Notes:
/// - Currently, all sampling fixtures keep sampling, even if
///   completion criteria are completed, until all are completed.
///   Reading final states, and using as input to state
///   generator is more complicated otherwise.
///
template <typename _ConfigType, typename _StatisticsType, typename _EngineType>
struct RunManager {
  typedef _ConfigType config_type;
  typedef _StatisticsType stats_type;
  typedef _EngineType engine_type;
  typedef State<config_type> state_type;

  typedef SamplingFixtureParams<config_type, stats_type>
      sampling_fixture_params_type;
  typedef SamplingFixture<config_type, stats_type, engine_type>
      sampling_fixture_type;

  /// A `run_index` used for status messages and results output
  Index run_index;

  /// Random number generator engine (not null)
  std::shared_ptr<engine_type> engine;

  /// Sampling fixtures
  std::vector<std::shared_ptr<sampling_fixture_type>> sampling_fixtures;

  /// \brief If true, the run is complete if any sampling fixture
  ///     is complete. Otherwise, all sampling fixtures must be
  ///     completed for the run to be completed
  bool global_cutoff;

  /// Next time-based sampling fixture, or nullptr if none
  sampling_fixture_type *next_sampling_fixture;

  /// Next time-based sampling sample time
  double next_sample_time;

  /// Default null action before / after sampling
  struct NullAction {
    void operator()(sampling_fixture_type const &fixture,
                    state_type const &state){
        // do nothing
    };
  };

  typedef std::function<bool(sampling_fixture_type const &, state_type const &)>
      BreakPointCheck;

  /// \brief Break point checks to perform when sampling the fixture with label
  /// matching key
  std::map<std::string, BreakPointCheck> break_point_checks;

  bool break_point_set;

  /// \brief Constructor
  ///
  /// \param _engine Random number generation engine (throw if null)
  /// \param _sampling_fixture_params Sampling fixture parameters
  /// \param _global_cutoff If true, the run is complete if any sampling fixture
  ///     is complete. Otherwise, all sampling fixtures must be
  ///     completed for the run to be completed.
  RunManager(
      std::shared_ptr<engine_type> _engine,
      std::vector<sampling_fixture_params_type> const &_sampling_fixture_params,
      bool _global_cutoff = true)
      : run_index(0),
        engine(_engine),
        global_cutoff(_global_cutoff),
        next_sampling_fixture(nullptr),
        next_sample_time(0.0),
        break_point_set(false) {
    if (!engine) {
      throw std::runtime_error(
          "Error constructing RunManager: engine==nullptr");
    }

    for (auto const &params : _sampling_fixture_params) {
      sampling_fixtures.emplace_back(
          std::make_shared<sampling_fixture_type>(params, engine));
    }
  }

  void initialize(Index steps_per_pass) {
    for (auto &fixture_ptr : sampling_fixtures) {
      fixture_ptr->initialize(steps_per_pass);
    }
    break_point_set = false;
  }

  bool is_break_point() const { return break_point_set; }

  bool is_complete() {
    // do not quit early, so that status
    // files can be printed with the latest completion
    // check results
    bool all_complete = true;
    bool any_complete = false;
    for (auto &fixture_ptr : sampling_fixtures) {
      if (fixture_ptr->is_complete()) {
        any_complete = true;
      } else {
        all_complete = false;
      }
    }
    if (this->global_cutoff && any_complete) {
      return true;
    }
    return all_complete;
  }

  void write_status_if_due() {
    for (auto &fixture_ptr : sampling_fixtures) {
      fixture_ptr->write_status_if_due(run_index);
    }
  }

  void increment_n_accept() {
    for (auto &fixture_ptr : sampling_fixtures) {
      fixture_ptr->increment_n_accept();
    }
  }

  void increment_n_reject() {
    for (auto &fixture_ptr : sampling_fixtures) {
      fixture_ptr->increment_n_reject();
    }
  }

  void increment_step() {
    for (auto &fixture_ptr : sampling_fixtures) {
      fixture_ptr->increment_step();
    }
  }

  void set_time(double event_time) {
    for (auto &fixture_ptr : sampling_fixtures) {
      fixture_ptr->set_time(event_time);
    }
  }

  // Collect event statistics - with configuration in the state before event,
  // but after step and time have been updated in sampling fixtures
  void set_selected_event_data(OccEvent const &selected_event,
                               double time_increment, double event_time,
                               state_type const &state) {
    for (auto &fixture_ptr : sampling_fixtures) {
      fixture_ptr->collect_event_stats(selected_event, time_increment,
                                       event_time, state);
    }
  }

  template <bool DebugMode = false, typename PreSampleActionType = NullAction,
            typename PostSampleActionType = NullAction>
  void sample_data_by_count_if_due(
      state_type const &state,
      PreSampleActionType pre_sample_f = PreSampleActionType(),
      PostSampleActionType post_sample_f = PostSampleActionType()) {
    for (auto &fixture_ptr : sampling_fixtures) {
      auto &fixture = *fixture_ptr;
      if (fixture.params().sampling_params.sample_mode !=
          SAMPLE_MODE::BY_TIME) {
        if (fixture.counter().count == fixture.next_sample_count()) {
          pre_sample_f(fixture, state);
          fixture.template sample_data<DebugMode>(state);
          post_sample_f(fixture, state);
          auto it = break_point_checks.find(fixture.label());
          if (it != break_point_checks.end()) {
            break_point_set = it->second(fixture, state);
          }
        }
      }
    }
  }

  template <bool DebugMode = false, typename PreSampleActionType = NullAction,
            typename PostSampleActionType = NullAction>
  void sample_data_by_time_if_due(
      TimeType event_time, state_type const &state,
      PreSampleActionType pre_sample_f = PreSampleActionType(),
      PostSampleActionType post_sample_f = PostSampleActionType()) {
    // Sample data, if a sample is due by time
    while (this->next_sampling_fixture != nullptr &&
           event_time >= this->next_sample_time) {
      auto &fixture = *this->next_sampling_fixture;

      pre_sample_f(fixture, state);
      fixture.set_time(this->next_sample_time);
      fixture.template sample_data<DebugMode>(state);
      post_sample_f(fixture, state);
      auto it = break_point_checks.find(fixture.label());
      if (it != break_point_checks.end()) {
        break_point_set = it->second(fixture, state);
      }
      this->update_next_sampling_fixture();
    }
  }

  void update_next_sampling_fixture() {
    // update next_sample_time and next_sampling_fixture
    next_sampling_fixture = nullptr;
    for (auto &fixture_ptr : sampling_fixtures) {
      auto &fixture = *fixture_ptr;
      if (fixture.params().sampling_params.sample_mode ==
          SAMPLE_MODE::BY_TIME) {
        if (next_sampling_fixture == nullptr ||
            fixture.next_sample_time() < next_sample_time) {
          next_sample_time = fixture.next_sample_time();
          next_sampling_fixture = &fixture;
        }
      }
    }
  }

  /// \brief Write results for each sampling fixtures and write completed runs
  ///
  /// Notes:
  /// - Calls `finalize` for all sampling fixtures
  void finalize(state_type const &final_state) {
    for (auto &fixture_ptr : sampling_fixtures) {
      fixture_ptr->finalize(final_state, run_index);
    }
  }
};

}  // namespace monte
}  // namespace CASM

#endif
