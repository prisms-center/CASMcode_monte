#ifndef CASM_monte_RunManager
#define CASM_monte_RunManager

#include "casm/monte/run_management/RunData.hh"
#include "casm/monte/run_management/SamplingFixture.hh"

// io
#include "casm/casm_io/SafeOfstream.hh"
#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/run_management/io/json/RunData_json_io.hh"

namespace CASM {
namespace monte {

struct RunManagerParams {
  /// \brief Save all initial_state in completed_runs
  bool do_save_all_initial_states = false;

  /// \brief Save all final_state in completed_runs
  bool do_save_all_final_states = false;

  /// \brief Save last final_state in completed_runs
  bool do_save_last_final_state = true;

  /// \brief Write saved initial_state to completed_runs.json
  bool do_write_initial_states = false;

  /// \brief Write saved final_state to completed_runs.json
  bool do_write_final_states = false;

  /// \brief Location to save completed_runs.json if not empty
  fs::path output_dir;

  /// \brief If true, the run is complete if any sampling fixture
  ///     is complete. Otherwise, all sampling fixtures must be
  ///     completed for the run to be completed
  bool global_cutoff = true;
};

/// \brief Holds sampling fixtures and checks for completion
///
/// Notes:
/// - Currently, all sampling fixtures keep sampling, even if
///   completion criteria are completed, until all are completed.
///   Reading final states, and using as input to state
///   generator is more complicated otherwise.
///
template <typename _ConfigType, typename _StatisticsType, typename _EngineType>
struct RunManager : public RunManagerParams {
  typedef _ConfigType config_type;
  typedef _StatisticsType stats_type;
  typedef _EngineType engine_type;
  typedef State<config_type> state_type;

  typedef SamplingFixtureParams<config_type, stats_type>
      sampling_fixture_params_type;
  typedef SamplingFixture<config_type, stats_type, engine_type>
      sampling_fixture_type;

  /// Random number generator engine
  std::shared_ptr<engine_type> engine;

  /// Sampling fixtures
  std::vector<sampling_fixture_type> sampling_fixtures;

  /// Current run data
  RunData<config_type> current_run;

  /// Completed runs
  std::vector<RunData<config_type>> completed_runs;

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

  RunManager(
      RunManagerParams const &run_manager_params,
      std::shared_ptr<engine_type> _engine,
      std::vector<sampling_fixture_params_type> const &sampling_fixture_params)
      : RunManagerParams(run_manager_params),
        engine(_engine),
        next_sampling_fixture(nullptr),
        next_sample_time(0.0),
        break_point_set(false) {
    for (auto const &params : sampling_fixture_params) {
      sampling_fixtures.emplace_back(params, engine);
    }
  }

  /// \brief Return completed runs from file
  ///
  /// Notes:
  /// - Reads from output_dir / "completed_runs.json", if exists
  void read_completed_runs() {
    completed_runs.clear();

    if (!this->output_dir.empty()) {
      fs::path completed_runs_path = this->output_dir / "completed_runs.json";
      if (!fs::exists(completed_runs_path)) {
        return;
      }
      jsonParser json(completed_runs_path);
      completed_runs = json.get<std::vector<RunData<config_type>>>();
    }
    return;
  }

  void initialize(state_type const &state, Index steps_per_pass) {
    current_run = RunData<config_type>();
    current_run.conditions = state.conditions;
    current_run.transformation_matrix_to_super =
        get_transformation_matrix_to_super(state);
    current_run.n_unitcells =
        current_run.transformation_matrix_to_super.determinant();
    if (this->do_save_all_initial_states) {
      current_run.initial_state = state;
    }

    for (auto &fixture : sampling_fixtures) {
      fixture.initialize(state, steps_per_pass);
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
    for (auto &fixture : sampling_fixtures) {
      if (fixture.is_complete()) {
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
    for (auto &fixture : sampling_fixtures) {
      fixture.write_status_if_due(completed_runs.size());
    }
  }

  void increment_n_accept() {
    for (auto &fixture : sampling_fixtures) {
      fixture.increment_n_accept();
    }
  }

  void increment_n_reject() {
    for (auto &fixture : sampling_fixtures) {
      fixture.increment_n_reject();
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

  template <typename PreSampleActionType = NullAction,
            typename PostSampleActionType = NullAction>
  void sample_data_by_count_if_due(
      state_type const &state,
      PreSampleActionType pre_sample_f = PreSampleActionType(),
      PostSampleActionType post_sample_f = PostSampleActionType()) {
    for (auto &fixture : sampling_fixtures) {
      auto const &ss = fixture.state_sampler();
      if (ss.sample_mode != SAMPLE_MODE::BY_TIME) {
        if (ss.count == ss.next_sample_count) {
          pre_sample_f(fixture, state);
          fixture.sample_data(state);
          post_sample_f(fixture, state);
          auto it = break_point_checks.find(fixture.label());
          if (it != break_point_checks.end()) {
            break_point_set = it->second(fixture, state);
          }
        }
      }
    }
  }

  template <typename PreSampleActionType = NullAction,
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
      fixture.sample_data(state);
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

  /// \brief Write results for each sampling fixtures and write completed runs
  ///
  /// Notes:
  /// - Writes completed runs to  `output_dir / "completed_runs.json"`
  /// - Calls `finalize` for all sampling fixtures
  void finalize(state_type const &final_state) {
    if (this->do_save_last_final_state || this->do_save_all_final_states) {
      current_run.final_state = final_state;
    }
    if (completed_runs.size() && this->do_save_last_final_state &&
        !this->do_save_all_final_states) {
      completed_runs.back().final_state.reset();
    }
    completed_runs.push_back(current_run);

    for (auto &fixture : sampling_fixtures) {
      fixture.finalize(final_state, completed_runs.size(), current_run);
    }

    // write completed_runs file
    if (!this->output_dir.empty()) {
      fs::path completed_runs_path = this->output_dir / "completed_runs.json";
      fs::create_directories(this->output_dir);
      SafeOfstream file;
      file.open(completed_runs_path);
      jsonParser json;
      to_json(completed_runs, json, this->do_write_initial_states,
              this->do_write_final_states);
      json.print(file.ofstream(), -1);
      file.close();
    }
  }
};

}  // namespace monte
}  // namespace CASM

#endif
