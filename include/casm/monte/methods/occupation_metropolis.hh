#ifndef CASM_monte_methods_occupation_metropolis
#define CASM_monte_methods_occupation_metropolis

#include <map>
#include <string>
#include <vector>

#include "casm/casm_io/Log.hh"
#include "casm/monte/Conversions.hh"
#include "casm/monte/MethodLog.hh"
#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/events/OccCandidate.hh"
#include "casm/monte/events/OccEventProposal.hh"
#include "casm/monte/events/OccLocation.hh"
#include "casm/monte/events/io/OccCandidate_stream_io.hh"
#include "casm/monte/methods/metropolis.hh"
#include "casm/monte/results/Results.hh"
#include "casm/monte/results/ResultsAnalysisFunction.hh"
#include "casm/monte/sampling/SamplingParams.hh"
#include "casm/monte/state/StateSampler.hh"

// debug
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/checks/io/json/CompletionCheck_json_io.hh"

namespace CASM {
namespace monte {

struct CompletionCheckParams;
class Conversions;
class OccLocation;
class OccSwap;
template <typename ConfigType>
struct Results;
struct SamplingParams;
template <typename ConfigType>
struct State;
template <typename ConfigType>
struct StateSamplingFunction;
template <typename ConfigType>
using StateSamplingFunctionMap =
    std::map<std::string, StateSamplingFunction<ConfigType>>;

template <typename ConfigType, typename CalculatorType,
          typename ProposeOccEventFuntionType, typename GeneratorType>
Results<ConfigType> occupation_metropolis(
    State<ConfigType> &state, OccLocation &occ_location,
    CalculatorType &potential, std::vector<OccSwap> const &possible_swaps,
    ProposeOccEventFuntionType propose_event_f,
    GeneratorType &random_number_generator,
    StateSampler<ConfigType> &state_sampler, CompletionCheck &completion_check,
    ResultsAnalysisFunctionMap<ConfigType> const &analysis_functions,
    MethodLog method_log = MethodLog());

// --- Implementation ---

/// \brief Run an occupation metropolis Monte Carlo calculation
///
/// \param state The state. Consists of both the initial
///     configuration and conditions. Conditions must include `temperature`
///     and any others required by `potential`.
/// \param occ_location An occupant location tracker, which enables efficient
///     event proposal. It must already be initialized with the input state.
/// \param potential A potential calculating method. Should match the interface
///     described below and already be set to calculate the potential for the
///     input state.
/// \param possible_swaps A vector of possible swap types,
///     indicated by the asymmetric unit index and occupant index of the
///     sites potentially being swapped. Typically constructed from
///     `make_canonical_swaps` which generates all possible canonical swaps, or
///     `make_grand_canonical_swaps` which generates all possible grand
///      canonical swaps. It can also be a subset to restrict which swaps are
///     allowed.
/// \param propose_event_f A function, typically one of
///     `propose_canonical_event` or `propose_grand_canonical_event`, which
///     proposes an event (of type `OccEvent`) based on the current occupation,
///     possible_swaps, and random_number_generator.
/// \param random_number_generator A random number generator
/// \param state_sampler A StateSampler<ConfigType>, determines what is sampled
///     and when, and holds sampled data until returned by results
/// \param completion_check A CompletionCheck method
/// \param analysis_functions Functions to evaluate after a run completes
/// \param method_log A MethedLog
///
/// \returns A Results<ConfigType> instance with run results.
///
/// Required interface for `State<ConfigType>`:
/// - `Eigen::VectorXi &get_occupation(State<ConfigType> const &configuration)`
/// - `Eigen::Matrix3l const &get_transformation_matrix_to_super(
///        State<ConfigType> const &state)`
///
/// Required interface for `CalculatorType potential`:
/// - `void set(CalculatorType &potential, State<ConfigType> const &state)`
/// - `double CalculatorType::extensive_value()`
/// - `double CalculatorType::occ_delta_extensive_value(
///        std::vector<Index> const &linear_site_index,
///        std::vector<int> const &new_occ)`
///
/// Required state conditions:
/// - scalar value `temperature`:
///   The temperature in K.
/// - any others required by `potential`
///
/// State properties that are set:
/// - scalar value `potential_energy`:
///   The intensive potential energy (eV / unit cell).
///
template <typename ConfigType, typename CalculatorType,
          typename ProposeOccEventFuntionType, typename GeneratorType>
Results<ConfigType> occupation_metropolis(
    State<ConfigType> &state, OccLocation &occ_location,
    CalculatorType &potential, std::vector<OccSwap> const &possible_swaps,
    ProposeOccEventFuntionType propose_event_f,
    GeneratorType &random_number_generator,
    StateSampler<ConfigType> &state_sampler, CompletionCheck &completion_check,
    ResultsAnalysisFunctionMap<ConfigType> const &analysis_functions,
    MethodLog method_log) {
  if (potential.get() != &state) {
    throw std::runtime_error(
        "Error in monte::occupation_metropolis: potential not set to correct "
        "state");
  }

  State<ConfigType> initial_state = state;

  CountType steps_per_pass = occ_location.mol_size();
  double n_unitcells = get_transformation_matrix_to_super(state).determinant();

  // Prepare properties
  state.properties.scalar_values["potential_energy"] = 0.;
  double &potential_energy_intensive =
      state.properties.scalar_values["potential_energy"];

  // Set formation energy calculator (so it evaluates state)
  // and calculate initial potential energy
  potential_energy_intensive = potential.extensive_value() / n_unitcells;

  // Reset state_sampler
  state_sampler.reset(steps_per_pass);

  // Used within the main loop:
  OccEvent event;
  double beta =
      1.0 / (CASM::KB * state.conditions.scalar_values.at("temperature"));

  // Log method status
  Log &log = method_log.log;
  std::optional<double> &log_frequency = method_log.log_frequency;
  log.restart_clock();
  log.begin_lap();

  // Sample initial state, if requested by sampling_params
  state_sampler.sample_data_if_due(state, log.time_s());

  // Main loop
  while (!completion_check.is_complete(state_sampler.samplers,
                                       state_sampler.count, log.time_s())) {
    // Log method status
    if (log_frequency.has_value() && log.lap_time() > *log_frequency) {
      method_log.reset();
      jsonParser json;
      json["status"] = "incomplete";
      json["time"] = log.time_s();
      to_json(completion_check.results(), json["convergence_check_results"]);
      // for (auto const &pair : state_sampler.samplers) {
      //   json[pair.first] = pair.second->values();
      // }
      log << json << std::endl;
      log.begin_lap();
    }

    // Propose an event
    propose_event_f(event, occ_location, possible_swaps,
                    random_number_generator);

    // Calculate change in potential energy (extensive) due to event
    double delta_potential_energy = potential.occ_delta_extensive_value(
        event.linear_site_index, event.new_occ);

    // Accept or reject event
    bool accept = metropolis_acceptance(delta_potential_energy, beta,
                                        random_number_generator);

    // Apply accepted event
    if (accept) {
      occ_location.apply(event, get_occupation(state));
      potential_energy_intensive += (delta_potential_energy / n_unitcells);
    }

    // Increment count
    state_sampler.increment_step();

    // Sample data, if a sample is due
    state_sampler.sample_data_if_due(state, log.time_s());
  }

  // Log method status
  if (log_frequency.has_value()) {
    method_log.reset();
    jsonParser json;
    json["status"] = "complete";
    json["time"] = log.time_s();
    to_json(completion_check.results(), json["convergence_check_results"]);
    log << json << std::endl;
  }

  Results<ConfigType> results;
  results.initial_state = initial_state;
  results.final_state = state;
  results.elapsed_clocktime = log.time_s();
  results.samplers = std::move(state_sampler.samplers);
  results.sample_count = std::move(state_sampler.sample_count);
  results.sample_clocktime = std::move(state_sampler.sample_clocktime);
  results.sample_trajectory = std::move(state_sampler.sample_trajectory);
  results.completion_check_results = completion_check.results();
  results.analysis = make_analysis(results, analysis_functions);
  return results;
}

}  // namespace monte
}  // namespace CASM

#endif
