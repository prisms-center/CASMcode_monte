#ifndef CASM_monte_methods_occupation_metropolis
#define CASM_monte_methods_occupation_metropolis

#include <map>
#include <string>
#include <vector>

#include "casm/monte/Conversions.hh"
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

namespace CASM {
namespace monte {

template <typename ConfigType, typename CalculatorType,
          typename ProposeOccEventFuntionType, typename GeneratorType>
void occupation_metropolis(State<ConfigType> &state, OccLocation &occ_location,
                           CalculatorType &potential,
                           std::vector<OccSwap> const &possible_swaps,
                           ProposeOccEventFuntionType propose_event_f,
                           GeneratorType &random_number_generator,
                           RunManager<ConfigType> &run_manager);

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
/// \param run_manager Contains sampling fixtures and after completion holds
///     final results
///
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
void occupation_metropolis(State<ConfigType> &state, OccLocation &occ_location,
                           CalculatorType &potential,
                           std::vector<OccSwap> const &possible_swaps,
                           ProposeOccEventFuntionType propose_event_f,
                           GeneratorType &random_number_generator,
                           RunManager<ConfigType> &run_manager) {
  // --- Track potential energy in state properties ---
  if (potential.state() != &state) {
    throw std::runtime_error(
        "Error in monte::occupation_metropolis: potential not set to correct "
        "state");
  }
  double n_unitcells =
      get_transformation_matrix_to_super(state).determinant();
  state.properties.scalar_values["potential_energy"] = 0.;
  double &potential_energy_intensive =
      state.properties.scalar_values["potential_energy"];
  potential_energy_intensive = potential.extensive_value() / n_unitcells;

  // Used within the main loop:
  OccEvent event;
  double beta =
      1.0 / (CASM::KB * state.conditions.scalar_values.at("temperature"));

  // Main loop
  run_manager.initialize(state, occ_location.mol_size());
  run_manager.sample_data_by_count_if_due(state);
  while (!run_manager.is_complete()) {
    run_manager.write_status_if_due();

    // Propose an event
    try {
      propose_event_f(event, occ_location, possible_swaps,
                      random_number_generator);
    } catch (std::exception &e) {
      CASM::err_log() << std::endl
                      << "Error proposing event: " << e.what() << std::endl;
      break;
    }

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
    run_manager.increment_step();

    // Sample data, if a sample is due by count
    run_manager.sample_data_by_count_if_due(state);
  }

  run_manager.finalize(state);
}

}  // namespace monte
}  // namespace CASM

#endif
