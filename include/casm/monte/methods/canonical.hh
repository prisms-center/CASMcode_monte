#ifndef CASM_monte_methods_canonical
#define CASM_monte_methods_canonical

#include <map>
#include <string>
#include <vector>

#include "casm/casm_io/Log.hh"
#include "casm/monte/Conversions.hh"
#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/events/OccCandidate.hh"
#include "casm/monte/events/OccEventProposal.hh"
#include "casm/monte/events/OccLocation.hh"
#include "casm/monte/events/io/OccCandidate_stream_io.hh"
#include "casm/monte/methods/metropolis.hh"
#include "casm/monte/results/Results.hh"
#include "casm/monte/sampling/SamplingParams.hh"
#include "casm/monte/state/StateSampler.hh"

namespace CASM {
namespace monte {

struct CompletionCheckParams;
class Conversions;
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

namespace canonical_impl {

// template <typename ConfigType, typename CalculatorType>
// void throw_if_invalid(
//     State<ConfigType> const &initial_state,
//     CalculatorType potential_energy_calculator, Conversions const &convert,
//     std::vector<OccSwap> const &canonical_swaps,
//     MTRand &random_number_generator, SamplingParams const &sampling_params,
//     StateSamplingFunctionMap<ConfigType> const &sampling_functions,
//     CompletionCheck &completion_check);

}

template <typename ConfigType, typename CalculatorType>
Results<ConfigType> canonical(State<ConfigType> const &initial_state,
                              CalculatorType potential_energy_calculator,
                              Conversions const &convert,
                              std::vector<OccSwap> const &canonical_swaps,
                              MTRand &random_number_generator,
                              StateSampler<ConfigType> &state_sampler,
                              CompletionCheck &completion_check);

// --- Definitions ---

namespace canonical_impl {

// template <typename ConfigType, typename CalculatorType>
// void throw_if_invalid(
//     State<ConfigType> const &initial_state,
//     CalculatorType potential_energy_calculator, Conversions const &convert,
//     std::vector<OccSwap> const &canonical_swaps,
//     MTRand &random_number_generator, SamplingParams const &sampling_params,
//     StateSamplingFunctionMap<ConfigType> const &sampling_functions,
//     CompletionCheckParams const &completion_check_params) {
//   // validate initial_state.configuration against convert
//   {
//     Eigen::VectorXi const &occupation =
//         get_occupation(initial_state.configuration);
//     if (occupation.size() != convert.l_size()) {
//       throw std::runtime_error(
//           "Error in monte::canonical: occupation vector size error.");
//     }
//     for (Index l = 0; l < occupation.size(); ++l) {
//       Index asym = convert.l_to_asym(l);
//       if (occupation(l) < 0 || occupation(l) >= convert.occ_size(asym)) {
//         throw std::runtime_error(
//             "Error in monte::canonical: occupation vector is not valid. "
//             "Initial occupation values are outside the expected range.");
//       }
//     }
//   }
//   // validate initial_state.conditions
//   {
//     auto const &conditions = initial_state.conditions;
//     if (!conditions.count("temperature")) {
//       throw std::runtime_error(
//           "Error in monte::canonical: \"temperature\" is a required
//           condition");
//     }
//     if (conditions.at("temperature").size() != 1) {
//       throw std::runtime_error(
//           "Error in monte::canonical: \"temperature\"  size != 1");
//     }
//   }
//   // validate canonical_swaps against convert
//   {
//     Index i = 0;
//     for (OccSwap const &swap : canonical_swaps) {
//       typedef std::pair<OccSwap const &, Conversions const &> swap_pair;
//       if (!is_valid(convert, swap)) {
//         CASM::err_log() << "Error in monte::canonical: Invalid swap (index="
//                         << i << "):\n"
//                         << swap_pair(swap, convert) << std::endl;
//         throw std::runtime_error("Error in monte::canonical: invalid swap");
//       }
//       if (!is_valid(convert, swap)) {
//         CASM::err_log()
//             << "Error in monte::canonical: Non-canonical swap (index=" << i
//             << "):\n"
//             << swap_pair(swap, convert) << std::endl;
//         throw std::runtime_error(
//             "Error in monte::canonical: non-canonical swap");
//       }
//       ++i;
//     }
//   }
//   // validate sampling_params.sampler_names against sampling_functions
//   {
//     for (std::string quantity : sampling_params.sampler_names) {
//       if (sampling_functions.find(quantity) == sampling_functions.end()) {
//         CASM::err_log()
//             << "Error in monte::canonical: sampling_params.sampler_names "
//                "contains the quantity \""
//             << quantity
//             << "\", but there is no function by that name in "
//                "sampling_functions."
//             << std::endl;
//
//         throw std::runtime_error(
//             "Error in monte::canonical: sampling_params.sampler_names is not
//             " "consistent with sampling_functions");
//       }
//     }
//   }
// }

}  // namespace canonical_impl

/// \brief Run a canonical Monte Carlo calculation
///
/// \param initial_state The initial state
/// \param potential_energy_calculator A potential energy calculating method.
///     Should match the interface example given by `CalculatorTemplate`.
/// \param convert A monte::Conversions instance, provides necessary index
///     conversions. Must be consistent with `initial_state` and
///     `canonical_swaps`.
/// \param random_number_generator A random number generator
/// \param state_sampler A StateSampler<ConfigType>, determines what is sampled
///     and when, and holds sampled data until returned by results
/// \param completion_check A CompletionCheck method
///
/// Requires:
/// - `Eigen::VectorXi &get_occupation(ConfigType const &configuration)`
/// - `Eigen::Matrix3l const &get_transformation_matrix_to_super(
///        ConfigType const &configuration)`
/// - `void set(CalculatorType &calculator,
///             State<ConfigType> const &state)`
template <typename ConfigType, typename CalculatorType>
Results<ConfigType> canonical(State<ConfigType> const &initial_state,
                              CalculatorType potential_energy_calculator,
                              Conversions const &convert,
                              std::vector<OccSwap> const &canonical_swaps,
                              MTRand &random_number_generator,
                              StateSampler<ConfigType> &state_sampler,
                              CompletionCheck &completion_check) {
  // Prepare state
  State<ConfigType> state = initial_state;

  // Initialize occupation tracking
  // - The OccCandidateList holds a list of OccCandidate, which are
  //   pairs of (asymmetric unit index, species index) indicating symmetrically
  //   distinct types of occupants.
  // - The OccLocation object holds information on where all the occupants are,
  //   organized by OccCandidate type. This allows for efficient event proposal
  //   even in dilute compositions.
  // - The OccLocation::mol_size() is the number of "mol" (possibly molecular
  //   occupants) that may mutate. Use this for `steps_per_pass`.
  OccCandidateList occ_candidate_list(convert);
  OccLocation occ_location(convert, occ_candidate_list);
  occ_location.initialize(get_occupation(state.configuration));
  CountType steps_per_pass = occ_location.mol_size();

  // Prepare properties
  state.properties["potential_energy"] = Eigen::VectorXd(1);
  double &potential_energy_intensive = state.properties["potential_energy"](0);

  // Set calculator (so it evaluates state)
  // and calculate initial potential energy
  set(potential_energy_calculator, state);
  potential_energy_intensive = potential_energy_calculator.intensive_value();

  // Reset state_sampler
  state_sampler.reset(steps_per_pass);

  // Sample initial state, if requested by sampling_params
  state_sampler.sample_data_if_due(state);

  // Used within the main loop:
  OccEvent event;
  double beta = 1.0 / (CASM::KB * state.conditions.at("temperature")(0));
  double n_unitcells =
      get_transformation_matrix_to_super(state.configuration).determinant();

  // Main loop
  while (!completion_check.is_complete(state_sampler.samplers,
                                       state_sampler.count)) {
    // Propose an event
    propose_canonical_event(event, occ_location, canonical_swaps,
                            random_number_generator);

    // Calculate change in potential energy (extensive) due to event
    double delta_potential_energy = potential_energy_calculator.occ_delta_value(
        event.linear_site_index, event.new_occ);

    // Accept or reject event
    bool accept = metropolis_acceptance(delta_potential_energy, beta,
                                        random_number_generator);

    // Apply accepted event
    if (accept) {
      occ_location.apply(event, get_occupation(state.configuration));
      potential_energy_intensive += (delta_potential_energy / n_unitcells);
    }

    // Increment count
    state_sampler.increment_step();

    // Sample data, if a sample is due
    state_sampler.sample_data_if_due(state);
  }

  Results<ConfigType> results;
  results.initial_state = initial_state;
  results.final_state = state;
  results.samplers = std::move(state_sampler.samplers);
  results.sample_count = std::move(state_sampler.sample_count);
  results.sample_trajectory = std::move(state_sampler.sample_trajectory);
  results.completion_check_results = completion_check.results();
  return results;
}

}  // namespace monte
}  // namespace CASM

#endif
