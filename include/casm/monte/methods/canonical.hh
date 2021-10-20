#ifndef CASM_monte_methods_canonical
#define CASM_monte_methods_canonical

#include <map>
#include <string>
#include <vector>
class MTRand;

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

template <typename ConfigType, typename CalculatorType>
Results<ConfigType> canonical(
    State<ConfigType> const &initial_state,
    CalculatorType const &formation_energy_calculator,
    Conversions const &convert, OccCandidateList const &occ_candidate_list,
    std::vector<OccSwap> const &canonical_swaps, MTRand &mtrand,
    SamplingParams const &sampling_params,
    StateSamplingFunctionMap<ConfigType> const &sampling_functions,
    CompletionCheckParams const &completion_check_params);

}  // namespace monte
}  // namespace CASM

#endif
