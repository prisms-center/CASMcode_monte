#ifndef CASM_monte_sampling_SamplingParams_json_io
#define CASM_monte_sampling_SamplingParams_json_io

#include <set>
#include <string>

namespace CASM {

template <typename T>
class InputParser;

namespace monte {
struct SamplingParams;

/// \brief Construct SamplingParams from JSON
void parse(InputParser<SamplingParams> &parser,
           std::set<std::string> const &sampling_function_names,
           bool time_sampling_allowed);

}  // namespace monte
}  // namespace CASM

#endif
