#ifndef CASM_monte_sampling_SamplingParams_json_io
#define CASM_monte_sampling_SamplingParams_json_io

#include <set>
#include <string>

namespace CASM {

template <typename T>
class InputParser;
class jsonParser;

namespace monte {
struct SamplingParams;

/// \brief Construct SamplingParams from JSON
void parse(InputParser<SamplingParams> &parser,
           std::set<std::string> const &sampling_function_names,
           std::set<std::string> const &json_sampling_function_names,
           bool time_sampling_allowed);

/// \brief Convert SamplingParams to JSON
jsonParser &to_json(SamplingParams const &sampling_params, jsonParser &json);

}  // namespace monte
}  // namespace CASM

#endif
