#ifndef CASM_monte_sampling_SamplingParams_json_io
#define CASM_monte_sampling_SamplingParams_json_io

namespace CASM {

template <typename T>
class InputParser;

namespace monte {
struct SamplingParams;

/// \brief Construct SamplingParams from JSON
void parse(InputParser<SamplingParams> &parser);

}  // namespace monte
}  // namespace CASM

#endif
