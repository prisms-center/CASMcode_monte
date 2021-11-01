#ifndef CASM_monte_checks_CutoffCheck_json_io
#define CASM_monte_checks_CutoffCheck_json_io

namespace CASM {

template <typename T>
class InputParser;

namespace monte {
struct CutoffCheckParams;

/// \brief Construct CutoffCheckParams from JSON
void parse(InputParser<CutoffCheckParams> &parser);

}  // namespace monte
}  // namespace CASM

#endif
