#ifndef CASM_monte_checks_CutoffCheck_json_io
#define CASM_monte_checks_CutoffCheck_json_io

namespace CASM {

template <typename T>
class InputParser;
class jsonParser;

namespace monte {
struct CutoffCheckParams;

/// \brief Convert CutoffCheckParams to JSON
jsonParser &to_json(CutoffCheckParams const &params, jsonParser &json);

/// \brief Construct CutoffCheckParams from JSON
void parse(InputParser<CutoffCheckParams> &parser);

}  // namespace monte
}  // namespace CASM

#endif
