#ifndef CASM_monte_checks_EquilibrationCheck_json_io
#define CASM_monte_checks_EquilibrationCheck_json_io

namespace CASM {

class jsonParser;

namespace monte {
struct IndividualEquilibrationCheckResult;
struct EquilibrationCheckResults;

/// \brief IndividualEquilibrationCheckResult to JSON
jsonParser &to_json(IndividualEquilibrationCheckResult const &value,
                    jsonParser &json);

/// \brief EquilibrationCheckResults to JSON
jsonParser &to_json(EquilibrationCheckResults const &value, jsonParser &json);

}  // namespace monte
}  // namespace CASM

#endif
