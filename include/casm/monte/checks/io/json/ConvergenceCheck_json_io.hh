#ifndef CASM_monte_checks_ConvergenceCheck_json_io
#define CASM_monte_checks_ConvergenceCheck_json_io

namespace CASM {

class jsonParser;

namespace monte {
struct IndividualConvergenceCheckResult;
struct ConvergenceCheckResults;

/// \brief IndividualConvergenceCheckResult to JSON
jsonParser &to_json(IndividualConvergenceCheckResult const &value,
                    jsonParser &json);

/// \brief ConvergenceCheckResults to JSON
jsonParser &to_json(ConvergenceCheckResults const &value, jsonParser &json);

}  // namespace monte
}  // namespace CASM

#endif
