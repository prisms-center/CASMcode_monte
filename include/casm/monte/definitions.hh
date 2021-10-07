#ifndef CASM_monte_Definitions
#define CASM_monte_Definitions

#include <map>
#include <string>

#include "casm/global/definitions.hh"
#include "casm/global/eigen.hh"

namespace CASM {
namespace monte {

typedef long CountType;
typedef double TimeType;

/// Map of value name to vector value
typedef std::map<std::string, Eigen::VectorXd> VectorValueMap;

}  // namespace monte
}  // namespace CASM

#endif
