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

struct CompletionCheckParams;
struct SamplingParams;

template <typename _ConfigType, typename _RunInfoType>
class ConfigGenerator;
template <typename _ConfigType>
class FixedConfigGenerator;

template <typename _ConfigType>
struct Results;

template <typename _ConfigType>
class ResultsIO;

template <typename _ConfigType>
struct State;

template <typename _ConfigType, typename _RunInfoType>
class StateGenerator;
template <typename _ConfigType>
class IncrementalConditionsStateGenerator;

template <typename _ConfigType>
class StateSampler;

template <typename _ConfigType>
struct StateSamplingFunction;

template <typename _ConfigType>
using StateSamplingFunctionMap =
    std::map<std::string, StateSamplingFunction<_ConfigType>>;

}  // namespace monte
}  // namespace CASM

#endif
