#ifndef CASM_monte_Definitions
#define CASM_monte_Definitions

#include <map>
#include <string>

#include "casm/global/definitions.hh"
#include "casm/global/eigen.hh"

namespace CASM {
namespace monte {

/// How often to sample runs
enum class SAMPLE_MODE { BY_STEP, BY_PASS, BY_TIME };

/// How to sample by time
enum class SAMPLE_METHOD { LINEAR, LOG };

typedef long CountType;
typedef double TimeType;

struct CompletionCheckParams;
class CompletionCheck;

template <typename _ConfigType, typename _RunInfoType>
class ConfigGenerator;
template <typename _ConfigType>
class FixedConfigGenerator;

template <typename _ConfigType>
struct Results;

template <typename _ConfigType>
class ResultsIO;
template <typename _ConfigType>
class jsonResultsIO;

struct SamplingParams;
template <typename _ConfigType>
struct State;

struct ValueMap;

template <typename _ConfigType, typename _RunInfoType>
class StateGenerator;
template <typename _ConfigType>
class IncrementalConditionsStateGenerator;

template <typename _ConfigType>
struct StateSampler;

template <typename _ConfigType>
struct StateSamplingFunction;

template <typename _ConfigType>
using StateSamplingFunctionMap =
    std::map<std::string, StateSamplingFunction<_ConfigType>>;

template <typename _ConfigType>
struct ResultsAnalysisFunction;

template <typename ConfigType>
using ResultsAnalysisFunctionMap =
    std::map<std::string, ResultsAnalysisFunction<ConfigType>>;

}  // namespace monte
}  // namespace CASM

#endif
