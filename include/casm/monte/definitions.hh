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

template <typename _ConfigType, typename _RunInfoType>
class ConfigGenerator;
template <typename _ConfigType>
class FixedConfigGenerator;

struct SamplingParams;
class Sampler;
struct SamplerComponent;
struct RequestedPrecision;

template <typename _ConfigType>
struct State;

struct ValueMap;

template <typename _ConfigType>
struct RunData;

template <typename _ConfigType, typename _RunInfoType>
class StateGenerator;
template <typename _ConfigType>
class IncrementalConditionsStateGenerator;
template <typename ConfigType>
struct StateModifyingFunction;
template <typename _ConfigType>
using StateModifyingFunctionMap =
    std::map<std::string, StateModifyingFunction<_ConfigType>>;

template <typename _ConfigType, typename _EngineType>
struct StateSampler;

struct StateSamplingFunction;
typedef std::map<std::string, StateSamplingFunction> StateSamplingFunctionMap;

struct BasicStatistics;
template <typename StatisticsType>
using CalcStatisticsFunction = std::function<StatisticsType(
    Eigen::VectorXd const &observations, Eigen::VectorXd const &sample_weight)>;

struct IndividualEquilibrationCheckResult;
struct EquilibrationCheckResults;
typedef std::function<IndividualEquilibrationCheckResult(
    Eigen::VectorXd const &observations, Eigen::VectorXd const &sample_weight,
    RequestedPrecision requested_precision)>
    EquilibrationCheckFunction;

template <typename _ConfigType, typename _StatisticsType>
struct Results;

template <typename _StatisticsType>
struct CompletionCheckParams;
template <typename _StatisticsType>
class CompletionCheck;

template <typename _ResultsType>
class ResultsIO;
template <typename _ResultsType>
class jsonResultsIO;

template <typename _ConfigType, typename _StatisticsType>
struct ResultsAnalysisFunction;

template <typename ConfigType, typename StatisticsType>
using ResultsAnalysisFunctionMap =
    std::map<std::string, ResultsAnalysisFunction<ConfigType, StatisticsType>>;

class Conversions;

struct OccCandidate;
class OccCandidateList;
struct OccEvent;
class OccLocation;
class OccSwap;

template <typename ConfigType, typename StatisticsType>
struct SamplingFixtureParams;
template <typename ConfigType, typename StatisticsType, typename EngineType>
class SamplingFixture;
template <typename ConfigType, typename StatisticsType, typename EngineType>
struct RunManager;

}  // namespace monte
}  // namespace CASM

#endif
