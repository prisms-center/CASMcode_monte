#ifndef CASM_monte_Definitions
#define CASM_monte_Definitions

#include <map>
#include <memory>
#include <random>
#include <string>

#include "casm/casm_io/json/jsonParser.hh"
#include "casm/global/definitions.hh"
#include "casm/global/eigen.hh"
#include "casm/monte/MTRandEngine.hh"

namespace CASM {
namespace monte {

typedef std::mt19937_64 default_engine_type;

/// How often to sample runs
enum class SAMPLE_MODE { BY_STEP, BY_PASS, BY_TIME };

/// How to sample by time
enum class SAMPLE_METHOD { LINEAR, LOG, CUSTOM };

typedef long CountType;
typedef long long BigCountType;
typedef double TimeType;

struct SamplingParams;
class Sampler;
typedef std::map<std::string, std::shared_ptr<Sampler>> SamplerMap;
struct SamplerComponent;
struct RequestedPrecision;
typedef std::map<SamplerComponent, RequestedPrecision> RequestedPrecisionMap;

template <typename _ConfigType>
struct State;

struct ValueMap;

struct StateSamplingFunction;
typedef std::map<std::string, StateSamplingFunction> StateSamplingFunctionMap;

struct jsonStateSamplingFunction;
typedef std::map<std::string, jsonStateSamplingFunction>
    jsonStateSamplingFunctionMap;

struct BasicStatistics;
template <typename StatisticsType>
using CalcStatisticsFunction = std::function<StatisticsType(
    Eigen::VectorXd const &observations, Eigen::VectorXd const &sample_weight)>;

template <typename StatisticsType>
CalcStatisticsFunction<StatisticsType> default_statistics_calculator();

struct IndividualEquilibrationCheckResult;
struct EquilibrationCheckResults;
typedef std::function<IndividualEquilibrationCheckResult(
    Eigen::VectorXd const &observations, Eigen::VectorXd const &sample_weight,
    RequestedPrecision requested_precision)>
    EquilibrationCheckFunction;

template <typename StatisticsType>
struct IndividualConvergenceCheckResult;
template <typename StatisticsType>
using ConvergenceResultMap =
    std::map<SamplerComponent,
             IndividualConvergenceCheckResult<StatisticsType>>;
typedef std::map<SamplerComponent, IndividualEquilibrationCheckResult>
    EquilibrationResultMap;

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

template <typename PtrType>
PtrType throw_if_null(PtrType ptr, std::string const &what) {
  if (ptr == nullptr) {
    throw std::runtime_error(what);
  }
  return ptr;
}

}  // namespace monte
}  // namespace CASM

#endif
