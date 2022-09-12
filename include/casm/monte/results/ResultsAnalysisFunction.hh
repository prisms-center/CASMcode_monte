#ifndef CASM_monte_ResultsAnalysisFunction
#define CASM_monte_ResultsAnalysisFunction

#include <vector>

#include "casm/monte/definitions.hh"
#include "casm/monte/misc/math.hh"
#include "casm/monte/results/Results.hh"

namespace CASM {
namespace monte {

template <typename ConfigType>
struct Results;

/// \brief Use to calculate functions of the sampled data at the
///     end of a run (ex. covariance)
template <typename ConfigType>
struct ResultsAnalysisFunction {
  /// \brief Constructor - default component names
  ResultsAnalysisFunction(
      std::string _name, std::string _description, std::vector<Index> _shape,
      std::function<Eigen::VectorXd(Results<ConfigType> const &)> _function);

  /// \brief Constructor - custom component names
  ResultsAnalysisFunction(
      std::string _name, std::string _description,
      std::vector<std::string> const &_component_names,
      std::vector<Index> _shape,
      std::function<Eigen::VectorXd(Results<ConfigType> const &)> _function);

  /// \brief Function name
  std::string name;

  /// \brief Description of the function
  std::string description;

  /// \brief Shape of resulting value, with column-major unrolling
  ///
  /// Scalar: [], Vector: [n], Matrix: [m, n], etc.
  std::vector<Index> shape;

  /// \brief A name for each component of the resulting Eigen::VectorXd
  ///
  /// Can be string representing an index (i.e "0", "1", "2", etc.) or can
  /// be a descriptive string (i.e. "susc(Ni,Ni)", "susc(Ni,Al)", etc.)
  std::vector<std::string> component_names;

  /// \brief The function to be evaluated
  std::function<Eigen::VectorXd(Results<ConfigType> const &)> function;

  /// \brief Evaluates `function`
  Eigen::VectorXd operator()(Results<ConfigType> const &results) const;
};

/// \brief Make variance for all components of a sampled quantity
template <typename ConfigType>
ResultsAnalysisFunction<ConfigType> make_variance_f(
    StateSamplingFunction<ConfigType> sampling_function,
    double normalization_constant = 1.0,
    std::optional<std::string> name = std::nullopt,
    std::optional<std::string> description = std::nullopt);

/// \brief Make covariance matrix
template <typename ConfigType>
ResultsAnalysisFunction<ConfigType> make_covariance_f(
    StateSamplingFunction<ConfigType> first,
    std::optional<StateSamplingFunction<ConfigType>> second = std::nullopt,
    double normalization_constant = 1.0,
    std::optional<std::string> name = std::nullopt,
    std::optional<std::string> description = std::nullopt);

/// \brief Make covariance of two components of sampled quantities
template <typename ConfigType>
ResultsAnalysisFunction<ConfigType> make_component_covariance_f(
    SamplerComponent first, SamplerComponent second,
    double normalization_constant = 1.0,
    std::optional<std::string> name = std::nullopt,
    std::optional<std::string> description = std::nullopt);

/// \brief Evaluate all analysis functions
template <typename ConfigType>
std::map<std::string, Eigen::VectorXd> make_analysis(
    Results<ConfigType> const &results,
    ResultsAnalysisFunctionMap<ConfigType> const &analysis_functions);

// --- Implementation ---

/// \brief Constructor - default component names
template <typename ConfigType>
ResultsAnalysisFunction<ConfigType>::ResultsAnalysisFunction(
    std::string _name, std::string _description, std::vector<Index> _shape,
    std::function<Eigen::VectorXd(Results<ConfigType> const &)> _function)
    : name(_name),
      description(_description),
      shape(_shape),
      component_names(default_component_names(shape)),
      function(_function) {}

/// \brief Constructor - custom component names
template <typename ConfigType>
ResultsAnalysisFunction<ConfigType>::ResultsAnalysisFunction(
    std::string _name, std::string _description,
    std::vector<std::string> const &_component_names, std::vector<Index> _shape,
    std::function<Eigen::VectorXd(Results<ConfigType> const &)> _function)
    : name(_name),
      description(_description),
      shape(_shape),
      component_names(_component_names),
      function(_function) {}

/// \brief Evaluates `function`
template <typename ConfigType>
Eigen::VectorXd ResultsAnalysisFunction<ConfigType>::operator()(
    Results<ConfigType> const &results) const {
  return function(results);
}

/// \brief Make variance for all components of a sampled quantity
template <typename ConfigType>
ResultsAnalysisFunction<ConfigType> make_variance_f(
    StateSamplingFunction<ConfigType> sampling_function,
    double normalization_constant, std::optional<std::string> name,
    std::optional<std::string> description) {
  if (!name.has_value()) {
    std::stringstream ss;
    ss << "var(" << sampling_function.name << ")";
    name = ss.str();
  }
  if (!description.has_value()) {
    std::stringstream ss;
    ss << "Variance of " << sampling_function.name;
    description = ss.str();
  }
  std::vector<std::string> var_component_names;
  Index n = sampling_function.component_names.size();
  for (std::string name : sampling_function.component_names) {
    var_component_names.push_back(name);
  }
  return ResultsAnalysisFunction(
      name, description, var_component_names, {n},  // shape
      [=](Results<ConfigType> const &results) {
        auto it = find_or_throw(results.samplers, sampling_function.name);

        Eigen::VectorXd var = Eigen::MatrixXd::Zero(n);
        for (Index i = 0; i < n; ++i) {
          Eigen::VectorXd x = it->second->component(i);
          var(i) = variance(x) / normalization_constant;
        }

        return var;
      });
}

/// \brief Make covariance matrix
template <typename ConfigType>
ResultsAnalysisFunction<ConfigType> make_covariance_f(
    StateSamplingFunction<ConfigType> first,
    std::optional<StateSamplingFunction<ConfigType>> second,
    double normalization_constant, std::optional<std::string> name,
    std::optional<std::string> description) {
  if (!name.has_value()) {
    std::stringstream ss;
    ss << "cov(" << first.name;
    if (second.has_value()) {
      ss << "," << second->name << ")";
    }
    ss << ")";
    name = ss.str();
  }
  if (!description.has_value()) {
    std::stringstream ss;
    ss << "Covariance matrix for " << first.name;
    if (second.has_value()) {
      ss << " (rows) and " << second->name << " (columns)";
    }
    description = ss.str();
  }
  if (!second.has_value()) {
    second = first;
  }
  std::vector<std::string> cov_matrix_component_names;
  Index m = first.component_names.size();
  Index n = second->component_names.size();
  for (std::string col_name : second->component_names) {
    for (std::string row_name : first.component_names) {
      cov_matrix_component_names.push_back(row_name + "," + col_name);
    }
  }
  return ResultsAnalysisFunction(
      name, description, cov_matrix_component_names, {m, n},  // shape
      [=](Results<ConfigType> const &results) {
        auto first_it = find_or_throw(results.samplers, first.name);
        auto second_it = find_or_throw(results.samplers, second->name);

        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(m, n);
        for (Index i = 0; i < m; ++i) {
          for (Index j = 0; j < n; ++j) {
            Eigen::VectorXd x = first_it->second->component(i);
            Eigen::VectorXd y = second_it->second->component(j);
            cov(i, j) = covariance(x, y) / normalization_constant;
          }
        }
        return reshaped(cov);
      });
}

/// \brief Make covariance of two components of sampled quantities
template <typename ConfigType>
ResultsAnalysisFunction<ConfigType> make_component_covariance_f(
    SamplerComponent first, SamplerComponent second,
    double normalization_constant, std::optional<std::string> name,
    std::optional<std::string> description) {
  if (!name.has_value()) {
    std::stringstream ss;
    ss << "cov(" << first.sampler_name << "(" << first.component_name << "),"
       << second.sampler_name << "(" << second.component_name << "))";
    name = ss.str();
  }
  if (!description.has_value()) {
    std::stringstream ss;
    ss << "Covariance of " << first.sampler_name << "(" << first.component_name
       << ") and " << second.sampler_name << "(" << second.component_name
       << ")";
    description = ss.str();
  }
  return ResultsAnalysisFunction(
      *name, *description, {},  // shape: empty vector for scalar
      [=](Results<ConfigType> const &results) {
        auto it_first = find_or_throw(results.samplers, first);
        Eigen::VectorXd x = it_first->second->component(first.component_index);

        auto it_second = find_or_throw(results.samplers, second);
        Eigen::VectorXd y =
            it_second->second->component(second.component_index);

        return covariance(x, y) / normalization_constant;
      });
}

/// \brief Evaluate all analysis functions
template <typename ConfigType>
std::map<std::string, Eigen::VectorXd> make_analysis(
    Results<ConfigType> const &results,
    ResultsAnalysisFunctionMap<ConfigType> const &analysis_functions) {
  std::map<std::string, Eigen::VectorXd> analysis;
  for (auto const &pair : analysis_functions) {
    auto const &f = pair.second;
    try {
      analysis.emplace(f.name, f(results));
    } catch (std::exception &e) {
      CASM::err_log() << "Results analysis '" << pair.first
                      << "' failed: " << e.what() << std::endl;
      Eigen::VectorXd nan_vector = Eigen::VectorXd::Constant(
          f.component_names.size(),
          std::numeric_limits<double>::signaling_NaN());
      analysis.emplace(f.name, nan_vector);
    }
  }
  return analysis;
}

}  // namespace monte
}  // namespace CASM

#endif
