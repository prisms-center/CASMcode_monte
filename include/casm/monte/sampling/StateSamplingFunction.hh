#ifndef CASM_monte_StateSamplingFunction
#define CASM_monte_StateSamplingFunction

#include "casm/monte/definitions.hh"
#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

/// \brief A function to be evaluated when taking a sample of a Monte Carlo
///     calculation state
///
/// - Each StateSamplingFunction returns an Eigen::VectorXd
/// - A StateSamplingFunction has additional information (name, description,
///   component_names) to enable specifying convergence criteria, allow input
///   and output descriptions, help and error messages, etc.
/// - Use `reshaped` (in casm/monte/sampling/Sampler.hh) to output scalars or
///   matrices as vectors.
///
struct StateSamplingFunction {
  /// \brief Constructor - default component names
  StateSamplingFunction(std::string _name, std::string _description,
                        std::vector<Index> _shape,
                        std::function<Eigen::VectorXd()> _function);

  /// \brief Constructor - custom component names
  StateSamplingFunction(std::string _name, std::string _description,
                        std::vector<std::string> const &_component_names,
                        std::vector<Index> _shape,
                        std::function<Eigen::VectorXd()> _function);

  /// \brief Function name (and quantity to be sampled)
  std::string name;

  /// \brief Description of the function
  std::string description;

  /// \brief Shape of quantity, with column-major unrolling
  ///
  /// Scalar: [], Vector: [n], Matrix: [m, n], etc.
  std::vector<Index> shape;

  /// \brief A name for each component of the resulting Eigen::VectorXd
  ///
  /// Can be string representing an index (i.e "0", "1", "2", etc.) or can
  /// be a descriptive string (i.e. "Mg", "Va", "O", etc.)
  std::vector<std::string> component_names;

  /// \brief The function to be evaluated
  std::function<Eigen::VectorXd()> function;

  /// \brief Evaluates `function`
  Eigen::VectorXd operator()() const;
};

/// \brief Get component names for a particular function, else use defaults
std::vector<std::string> get_scalar_component_names(
    std::string const &function_name, double const &value,
    StateSamplingFunctionMap const &sampling_functions);

/// \brief Get component names for a particular function, else use defaults
std::vector<std::string> get_vector_component_names(
    std::string const &function_name, Eigen::VectorXd const &value,
    StateSamplingFunctionMap const &sampling_functions);

/// \brief Get component names for a particular function, else use defaults
std::vector<std::string> get_matrix_component_names(
    std::string const &function_name, Eigen::MatrixXd const &value,
    StateSamplingFunctionMap const &sampling_functions);

/// \brief A function, returning JSON, to be evaluated when taking a sample
///     of a Monte Carlo calculation state
///
/// - Each StateSamplingFunction returns a jsonParser
/// - A StateSamplingFunction has additional information (name, description,
///   component_names) to enable specifying convergence criteria, allow input
///   and output descriptions, help and error messages, etc.
///
struct jsonStateSamplingFunction {
  /// \brief Constructor
  jsonStateSamplingFunction(std::string _name, std::string _description,
                            std::function<jsonParser()> _function);

  /// \brief Function name (and quantity to be sampled)
  std::string name;

  /// \brief Description of the function
  std::string description;

  /// \brief The function to be evaluated
  std::function<jsonParser()> function;

  /// \brief Evaluates `function`
  jsonParser operator()() const;
};

}  // namespace monte
}  // namespace CASM

// --- Inline implementations ---

namespace CASM {
namespace monte {

/// \brief Constructor - default component names
inline StateSamplingFunction::StateSamplingFunction(
    std::string _name, std::string _description, std::vector<Index> _shape,
    std::function<Eigen::VectorXd()> _function)
    : name(_name),
      description(_description),
      shape(_shape),
      component_names(default_component_names(shape)),
      function(_function) {}

/// \brief Constructor - custom component names
inline StateSamplingFunction::StateSamplingFunction(
    std::string _name, std::string _description,
    std::vector<std::string> const &_component_names, std::vector<Index> _shape,
    std::function<Eigen::VectorXd()> _function)
    : name(_name),
      description(_description),
      shape(_shape),
      component_names(_component_names),
      function(_function) {}

/// \brief Take a sample
inline Eigen::VectorXd StateSamplingFunction::operator()() const {
  return function();
}

/// \brief Get component names for a particular function, else use defaults
///
/// Notes:
/// - Used for naming conditions vector components using a sampling function
///   of the same name.
/// - If function not found, returns default component names ("0")
/// - Throws if function found, but component_names dimension does not match
inline std::vector<std::string> get_scalar_component_names(
    std::string const &function_name, double const &value,
    StateSamplingFunctionMap const &sampling_functions) {
  std::vector<Index> shape({});
  auto function_it = sampling_functions.find(function_name);
  if (function_it == sampling_functions.end()) {
    return default_component_names(shape);
  } else {
    if (function_it->second.component_names.size() != 1) {
      std::stringstream msg;
      msg << "Error in get_scalar_component_names: Dimension of \""
          << function_name << "\" (" << 1
          << ") does not match the corresponding sampling function.";
      throw std::runtime_error(msg.str());
    }
    return function_it->second.component_names;
  }
}

/// \brief Get component names for a particular function, else use defaults
///
/// Notes:
/// - Used for naming conditions vector components using a sampling function
///   of the same name.
/// - If function not found, returns default component names ("0", "1", "2",
///   ...)
/// - Throws if function found, but component_names dimension does not match
///   value.size().
inline std::vector<std::string> get_vector_component_names(
    std::string const &function_name, Eigen::VectorXd const &value,
    StateSamplingFunctionMap const &sampling_functions) {
  std::vector<Index> shape({value.size()});
  auto function_it = sampling_functions.find(function_name);
  if (function_it == sampling_functions.end()) {
    return default_component_names(shape);
  } else {
    if (function_it->second.component_names.size() != value.size()) {
      std::stringstream msg;
      msg << "Error in get_vector_component_names: Dimension of \""
          << function_name << "\" (" << value.size()
          << ") does not match the corresponding sampling function.";
      throw std::runtime_error(msg.str());
    }
    return function_it->second.component_names;
  }
}

/// \brief Get component names for a particular function, else use defaults
///
/// Notes:
/// - Used for naming conditions vector components using a sampling function
///   of the same name.
/// - If function not found, returns default component names ("0", "1", "2",
///   ...)
/// - Throws if function found, but component_names dimension does not match
///   value.size().
inline std::vector<std::string> get_matrix_component_names(
    std::string const &function_name, Eigen::MatrixXd const &value,
    StateSamplingFunctionMap const &sampling_functions) {
  std::vector<Index> shape({value.rows(), value.cols()});
  auto function_it = sampling_functions.find(function_name);
  if (function_it == sampling_functions.end()) {
    return default_component_names(shape);
  } else {
    if (function_it->second.component_names.size() != value.size()) {
      std::stringstream msg;
      msg << "Error in get_matrix_component_names: Dimension of \""
          << function_name << "\" (" << value.size()
          << ") does not match the corresponding sampling function.";
      throw std::runtime_error(msg.str());
    }
    return function_it->second.component_names;
  }
}

/// \brief Constructor - custom component names
inline jsonStateSamplingFunction::jsonStateSamplingFunction(
    std::string _name, std::string _description,
    std::function<jsonParser()> _function)
    : name(_name), description(_description), function(_function) {}

/// \brief Take a sample
inline jsonParser jsonStateSamplingFunction::operator()() const {
  return function();
}

}  // namespace monte
}  // namespace CASM

#endif
