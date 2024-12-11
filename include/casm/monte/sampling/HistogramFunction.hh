#ifndef CASM_monte_HistogramFunction
#define CASM_monte_HistogramFunction

#include <optional>

#include "casm/monte/definitions.hh"
#include "casm/monte/misc/LexicographicalCompare.hh"
#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

/// \brief A function to be evaluated during a Monte Carlo calculation
///
/// - Each SamplingFunction returns a ValueType (e.g. Eigen::VectorXl,
///   Eigen::VectorXd)
/// - A StateSamplingFunction has additional information (name, description,
///   component_names) to enable specifying convergence criteria, allow input
///   and output descriptions, help and error messages, etc.
/// - Use `reshaped` (in casm/monte/sampling/Sampler.hh) to output scalars or
///   matrices as vectors.
///
template <typename ValueType, typename CompareType>
class HistogramFunctionT {
 public:
  /// \brief Constructor
  HistogramFunctionT(std::string _name, std::string _description,
                     std::vector<Index> _shape,
                     std::optional<std::vector<std::string>> _component_names,
                     bool _requires_event_state,
                     std::function<ValueType()> _function,
                     std::function<bool()> _has_value_function, Index _max_size,
                     double _tol = CASM::TOL);

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

  /// \brief Does the function require the event state?
  bool requires_event_state;

  /// \brief The function to be evaluated
  std::function<ValueType()> function;

  /// \brief Returns true if the function has a value
  std::function<bool()> has_value_function;

  /// \brief Maximum number of bins in the histogram
  Index max_size;

  /// \brief Tolerance for comparing values (if applicable)
  double tol;

  /// \brief Optional labels for each value in the histogram
  std::optional<std::map<ValueType, std::string, CompareType>> value_labels;

  /// \brief Evaluates `function`
  ValueType operator()() const { return function(); }

  /// \brief Evaluates `has_value_function`
  bool has_value() const { return has_value_function(); }
};

typedef HistogramFunctionT<Eigen::VectorXl, LexicographicalCompare>
    DiscreteVectorIntHistogramFunction;

typedef HistogramFunctionT<Eigen::VectorXd, FloatLexicographicalCompare>
    DiscreteVectorFloatHistogramFunction;

template <typename ValueType>
class PartitionedHistogramFunction {
 public:
  /// \brief Constructor
  PartitionedHistogramFunction(std::string _name, std::string _description,
                               bool _requires_event_state,
                               std::function<ValueType()> _function,
                               std::vector<std::string> const &_partition_names,
                               std::function<int()> _get_partition,
                               bool _is_log, double _initial_begin,
                               double _bin_width, Index _max_size);

  /// \brief Function name (and quantity to be sampled)
  std::string name;

  /// \brief Description of the function
  std::string description;

  /// \brief Does the function require the event state?
  bool requires_event_state;

  /// \brief The function to be evaluated
  std::function<ValueType()> function;

  /// \brief Evaluates `function`
  ValueType operator()() const { return function(); }

  /// \brief A name for each partition
  std::vector<std::string> partition_names;

  /// \brief Get the partition value
  std::function<int()> get_partition;

  /// \brief Evaluates `get_partition`
  int partition() const { return get_partition(); }

  /// \brief Is the function log-scaled?
  bool is_log;

  /// \brief The initial value of the first bin
  double initial_begin;

  /// \brief The width of each bin
  double bin_width;

  /// \brief The maximum number of bins in the histogram
  Index max_size;
};

}  // namespace monte
}  // namespace CASM

// --- Inline implementations ---

namespace CASM {
namespace monte {

/// \brief Constructor
template <typename ValueType, typename CompareType>
HistogramFunctionT<ValueType, CompareType>::HistogramFunctionT(
    std::string _name, std::string _description, std::vector<Index> _shape,
    std::optional<std::vector<std::string>> _component_names,
    bool _requires_event_state, std::function<ValueType()> _function,
    std::function<bool()> _has_value_function, Index _max_size, double _tol)
    : name(_name),
      description(_description),
      shape(_shape),
      component_names(_component_names.has_value()
                          ? _component_names.value()
                          : default_component_names(shape)),
      requires_event_state(_requires_event_state),
      function(_function),
      has_value_function(_has_value_function),
      max_size(_max_size),
      tol(_tol) {
  if (!function) {
    throw std::runtime_error("HistogramFunction: function is empty");
  }
  if (!has_value_function) {
    throw std::runtime_error("HistogramFunction: has_value_function is empty");
  }
}

/// \brief Constructor
template <typename ValueType>
PartitionedHistogramFunction<ValueType>::PartitionedHistogramFunction(
    std::string _name, std::string _description, bool _requires_event_state,
    std::function<ValueType()> _function,
    std::vector<std::string> const &_partition_names,
    std::function<int()> _get_partition, bool _is_log, double _initial_begin,
    double _bin_width, Index _max_size)
    : name(_name),
      description(_description),
      requires_event_state(_requires_event_state),
      function(_function),
      partition_names(_partition_names),
      get_partition(_get_partition),
      is_log(_is_log),
      initial_begin(_initial_begin),
      bin_width(_bin_width),
      max_size(_max_size) {
  if (!function) {
    throw std::runtime_error("PartitionedHistogramFunction: function is empty");
  }
  if (!get_partition) {
    throw std::runtime_error(
        "PartitionedHistogramFunction: get_partition is empty");
  }
}

}  // namespace monte
}  // namespace CASM

#endif
