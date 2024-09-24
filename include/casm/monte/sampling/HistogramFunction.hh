#ifndef CASM_monte_HistogramFunction
#define CASM_monte_HistogramFunction

#include "casm/monte/definitions.hh"
#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

/// \brief A function to be evaluated during a Monte Carlo calculation
///
/// - Each SamplingFunction returns a ValueType (e.g. Eigen::VectorXi,
///   Eigen::VectorXd)
/// - A StateSamplingFunction has additional information (name, description,
///   component_names) to enable specifying convergence criteria, allow input
///   and output descriptions, help and error messages, etc.
/// - Use `reshaped` (in casm/monte/sampling/Sampler.hh) to output scalars or
///   matrices as vectors.
///
template <typename ValueType>
class HistogramFunction {
 public:
  /// \brief Constructor - default component names
  HistogramFunction(std::string _name, std::string _description,
                    std::vector<Index> _shape,
                    std::function<ValueType()> _function, Index _max_size,
                    double _tol = CASM::TOL);

  /// \brief Constructor - custom component names
  HistogramFunction(std::string _name, std::string _description,
                    std::vector<std::string> const &_component_names,
                    std::vector<Index> _shape,
                    std::function<ValueType()> _function, Index _max_size,
                    double _tol = CASM::TOL);

  /// \brief Function name (and quantity to be sampled)
  std::string name;

  /// \brief Description of the function
  std::string description;

  /// \brief The function to be evaluated
  std::function<ValueType()> function;

  /// \brief Shape of quantity, with column-major unrolling
  ///
  /// Scalar: [], Vector: [n], Matrix: [m, n], etc.
  std::vector<Index> shape;

  /// \brief A name for each component of the resulting Eigen::VectorXd
  ///
  /// Can be string representing an index (i.e "0", "1", "2", etc.) or can
  /// be a descriptive string (i.e. "Mg", "Va", "O", etc.)
  std::vector<std::string> component_names;

  /// \brief Maximum number of bins in the histogram
  Index max_size;

  /// \brief Tolerance for comparing values (if applicable)
  double tol;

  /// \brief Evaluates `function`
  ValueType operator()() const { return function(); }
};

template <typename ValueType>
class PartitionedHistogramFunction {
 public:
  /// \brief Constructor
  PartitionedHistogramFunction(std::string _name, std::string _description,
                               std::function<ValueType()> _function,
                               std::vector<std::string> const &_partition_names,
                               std::function<int()> _get_partition,
                               bool _is_log, double _initial_begin,
                               double _bin_width, Index _max_size);

  /// \brief Function name (and quantity to be sampled)
  std::string name;

  /// \brief Description of the function
  std::string description;

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

/// \brief Constructor - default component names
template <typename ValueType>
HistogramFunction<ValueType>::HistogramFunction(
    std::string _name, std::string _description, std::vector<Index> _shape,
    std::function<ValueType()> _function, Index _max_size, double _tol)
    : name(_name),
      description(_description),
      shape(_shape),
      component_names(default_component_names(shape)),
      function(_function),
      max_size(_max_size),
      tol(_tol) {}

/// \brief Constructor - custom component names
template <typename ValueType>
HistogramFunction<ValueType>::HistogramFunction(
    std::string _name, std::string _description,
    std::vector<std::string> const &_component_names, std::vector<Index> _shape,
    std::function<ValueType()> _function, Index _max_size, double _tol)
    : name(_name),
      description(_description),
      shape(_shape),
      component_names(_component_names),
      function(_function),
      max_size(_max_size),
      tol(_tol) {}

/// \brief Constructor
template <typename ValueType>
PartitionedHistogramFunction<ValueType>::PartitionedHistogramFunction(
    std::string _name, std::string _description,
    std::function<ValueType()> _function,
    std::vector<std::string> const &_partition_names,
    std::function<int()> _get_partition, bool _is_log, double _initial_begin,
    double _bin_width, Index _max_size)
    : name(_name),
      description(_description),
      function(_function),
      partition_names(_partition_names),
      get_partition(_get_partition),
      is_log(_is_log),
      initial_begin(_initial_begin),
      bin_width(_bin_width),
      max_size(_max_size) {}

}  // namespace monte
}  // namespace CASM

#endif
