#ifndef CASM_monte_Sampler
#define CASM_monte_Sampler

#include <map>
#include <memory>
#include <vector>

#include "casm/casm_io/json/jsonParser.hh"
#include "casm/global/eigen.hh"
#include "casm/monte/definitions.hh"

namespace CASM {
namespace monte {

/// \brief Return scalar as size=1 Eigen::VectorXd
inline Eigen::VectorXd reshaped(double value) {
  return Eigen::VectorXd::Constant(1, value);
}

/// \brief Return vector or matrix as column-major Eigen::VectorXd
inline Eigen::VectorXd reshaped(Eigen::MatrixXd const &value) {
  return value.reshaped();
}

/// \brief Sampler stores vector valued samples in a matrix
///
/// \note For sampling scalars, a size=1 vector is expected. For
///     sampling matrices, column-major order is expected. These
///     can be obtained with `reshaped(value)`.
class Sampler {
 public:
  /// \brief Return type for a column vector block
  typedef const Eigen::VectorBlock<
      const Eigen::Block<const Eigen::MatrixXd, -1, 1, true>, -1>
      const_vector_block_type;

  /// \brief Sampler constructor - default component names
  Sampler(std::vector<Index> _shape, CountType _capacity_increment = 1000);

  /// \brief Sampler constructor - custom component names
  Sampler(std::vector<Index> _shape,
          std::vector<std::string> const &_component_names,
          CountType _capacity_increment = 1000);

  /// \brief Add a scalar sample
  void push_back(double const &value);

  /// \brief Add a new sample - any shape, unrolled
  void push_back(Eigen::VectorXd const &vector);

  /// \brief Set all values directly
  void set_values(Eigen::MatrixXd const &values);

  /// Clear values - preserves n_components, set n_samples to 0
  void clear();

  /// Conservative resize, to increase capacity for more samples
  void set_sample_capacity(CountType sample_capacity);

  /// Set capacity increment (used when push_back requires more capacity)
  void set_capacity_increment(CountType _capacity_increment);

  /// Return sampled vector component names
  std::vector<std::string> const &component_names() const;

  /// Return sampled quantity shape before unrolling
  std::vector<Index> const &shape() const;

  /// Number of components (vector size) of samples
  Index n_components() const;

  /// Current number of samples taken
  CountType n_samples() const;

  /// Current sample capacity
  CountType sample_capacity() const;

  /// \brief Get sampled values as a matrix
  ///
  /// Stores individual vector samples in rows,
  /// use columns to check convergence of individual components
  Eigen::Block<const Eigen::MatrixXd> values() const;

  /// \brief Get all samples of a particular component (a column of `values()`)
  const_vector_block_type component(Index component_index) const;

  /// \brief Get a sample (a row of `values()`)
  Eigen::MatrixXd::ConstRowXpr sample(CountType sample_index) const;

 private:
  /// Size of vectors to be sampled
  Index m_n_components;

  /// Names to use for components. Size == m_n_components
  std::vector<std::string> m_component_names;

  /// Quantity shape before unrolling
  std::vector<Index> m_shape;

  /// Current number of samples taken
  Index m_n_samples;

  /// Used when push_back requires more capacity
  CountType m_capacity_increment;

  /// Stores individual samples in rows.
  /// Total number of rows is `sample_capacity` which to avoid constant re-
  /// sizing may be greater than the current number of samples taken,
  /// `m_n_samples`
  /// Use columns to check convergence of individual components.
  Eigen::MatrixXd m_values;
};

typedef std::map<std::string, std::shared_ptr<monte::Sampler>> SamplerMap;

/// \brief jsonSampler stores a vector of JSON-valued samples
struct jsonSampler {
  std::vector<jsonParser> values;
};

/// \brief Construct vector of component_names
std::vector<std::string> default_component_names(std::vector<Index> shape);

/// \brief Construct vector of (row,col) names ["0,0", "1,0", ...,
/// "n_rows-1,n_cols-1"]
std::vector<std::string> colmajor_component_names(Index n_rows, Index n_cols);

struct SamplerComponent {
  /// \brief Constructor
  ///
  /// \param _sampler_name Name of Sampler
  /// \param _component_index Index into sampler output vector
  SamplerComponent(std::string _sampler_name, Index _component_index,
                   std::string _component_name)
      : sampler_name(_sampler_name),
        component_index(_component_index),
        component_name(_component_name) {}

  /// Sampler name (i.e. "comp_n", "corr", etc.)
  std::string sampler_name = "";

  /// Sampler component index (i.e. 0, 1, etc.)
  Index component_index = 0;

  /// Sampler component name (i.e. "0", "1", "Mg", "O", etc.)
  std::string component_name = 0;

  bool operator<(SamplerComponent const &other) const;
};

struct RequestedPrecision {
  bool abs_convergence_is_required = false;
  double abs_precision;

  bool rel_convergence_is_required = false;
  double rel_precision;

  static RequestedPrecision abs_and_rel(double abs_value, double rel_value) {
    RequestedPrecision x;
    x.abs_convergence_is_required = true;
    x.abs_precision = abs_value;
    x.rel_convergence_is_required = true;
    x.rel_precision = rel_value;
    return x;
  }

  static RequestedPrecision abs(double value) {
    RequestedPrecision x;
    x.abs_convergence_is_required = true;
    x.abs_precision = value;
    return x;
  }

  static RequestedPrecision rel(double value) {
    RequestedPrecision x;
    x.rel_convergence_is_required = true;
    x.rel_precision = value;
    return x;
  }
};

/// \brief Find sampler by name and throw if not found
inline std::map<std::string, std::shared_ptr<Sampler>>::const_iterator
find_or_throw(std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
              std::string const &sampler_name);

/// \brief Find sampler by name, and throw if not found or component index is
///     not in a valid range
std::map<std::string, std::shared_ptr<Sampler>>::const_iterator find_or_throw(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
    SamplerComponent const &key);

/// \brief Get Sampler::n_samples() value (assumes same for all)
/// (else 0)
CountType get_n_samples(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers);

/// \brief Holds sampled JSON data
typedef std::map<std::string, jsonSampler> jsonSamplerMap;

}  // namespace monte
}  // namespace CASM

// --- Inline implementations ---

namespace CASM {
namespace monte {

inline Index calc_n_components(std::vector<Index> shape) {
  Index result = 1;
  for (Index x : shape) {
    result *= x;
  }
  return result;
}

/// \brief Sampler constructor - default component names
///
/// \param _shape Shape of quantity to be sampled
/// \param _capacity_increment How much to resize the underlying matrix by
///     whenever space runs out.
///
/// Notes:
/// - Components are given default names ["0", "1", "2", ...)
inline Sampler::Sampler(std::vector<Index> _shape,
                        CountType _capacity_increment)
    : m_n_components(calc_n_components(_shape)),
      m_component_names(default_component_names(_shape)),
      m_shape(_shape),
      m_n_samples(0),
      m_capacity_increment(_capacity_increment) {
  clear();
}

/// \brief Sampler constructor - custom component names
///
/// \param _component_names Names to give to each sampled vector element
/// \param _capacity_increment How much to resize the underlying matrix by
///     whenever space runs out.
inline Sampler::Sampler(std::vector<Index> _shape,
                        std::vector<std::string> const &_component_names,
                        CountType _capacity_increment)
    : m_n_components(_component_names.size()),
      m_component_names(_component_names),
      m_shape(_shape),
      m_n_samples(0),
      m_capacity_increment(_capacity_increment) {
  clear();
}

/// \brief Add a new sample of a scalar quantity
inline void Sampler::push_back(double const &value) {
  if (n_samples() == sample_capacity()) {
    set_sample_capacity(sample_capacity() + m_capacity_increment);
  }
  m_values(m_n_samples, 0) = value;
  ++m_n_samples;
}

/// \brief Add a new sample - any shape, unrolled
inline void Sampler::push_back(Eigen::VectorXd const &vector) {
  if (n_samples() == sample_capacity()) {
    set_sample_capacity(sample_capacity() + m_capacity_increment);
  }
  m_values.row(m_n_samples) = vector;
  ++m_n_samples;
}

/// \brief Set all values directly
inline void Sampler::set_values(Eigen::MatrixXd const &values) {
  m_values = values;
  m_n_samples = m_values.rows();
}

/// Clear values - preserves n_components, set n_samples to 0
inline void Sampler::clear() {
  m_values.resize(m_capacity_increment, m_n_components);
  m_n_samples = 0;
}

/// Conservative resize, to increase capacity for more samples
inline void Sampler::set_sample_capacity(CountType sample_capacity) {
  m_values.conservativeResize(sample_capacity, Eigen::NoChange_t());
}

/// Set capacity increment (used when push_back requires more capacity)
inline void Sampler::set_capacity_increment(CountType _capacity_increment) {
  m_capacity_increment = _capacity_increment;
}

/// Return sampled vector component names
inline std::vector<std::string> const &Sampler::component_names() const {
  return m_component_names;
}

/// Return sampled quantity shape before unrolling
inline std::vector<Index> const &Sampler::shape() const { return m_shape; }

/// Number of components (vector size) of samples
inline Index Sampler::n_components() const { return m_n_components; }

/// Current number of samples taken
inline CountType Sampler::n_samples() const { return m_n_samples; }

/// Current sample capacity
inline CountType Sampler::sample_capacity() const { return m_values.rows(); }

/// \brief Get sampled values as a matrix
///
/// Stores individual vector samples in rows,
/// use columns to check convergence of individual components
inline Eigen::Block<const Eigen::MatrixXd> Sampler::values() const {
  return m_values.block(0, 0, m_n_samples, m_values.cols());
}

/// \brief Get all samples of a particular component (a column of `values()`)
inline Sampler::const_vector_block_type Sampler::component(
    Index component_index) const {
  return m_values.col(component_index).head(m_n_samples);
}

/// \brief Get a sample (a row of `values()`)
inline Eigen::MatrixXd::ConstRowXpr Sampler::sample(
    CountType sample_index) const {
  return m_values.row(sample_index);
}

/// \brief Construct vector of component_names
///
/// Shape = [] (scalar) -> {"0"}
/// Shape = [n] (vector) -> {"0", "1", ..., "n-1"}
/// Shape = [m, n] (matrix) -> {"0,0", "1,0", ..., "m-1,n-1"}
inline std::vector<std::string> default_component_names(
    std::vector<Index> shape) {
  if (shape.size() == 0) {
    return {"0"};
  } else if (shape.size() == 1) {
    std::vector<std::string> result;
    for (Index i = 0; i < shape[0]; ++i) {
      result.push_back(std::to_string(i));
    }
    return result;
  } else if (shape.size() == 2) {
    return colmajor_component_names(shape[0], shape[1]);
  } else {
    throw std::runtime_error(
        "Error constructing sampler component names: >2 dimensions is not "
        "supported");
  }
}

/// \brief Construct vector of (row,col) names ["0,0", "1,0", ...,
/// std::string(n_rows*n_cols-1)]
inline std::vector<std::string> colmajor_component_names(Index n_rows,
                                                         Index n_cols) {
  std::vector<std::string> result;
  for (Index i = 0; i < n_cols; ++i) {
    for (Index j = 0; j < n_rows; ++j) {
      result.push_back(std::to_string(j) + "," + std::to_string(i));
    }
  }
  return result;
}

inline bool SamplerComponent::operator<(SamplerComponent const &other) const {
  if (this->sampler_name == other.sampler_name) {
    return this->component_index < other.component_index;
  }
  return this->sampler_name < other.sampler_name;
}

/// \brief Find sampler by name and throw if not found
inline std::map<std::string, std::shared_ptr<Sampler>>::const_iterator
find_or_throw(std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
              std::string const &sampler_name) {
  // find and validate sampler name && component index
  auto sampler_it = samplers.find(sampler_name);
  if (sampler_it == samplers.end()) {
    std::stringstream msg;
    msg << "Error finding sampler component: Sampler '" << sampler_name
        << "' not found." << std::endl;
    throw std::runtime_error(msg.str());
  }
  return sampler_it;
}

/// \brief Find sampler by name, and throw if not found or component index is
///     not in a valid range
inline std::map<std::string, std::shared_ptr<Sampler>>::const_iterator
find_or_throw(std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
              SamplerComponent const &key) {
  // find and validate sampler name && component index
  auto sampler_it = samplers.find(key.sampler_name);
  if (sampler_it == samplers.end()) {
    std::stringstream msg;
    msg << "Error finding sampler component: Sampler '" << key.sampler_name
        << "' not found." << std::endl;
    throw std::runtime_error(msg.str());
  }
  if (key.component_index >= sampler_it->second->n_components()) {
    std::stringstream msg;
    msg << "Error finding sampler component: Requested component index "
        << key.component_index << ", but '" << key.sampler_name << "' has "
        << sampler_it->second->n_components() << "components." << std::endl;
    throw std::runtime_error(msg.str());
  }
  return sampler_it;
}

inline CountType get_n_samples(
    std::map<std::string, std::shared_ptr<Sampler>> const &samplers) {
  if (samplers.size()) {
    return samplers.begin()->second->n_samples();
  }
  return CountType(0);
}

}  // namespace monte
}  // namespace CASM

#endif
