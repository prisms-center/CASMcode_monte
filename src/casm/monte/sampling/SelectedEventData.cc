#include "casm/monte/sampling/SelectedEventData.hh"

namespace CASM::monte {

void CorrelationsData::initialize(Index _n_atoms,
                                  Index _jumps_per_position_sample,
                                  Index _max_n_position_samples,
                                  bool _output_incomplete_samples) {
  this->jumps_per_position_sample = _jumps_per_position_sample;
  this->max_n_position_samples = _max_n_position_samples;
  this->output_incomplete_samples = _output_incomplete_samples;
  this->n_position_samples = std::vector<CountType>(_n_atoms, 0);
  this->n_complete_samples = 0;

  this->step = Eigen::MatrixXl::Zero(_max_n_position_samples, _n_atoms);
  this->pass = Eigen::MatrixXl::Zero(_max_n_position_samples, _n_atoms);
  this->sample = Eigen::MatrixXl::Zero(_max_n_position_samples, _n_atoms);
  this->time = Eigen::MatrixXd::Zero(_max_n_position_samples, _n_atoms);

  this->atom_positions_cart = std::vector<Eigen::MatrixXd>(
      _max_n_position_samples, Eigen::MatrixXd::Zero(3, _n_atoms));
}

/// \brief Insert a new position sample for an atom, if the atom has jumped
///     the necessary number of times
void CorrelationsData::insert(Index atom_id, CountType n_jumps,
                              Eigen::VectorXd const &position_cart,
                              CountType _step, CountType _pass,
                              CountType _sample, double _time) {
  if (n_jumps % this->jumps_per_position_sample != 0) {
    return;
  }

  CountType n_samples = this->n_position_samples[atom_id];

  if (n_samples >= max_n_position_samples) {
    return;
  }

  this->step(n_samples, atom_id) = _step;
  this->pass(n_samples, atom_id) = _pass;
  this->sample(n_samples, atom_id) = _sample;
  this->time(n_samples, atom_id) = _time;
  this->atom_positions_cart[n_samples].col(atom_id) = position_cart;

  this->n_position_samples[atom_id]++;
  this->_update_n_complete_samples(n_samples);
}

/// \brief Update the number of complete samples
///
/// \param n_samples The number of position samples that have been taken for
///     most recent atom (before the current sample)
void CorrelationsData::_update_n_complete_samples(Index n_samples) {
  if (n_samples == this->n_complete_samples) {
    for (CountType x : this->n_position_samples) {
      if (x == this->n_complete_samples) {
        return;
      }
    }
    this->n_complete_samples++;
  }
};

// -- DiscreteVectorIntHistogram --

DiscreteVectorIntHistogram::DiscreteVectorIntHistogram(
    std::vector<std::string> const &_component_names, std::vector<Index> _shape,
    Index _max_size)
    : m_shape(_shape),
      m_component_names(_component_names),
      m_max_size(_max_size),
      m_max_size_exceeded(false),
      m_out_of_range_count(0.0) {}

/// \brief Insert a value into the histogram, with an optional weight
void DiscreteVectorIntHistogram::insert(Eigen::VectorXi const &value,
                                        double weight) {
  // If the value is not already in the histogram, insert it with a count of 0
  auto it = m_count.find(value);
  if (it == m_count.end()) {
    if (m_count.size() == m_max_size) {
      m_max_size_exceeded = true;
      m_out_of_range_count += weight;
      return;
    }
    it = m_count.emplace(value, 0.0).first;
  }
  it->second += weight;
}

/// \brief Return the sum of bin counts + out-of-range counts
double DiscreteVectorIntHistogram::sum() const {
  double _sum = m_out_of_range_count;
  for (auto const &x : m_count) {
    _sum += x.second;
  }
  return _sum;
}

/// \brief Return the values as a vector
std::vector<Eigen::VectorXi> DiscreteVectorIntHistogram::values() const {
  std::vector<Eigen::VectorXi> _keys;
  for (auto const &x : m_count) {
    _keys.push_back(x.first);
  }
  return _keys;
}

/// \brief Return the count as a vector
std::vector<double> DiscreteVectorIntHistogram::count() const {
  std::vector<double> _count;
  for (auto const &x : m_count) {
    _count.push_back(x.second);
  }
  return _count;
}

/// \brief Return the count as a vector containing fractions of the sum
std::vector<double> DiscreteVectorIntHistogram::fraction() const {
  std::vector<double> _fraction;
  double _sum = this->sum();
  for (auto const &x : m_count) {
    _fraction.push_back(x.second / _sum);
  }
  return _fraction;
}

// -- DiscreteVectorFloatHistogram --

DiscreteVectorFloatHistogram::DiscreteVectorFloatHistogram(
    std::vector<std::string> const &_component_names, std::vector<Index> _shape,
    double _tol, Index _max_size)
    : m_shape(_shape),
      m_component_names(_component_names),
      m_max_size(_max_size),
      m_max_size_exceeded(false),
      m_count(FloatLexicographicalCompare(_tol)),
      m_out_of_range_count(0.0) {}

/// \brief Insert a value into the histogram, with an optional weight
void DiscreteVectorFloatHistogram::insert(Eigen::VectorXd const &value,
                                          double weight) {
  // If the value is not already in the histogram, insert it with a count of 0
  auto it = m_count.find(value);
  if (it == m_count.end()) {
    if (m_count.size() == m_max_size) {
      m_max_size_exceeded = true;
      m_out_of_range_count += weight;
      return;
    }
    it = m_count.emplace(value, 0.0).first;
  }
  it->second += weight;
}

/// \brief Return the sum of bin counts + out-of-range counts
double DiscreteVectorFloatHistogram::sum() const {
  double _sum = m_out_of_range_count;
  for (auto const &x : m_count) {
    _sum += x.second;
  }
  return _sum;
}

/// \brief Return the values as a vector
std::vector<Eigen::VectorXd> DiscreteVectorFloatHistogram::values() const {
  std::vector<Eigen::VectorXd> _keys;
  for (auto const &x : m_count) {
    _keys.push_back(x.first);
  }
  return _keys;
}

/// \brief Return the count as a vector
std::vector<double> DiscreteVectorFloatHistogram::count() const {
  std::vector<double> _count;
  for (auto const &x : m_count) {
    _count.push_back(x.second);
  }
  return _count;
}

/// \brief Return the count as a vector containing fractions of the sum
std::vector<double> DiscreteVectorFloatHistogram::fraction() const {
  std::vector<double> _fraction;
  double _sum = this->sum();
  for (auto const &x : m_count) {
    _fraction.push_back(x.second / _sum);
  }
  return _fraction;
}

// -- Histogram1D --

/// \brief Constructor
Histogram1D::Histogram1D(double _initial_begin, double _bin_width, bool _is_log,
                         Index _max_size)
    : m_initial_begin(_initial_begin),
      m_bin_width(_bin_width),
      m_is_log(_is_log),
      m_max_size(_max_size),
      m_begin(_initial_begin) {}

/// \brief Insert a value into the histogram, with an optional weight
void Histogram1D::insert(double value, double weight) {
  if (m_is_log) {
    value = std::log10(value);
  }
  if (value < m_begin || m_count.empty()) {
    _reset_bins(value);
  }
  if (value < m_begin && m_max_size_exceeded) {
    m_out_of_range_count += weight;
    return;
  }

  int bin = (value - m_begin) / m_bin_width;

  while (bin >= m_count.size()) {
    if (m_count.size() == m_max_size) {
      m_max_size_exceeded = true;
      m_out_of_range_count += weight;
      return;
    }
    m_count.push_back(0);
  }

  m_count[bin] += weight;
}

/// \brief Return the coordinates of the beginning of each bin range
std::vector<double> Histogram1D::bin_coords() const {
  std::vector<double> _bin_coords;
  for (Index i = 0; i < m_count.size(); ++i) {
    _bin_coords.push_back(m_begin + i * m_bin_width);
  }
  return _bin_coords;
}

/// \brief Return the sum of bin counts
double Histogram1D::sum() const {
  double _sum = 0.0;
  for (double x : m_count) {
    _sum += x;
  }
  return _sum;
}

/// \brief Return the count as a probability density, such that the area
///     under the histogram integrates to 1 (if no out-of-range count)
std::vector<double> Histogram1D::density() const {
  std::vector<double> _density;
  double _sum = this->sum();
  for (double x : m_count) {
    _density.push_back(x / (_sum * m_bin_width));
  }
  return _density;
}

/// \brief Merge another histogram into this one
///
/// Notes:
/// - The other histogram must have the same `is_log`, `bin_width`, and
///   `initial_begin` values.
///
/// \param other The other histogram.
void Histogram1D::merge(Histogram1D const &other) {
  if (m_is_log != other.m_is_log) {
    throw std::runtime_error(
        "Error in Histogram1D::merge: cannot merge histograms with "
        "different log settings");
  }
  if (m_bin_width != other.m_bin_width) {
    throw std::runtime_error(
        "Error in Histogram1D::merge: cannot merge histograms with "
        "different bin_width values");
  }
  if (m_initial_begin != other.m_initial_begin) {
    throw std::runtime_error(
        "Error in Histogram1D::merge: cannot merge histograms with "
        "different initial_begin values");
  }

  // Merge the counts
  std::vector<double> other_bin_coords = other.bin_coords();
  for (Index i = 0; i < other.m_count.size(); ++i) {
    this->insert(other_bin_coords[i], other.m_count[i]);
  }
}

/// \brief Reset histogram bins if this is the first value being added,
/// or if `value` is less than `begin`
///
/// \param value The value to add to the histogram. If `is_log` is true,
///     the value should already be in log space.
void Histogram1D::_reset_bins(double value) {
  if (m_count.empty()) {
    while (value < m_begin) {
      m_begin -= m_bin_width;
    }
    while (value > m_begin + m_bin_width) {
      m_begin += m_bin_width;
    }
    return;
  }

  std::vector<double> prepended_bins;
  while (value < m_begin) {
    if (prepended_bins.size() + m_count.size() == m_max_size) {
      m_max_size_exceeded = true;
      break;
    }
    m_begin -= m_bin_width;
    prepended_bins.push_back(0);
  }

  if (prepended_bins.empty()) {
    return;
  }

  prepended_bins.insert(prepended_bins.end(), m_count.begin(), m_count.end());
  m_count = std::move(prepended_bins);
}

/// \brief Combine 1D histograms from multiple partitions into a single 1d
/// histogram
///
/// Notes:
/// - The histograms must have the same `is_log`, `bin_width`, and
///   `initial_begin` values.
///
/// \param histograms
///
/// \return The combined histogram.
///
Histogram1D combine(std::vector<Histogram1D> const &histograms) {
  if (histograms.empty()) {
    throw std::runtime_error(
        "Error in combine: cannot combine empty vector of histograms");
  }

  Histogram1D combined = histograms[0];
  for (Index i = 1; i < histograms.size(); ++i) {
    combined.merge(histograms[i]);
  }
  return combined;
}

}  // namespace CASM::monte
