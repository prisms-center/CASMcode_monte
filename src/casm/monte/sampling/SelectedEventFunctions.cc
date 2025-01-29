#include "casm/monte/sampling/SelectedEventFunctions.hh"

// debug
#include <iostream>

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
  this->time = Eigen::MatrixXd::Zero(_max_n_position_samples, _n_atoms);

  this->atom_positions_cart = std::vector<Eigen::MatrixXd>(
      _max_n_position_samples, Eigen::MatrixXd::Zero(3, _n_atoms));
}

/// \brief Insert a new position sample for an atom, if the atom has jumped
///     the necessary number of times
void CorrelationsData::insert(Index atom_id, CountType n_jumps,
                              Eigen::VectorXd const &position_cart,
                              CountType _step, CountType _pass, double _time) {
  if (n_jumps % this->jumps_per_position_sample != 0) {
    return;
  }

  CountType n_samples = this->n_position_samples[atom_id];

  if (n_samples >= max_n_position_samples) {
    return;
  }

  this->step(n_samples, atom_id) = _step;
  this->pass(n_samples, atom_id) = _pass;
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
    Index _max_size,
    std::optional<
        std::map<Eigen::VectorXl, std::string, LexicographicalCompare>>
        _value_labels)
    : m_shape(_shape),
      m_component_names(_component_names),
      m_max_size(_max_size),
      m_max_size_exceeded(false),
      m_out_of_range_count(0.0),
      m_value_labels(_value_labels) {}

/// \brief Insert a value into the histogram, with an optional weight
void DiscreteVectorIntHistogram::insert(Eigen::VectorXl const &value,
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
std::vector<Eigen::VectorXl> DiscreteVectorIntHistogram::values() const {
  std::vector<Eigen::VectorXl> _keys;
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
    double _tol, Index _max_size,
    std::optional<
        std::map<Eigen::VectorXd, std::string, FloatLexicographicalCompare>>
        _value_labels)
    : m_shape(_shape),
      m_component_names(_component_names),
      m_max_size(_max_size),
      m_max_size_exceeded(false),
      m_count(FloatLexicographicalCompare(_tol)),
      m_out_of_range_count(0.0),
      m_value_labels(_value_labels) {}

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
      m_max_size_exceeded(false),
      m_begin(_initial_begin),
      m_out_of_range_count(0.0) {}

/// \brief Insert a value into the histogram, with an optional weight
///
/// Notes:
/// - This function takes the "real value" being inserted into the histogram,
///   whether the histogram is in linear or log space.
///
/// \param log_value The "real value" being inserted into the histogram. If the
///     histogram is in log space, then the logarithm (std::log10) of `value` is
///     evaluated and inserted.
/// \param weight The weight to assign to the value

void Histogram1D::insert(double value, double weight) {
  if (m_is_log) {
    this->_insert(std::log10(value), weight);
  } else {
    this->_insert(value, weight);
  }
}

/// \brief Insert the log of a value into a log space histogram directly,
///     with an optional weight.
///
/// Notes:
/// - This function is used to update the histogram when the logarithm
///   (std::log10) of the "real value" has already been evaluated.
/// - This function throws if the histogram is not in log space.
///
/// \param log_value The logarithm (std::log10) of the "real value"
/// \param weight The weight to assign to the value
void Histogram1D::insert_log_value(double log_value, double weight) {
  if (!m_is_log) {
    throw std::runtime_error(
        "Error in Histogram1D::insert_log_value: histogram is not in log "
        "space");
  }
  this->_insert(log_value, weight);
}

/// \brief Return the coordinates of the beginning of each bin range
std::vector<double> Histogram1D::bin_coords() const {
  std::vector<double> _bin_coords;
  for (Index i = 0; i < m_count.size(); ++i) {
    _bin_coords.push_back(m_begin + i * m_bin_width);
  }
  return _bin_coords;
}

/// \brief Return the sum of bin counts + out-of-range counts
double Histogram1D::sum() const {
  double _sum = m_out_of_range_count;
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
    if (m_is_log) {
      this->insert_log_value(other_bin_coords[i], other.m_count[i]);
    } else {
      this->insert(other_bin_coords[i], other.m_count[i]);
    }
  }
  this->m_out_of_range_count += other.m_out_of_range_count;
}

/// \brief Insert a value into the histogram, with an optional weight
///
/// \param value The value to add to the histogram. If `is_log` is true,
///     the value should already be in log space.
/// \param weight The weight to assign to the value
void Histogram1D::_insert(double value, double weight) {
  if (value < m_begin || m_count.empty()) {
    _reset_bins(value);
  }
  if (value < m_begin && m_max_size_exceeded) {
    m_out_of_range_count += weight;
    return;
  }

  double _tol = 1e-10;
  int bin = std::floor((value - m_begin) / m_bin_width + _tol);

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

/// \brief Reset histogram bins if this is the first value being added,
/// or if `value` is less than `begin`
///
/// \param value The value to add to the histogram. If `is_log` is true,
///     the value should already be in log space.
void Histogram1D::_reset_bins(double value) {
  if (!std::isfinite(value)) {
    std::stringstream msg;
    msg << "Error in Histogram1D::_reset_bins: value (=" << value
        << ") is not finite.";
    throw std::runtime_error(msg.str());
  }
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

// -- PartitionedHistogram1D --

PartitionedHistogram1D::PartitionedHistogram1D(
    std::vector<std::string> const &_partion_names, double _initial_begin,
    double _bin_width, bool _is_log, Index _max_size)
    : m_partition_names(_partion_names),
      m_histograms(_partion_names.size(),
                   Histogram1D(_initial_begin, _bin_width, _is_log, _max_size)),
      m_up_to_date(false) {}

/// \brief Make the combined histogram from the partitioned histograms
void PartitionedHistogram1D::_make_combined_histogram() const {
  m_combined_histogram = combine(m_histograms);
  std::vector<double> bin_coords = m_combined_histogram->bin_coords();
  if (!bin_coords.empty()) {
    auto &hist = const_cast<std::vector<Histogram1D> &>(m_histograms);
    for (Index i = 0; i < hist.size(); ++i) {
      if (m_combined_histogram->is_log()) {
        hist[i].insert_log_value(bin_coords.front(), 0.0);
        hist[i].insert_log_value(bin_coords.back(), 0.0);
      } else {
        hist[i].insert(bin_coords.front(), 0.0);
        hist[i].insert(bin_coords.back(), 0.0);
      }
    }
  }
  m_up_to_date = true;
}

// -- GenericSelectedEventFunction --

/// \brief Constructor - default component names
GenericSelectedEventFunction::GenericSelectedEventFunction(
    std::string _name, std::string _description, bool _requires_event_state,
    std::function<void()> _function, std::function<bool()> _has_value_function,
    Index _order)
    : name(_name),
      description(_description),
      requires_event_state(_requires_event_state),
      function(_function),
      has_value_function(_has_value_function),
      order(_order) {}

namespace {

bool try_construct_generic_f(
    SelectedEventDataCollector &collector, std::string name,
    monte::SelectedEventFunctions const &functions,
    monte::SelectedEventFunctionParams const &params,
    std::map<std::pair<Index, std::string>, GenericSelectedEventFunction>
        &ordered_generic_f) {
  if (!functions.generic_functions.count(name)) {
    return false;
  }

  auto f = functions.generic_functions.at(name);

  if (params.order.count(name)) {
    f.order = params.order.at(name);
  }

  ordered_generic_f.emplace(std::make_pair(f.order, f.name), f);

  collector.requires_event_state =
      (collector.requires_event_state || f.requires_event_state);

  return true;
}

bool try_construct_vector_int_hist(
    SelectedEventDataCollector &collector, std::string name,
    monte::SelectedEventFunctions const &functions,
    monte::SelectedEventFunctionParams const &params) {
  if (!functions.discrete_vector_int_functions.count(name)) {
    return false;
  }

  collector.discrete_vector_int_f.push_back(
      functions.discrete_vector_int_functions.at(name));

  // user override of default parameters:
  auto &f = collector.discrete_vector_int_f.back();

  collector.requires_event_state =
      (collector.requires_event_state || f.requires_event_state);

  if (params.max_size.count(name)) {
    f.max_size = params.max_size.at(name);
  }

  // construct Histogram
  collector.data->discrete_vector_int_histograms.emplace(
      name, monte::DiscreteVectorIntHistogram(f.component_names, f.shape,
                                              f.max_size, f.value_labels));
  collector.discrete_vector_int_hist.push_back(
      &collector.data->discrete_vector_int_histograms.at(name));
  return true;
}

bool try_construct_vector_float_hist(
    SelectedEventDataCollector &collector, std::string name,
    monte::SelectedEventFunctions const &functions,
    monte::SelectedEventFunctionParams const &params) {
  if (!functions.discrete_vector_float_functions.count(name)) {
    return false;
  }

  collector.discrete_vector_float_f.push_back(
      functions.discrete_vector_float_functions.at(name));

  // user override of default parameters:
  auto &f = collector.discrete_vector_float_f.back();

  collector.requires_event_state =
      (collector.requires_event_state || f.requires_event_state);

  if (params.tol.count(name)) {
    f.tol = params.tol.at(name);
  }
  if (params.max_size.count(name)) {
    f.max_size = params.max_size.at(name);
  }

  // construct Histogram
  collector.data->discrete_vector_float_histograms.emplace(
      name, monte::DiscreteVectorFloatHistogram(
                f.component_names, f.shape, f.tol, f.max_size, f.value_labels));
  collector.discrete_vector_float_hist.push_back(
      &collector.data->discrete_vector_float_histograms.at(name));
  return true;
}

bool try_construct_continuous_1d_hist(
    SelectedEventDataCollector &collector, std::string name,
    monte::SelectedEventFunctions const &functions,
    monte::SelectedEventFunctionParams const &params) {
  if (!functions.continuous_1d_functions.count(name)) {
    return false;
  }

  collector.continuous_1d_f.push_back(
      functions.continuous_1d_functions.at(name));

  // user override of default parameters:
  auto &f = collector.continuous_1d_f.back();

  collector.requires_event_state =
      (collector.requires_event_state || f.requires_event_state);

  if (params.initial_begin.count(name)) {
    f.initial_begin = params.initial_begin.at(name);
  }
  if (params.bin_width.count(name)) {
    f.bin_width = params.bin_width.at(name);
  }
  if (params.is_log.count(name)) {
    f.is_log = params.is_log.at(name);
  }
  if (params.max_size.count(name)) {
    f.max_size = params.max_size.at(name);
  }

  // construct Histogram
  collector.data->continuous_1d_histograms.emplace(
      name, monte::PartitionedHistogram1D(f.partition_names, f.initial_begin,
                                          f.bin_width, f.is_log, f.max_size));
  collector.continuous_1d_hist.push_back(
      &collector.data->continuous_1d_histograms.at(name));
  return true;
}

}  // namespace

/// \brief Constructor
///
/// \param selected_event_data Where data will be stored. Will be reset. Must
///     not be null, else will throw a runtime_error.
SelectedEventDataCollector::SelectedEventDataCollector(
    monte::SelectedEventFunctions const &selected_event_data_functions,
    monte::SelectedEventFunctionParams const &selected_event_data_params,
    std::shared_ptr<monte::SelectedEventData> selected_event_data)
    : data(selected_event_data), requires_event_state(false) {
  auto const &functions = selected_event_data_functions;
  auto const &params = selected_event_data_params;

  if (!data) {
    throw std::runtime_error(
        "Error in SelectedEventDataCollector: null "
        "selected_event_data");
  }

  data->reset();

  // Use this to sort the generic functions by {order, name}
  std::map<std::pair<Index, std::string>, GenericSelectedEventFunction>
      ordered_generic_f;
  for (std::string const &name : selected_event_data_params.function_names) {
    if (try_construct_generic_f(*this, name, functions, params,
                                ordered_generic_f)) {
      continue;
    } else if (try_construct_vector_int_hist(*this, name, functions, params)) {
      continue;
    } else if (try_construct_vector_float_hist(*this, name, functions,
                                               params)) {
      continue;
    } else if (try_construct_continuous_1d_hist(*this, name, functions,
                                                params)) {
      continue;
    } else {
      throw std::runtime_error(
          "Error in SelectedEventDataCollector: unknown quantity in "
          "selected_event_data_params: " +
          name);
    }
  }

  // Copy the sorted generic functions
  for (auto const &pair : ordered_generic_f) {
    generic_f.push_back(pair.second);
  }
}

void SelectedEventDataCollector::collect_vector_int_data() {
  double weight = 1.0;

  auto hist_it = discrete_vector_int_hist.begin();
  auto f_it = discrete_vector_int_f.begin();
  auto f_end = discrete_vector_int_f.end();
  while (f_it != f_end) {
    if (f_it->has_value()) {
      (*hist_it)->insert(f_it->function(), weight);
    }
    ++f_it;
    ++hist_it;
  }
}

void SelectedEventDataCollector::collect_vector_float_data() {
  double weight = 1.0;

  auto hist_it = discrete_vector_float_hist.begin();
  auto f_it = discrete_vector_float_f.begin();
  auto f_end = discrete_vector_float_f.end();
  while (f_it != f_end) {
    if (f_it->has_value()) {
      (*hist_it)->insert(f_it->function(), weight);
    }
    ++f_it;
    ++hist_it;
  }
}

void SelectedEventDataCollector::collect_continuous_1d_data() {
  int partition;
  double value;
  double log_value;
  double weight = 1.0;

  auto hist_it = continuous_1d_hist.begin();
  auto f_it = continuous_1d_f.begin();
  auto f_end = continuous_1d_f.end();
  while (f_it != f_end) {
    // Get partition and value to insert
    partition = f_it->partition();
    value = f_it->function();

    // Validation so we can throw a more informative error message:
    if (partition < 0 || partition >= f_it->partition_names.size()) {
      std::stringstream msg;
      msg << "Error in PartitionedHistogram1D::insert: "
          << "function (name=\"" << f_it->name << "\") returned a "
          << "partition index (=" << partition << ") that is out of range.";
      throw std::runtime_error(msg.str());
    }
    if (f_it->is_log) {
      log_value = std::log10(value);

      if (!std::isfinite(log_value)) {
        std::stringstream msg;
        msg << "Error in PartitionedHistogram1D::insert: "
            << "function (name=\"" << f_it->name << "\", "
            << "partition=\"" << f_it->partition_names[partition] << "\") "
            << "returned a value (=" << value << ") "
            << "with log10(value) (=" << log_value << ") that is not finite.";
        throw std::runtime_error(msg.str());
      }

    } else {
      if (!std::isfinite(value)) {
        std::stringstream msg;
        msg << "Error in PartitionedHistogram1D::insert: "
            << "function (name=\"" << f_it->name << "\", "
            << "partition=\"" << f_it->partition_names[partition] << "\") "
            << " returned a value (=" << value << ") that is not finite.";
        throw std::runtime_error(msg.str());
      }
    }

    // Insert value
    (*hist_it)->insert(partition, value, weight);
    ++f_it;
    ++hist_it;
  }
}

void SelectedEventDataCollector::evaluate_generic_functions() {
  auto f_it = generic_f.begin();
  auto f_end = generic_f.end();
  while (f_it != f_end) {
    f_it->function();
    ++f_it;
  }
}

void SelectedEventDataCollector::collect() {
  evaluate_generic_functions();
  collect_vector_int_data();
  collect_vector_float_data();
  collect_continuous_1d_data();
}

}  // namespace CASM::monte
