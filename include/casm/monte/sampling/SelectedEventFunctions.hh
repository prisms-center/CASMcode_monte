#ifndef CASM_monte_SelectedEventData
#define CASM_monte_SelectedEventData

#include <map>
#include <vector>

#include "casm/global/eigen.hh"
#include "casm/monte/definitions.hh"
#include "casm/monte/misc/LexicographicalCompare.hh"
#include "casm/monte/sampling/HistogramFunction.hh"

namespace CASM {
namespace monte {

/// \brief Parameters for collecting hop correlations data (not basis function
/// correlations)
struct CorrelationsDataParams {
  /// Every `jumps_per_position_sample` steps of an individual atom, store
  /// its position.
  Index jumps_per_position_sample = 1;

  /// The maximum number of positions to store for each atom.
  Index max_n_position_samples = 100;

  /// If false, only output data when all atoms have jumped the necessary
  /// number of times. If true, output matrices with 0.0 values for atoms that
  /// have not jumped enough times to be sampled.
  bool output_incomplete_samples = false;

  /// If true, stop the run when the maximum number of positions have been
  /// sampled for all atoms. If false, continue running until the standard
  /// completion check is met, but do not collect any more position samples.
  bool stop_run_when_complete = false;
};

/// \brief Hop correlations data (not basis function correlations)
struct CorrelationsData {
  /// Every `jumps_per_position_sample` steps of an individual atom, store
  /// its position in this object
  Index jumps_per_position_sample;

  /// Maximum number of position samples for any atom
  CountType max_n_position_samples;

  /// If false, only output data when all atoms have jumped the necessary
  /// number of times. If true, output matrices with 0.0 values for atoms that
  /// have not jumped enough times to be sampled.
  bool output_incomplete_samples;

  /// If true, stop the run when the maximum number of positions have been
  /// sampled for all atoms. If false, continue running until the standard
  /// completion check is met, but do not collect any more position samples.
  bool stop_run_when_complete;

  /// For each atom, the number of positions stored in this object
  std::vector<CountType> n_position_samples;

  /// Number of position samples completed for all atoms
  CountType n_complete_samples;

  /// Store when atom positions were sampled (step, pass, sample, time)
  // X[position_sample_index][atom_index]

  Eigen::MatrixXl step;
  Eigen::MatrixXl pass;
  Eigen::MatrixXd time;

  /// Store atom positions
  // X[position_sample_index][x/y/z][atom_index]
  std::vector<Eigen::MatrixXd> atom_positions_cart;

  void initialize(Index _n_atoms, Index _jumps_per_position_sample,
                  Index _max_n_position_samples,
                  bool _output_incomplete_samples);

  void insert(Index atom_id, Index n_jumps,
              Eigen::VectorXd const &position_cart, CountType _step,
              CountType _pass, double _time);

 private:
  /// \brief Update the number of complete samples
  ///
  /// \param n_samples The number of position samples that have been taken for
  ///     most recent atom (before the current sample)
  void _update_n_complete_samples(Index n_samples);
};

/// \brief Histogram of a discrete integral vector variable
///
/// - The histogram is stored as a map of counts, where the key is the vector
///   value and the value is the count.
/// - A maximum size restricts the number of unique values that can be stored.
/// - If the maximum size is reached, a flag is set and the count for new key
///   values are stored in the `out_of_range_count`.
class DiscreteVectorIntHistogram {
 public:
  DiscreteVectorIntHistogram(
      std::vector<std::string> const &_component_names,
      std::vector<Index> _shape, Index _max_size = 10000,
      std::optional<
          std::map<Eigen::VectorXl, std::string, LexicographicalCompare>>
          _value_labels = std::nullopt);

  /// \brief Return the shape of the quantity
  std::vector<Index> const &shape() const { return this->m_shape; }

  /// \brief Return the component names of the quantity
  std::vector<std::string> const &component_names() const {
    return this->m_component_names;
  }

  /// \brief Return the number of unique values in the histogram
  Index size() const { return this->m_count.size(); }

  /// \brief Return the maximum number of unique values that can be stored
  Index max_size() const { return this->m_max_size; }

  /// \brief Return true if the maximum number of unique values has been reached
  bool max_size_exceeded() const { return this->m_max_size_exceeded; }

  /// \brief Insert a value into the histogram, with an optional weight
  void insert(Eigen::VectorXl const &value, double weight = 1.0);

  /// \brief Return the sum of bin counts + out-of-range counts
  double sum() const;

  /// \brief Return the values as a vector
  std::vector<Eigen::VectorXl> values() const;

  /// \brief Return the count as a vector
  std::vector<double> count() const;

  /// The number of values (total weight) that was not binned because the
  /// max_size was exceeded
  double out_of_range_count() const { return this->m_out_of_range_count; }

  /// \brief Return the count as a vector containing fractions of the sum
  std::vector<double> fraction() const;

  /// Optional labels for each value in the histogram
  std::optional<
      std::map<Eigen::VectorXl, std::string, LexicographicalCompare>> const &
  value_labels() const {
    return this->m_value_labels;
  }

  /// The number of values (total weight) in each bin
  std::map<Eigen::VectorXl, double, LexicographicalCompare> const &
  value_counts() const {
    return this->m_count;
  }

 private:
  /// \brief Shape of quantity, with column-major unrolling
  ///
  /// Scalar: [], Vector: [n], Matrix: [m, n], etc.
  std::vector<Index> m_shape;

  /// \brief A name for each component of the resulting Eigen::VectorXd
  ///
  /// Can be string representing an index (i.e "0", "1", "2", etc.) or can
  /// be a descriptive string (i.e. "Mg", "Va", "O", etc.)
  std::vector<std::string> m_component_names;

  /// The maximum number of key values to store
  /// - If the max_size is reached, a flag is set and new key values are ignored
  Index m_max_size;

  /// Flag to indicate that the max_size has been reached
  bool m_max_size_exceeded;

  /// The number of values (total weight) in each bin
  std::map<Eigen::VectorXl, double, LexicographicalCompare> m_count;

  /// The number of values (total weight) that was not binned because the
  /// max_size was exceeded
  double m_out_of_range_count;

  /// Optional labels for each value in the histogram
  std::optional<std::map<Eigen::VectorXl, std::string, LexicographicalCompare>>
      m_value_labels;
};

class DiscreteVectorFloatHistogram {
 public:
  DiscreteVectorFloatHistogram(
      std::vector<std::string> const &_component_names,
      std::vector<Index> _shape, double _tol, Index _max_size = 10000,
      std::optional<
          std::map<Eigen::VectorXd, std::string, FloatLexicographicalCompare>>
          _value_labels = std::nullopt);

  /// \brief Return the shape of the quantity
  std::vector<Index> const &shape() const { return this->m_shape; }

  /// \brief Return the component names of the quantity
  std::vector<std::string> const &component_names() const {
    return this->m_component_names;
  }

  /// \brief Return the number of unique values in the histogram
  Index size() const { return this->m_count.size(); }

  /// \brief Return the maximum number of unique values that can be stored
  Index max_size() const { return this->m_max_size; }

  /// \brief Return true if the maximum number of unique values has been reached
  bool max_size_exceeded() const { return this->m_max_size_exceeded; }

  /// \brief Return the tolerance for comparing floating point values
  double tol() const { return this->m_count.key_comp().tol; }

  /// \brief Insert a value into the histogram, with an optional weight
  void insert(Eigen::VectorXd const &value, double weight = 1.0);

  /// \brief Return the sum of bin counts + out-of-range counts
  double sum() const;

  /// \brief Return the values as a vector
  std::vector<Eigen::VectorXd> values() const;

  /// \brief Return the count as a vector
  std::vector<double> count() const;

  /// The number of values (total weight) that was not binned because the
  /// max_size was exceeded
  double out_of_range_count() const { return this->m_out_of_range_count; }

  /// \brief Return the count as a vector containing fractions of the sum
  std::vector<double> fraction() const;

  /// Optional labels for each value in the histogram
  std::optional<std::map<Eigen::VectorXd, std::string,
                         FloatLexicographicalCompare>> const &
  value_labels() const {
    return this->m_value_labels;
  }

  /// The number of values (total weight) in each bin
  std::map<Eigen::VectorXd, double, FloatLexicographicalCompare> const &
  value_counts() const {
    return this->m_count;
  }

 private:
  /// \brief Shape of quantity, with column-major unrolling
  ///
  /// Scalar: [], Vector: [n], Matrix: [m, n], etc.
  std::vector<Index> m_shape;

  /// \brief A name for each component of the resulting Eigen::VectorXd
  ///
  /// Can be string representing an index (i.e "0", "1", "2", etc.) or can
  /// be a descriptive string (i.e. "Mg", "Va", "O", etc.)
  std::vector<std::string> m_component_names;

  /// The maximum number of key values to store
  Index m_max_size;

  /// Flag to indicate that the max_size has been reached
  bool m_max_size_exceeded;

  /// The number of values (total weight) in each bin
  std::map<Eigen::VectorXd, double, FloatLexicographicalCompare> m_count;

  /// The number of values (total weight) that was not binned because the
  /// max_size was exceeded
  double m_out_of_range_count;

  /// Optional labels for each value in the histogram
  std::optional<
      std::map<Eigen::VectorXd, std::string, FloatLexicographicalCompare>>
      m_value_labels;
};

/// \brief Histogram of a single continuous variable with fixed bin width
///
/// - The histogram is stored as a vector of counts, where the index of the
///     count corresponds to the bin number.
/// - The bin number is calculated as `(value - begin) / bin_width`, so the
///   range for bin `i` is [begin, begin + i*bin_width).
/// - If the value is less than `begin`, the bins are reset to prepend the
///     necessary number of bins to the beginning of the histogram.
class Histogram1D {
 public:
  /// \brief Constructor
  Histogram1D(double _initial_begin, double _bin_width, bool _is_log,
              Index _max_size = 10000);

  /// The width of each bin in the histogram
  double bin_width() const { return this->m_bin_width; }

  /// If true, the histogram (including bin width and `begin` value)
  /// is in log space (using base 10)
  bool is_log() const { return this->m_is_log; }

  /// \brief Return the number of bins in the histogram
  Index size() const { return this->m_count.size(); }

  /// The maximum number of bins to store
  Index max_size() const { return this->m_max_size; }

  /// Flag to indicate that the max_size has been reached
  bool max_size_exceeded() const { return this->m_max_size_exceeded; }

  /// The first bin is for the range `[begin, begin + bin_width)`
  double begin() const { return this->m_begin; }

  /// The number of values (total weight) in each bin
  std::vector<double> const &count() const { return this->m_count; }

  /// The number of values (total weight) that was not binned because the
  /// max_size was exceeded
  double out_of_range_count() const { return this->m_out_of_range_count; }

  /// \brief Insert a value into the histogram, with an optional weight
  void insert(double value, double weight = 1.0);

  /// \brief Insert the log of a value into a log space histogram directly,
  ///     with an optional weight.
  void insert_log_value(double log_value, double weight = 1.0);

  /// \brief Return the coordinates of the beginning of each bin range
  std::vector<double> bin_coords() const;

  /// \brief Return the sum of bin counts + out-of-range counts
  double sum() const;

  /// \brief Return the count as a probability density, such that the area
  ///     under the histogram integrates to 1 (if no out-of-range count)
  std::vector<double> density() const;

  /// \brief Merge another histogram into this one
  void merge(Histogram1D const &other);

 private:
  /// \brief Insert a value into the histogram, with an optional weight
  void _insert(double value, double weight);

  /// \brief Reset histogram bins if this is the first value being added,
  /// or if `value` is less than `begin`
  void _reset_bins(double value);

  /// The type used to store counts in the histogram
  double m_initial_begin;

  /// The width of each bin in the histogram
  double m_bin_width;

  /// If true, the histogram (including bin width and `begin` value)
  /// is in log space (using base 10)
  bool m_is_log;

  /// The maximum number of bins to store
  Index m_max_size;

  /// Flag to indicate that the max_size has been reached
  bool m_max_size_exceeded;

  /// The first bin is for the range `[begin, begin + bin_width)`
  double m_begin;

  /// The number of values (total weight) in each bin
  std::vector<double> m_count;

  /// The number of values (total weight) that was not binned because the
  /// max_size was exceeded
  double m_out_of_range_count;
};

/// \brief Combine 1D histograms from multiple partitions into a single 1d
/// histogram
Histogram1D combine(std::vector<Histogram1D> const &histograms);

/// \brief Histogram of a single continuous variable with fixed bin width
///
/// - The histogram is stored as a vector of counts, where the index of the
///     count corresponds to the bin number.
/// - The bin number is calculated as `(value - begin) / bin_width`, so the
///   range for bin `i` is [begin, begin + i*bin_width).
/// - If the value is less than `begin`, the bins are reset to prepend the
///   necessary number of bins to the beginning of the histogram.
/// - The histogram is partitioned into multiple individual histograms, where
///   each partition corresponds to a different value of the partition index.
/// - A combined histogram is generated on request.
/// - When the combined or individual histograms are requested, the combined
///   histogram is generated if it is not up to date, and the bin ranges for
///   each partition are updated to match the bin ranges of the combined
///   histogram.
class PartitionedHistogram1D {
 public:
  /// \brief Constructor
  PartitionedHistogram1D(std::vector<std::string> const &_partion_names,
                         double _initial_begin, double _bin_width, bool _is_log,
                         Index _max_size = 10000);

  /// \brief Return the names of the partitions
  std::vector<std::string> const &partition_names() const {
    return this->m_partition_names;
  }

  /// \brief Access the individual histograms
  std::vector<Histogram1D> const &histograms() const {
    if (m_histograms.size() == 1) {
      return this->m_histograms;
    }
    if (!m_up_to_date) {
      _make_combined_histogram();
    }
    return this->m_histograms;
  }

  /// \brief Insert a value into an individual histogram, with an optional
  /// weight
  void insert(int partition, double value, double weight = 1.0) {
    m_up_to_date = false;
    if (partition < 0 || partition >= m_histograms.size()) {
      throw std::runtime_error("Partition index out of range");
    }
    if (!std::isfinite(value)) {
      std::stringstream msg;
      msg << "Error in PartitionedHistogram1D::insert: "
          << "for partition=\"" << m_partition_names[partition] << "\" "
          << "value (=" << value << ") is not finite.";
      throw std::runtime_error(msg.str());
    }
    m_histograms[partition].insert(value, weight);
  }

  /// \brief Access the combined histogram
  ///
  /// If there is only a single partition, that histogram is returned directly.
  Histogram1D const &combined_histogram() const {
    if (m_histograms.size() == 1) {
      return m_histograms.front();
    }
    if (!m_up_to_date) {
      _make_combined_histogram();
    }
    return this->m_combined_histogram.value();
  }

 private:
  /// \brief Make the combined histogram from the partitioned histograms
  void _make_combined_histogram() const;

  /// The names of the partitions
  std::vector<std::string> m_partition_names;

  /// The histograms for each partition
  std::vector<Histogram1D> m_histograms;

  /// If true, the combined histogram is up to date
  mutable bool m_up_to_date;

  /// The combined histogram
  mutable std::optional<Histogram1D> m_combined_histogram;
};

/// \brief A function to be evaluated during a Monte Carlo calculation after
/// every event selection
///
/// - Can request to have the event state of the selected event calculated
/// - Can be customized to only evaluate after particular event types are
///   selected.
/// - Will be evaluated before selected event data collection functions are
///   evaluated, in user-specified order (defaults to lexicographical by
///   function name).
///
class GenericSelectedEventFunction {
 public:
  /// \brief Constructor - default component names
  GenericSelectedEventFunction(std::string _name, std::string _description,
                               bool _requires_event_state,
                               std::function<void()> _function,
                               std::function<bool()> _has_value_function,
                               Index _order);

  /// \brief Function name (and quantity to be sampled)
  std::string name;

  /// \brief Description of the function
  std::string description;

  /// \brief Does the function require the event state?
  bool requires_event_state;

  /// \brief The function to be evaluated
  std::function<void()> function;

  /// \brief Returns true if the function has a value
  std::function<bool()> has_value_function;

  /// \brief The order in which the function should be evaluated (ties are
  /// broken by function name)
  Index order;

  /// \brief Evaluates `function`
  void operator()() const { return function(); }

  /// \brief Evaluates `has_value_function`
  bool has_value() const { return has_value_function(); }
};

struct SelectedEventFunctions {
  std::map<std::string, GenericSelectedEventFunction> generic_functions;
  std::map<std::string, DiscreteVectorIntHistogramFunction>
      discrete_vector_int_functions;
  std::map<std::string, DiscreteVectorFloatHistogramFunction>
      discrete_vector_float_functions;
  std::map<std::string, PartitionedHistogramFunction<double>>
      continuous_1d_functions;

  void insert(GenericSelectedEventFunction f) {
    this->generic_functions.emplace(f.name, f);
  }

  void insert(DiscreteVectorIntHistogramFunction f) {
    this->discrete_vector_int_functions.emplace(f.name, f);
  }

  void insert(DiscreteVectorFloatHistogramFunction f) {
    this->discrete_vector_float_functions.emplace(f.name, f);
  }

  void insert(PartitionedHistogramFunction<double> f) {
    this->continuous_1d_functions.emplace(f.name, f);
  }

  void reset() {
    generic_functions.clear();
    discrete_vector_int_functions.clear();
    discrete_vector_float_functions.clear();
    continuous_1d_functions.clear();
  }
};

struct SelectedEventFunctionParams {
  // -- Jump Correlations --------------------------------------------

  /// Optional parameters for collecting correlations data
  std::optional<CorrelationsDataParams> correlations_data_params;

  // -- Histograms ----------------------------------------------

  // -- Which histograms to collect / generic functions to evaluate --

  /// The data to collect and construct histograms for or generic functions to
  /// evaluate
  std::vector<std::string> function_names;

  // -- Allow overriding default values for the generic functions --

  std::map<std::string, Index> order;

  // -- The following allow overriding default values for the histograms --

  /// Tolerances for comparing floating point values for discrete float values,
  /// by function name
  std::map<std::string, double> tol;

  /// Bin width for continuous variables, by function name
  std::map<std::string, double> bin_width;

  /// Initial value for continuous variables, by function name
  std::map<std::string, double> initial_begin;

  /// If true, the histogram (including bin width and `begin` value)
  /// is in log space (using base 10), by function name
  std::map<std::string, bool> is_log;

  /// Maximum number of bins / discrete values, by function name
  std::map<std::string, Index> max_size;

  SelectedEventFunctionParams &evaluate(std::string name,
                                        std::optional<Index> order) {
    this->function_names.push_back(name);
    if (order.has_value()) {
      this->order[name] = order.value();
    }
    return *this;
  }

  SelectedEventFunctionParams &collect(std::string name,
                                       std::optional<double> tol,
                                       std::optional<double> bin_width,
                                       std::optional<double> initial_begin,
                                       std::optional<std::string> spacing,
                                       std::optional<Index> max_size) {
    this->function_names.push_back(name);
    if (tol.has_value()) {
      this->tol[name] = tol.value();
    }
    if (bin_width.has_value()) {
      this->bin_width[name] = bin_width.value();
    }
    if (initial_begin.has_value()) {
      this->initial_begin[name] = initial_begin.value();
    }
    if (spacing.has_value()) {
      std::string _spacing = spacing.value();
      if (_spacing == "log") {
        this->is_log[name] = true;
      } else if (_spacing == "linear") {
        this->is_log[name] = false;
      } else {
        throw std::runtime_error(
            "Error in SelectedEventFunctionParams::set: spacing must be "
            "'log' or 'linear'");
      }
    }
    if (max_size.has_value()) {
      this->max_size[name] = max_size.value();
    }
    return *this;
  }

  SelectedEventFunctionParams &do_not_collect(std::string name) {
    this->function_names.erase(std::remove(this->function_names.begin(),
                                           this->function_names.end(), name),
                               this->function_names.end());
    this->order.erase(name);
    this->tol.erase(name);
    this->bin_width.erase(name);
    this->initial_begin.erase(name);
    this->is_log.erase(name);
    this->max_size.erase(name);
    return *this;
  }

  void reset() {
    correlations_data_params.reset();
    function_names.clear();
    order.clear();
    tol.clear();
    bin_width.clear();
    initial_begin.clear();
    is_log.clear();
    max_size.clear();
  }
};

/// \brief Statistics for events that have been selected
struct SelectedEventData {
  /// \brief Hop correlations data (not basis function correlations)
  std::optional<CorrelationsData> correlations_data;

  /// \brief Histogram of discrete integer vector variables
  std::map<std::string, DiscreteVectorIntHistogram>
      discrete_vector_int_histograms;

  /// \brief Histogram of discrete floating point vector variables
  std::map<std::string, DiscreteVectorFloatHistogram>
      discrete_vector_float_histograms;

  /// \brief Histograms of continuous variables
  ///
  /// - The key is the name of the function that was evaluated
  /// - The value is a PartitionedHistogram1D.
  /// - For example, the partition value may always be 1, or may correspond to
  ///   the event type, or the event equivalent index, to save a single
  ///   histogram of Ekra or to save separate histograms by event type, or
  ///   by event orientation.
  ///
  std::map<std::string, PartitionedHistogram1D> continuous_1d_histograms;

  void reset() {
    correlations_data.reset();
    discrete_vector_int_histograms.clear();
    discrete_vector_float_histograms.clear();
    continuous_1d_histograms.clear();
  }
};

struct MonteCounter;

struct SelectedEventDataCollector {
  /// \brief Constructor
  SelectedEventDataCollector(
      monte::SelectedEventFunctions const &selected_event_data_functions,
      monte::SelectedEventFunctionParams const &selected_event_data_params,
      std::shared_ptr<monte::SelectedEventData> selected_event_data);

  // -- Must not be null, else constructor will throw --
  std::shared_ptr<monte::SelectedEventData> data;

  /// \brief If True, the event state must be calculated for at least one of
  ///     the requested functions
  bool requires_event_state;

  // -- Discrete vector int histograms --

  std::vector<DiscreteVectorIntHistogramFunction> discrete_vector_int_f;
  std::vector<DiscreteVectorIntHistogram *> discrete_vector_int_hist;

  void collect_vector_int_data();

  // -- Discrete vector float histograms --

  std::vector<DiscreteVectorFloatHistogramFunction> discrete_vector_float_f;
  std::vector<DiscreteVectorFloatHistogram *> discrete_vector_float_hist;

  void collect_vector_float_data();

  // -- Continuous 1d histograms --

  std::vector<PartitionedHistogramFunction<double>> continuous_1d_f;
  std::vector<PartitionedHistogram1D *> continuous_1d_hist;

  void collect_continuous_1d_data();

  // -- Generic functions --

  std::vector<GenericSelectedEventFunction> generic_f;
  void evaluate_generic_functions();

  // -- Collect selected event data && evaluate generic functions --
  void collect();
};

}  // namespace monte
}  // namespace CASM

#endif
