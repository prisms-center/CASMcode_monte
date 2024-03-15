#ifndef CASM_monte_BasicStatistics
#define CASM_monte_BasicStatistics

#include <optional>

#include "casm/monte/definitions.hh"

namespace CASM {
class jsonParser;

namespace monte {

/// \brief Basic version of a StatisticsType
///
/// Interface that must be implemented to allow auto convergence checking and
/// JSON output:
/// - double get_calculated_precision(StatisticsType const &stats);
/// - template<> CalcStatisticsFunction<StatisticsType>
/// default_calc_statistics_f();
/// - void append_statistics_to_json_arrays(std::optional<StatisticsType> const
/// &stats, jsonParser &json);
/// - void to_json(StatisticsType const &stats, jsonParser &json);
///
struct BasicStatistics {
  BasicStatistics()
      : mean(0.0), calculated_precision(std::numeric_limits<double>::max()) {}

  /// \brief Mean of property (<X>)
  double mean;

  /// \brief Calculated absolute precision in <X>
  ///
  /// Notes:
  /// - See `convergence_check` function for calculation details
  double calculated_precision;
};

inline double get_calculated_precision(BasicStatistics const &stats) {
  return stats.calculated_precision;
}

inline double get_calculated_relative_precision(BasicStatistics const &stats) {
  return std::abs(stats.calculated_precision / stats.mean);
}

double autocorrelation_factor(Eigen::VectorXd const &observations,
                              double increment = 1.0);

Eigen::VectorXd resample(Eigen::VectorXd const &observations,
                         Eigen::VectorXd const &sample_weight,
                         double sample_weight_sum, Index n_equally_spaced);

struct BasicStatisticsCalculator {
  BasicStatisticsCalculator(double _confidence = 0.95, Index _method = 1,
                            Index _n_resamples = 10000);

  /// \brief Confidence level used to calculate error interval
  double confidence;

  /// \brief Method selection
  ///
  /// Options when sample_weight.size() != 0:
  /// 1) Calculate weighted sample variance directly from weighted samples
  ///    and only autocorrelation factor (1+rho)/(1-rho) from resampled
  ///    observations
  /// 2) Calculate all statistics from resampled observations
  Index method;

  /// \brief Number of resamples when calculating autocovariance of weighted
  ///     observations
  Index n_resamples;

  /// \Brief Calculate statistics for a range of observations
  BasicStatistics operator()(Eigen::VectorXd const &observations) const;

  /// \brief Calculate statistics for a range of weighted observations
  BasicStatistics operator()(Eigen::VectorXd const &observations,
                             Eigen::VectorXd const &sample_weight) const;
};

template <typename StatisticsType>
CalcStatisticsFunction<StatisticsType> default_statistics_calculator();

template <>
inline CalcStatisticsFunction<BasicStatistics>
default_statistics_calculator<BasicStatistics>() {
  return BasicStatisticsCalculator();
}

void append_statistics_to_json_arrays(
    std::optional<BasicStatistics> const &stats, jsonParser &json);

void to_json(BasicStatistics const &stats, jsonParser &json);

}  // namespace monte
}  // namespace CASM

#endif
