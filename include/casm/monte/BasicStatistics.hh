#ifndef CASM_monte_BasicStatistics
#define CASM_monte_BasicStatistics

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
/// - void append_statistics_to_json_arrays(Statistics const &stats, jsonParser
/// &json);
/// - void to_json(StatisticsType const &stats, jsonParser &json);
///
struct BasicStatistics {
  /// \brief Mean of property (<X>)
  double mean;

  /// \brief Calculated absolute precision in <X>
  ///
  /// Notes:
  /// - See `convergence_check` function for calculation details
  double calculated_precision;
};

double get_calculated_precision(BasicStatistics const &stats) {
  return stats.calculated_precision;
}

/// \brief Calculated statistics for a range of observations
BasicStatistics calc_basic_statistics(Eigen::VectorXd const &observations,
                                      double confidence);

template <>
CalcStatisticsFunction<BasicStatistics>
default_calc_statistics_f<BasicStatistics>() {
  return calc_basic_statistics;
}

void append_statistics_to_json_arrays(BasicStatistics const &stats,
                                      jsonParser &json);

void to_json(BasicStatistics const &stats, jsonParser &json);

}  // namespace monte
}  // namespace CASM

#endif
