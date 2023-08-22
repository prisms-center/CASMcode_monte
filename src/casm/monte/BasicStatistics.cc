#include "casm/monte/BasicStatistics.hh"

#include <cmath>
#include <iostream>

#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/misc/math.hh"

namespace CASM {
namespace monte {

/// \brief Calculate the autocorrelaction factor
///
/// autocorrelation_factor = (1.0 + rho) / (1.0 - rho)
/// CoVar(k) = lag-k autocovariance,
/// rho = pow(2.0, (-1.0 / (k_star*increment))),
/// where k_star is the the k where the CoVar(k) <= 0.5 * CoVar(0)
///
/// \param observations Observations
/// \param increment Interval between resampled observations, if observations
///     is resampled from weighted observations. Use 1.0 for non-weighted
///     observations.
///
double autocorrelation_factor(Eigen::VectorXd const &observations,
                              double increment) {
  Index N = observations.size();
  double mean = observations.mean();
  double CoVar0 = variance(observations, mean);

  // if there is essentially no variation, return 1.0
  if (std::abs(CoVar0 / mean) < 1e-8 || CoVar0 == 0.0) {
    return 1.0;
  }

  // simple incremental search for std::abs(cov(i) / CoVar0) <= 0.5
  for (CountType i = 1; i < N; ++i) {
    CountType range_size = N - i;
    double cov = covariance(observations.segment(0, range_size),
                            observations.segment(i, range_size), mean);
    if (std::abs(cov / CoVar0) <= 0.5) {
      double rho = pow(2.0, (-1.0 / (i * increment)));
      return (1.0 + rho) / (1.0 - rho);
    }
  }

  // if could not find:
  return std::numeric_limits<double>::max();
}

Eigen::VectorXd resample(Eigen::VectorXd const &observations,
                         Eigen::VectorXd const &sample_weight,
                         double sample_weight_sum, Index n_equally_spaced) {
  // weighted observations
  // | 0 -------------- | 1 ---- | 2 ------- | 3 ----- |

  // n_resamples equally spaced observations
  // 0    0    0    0    1    1    2    2    3    3    )

  double increment = sample_weight_sum / n_equally_spaced;
  Eigen::VectorXd equally_spaced(n_equally_spaced);
  Index j = 0;
  double W_j = 0.0;
  double W_target;
  for (Index i = 0; i < n_equally_spaced; ++i) {
    W_target = i * increment;
    while (W_j + sample_weight(j) < W_target) {
      W_j += sample_weight(j);
      ++j;
    }
    equally_spaced(i) = observations(j);
  }
  return equally_spaced;
}

/// \brief Constructor
///
/// \param _confidence Confidence level for determining precision
///     in the sample mean. Expected in [0.0, 1.0).
/// \param _method Options: 1 = Calculate weighted sample variance of the
///     mean directly from weighted samples and only autocorrelation factor
///     (1+rho)/(1-rho) from resampled observations; 2 = Calculate all
///     statistics from resampled observations
/// \param _n_resamples Number of resamples to perform when calculating
///     statistics of weighted observations. The approach treats the
///     the weighted observations as a time series which is sampled
///     at equally spaced time intervals and then lag-k covariances are
///     calculated from the resampled observations.
BasicStatisticsCalculator::BasicStatisticsCalculator(double _confidence,
                                                     Index _method,
                                                     Index _n_resamples)
    : confidence(_confidence), method(_method), n_resamples(_n_resamples) {}

/// \brief Calculate statistics for a range of observations
///
/// Precision in the mean is calculated using the algorithm of:
///  Van de Walle and Asta, Modelling Simul. Mater. Sci. Eng. 10 (2002) 521–538.
///
/// The observations are considered converged to the desired precision at a
/// particular confidence level if:
///
///     calculated_precision <= requested_precision,
///
/// where:
/// - calculated_precision = z_alpha*sqrt(var_of_mean),
/// - z_alpha = sqrt(2.0)*inv_erf(1.0-conf)
/// - var_of_mean = (CoVar[0]/observations.size())*((1.0+rho)/(1.0-rho))
/// - CoVar[i] = ( (1.0/(observations.size()-i))*sum_j(j=0:L-i-1,
/// observations(j)*observations(j+1)) ) - sqr(observations.mean());
/// - rho = pow(2.0, -1.0/i), using min i such that CoVar[i]/CoVar[0] < 0.5
///
/// \param observations An Eigen::VectorXd of observations. Should only include
///     samples after the calculation has equilibrated.
///
BasicStatistics BasicStatisticsCalculator::operator()(
    Eigen::VectorXd const &observations) const {
  if (observations.size() == 0) {
    throw std::runtime_error(
        "Error in BasicStatisticsCalculator: observations.size()==0");
  }
  CountType N = observations.size();

  BasicStatistics stats;
  stats.mean = observations.mean();

  double CoVar0 = variance(observations, stats.mean);
  double f_autocorr = autocorrelation_factor(observations);
  double f_confidence = sqrt(2.0) * approx_erf_inv(this->confidence);
  stats.calculated_precision = f_confidence * sqrt(f_autocorr * CoVar0 / N);

  return stats;
}

/// \brief Calculate statistics for a range of weighted observations
///
/// The method treats the the weighted observations as a time series
/// which is resampled at equally spaced time intervals determined by
/// the n_resamples parameter. Then statistics are determined as if
/// the observations are not weighted.
///
/// \param observations An Eigen::VectorXd of observations. Should only include
///     samples after the calculation has equilibrated.
/// \param sample_weight Sample weights associated with observations.
///
BasicStatistics BasicStatisticsCalculator::operator()(
    Eigen::VectorXd const &observations,
    Eigen::VectorXd const &sample_weight) const {
  if (observations.size() == 0) {
    throw std::runtime_error(
        "Error in BasicStatisticsCalculator: observations.size()==0");
  }

  // Unweighted observations
  if (sample_weight.size() == 0) {
    return (*this)(observations);
  }

  // Weighted observations
  if (observations.size() != sample_weight.size()) {
    throw std::runtime_error(
        "Error in BasicStatisticsCalculator: observations.size() != "
        "sample_weight.size()");
  }

  double W = sample_weight.sum();
  double increment = W / this->n_resamples;

  Eigen::VectorXd equally_spaced =
      resample(observations, sample_weight, W, this->n_resamples);

  if (method == 1) {
    BasicStatistics stats;
    stats.mean = observations.dot(sample_weight) / W;

    double weighted_var =
        weighted_variance(observations, stats.mean, sample_weight, W);
    double f_autocorr = autocorrelation_factor(equally_spaced, increment);
    double f_confidence = sqrt(2.0) * approx_erf_inv(confidence);
    stats.calculated_precision =
        f_confidence * sqrt(f_autocorr * weighted_var / W);

    return stats;
  } else if (method == 2) {
    return (*this)(equally_spaced);
  } else {
    throw std::runtime_error(
        "Error in BasicStatisticsCalculator: invalid method");
  }
}

void append_statistics_to_json_arrays(
    std::optional<BasicStatistics> const &stats, jsonParser &json) {
  auto ensure = [&](std::string key) {
    if (!json.contains(key)) {
      json[key] = jsonParser::array();
    }
  };

  ensure("mean");
  ensure("calculated_precision");

  if (stats.has_value()) {
    json["mean"].push_back(stats->mean);
    json["calculated_precision"].push_back(stats->calculated_precision);
  } else {
    json["mean"].push_back(jsonParser::null());
    json["calculated_precision"].push_back(jsonParser::null());
  }
}

void to_json(BasicStatistics const &stats, jsonParser &json) {
  json.put_obj();
  json["mean"] = stats.mean;
  json["calculated_precision"] = stats.calculated_precision;
}

}  // namespace monte
}  // namespace CASM
