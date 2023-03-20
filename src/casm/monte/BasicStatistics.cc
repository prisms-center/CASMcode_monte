#include "casm/monte/BasicStatistics.hh"

#include <cmath>
#include <iostream>

#include "casm/casm_io/json/jsonParser.hh"

namespace CASM {
namespace monte {

namespace {  // anonymous

double _covariance(const Eigen::VectorXd &X, const Eigen::VectorXd &Y,
                   double mean) {
  Eigen::VectorXd vmean = Eigen::VectorXd::Ones(X.size()) * mean;
  return (X - vmean).dot(Y - vmean) / X.size();
}

/// \brief Error function inverse
///
/// Notes:
/// - From "A handy approximation for the error function and its inverse" by
///   Sergei Winitzk
/// - Maximum relative error is about 0.00013.
double _approx_erf_inv(double x) {
  const double one = 1.0;
  const double PI = 3.141592653589793238463;
  const double a = 0.147;

  double sgn = (x < 0.0) ? -one : one;
  double b = std::log((one - x) * (one + x));
  double c = 2.0 / (PI * a) + b * 0.5;
  double d = b / a;
  return sgn * std::sqrt(std::sqrt(c * c - d) - c);
}

/// \brief Try to find rho = pow(2.0, -1.0/i), using min i such that
/// CoVar[i]/CoVar[0] <= 0.5
///
/// \returns std::tuple<bool, double, double> : (found_rho?, rho, CoVar[0])
///
std::tuple<bool, double, double> _calc_rho(Eigen::VectorXd const &observations,
                                           double mean, double squared_norm) {
  CountType N = observations.size();
  // double CoVar0 = squared_norm / N - mean * mean;
  double CoVar0 = _covariance(observations, observations, mean);

  // if there is essentially no variation, return rho(l==1)
  if (std::abs(CoVar0 / mean) < 1e-8 || CoVar0 == 0.0) {
    CoVar0 = 0.0;
    return std::make_tuple(true, pow(2.0, -1.0 / 1), CoVar0);
  }

  // simple incremental search for now, could try bracketing / bisection
  for (CountType i = 1; i < N; ++i) {
    CountType range_size = N - i;

    double cov = _covariance(observations.segment(0, range_size),
                             observations.segment(i, range_size), mean);

    if (std::abs(cov / CoVar0) <= 0.5) {
      return std::make_tuple(true, pow(2.0, (-1.0 / i)), CoVar0);
    }
  }

  // if could not find:
  return std::make_tuple(false, 0.0, CoVar0);
}

double autocorrelation_factor(Eigen::VectorXd const &observations) {
  Index N = observations.size();
  double mean = observations.mean();
  double CoVar0 = _covariance(observations, observations, mean);

  // if there is essentially no variation, use rho(l==1)
  if (std::abs(CoVar0 / mean) < 1e-8 || CoVar0 == 0.0) {
    double rho = pow(2.0, -1.0 / 1);
    return (1.0 + rho) / (1.0 - rho);
  }

  // simple incremental search for std::abs(cov(i) / CoVar0) <= 0.5
  for (CountType i = 1; i < N; ++i) {
    CountType range_size = N - i;
    double cov = _covariance(observations.segment(0, range_size),
                             observations.segment(i, range_size), mean);
    if (std::abs(cov / CoVar0) <= 0.5) {
      double rho = pow(2.0, (-1.0 / i));
      return (1.0 + rho) / (1.0 - rho);
    }
  }

  // if could not find:
  return std::numeric_limits<double>::max();
}

double autocorrelation_factor(Eigen::VectorXd const &observations,
                              Eigen::VectorXd const &sample_weight,
                              Index n_equally_spaced) {
  // weighted observations
  // | 0 -------------- | 1 ---- | 2 ------- | 3 ----- |

  // n_resamples equally spaced observations
  // 0    0    0    0    1    1    2    2    3    3    )

  double W = sample_weight.sum();
  double increment = W / n_equally_spaced;
  Eigen::VectorXd equally_spaced(n_equally_spaced);
  Index j = 0;
  double W_j = 0.0;
  double W_target;
  for (Index i = 0; i < n_equally_spaced; ++i) {
    W_target = W * i / n_equally_spaced;
    while (W_j + sample_weight(j) < W_target) {
      W_j += sample_weight(j);
      ++j;
    }
    equally_spaced(i) = observations(j);
  }

  double mean = equally_spaced.mean();
  double CoVar0 = _covariance(equally_spaced, equally_spaced, mean);

  // simple incremental search for std::abs(cov(i) / CoVar0) <= 0.5
  Eigen::VectorXd _cov(n_equally_spaced);
  _cov(0) = CoVar0;
  for (CountType i = 1; i < n_equally_spaced; ++i) {
    CountType range_size = n_equally_spaced - i;
    double cov = _covariance(equally_spaced.segment(0, range_size),
                             equally_spaced.segment(i, range_size), mean);
    _cov(i) = cov;
    if (std::abs(cov / CoVar0) <= 0.5) {
      double rho = std::pow(2.0, (-1.0 / (i * increment)));
      // double tau = -1.0 / std::log(rho);
      return (1.0 + rho) / (1.0 - rho);
    }
  }

  // if could not find:
  return std::numeric_limits<double>::max();
}

}  // namespace

/// \brief Calculated statistics for a range of observations
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
/// \param confidence Desired confidence level
///
/// \returns A Statistics instance
///
BasicStatistics _calc_basic_statistics(Eigen::VectorXd const &observations,
                                       double confidence) {
  if (observations.size() == 0) {
    throw std::runtime_error(
        "Error in default_calc_statistics: observations.size()==0");
  }
  BasicStatistics stats;
  stats.mean = observations.mean();
  double squared_norm = observations.squaredNorm();

  // try to calculate variance taking into account correlations
  bool found_rho;
  double rho;
  double CoVar0;
  std::tie(found_rho, rho, CoVar0) =
      _calc_rho(observations, stats.mean, squared_norm);

  if (!found_rho) {
    stats.calculated_precision = std::numeric_limits<double>::max();
  } else {
    // calculated precision:
    CountType N = observations.size();
    double z_alpha = sqrt(2.0) * _approx_erf_inv(confidence);
    double var_of_mean = (CoVar0 / N) * (1.0 + rho) / (1.0 - rho);
    stats.calculated_precision = z_alpha * sqrt(var_of_mean);
  }

  return stats;
}

/// \param observations An Eigen::VectorXd of observations. Should only include
///     samples after the calculation has equilibrated.
/// \param sample_weight Sample weights (includes samples from equilibration
/// range) \param confidence Desired confidence level
///
/// \returns A Statistics instance

BasicStatistics calc_basic_statistics(Eigen::VectorXd const &observations,
                                      Eigen::VectorXd const &sample_weight,
                                      double confidence) {
  if (sample_weight.size() == 0) {
    return _calc_basic_statistics(observations, confidence);
  } else {
    if (observations.size() != sample_weight.size()) {
      throw std::runtime_error(
          "Error in calc_basic_statistics: observations.size() != "
          "sample_weight.size()");
    }

    Index N_stats = observations.size();
    double W = sample_weight.sum();

    BasicStatistics stats;
    stats.mean = observations.dot(sample_weight) / W;

    double weighted_sample_var_of_mean = 0.0;
    for (Index i = 0; i < observations.size(); ++i) {
      double d = observations(i) - stats.mean;
      weighted_sample_var_of_mean += sample_weight(i) * d * d;
    }
    weighted_sample_var_of_mean /= W;

    Index n_resamples = 10000;
    double f_autocorr =
        autocorrelation_factor(observations, sample_weight, n_resamples);
    double f_confidence = sqrt(2.0) * _approx_erf_inv(confidence);
    stats.calculated_precision =
        f_confidence * sqrt(f_autocorr * weighted_sample_var_of_mean / W);

    return stats;
  }
}

template <>
CalcStatisticsFunction<BasicStatistics>
default_calc_statistics_f<BasicStatistics>() {
  return calc_basic_statistics;
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
