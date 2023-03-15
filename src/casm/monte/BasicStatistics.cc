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
  double CoVar0 = squared_norm / N - mean * mean;

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
BasicStatistics calc_basic_statistics(Eigen::VectorXd const &observations,
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

void append_statistics_to_json_arrays(BasicStatistics const &stats,
                                      jsonParser &json) {
  auto ensure = [&](std::string key) {
    if (!json.contains(key)) {
      json[key] = jsonParser::array();
    }
  };

  ensure("mean");
  json["mean"].push_back(stats.mean);

  ensure("calculated_precision");
  json["calculated_precision"].push_back(stats.calculated_precision);
}

void to_json(BasicStatistics const &stats, jsonParser &json) {
  json.put_obj();
  json["mean"] = stats.mean;
  json["calculated_precision"] = stats.calculated_precision;
}

}  // namespace monte
}  // namespace CASM
