#include "casm/monte/sampling/io/json/SamplingParams_json_io.hh"

#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/InputParser_impl.hh"
#include "casm/monte/sampling/SamplingParams.hh"

namespace CASM {
namespace monte {

/// \brief Construct SamplingParams from JSON
///
/// Expected:
///   sample_by: string (optional, default=(depends on calculation type))
///     What to count when determining when to sample the Monte Carlo state.
///     One of "pass", "step", "time" (not valid for all Monte Carlo methods).
///     A "pass" is a number of steps, equal to one step per site with degrees
///     of freedom (DoF).
///
///   spacing: string (optional, default="linear")
///     The spacing of samples in the specified `"period"`. One of "linear"
///     or "log". Custom functions must be specified using the Python interface.
///
///     For "linear" spacing, the n-th (n=0,1,2,...) sample will be taken when:
///
///         sample/pass = round( begin + period * n )
///                time = begin + period * n
///
///     For "log" spacing, the n-th sample will be taken when:
///
///         sample/pass = round( begin + base ^ (n + shift)
///                time = begin + base ^ (n + shift)
///
///   begin: number (optional, default=0.0)
///     The number of pass/step or amount of time at which to begin
///     sampling.
///
///   period: number (required)
///     A number of pass/step or amount of time.
///
///   base: number (optional, default=10^(1/10))
///     The base of logarithmic spaced sampling.
///
///   shift: number (optional, default=10.0)
///     Used with `"spacing": "log"`.
///
///   stochastic_sample_period: bool (optional, default=false)
///     If true, then instead of setting the sample time / count
///     deterministically, use the sampling period to determine the
///     sampling rate and determine the next sample time / count
///     stochastically with equivalent mean rate.
///
///   quantities: array of string (optional)
///     Specifies which quantities will be sampled. Options depend on the
///     type of Monte Carlo calculation and should be keys in the
///     sampling_functions map.
///
///   json_quantities: array of string (optional)
///     Specifies which JSON quantities will be sampled. Options depend on the
///     type of Monte Carlo calculation and should be keys in the
///     json_sampling_functions map.
///
///   sample_trajectory: bool (optional, default=false)
///     If true, request that the entire configuration is saved each time
///     samples are taken.
///
void parse(InputParser<SamplingParams> &parser,
           std::set<std::string> const &sampling_function_names,
           std::set<std::string> const &json_sampling_function_names,
           bool time_sampling_allowed) {
  SamplingParams sampling_params;

  // "sample_by"
  std::unique_ptr<std::string> sample_mode =
      parser.require<std::string>("sample_by");
  if (sample_mode == nullptr) {
    return;
  }
  if (*sample_mode == "pass") {
    sampling_params.sample_mode = SAMPLE_MODE::BY_PASS;
  } else if (*sample_mode == "step") {
    sampling_params.sample_mode = SAMPLE_MODE::BY_STEP;
  } else if (time_sampling_allowed && *sample_mode == "time") {
    sampling_params.sample_mode = SAMPLE_MODE::BY_TIME;
  } else {
    if (time_sampling_allowed) {
      parser.insert_error("sample_by",
                          "Error: \"sample_mode\" must be one of \"pass\", "
                          "\"step\", or \"time\".");
    } else {
      parser.insert_error(
          "sample_by",
          "Error: \"sample_mode\" must be one of \"pass\" or \"step\".");
    }
  }

  // "spacing"
  std::string sample_method = "linear";
  parser.optional(sample_method, "spacing");
  if (sample_method == "linear") {
    sampling_params.sample_method = SAMPLE_METHOD::LINEAR;
  } else if (sample_method == "log") {
    sampling_params.sample_method = SAMPLE_METHOD::LOG;
  } else {
    parser.insert_error(
        "spacing", "Error: \"spacing\" must be one of \"linear\", \"log\".");
  }

  // "begin"
  sampling_params.begin = 0.0;
  parser.optional(sampling_params.begin, "begin");

  // "period"
  parser.require(sampling_params.period, "period");
  if (sampling_params.sample_method == SAMPLE_METHOD::LOG &&
      sampling_params.period <= 1.0) {
    parser.insert_error(
        "period", "Error: For \"spacing\"==\"log\", \"period\" must > 1.0.");
  }
  if (sampling_params.sample_method == SAMPLE_METHOD::LINEAR &&
      sampling_params.period <= 0.0) {
    parser.insert_error(
        "period", "Error: For \"spacing\"==\"log\", \"period\" must > 0.0.");
  }

  // "base"
  sampling_params.base = std::pow(10.0, 1.0 / 10.0);
  parser.optional(sampling_params.base, "base");

  // "shift"
  sampling_params.shift = 10.0;
  parser.optional(sampling_params.shift, "shift");

  // "stochastic_sample_period"
  sampling_params.stochastic_sample_period = false;
  parser.optional(sampling_params.stochastic_sample_period,
                  "stochastic_sample_period");

  // "quantities"
  parser.optional(sampling_params.sampler_names, "quantities");
  for (std::string name : sampling_params.sampler_names) {
    if (!sampling_function_names.count(name)) {
      std::stringstream msg;
      msg << "Error: \"" << name << "\" is not a sampling option.";
      parser.insert_error("quantities", msg.str());
    }
  }

  // "json_quantities"
  parser.optional(sampling_params.json_sampler_names, "json_quantities");
  for (std::string name : sampling_params.json_sampler_names) {
    if (!json_sampling_function_names.count(name)) {
      std::stringstream msg;
      msg << "Error: \"" << name << "\" is not a JSON sampling option.";
      parser.insert_error("json_quantities", msg.str());
    }
  }

  // "sample_trajectory"
  parser.optional(sampling_params.do_sample_trajectory, "sample_trajectory");

  sampling_params.do_sample_time = time_sampling_allowed;

  if (parser.valid()) {
    parser.value = std::make_unique<SamplingParams>(sampling_params);
  }
}

/// \brief Convert SamplingParams to JSON
jsonParser &to_json(SamplingParams const &sampling_params, jsonParser &json) {
  json.put_obj();

  // "sample_by"
  if (sampling_params.sample_mode == SAMPLE_MODE::BY_PASS) {
    json["sample_by"] = "pass";
  } else if (sampling_params.sample_mode == SAMPLE_MODE::BY_STEP) {
    json["sample_by"] = "step";
  } else if (sampling_params.sample_mode == SAMPLE_MODE::BY_TIME) {
    json["sample_by"] = "time";
  } else {
    throw std::runtime_error(
        "Error converting SamplingParams to json: invalid sample_mode");
  }

  // "spacing"
  if (sampling_params.sample_method == SAMPLE_METHOD::LINEAR) {
    json["spacing"] = "linear";

    auto check_if_integral = [&](double value, std::string key) {
      if (std::abs(value - std::round(value)) < CASM::TOL) {
        json[key] = static_cast<CountType>(std::round(value));
      } else {
        json[key] = value;
      }
    };
    check_if_integral(sampling_params.begin, "begin");
    check_if_integral(sampling_params.period, "period");
  } else if (sampling_params.sample_method == SAMPLE_METHOD::LOG) {
    json["spacing"] = "log";
    json["begin"] = sampling_params.begin;
    json["base"] = sampling_params.base;
    json["shift"] = sampling_params.shift;
  } else {
    throw std::runtime_error(
        "Error converting SamplingParams to json: invalid sample_method");
  }

  if (sampling_params.stochastic_sample_period == true) {
    json["stochastic_sample_period"] = sampling_params.stochastic_sample_period;
  }
  if (!sampling_params.sampler_names.empty()) {
    json["quantities"] = sampling_params.sampler_names;
  }
  if (!sampling_params.json_sampler_names.empty()) {
    json["json_quantities"] = sampling_params.json_sampler_names;
  }

  if (sampling_params.do_sample_trajectory == true) {
    json["sample_trajectory"] = sampling_params.do_sample_trajectory;
  }

  return json;
}

}  // namespace monte
}  // namespace CASM
