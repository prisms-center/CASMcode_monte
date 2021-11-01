#include "casm/monte/sampling/io/json/SamplingParams_json_io.hh"

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
///     or "log".
///
///     For "linear" spacing, the n-th sample will be taken when:
///
///         count = ceil(begin + (period / n_samples) * (n))
///
///     For "log" spacing, the n-th sample will be taken when:
///
///         count = ceil(begin + period ^ ( (n + shift) / n_samples))
///
///   begin: number (optional, default=0.0)
///     The number of pass/step at which to begin sampling.
///
///   period: number (required)
///     The number of pass/step in which the first `"n_samples"` samples should
///     be taken.
///
///   samples_per_period: number (optional, default=1.0)
///    The number of samples to be taken in the specified `"period"`.
///
///   shift: number (optional, default=0.0)
///     Used with `"spacing": "log"`.
///
///   quantities: array of string (optional)
///     Specifies which quantities will be sampled. Options depend on the
///     type of Monte Carlo calculation and should be keys in the
///     sampling_functions map.
///
///   sample_trajectory: bool (optional, default=false)
///     If true, request that the entire configuration is saved each time
///     samples are taken.
///
template <typename ConfigType>
void parse(InputParser<SamplingParams> &parser,
           StateSamplingFunctionMap<ConfigType> const &sampling_functions,
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
        "sample_method",
        "Error: \"sample_method\" must be one of \"linear\", \"log\".");
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

  // "samples_per_period"
  sampling_params.samples_per_period = 1.0;
  parser.optional(sampling_params.samples_per_period, "samples_per_period");

  // "quantities"
  parser.optional(sampling_params.sampler_names, "quantities");
  for (std::string name : sampling_params.sampler_names) {
    if (!sampling_functions.count(name)) {
      std::stringstream msg;
      msg << "Error: \"" << name << "\" is not a sampling option.";
      parser.insert_error("quantities", msg.str());
    }
  }

  // "sample_trajectory"
  parser.optional(sampling_params.sample_trajectory, "sample_trajectory");

  if (parser.valid()) {
    parser.value = std::make_unique<SamplingParams>(sampling_params);
  }
}

}  // namespace monte
}  // namespace CASM
