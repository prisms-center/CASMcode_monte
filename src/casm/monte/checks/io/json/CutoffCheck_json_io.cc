#include "casm/monte/checks/io/json/CutoffCheck_json_io.hh"

#include "casm/casm_io/json/InputParser_impl.hh"
#include "casm/casm_io/json/optional.hh"
#include "casm/monte/checks/CutoffCheck.hh"

namespace CASM {
namespace monte {

/// \brief Convert CutoffCheckParams to JSON
jsonParser &to_json(CutoffCheckParams const &params, jsonParser &json) {
  json.put_obj();
  if (params.min_count.has_value() || params.max_count.has_value()) {
    json["count"].put_obj();
    if (params.min_count.has_value()) {
      json["count"]["min"] = params.min_count.value();
    }
    if (params.max_count.has_value()) {
      json["count"]["max"] = params.max_count.value();
    }
  }
  if (params.min_time.has_value() || params.max_time.has_value()) {
    json["time"].put_obj();
    if (params.min_time.has_value()) {
      json["time"]["min"] = params.min_time.value();
    }
    if (params.max_time.has_value()) {
      json["time"]["max"] = params.max_time.value();
    }
  }
  if (params.min_sample.has_value() || params.max_sample.has_value()) {
    json["sample"].put_obj();
    if (params.min_sample.has_value()) {
      json["sample"]["min"] = params.min_sample.value();
    }
    if (params.max_sample.has_value()) {
      json["sample"]["max"] = params.max_sample.value();
    }
  }
  if (params.min_clocktime.has_value() || params.max_clocktime.has_value()) {
    json["clocktime"].put_obj();
    if (params.min_clocktime.has_value()) {
      json["clocktime"]["min"] = params.min_clocktime.value();
    }
    if (params.max_clocktime.has_value()) {
      json["clocktime"]["max"] = params.max_clocktime.value();
    }
  }
  return json;
}

/// \brief Construct CutoffCheckParams from JSON
///
/// Expected:
///   count: dict (optional, default={})
///     Sets a minimum and maximum for how many steps or passes the
///     calculation runs. If sampling by pass, then the count refers to the
///     number of passes, else the count refers to the number of steps. May
///     include:
///
///       min: int (optional, default=null)
///         Applies a minimum count, if not null.
///
///       max: int (optional, default=null)
///         Applies a maximum count, if not null.
///
///   sample: dict (optional, default={})
///     Sets a minimum and maximum for how many samples are taken. Options
///     are `min` and `max`, the same as for `count`.
///
///   time: dict (optional, default={})
///     If a time-based calculation, sets minimum and maximum cuttoffs for
///     time. Options are `min` and `max`, the same as for `count`.
///
///   clocktime: dict (optional, default={})
///     Sets minimum and maximum cuttoffs for elapsed calculation time in
///     seconds. Options are `min` and `max`, the same as for `count`.
///
void parse(InputParser<CutoffCheckParams> &parser) {
  CutoffCheckParams params;
  parser.optional(params.min_count, fs::path("count") / "min");
  parser.optional(params.max_count, fs::path("count") / "max");
  parser.optional(params.min_time, fs::path("time") / "min");
  parser.optional(params.max_time, fs::path("time") / "max");
  parser.optional(params.min_sample, fs::path("sample") / "min");
  parser.optional(params.max_sample, fs::path("sample") / "max");
  parser.optional(params.min_clocktime, fs::path("clocktime") / "min");
  parser.optional(params.max_clocktime, fs::path("clocktime") / "max");
  if (parser.valid()) {
    parser.value = std::make_unique<CutoffCheckParams>(params);
  }
}

}  // namespace monte
}  // namespace CASM
