#include "casm/monte/sampling/io/json/Sampler_json_io.hh"

#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {

/// \brief Sampler to JSON
jsonParser &to_json(Sampler const &value, jsonParser &json) {
  json = value.values();
  return json;
}

/// \brief Sampler to JSON
jsonParser &to_json(std::shared_ptr<Sampler> const &value, jsonParser &json) {
  json = value->values();
  return json;
}

/// \brief jsonSampler to JSON
jsonParser &to_json(jsonSampler const &value, jsonParser &json) {
  json = value.values;
  return json;
}

/// \brief jsonSampler to JSON
jsonParser &to_json(std::shared_ptr<jsonSampler> const &value,
                    jsonParser &json) {
  json = value->values;
  return json;
}

/// \brief RequestedPrecision to JSON
///
/// Notes:
/// - `json` must be an object
jsonParser &to_json(RequestedPrecision const &value, jsonParser &json) {
  if (json.is_obj() == false) {
    throw std::runtime_error(
        "Error writing RequestedPrecision to json: must write to an existing "
        "JSON object");
  }
  if (value.abs_convergence_is_required) {
    json["abs_precision"] = value.abs_precision;
  }
  if (value.rel_convergence_is_required) {
    json["rel_precision"] = value.rel_precision;
  }
  return json;
}

/// \brief RequestedPrecision from JSON
///
/// \param value Value to set
/// \param json JSON object
///
/// Expected format:
/// \code
/// {
///     "abs_precision": (optional) number
///         Requested absolute precision
///     "precision": (optional) number
///         Requested absolute precision
///     "rel_precision": (optional) number
///         Requested relative precision
/// }
/// \endcode
void from_json(RequestedPrecision &value, jsonParser const &json) {
  auto it = json.find("abs_precision");
  if (it != json.end()) {
    value.abs_convergence_is_required = true;
    value.abs_precision = it->get<double>();
  } else {
    // value.abs_convergence_is_required = false;

    // deprecated...
    it = json.find("precision");
    if (it != json.end()) {
      value.abs_convergence_is_required = true;
      value.abs_precision = it->get<double>();
    } else {
      value.abs_convergence_is_required = false;
    }
  }

  it = json.find("rel_precision");
  if (it != json.end()) {
    value.rel_convergence_is_required = true;
    value.rel_precision = it->get<double>();
  } else {
    value.rel_convergence_is_required = false;
  }
}

}  // namespace monte
}  // namespace CASM
