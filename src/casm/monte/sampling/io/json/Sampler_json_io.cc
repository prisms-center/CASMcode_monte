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

}  // namespace monte
}  // namespace CASM
