#ifndef CASM_monte_sampling_Sampler_json_io
#define CASM_monte_sampling_Sampler_json_io

#include <memory>

namespace CASM {
class jsonParser;
namespace monte {
class Sampler;
struct RequestedPrecision;
struct jsonSampler;

/// \brief Sampler to JSON
jsonParser &to_json(Sampler const &value, jsonParser &json);

/// \brief Sampler to JSON
jsonParser &to_json(std::shared_ptr<Sampler> const &value, jsonParser &json);

/// \brief jsonSampler to JSON
jsonParser &to_json(jsonSampler const &value, jsonParser &json);

/// \brief jsonSampler to JSON
jsonParser &to_json(std::shared_ptr<jsonSampler> const &value,
                    jsonParser &json);

/// \brief RequestedPrecision to JSON
jsonParser &to_json(RequestedPrecision const &value, jsonParser &json);

/// \brief RequestedPrecision from JSON
void from_json(RequestedPrecision &value, jsonParser const &json);

}  // namespace monte
}  // namespace CASM

#endif
