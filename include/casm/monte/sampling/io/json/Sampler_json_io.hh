#ifndef CASM_monte_sampling_Sampler_json_io
#define CASM_monte_sampling_Sampler_json_io

#include <memory>

namespace CASM {
class jsonParser;
namespace monte {
class Sampler;

/// \brief Sampler to JSON
jsonParser &to_json(Sampler const &value, jsonParser &json);

/// \brief Sampler to JSON
jsonParser &to_json(std::shared_ptr<Sampler> const &value, jsonParser &json);

}  // namespace monte
}  // namespace CASM

#endif
