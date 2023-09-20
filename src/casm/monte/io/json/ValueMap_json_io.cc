#include "casm/monte/io/json/ValueMap_json_io.hh"

#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/InputParser_impl.hh"
#include "casm/monte/ValueMap.hh"

namespace CASM {

jsonParser &to_json(monte::ValueMap const &value_map, jsonParser &json) {
  for (auto const &v : value_map.boolean_values) {
    json[v.first] = v.second;
  }
  for (auto const &v : value_map.scalar_values) {
    json[v.first] = v.second;
  }
  for (auto const &v : value_map.vector_values) {
    to_json(v.second, json[v.first], jsonParser::as_array());
  }
  for (auto const &v : value_map.matrix_values) {
    to_json(v.second, json[v.first]);
  }
  return json;
}

void parse(InputParser<monte::ValueMap> &parser) {
  auto value = std::make_unique<monte::ValueMap>();
  auto it = parser.self.begin();
  auto end = parser.self.end();
  for (; it != end; ++it) {
    if (it->is_bool()) {
      parser.require(value->boolean_values[it.name()], it.name());
    } else if (it->is_number()) {
      parser.require(value->scalar_values[it.name()], it.name());
    } else if (it->is_array()) {
      if (it->size() && it->begin()->is_array()) {
        parser.require(value->matrix_values[it.name()], it.name());
      } else {
        parser.require(value->vector_values[it.name()], it.name());
      }
    } else {
      parser.insert_error(it.name(), "Error: invalid type");
    }
  }
  if (parser.valid()) {
    parser.value = std::move(value);
  }
}

void from_json(monte::ValueMap &value_map, jsonParser const &json) {
  InputParser<monte::ValueMap> parser{json};
  std::stringstream ss;
  ss << "Error: Invalid monte::ValueMap object";
  report_and_throw_if_invalid(parser, err_log(), std::runtime_error{ss.str()});
  value_map = *parser.value;
}

}  // namespace CASM
