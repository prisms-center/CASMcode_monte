#ifndef CASM_monte_ValueMap_json_io
#define CASM_monte_ValueMap_json_io

namespace CASM {

class jsonParser;
template <typename T>
class InputParser;

namespace monte {

struct ValueMap;

}  // namespace monte

jsonParser &to_json(monte::ValueMap const &value_map, jsonParser &json);

void parse(InputParser<monte::ValueMap> &parser);

void from_json(monte::ValueMap &value_map, jsonParser const &json);

}  // namespace CASM

#endif
