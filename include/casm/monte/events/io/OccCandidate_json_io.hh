#ifndef CASM_monte_OccCandidate_json_io
#define CASM_monte_OccCandidate_json_io

namespace CASM {

class jsonParser;
template <typename T>
struct jsonConstructor;

namespace monte {

class Conversions;
struct OccCandidate;
class OccCandidateList;
class OccSwap;

}  // namespace monte

template <>
struct jsonConstructor<monte::OccCandidate> {
  static monte::OccCandidate from_json(jsonParser const &json,
                                       monte::Conversions const &convert);
};

jsonParser &to_json(monte::OccCandidate const &cand,
                    monte::Conversions const &convert, jsonParser &json);

template <>
struct jsonConstructor<monte::OccSwap> {
  static monte::OccSwap from_json(jsonParser const &json,
                                  monte::Conversions const &convert);
};

jsonParser &to_json(monte::OccSwap const &swap,
                    monte::Conversions const &convert, jsonParser &json);

template <>
struct jsonConstructor<monte::OccCandidateList> {
  static monte::OccCandidateList from_json(const jsonParser &json,
                                           const monte::Conversions &convert);
};

jsonParser &to_json(monte::OccCandidateList const &swap,
                    monte::Conversions const &convert, jsonParser &json);

}  // namespace CASM

#endif
