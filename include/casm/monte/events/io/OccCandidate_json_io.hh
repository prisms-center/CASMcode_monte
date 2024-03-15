#ifndef CASM_monte_OccCandidate_json_io
#define CASM_monte_OccCandidate_json_io

namespace CASM {

class jsonParser;
template <typename T>
struct jsonConstructor;
template <typename T>
class InputParser;

namespace monte {

class Conversions;
class MultiOccSwap;
struct OccCandidate;
class OccCandidateList;
class OccSwap;

}  // namespace monte

// ~~~ OccCandidate

template <>
struct jsonConstructor<monte::OccCandidate> {
  static monte::OccCandidate from_json(jsonParser const &json,
                                       monte::Conversions const &convert);
};

void parse(InputParser<monte::OccCandidate> &parser,
           monte::Conversions const &convert);

jsonParser &to_json(monte::OccCandidate const &cand, jsonParser &json,
                    monte::Conversions const &convert);

// ~~~ OccSwap

template <>
struct jsonConstructor<monte::OccSwap> {
  static monte::OccSwap from_json(jsonParser const &json,
                                  monte::Conversions const &convert);
};

void parse(InputParser<monte::OccSwap> &parser,
           monte::Conversions const &convert);

jsonParser &to_json(monte::OccSwap const &swap, jsonParser &json,
                    monte::Conversions const &convert);

// ~~~ MultiOccSwap

template <>
struct jsonConstructor<monte::MultiOccSwap> {
  static monte::MultiOccSwap from_json(jsonParser const &json,
                                       monte::Conversions const &convert);
};

void parse(InputParser<monte::MultiOccSwap> &parser,
           monte::Conversions const &convert);

jsonParser &to_json(monte::MultiOccSwap const &multiswap, jsonParser &json,
                    monte::Conversions const &convert);

// ~~~ OccCandidateList

template <>
struct jsonConstructor<monte::OccCandidateList> {
  static monte::OccCandidateList from_json(const jsonParser &json,
                                           const monte::Conversions &convert);
};

void parse(InputParser<monte::OccCandidateList> &parser,
           monte::Conversions const &convert);

jsonParser &to_json(monte::OccCandidateList const &swap, jsonParser &json,
                    monte::Conversions const &convert);

}  // namespace CASM

#endif
