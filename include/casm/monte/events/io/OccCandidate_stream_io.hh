#ifndef CASM_monte_OccCandidate_stream_io
#define CASM_monte_OccCandidate_stream_io

#include <iostream>

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

std::ostream &operator<<(
    std::ostream &sout,
    std::pair<monte::OccCandidate const &, monte::Conversions const &> value);

std::ostream &operator<<(
    std::ostream &sout,
    std::pair<monte::OccSwap const &, monte::Conversions const &> value);

std::ostream &operator<<(
    std::ostream &sout,
    std::pair<monte::OccCandidateList const &, monte::Conversions const &>
        value);

}  // namespace CASM

#endif
