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
}  // namespace CASM

namespace std {

std::ostream &operator<<(std::ostream &sout,
                         std::pair<CASM::monte::OccCandidate const &,
                                   CASM::monte::Conversions const &>
                             value);

std::ostream &operator<<(
    std::ostream &sout,
    std::pair<CASM::monte::OccSwap const &, CASM::monte::Conversions const &>
        value);

std::ostream &operator<<(std::ostream &sout,
                         std::pair<CASM::monte::OccCandidateList const &,
                                   CASM::monte::Conversions const &>
                             value);
}  // namespace std

#endif
