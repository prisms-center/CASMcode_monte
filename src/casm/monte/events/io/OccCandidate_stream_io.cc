#include "casm/monte/events/io/OccCandidate_stream_io.hh"

#include "casm/crystallography/UnitCellCoord.hh"
#include "casm/monte/Conversions.hh"
#include "casm/monte/events/OccCandidate.hh"

namespace std {

std::ostream &operator<<(std::ostream &sout,
                         std::pair<CASM::monte::OccCandidate const &,
                                   CASM::monte::Conversions const &>
                             value) {
  sout << "(" << value.second.species_name(value.first.species_index) << ", "
       << value.first.asym << ")";
  return sout;
}

std::ostream &operator<<(
    std::ostream &sout,
    std::pair<CASM::monte::OccSwap const &, CASM::monte::Conversions const &>
        value) {
  using namespace CASM::monte;
  sout << std::pair<OccCandidate const &, Conversions const &>(
              value.first.cand_a, value.second)
       << " <-> "
       << std::pair<OccCandidate const &, Conversions const &>(
              value.first.cand_b, value.second);
  return sout;
}

/// \brief Write OccCandidateList to stream, including all possible canonical
///     and grand canonical swaps
std::ostream &operator<<(std::ostream &sout,
                         std::pair<CASM::monte::OccCandidateList const &,
                                   CASM::monte::Conversions const &>
                             value) {
  using CASM::Index;
  using namespace CASM::monte;
  typedef std::pair<OccCandidate const &, Conversions const &> cand_pair;
  typedef std::pair<OccSwap const &, Conversions const &> swap_pair;
  Conversions const &convert = value.second;
  OccCandidateList const &list = value.first;

  sout << "Unit cell for determining equivalent swaps: \n"
       << convert.unit_transformation_matrix_to_super() << "\n\n";

  sout << "Asymmetric Unit: " << std::endl;
  for (Index asym = 0; asym != convert.asym_size(); ++asym) {
    sout << "  " << asym << ": ";
    for (Index i = 0; i != convert.occ_size(asym); ++i) {
      sout << convert.species_name(convert.species_index(asym, i)) << " ";
    }
    sout << "\n";

    const auto &set = convert.asym_to_unitl(asym);
    for (auto it = set.begin(); it != set.end(); ++it) {
      sout << "    " << convert.unitl_to_bijk(*it) << "\n";
    }
  }
  sout << "\n";

  sout << "Candidates: (Species, AsymUnit)" << std::endl;
  for (auto it = list.begin(); it != list.end(); ++it) {
    sout << "  " << cand_pair(*it, convert) << "\n";
  }
  sout << "\n";

  sout << "Canonical swaps: " << std::endl;
  auto canonical_swaps = make_canonical_swaps(convert, list);
  for (auto it = canonical_swaps.begin(); it != canonical_swaps.end(); ++it) {
    sout << "  " << swap_pair(*it, convert) << "\n";
  }
  sout << "\n";

  sout << "Grand canonical swaps: " << std::endl;
  auto semigrand_canonical_swaps =
      make_semigrand_canonical_swaps(convert, list);
  for (auto it = semigrand_canonical_swaps.begin();
       it != semigrand_canonical_swaps.end(); ++it) {
    sout << "  " << cand_pair(it->cand_a, convert) << " -> "
         << cand_pair(it->cand_b, convert) << "\n";
  }
  sout << "\n";
  return sout;
}

}  // namespace std
