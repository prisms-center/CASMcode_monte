#include "casm/monte/events/OccCandidate.hh"

#include "casm/crystallography/UnitCellCoord.hh"
#include "casm/monte/Conversions.hh"

namespace CASM {
namespace monte {

OccCandidateList::OccCandidateList(const Conversions &convert) {
  // create set of 'candidate' asym / species pairs
  m_candidate.clear();
  for (Index asym = 0; asym < convert.asym_size(); ++asym) {
    // hard code allowed sublattices: >1 allowed occupant
    if (convert.occ_size(asym) < 2) {
      continue;
    }

    // add candidates - only if allowed
    for (Index i = 0; i < convert.occ_size(asym); ++i) {
      Index species_index = convert.species_index(asym, i);
      m_candidate.push_back(OccCandidate(asym, species_index));
    }
  }

  // create lookup table of asym, species_index -> candidate index,
  //   will return {Nasym, Nspecies} if {asym, species_index} not allowed
  Index Nspecies = convert.species_size();
  Index Nasym = convert.asym_size();
  m_end = m_candidate.size();
  std::vector<Index> unallowed(Nspecies, m_end);
  m_species_to_cand_index = std::vector<std::vector<Index> >(Nasym, unallowed);

  Index index = 0;
  for (const auto &cand : m_candidate) {
    m_species_to_cand_index[cand.asym][cand.species_index] = index;
    ++index;
  }
}

/// \brief Check that OccCandidate is valid (won't cause segfaults)
bool is_valid(Conversions const &convert, OccCandidate const &cand) {
  return cand.asym >= 0 && cand.asym < convert.asym_size() &&
         cand.species_index >= 0 && cand.species_index < convert.species_size();
}

/// \brief Check that swap is valid (won't cause segfaults)
bool is_valid(Conversions const &convert, OccCandidate const &cand_a,
              OccCandidate const &cand_b) {
  return is_valid(convert, cand_a) && is_valid(convert, cand_b);
}

/// \brief Check that swap is valid (won't cause segfaults)
bool is_valid(Conversions const &convert, OccSwap const &swap) {
  return is_valid(convert, swap.cand_a, swap.cand_b);
}

/// \brief Check that species are different and allowed on both sites
bool allowed_canonical_swap(Conversions const &convert, OccCandidate cand_a,
                            OccCandidate cand_b) {
  return cand_a.species_index != cand_b.species_index &&
         convert.species_allowed(cand_a.asym, cand_b.species_index) &&
         convert.species_allowed(cand_b.asym, cand_a.species_index);
};

/// \brief Construct OccSwap allowed for canonical Monte Carlo
std::vector<OccSwap> make_canonical_swaps(
    Conversions const &convert, OccCandidateList const &occ_candidate_list) {
  // construct canonical swaps
  std::vector<OccSwap> canonical_swaps;

  // for each pair of candidates, check if they are allowed to swap
  for (const auto &cand_a : occ_candidate_list) {
    for (const auto &cand_b : occ_candidate_list) {
      // don't repeat a->b, b->a
      // and check that cand_b's species is allowed on cand_a's sublat && vice
      // versa
      if (cand_a < cand_b && allowed_canonical_swap(convert, cand_a, cand_b)) {
        canonical_swaps.push_back(OccSwap(cand_a, cand_b));
      }
    }
  }

  return canonical_swaps;
}

/// \brief Check that asym is the same and species_index is different
bool allowed_grand_canonical_swap(Conversions const &convert,
                                  OccCandidate cand_a, OccCandidate cand_b) {
  return cand_a.species_index != cand_b.species_index &&
         convert.species_allowed(cand_a.asym, cand_b.species_index) &&
         convert.species_allowed(cand_b.asym, cand_a.species_index);
};

/// \brief Construct OccSwap allowed for grand canonical Monte Carlo
std::vector<OccSwap> make_grand_canonical_swaps(
    const Conversions &convert, OccCandidateList const &occ_candidate_list) {
  // construct grand canonical swaps
  std::vector<OccSwap> grand_canonical_swaps;

  // for each pair of candidates, check if they are allowed to swap
  for (const auto &cand_a : occ_candidate_list) {
    for (const auto &cand_b : occ_candidate_list) {
      // allow a->b, b->a
      // check that asym is the same and species_index is different
      if (allowed_grand_canonical_swap(convert, cand_a, cand_b)) {
        grand_canonical_swaps.push_back(OccSwap(cand_a, cand_b));
      }
    }
  }
  return grand_canonical_swaps;
}

}  // namespace monte
}  // namespace CASM
