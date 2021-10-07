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

  // make canonical and grand canonical swaps
  _make_possible_swaps(convert);
}

/// \brief Construct m_canonical_swaps, m_grand_canonical_swaps
///
/// - Currently settings is not used, but we could add restrictions
void OccCandidateList::_make_possible_swaps(const Conversions &convert) {
  // construct canonical and grand canonical swaps
  m_canonical_swap.clear();
  m_grand_canonical_swap.clear();

  // check that species are different and allowed on both sites
  auto allowed_canonical_swap = [&](OccCandidate cand_a, OccCandidate cand_b) {
    return cand_a.species_index != cand_b.species_index &&
           convert.species_allowed(cand_a.asym, cand_b.species_index) &&
           convert.species_allowed(cand_b.asym, cand_a.species_index);
  };

  // check that asym is the same and species_index is different
  auto allowed_grand_canonical_swap = [&](OccCandidate cand_a,
                                          OccCandidate cand_b) {
    return cand_a.asym == cand_b.asym &&
           cand_a.species_index != cand_b.species_index;
  };

  // for each pair of candidates, check if they are allowed to swap
  for (const auto &cand_a : m_candidate) {
    for (const auto &cand_b : m_candidate) {
      // don't repeat a->b, b->a
      // and check that cand_b's species is allowed on cand_a's sublat && vice
      // versa
      if (cand_a < cand_b && allowed_canonical_swap(cand_a, cand_b)) {
        m_canonical_swap.push_back(OccSwap(cand_a, cand_b));
      }

      // allow a->b, b->a
      // check that asym is the same and species_index is different
      if (allowed_grand_canonical_swap(cand_a, cand_b)) {
        m_grand_canonical_swap.push_back(OccSwap(cand_a, cand_b));
      }
    }
  }
}

}  // namespace monte
}  // namespace CASM
