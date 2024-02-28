#include "casm/monte/events/OccCandidate.hh"

#include <map>

#include "casm/crystallography/UnitCellCoord.hh"
#include "casm/monte/Conversions.hh"

namespace CASM {
namespace monte {

/// \brief Construct with custom list of OccCandidate
OccCandidateList::OccCandidateList(std::vector<OccCandidate> candidates,
                                   const Conversions &convert)
    : m_candidate(candidates) {
  // create lookup table of asym, species_index -> candidate index,
  //   will return {Nasym, Nspecies} if {asym, species_index} not allowed
  Index Nspecies = convert.species_size();
  Index Nasym = convert.asym_size();
  m_end = m_candidate.size();
  std::vector<Index> unallowed(Nspecies, m_end);
  m_species_to_cand_index = std::vector<std::vector<Index>>(Nasym, unallowed);

  Index index = 0;
  for (const auto &cand : m_candidate) {
    m_species_to_cand_index[cand.asym][cand.species_index] = index;
    ++index;
  }
}

/// \brief Construct with all possible OccCandidate
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
  m_species_to_cand_index = std::vector<std::vector<Index>>(Nasym, unallowed);

  Index index = 0;
  for (const auto &cand : m_candidate) {
    m_species_to_cand_index[cand.asym][cand.species_index] = index;
    ++index;
  }
}

/// \brief Check that OccCandidate is valid
///
/// Checks that:
/// - indices are in allowed range
/// - species is allowed on the asymmetric unit site
bool is_valid(Conversions const &convert, OccCandidate const &cand) {
  return cand.asym >= 0 && cand.asym < convert.asym_size() &&
         cand.species_index >= 0 &&
         cand.species_index < convert.species_size() &&
         convert.species_allowed(cand.asym, cand.species_index);
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

/// \brief Check that candidates form an allowed canonical Monte Carlo event
///
/// Checks that:
/// - cand_a and cand_b are valid
/// - the species are different and allowed on both sites
bool allowed_canonical_swap(Conversions const &convert, OccCandidate cand_a,
                            OccCandidate cand_b) {
  return is_valid(convert, cand_a) && is_valid(convert, cand_b) &&
         cand_a.species_index != cand_b.species_index &&
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

/// \brief Check that candidates form an allowed semi-grand canonical Monte
/// Carlo event
///
/// Checks that:
/// - cand_a and cand_b are valid
/// - the asym index is the same
/// - the species are different and both allowed on the asym site
bool allowed_semigrand_canonical_swap(Conversions const &convert,
                                      OccCandidate cand_a,
                                      OccCandidate cand_b) {
  return is_valid(convert, cand_a) && is_valid(convert, cand_b) &&
         cand_a.asym == cand_b.asym &&
         cand_a.species_index != cand_b.species_index &&
         convert.species_allowed(cand_a.asym, cand_b.species_index);
};

/// \brief Construct OccSwap allowed for grand canonical Monte Carlo
std::vector<OccSwap> make_semigrand_canonical_swaps(
    const Conversions &convert, OccCandidateList const &occ_candidate_list) {
  // construct grand canonical swaps
  std::vector<OccSwap> semigrand_canonical_swaps;

  // for each pair of candidates, check if they are allowed to swap
  for (const auto &cand_a : occ_candidate_list) {
    for (const auto &cand_b : occ_candidate_list) {
      // allow a->b, b->a
      // check that asym is the same and species_index is different
      if (allowed_semigrand_canonical_swap(convert, cand_a, cand_b)) {
        semigrand_canonical_swaps.push_back(OccSwap(cand_a, cand_b));
      }
    }
  }
  return semigrand_canonical_swaps;
}

/// \brief For grand canonical swaps, get the number of possible events
///     that can be chosen from at any one time
Index get_n_allowed_per_unitcell(
    Conversions const &convert,
    std::vector<OccSwap> const &semigrand_canonical_swaps) {
  std::map<Index, Index> asym_to_n_swaps;
  for (Index asym = 0; asym < convert.asym_size(); ++asym) {
    asym_to_n_swaps.emplace(asym, 0);
  }
  for (OccSwap const &swap : semigrand_canonical_swaps) {
    asym_to_n_swaps[swap.cand_a.asym]++;
  }

  Index n_allowed_per_unitcell = 0.0;
  for (auto const &pair : asym_to_n_swaps) {
    if (pair.second > 0) {
      n_allowed_per_unitcell +=
          (pair.second - 1) * convert.asym_to_b(pair.first).size();
    }
  }
  return n_allowed_per_unitcell;
}

/// \brief Construct unique MultiOccSwap
std::vector<MultiOccSwap> make_multiswaps(
    std::vector<OccSwap> const &single_swaps, int max_total_count) {
  // Get MultiOccSwap with total_count = 1
  std::set<MultiOccSwap> multiswaps;
  for (auto const &single : single_swaps) {
    std::map<OccSwap, int> swaps;
    swaps.emplace(single, 1);
    multiswaps.emplace(swaps);
  }

  // Get MultiOccSwap with total_count > 1
  std::set<MultiOccSwap> last = multiswaps;  // last: total_count = n
  std::set<MultiOccSwap> next;               // next: total_count = n+1
  for (Index n = 1; n < max_total_count; ++n) {
    next.clear();
    for (auto const &base : last) {  // for each 'last' multi-occ swap
      for (auto const &single : single_swaps) {  // add each single swap
        std::map<OccSwap, int> swaps = base.swaps;
        auto it = swaps.find(single);
        if (it == swaps.end()) {
          swaps.emplace(single, 1);
        } else {
          it->second += 1;
        }
        next.emplace(swaps);
      }
    }
    last = next;
    multiswaps.insert(next.begin(), next.end());
  }
  return std::vector<MultiOccSwap>(multiswaps.begin(), multiswaps.end());
}

}  // namespace monte
}  // namespace CASM
