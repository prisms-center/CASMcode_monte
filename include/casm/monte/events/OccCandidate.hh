#ifndef CASM_monte_OccCandidate
#define CASM_monte_OccCandidate

#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "casm/global/definitions.hh"
#include "casm/misc/Comparisons.hh"

namespace CASM {
namespace monte {

class Conversions;
struct OccCandidate;
class OccSwap;

/// A pair of asymmetric unit index and species index, indicating a type of
/// occupant that may be chosen for Monte Carlo events
struct OccCandidate : public Comparisons<CRTPBase<OccCandidate>> {
  OccCandidate(Index _asym, Index _species_index)
      : asym(_asym), species_index(_species_index) {}

  Index asym;
  Index species_index;

  bool operator<(OccCandidate B) const {
    if (asym != B.asym) {
      return asym < B.asym;
    }
    return species_index < B.species_index;
  }
};

/// \brief Represents a Monte Carlo event that swaps occupants
///
/// This object does not specify which particular sites are changing, just the
/// type of change (which occupant types on which asymmetric unit sites).
/// Depending on the context this may be canonical or semi-grand canonical.
class OccSwap : public Comparisons<CRTPBase<OccSwap>> {
 public:
  OccSwap(const OccCandidate &_cand_a, const OccCandidate &_cand_b)
      : cand_a(_cand_a), cand_b(_cand_b) {}

  OccCandidate cand_a;
  OccCandidate cand_b;

  void reverse() {
    using std::swap;
    std::swap(cand_a, cand_b);
  }

  OccSwap &sort() {
    OccSwap B(*this);
    B.reverse();

    if (B._lt(*this)) {
      *this = B;
    }
    return *this;
  }

  OccSwap sorted() const {
    OccSwap res(*this);
    res.sort();
    return res;
  }

  bool operator<(const OccSwap &B) const { return this->_lt(B); }

 private:
  bool _lt(const OccSwap &B) const { return this->tuple() < B.tuple(); }

  typedef std::tuple<OccCandidate, OccCandidate> tuple_type;

  tuple_type tuple() const { return std::make_tuple(cand_a, cand_b); }
};

class MultiOccSwap : public Comparisons<CRTPBase<MultiOccSwap>> {
 public:
  MultiOccSwap(std::map<OccSwap, int> const &_swaps)
      : swaps(_swaps), total_count(0) {
    if (!swaps.size()) {
      throw std::runtime_error(
          "Error constructing MultiOccSwap: "
          "Empty multi-occ swap.");
    }
    for (auto const &pair : swaps) {
      total_count += pair.second;
    }
  }

  /// \brief {Swap, count}
  std::map<OccSwap, int> swaps;

  /// \brief Sum of swap counts (total number of single swaps)
  int total_count;

  void reverse() {
    std::map<OccSwap, int> _reverse_swaps;
    for (auto const &pair : swaps) {
      OccSwap tmp = pair.first;
      tmp.reverse();
      _reverse_swaps.emplace(tmp, pair.second);
    }
    *this = MultiOccSwap(_reverse_swaps);
  }

  MultiOccSwap &sort() {
    MultiOccSwap B(*this);
    B.reverse();

    if (B._lt(*this)) {
      *this = B;
    }
    return *this;
  }

  MultiOccSwap sorted() const {
    MultiOccSwap res(*this);
    res.sort();
    return res;
  }

  bool operator<(const MultiOccSwap &B) const { return this->_lt(B); }

 private:
  bool _lt(const MultiOccSwap &B) const { return this->swaps < B.swaps; }
};

/// List of asym / species_index pairs indicating allowed variable occupation
/// dof
class OccCandidateList {
 public:
  typedef std::vector<OccCandidate>::const_iterator const_iterator;

  OccCandidateList() {}

  /// \brief Construct with custom list of OccCandidate
  OccCandidateList(std::vector<OccCandidate> candidates,
                   const Conversions &convert);

  /// \brief Construct with all possible OccCandidate
  OccCandidateList(const Conversions &convert);

  /// Return index into std::vector<OccCandidate>, or _candidate.size() if not
  /// allowed
  Index index(const OccCandidate &cand) const {
    return m_species_to_cand_index[cand.asym][cand.species_index];
  }

  /// Return index into std::vector<OccCandidate>, or _candidate.size() if not
  /// allowed
  Index index(Index asym, Index species_index) const {
    return m_species_to_cand_index[asym][species_index];
  }

  const OccCandidate &operator[](Index candidate_index) const {
    return m_candidate[candidate_index];
  }

  const_iterator begin() const { return m_candidate.begin(); }

  const_iterator end() const { return m_candidate.end(); }

  Index size() const { return m_end; }

 private:
  /// m_converter[asym][species_index] -> candidate_index
  std::vector<std::vector<Index>> m_species_to_cand_index;

  std::vector<OccCandidate> m_candidate;

  /// Number of allowed candidates, what is returned if a candidate is not
  /// allowed
  Index m_end;
};

/// \brief Check that OccCandidate is valid (won't cause segfaults)
bool is_valid(Conversions const &convert, OccCandidate const &cand);

/// \brief Check that swap is valid (won't cause segfaults)
bool is_valid(Conversions const &convert, OccCandidate const &cand_a,
              OccCandidate const &cand_b);

/// \brief Check that swap is valid (won't cause segfaults)
bool is_valid(Conversions const &convert, OccSwap const &swap);

/// \brief Check that candidates form an allowed canonical Monte Carlo event
bool allowed_canonical_swap(Conversions const &convert, OccCandidate cand_a,
                            OccCandidate cand_b);

/// \brief Construct OccSwap allowed for canonical Monte Carlo
std::vector<OccSwap> make_canonical_swaps(
    Conversions const &convert, OccCandidateList const &occ_candidate_list);

/// \brief Check that candidates form an allowed semi-grand canonical Monte
/// Carlo event
bool allowed_semigrand_canonical_swap(Conversions const &convert,
                                      OccCandidate cand_a, OccCandidate cand_b);

/// \brief Construct OccSwap allowed for grand canonical Monte Carlo
std::vector<OccSwap> make_semigrand_canonical_swaps(
    const Conversions &convert, OccCandidateList const &occ_candidate_list);

/// \brief For grand canonical swaps, get the number of possible events
///     that can be chosen from at any one time
Index get_n_allowed_per_unitcell(
    Conversions const &convert,
    std::vector<OccSwap> const &semigrand_canonical_swaps);

/// \brief Construct unique MultiOccSwap
std::vector<MultiOccSwap> make_multiswaps(
    std::vector<OccSwap> const &single_swaps, int max_total_count);

}  // namespace monte
}  // namespace CASM

#endif
