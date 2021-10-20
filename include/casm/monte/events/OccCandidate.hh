#ifndef CASM_monte_OccCandidate
#define CASM_monte_OccCandidate

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

/// \brief Store swap type, mutating sites, and info for keeping OccLocation
/// up-to-date
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

  bool operator<(const OccSwap &B) const {
    return this->sorted()._lt(B.sorted());
  }

 private:
  bool _lt(const OccSwap &B) const { return this->tuple() < B.tuple(); }

  typedef std::tuple<OccCandidate, OccCandidate> tuple_type;

  tuple_type tuple() const { return std::make_tuple(cand_a, cand_b); }
};

/// List of asym / species_index pairs indicating allowed variable occupation
/// dof
///
/// This stores lists of allowed swaps of potential canonical or grand
/// canonical events, by type only (asymmetric unit and type of occupant before
/// and after), not by which particular sites are changing.
class OccCandidateList {
 public:
  typedef std::vector<OccCandidate>::const_iterator const_iterator;

  OccCandidateList() {}

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

/// \brief Construct OccSwap allowed for canonical Monte Carlo
std::vector<OccSwap> make_canonical_swaps(
    Conversions const &convert, OccCandidateList const &occ_candidate_list);

/// \brief Construct OccSwap allowed for grand canonical Monte Carlo
std::vector<OccSwap> make_grand_canonical_swaps(
    const Conversions &convert, OccCandidateList const &occ_candidate_list);

}  // namespace monte
}  // namespace CASM

#endif
