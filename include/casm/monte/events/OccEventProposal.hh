#ifndef CASM_monte_OccEventProposal
#define CASM_monte_OccEventProposal

#include <vector>

#include "casm/monte/Conversions.hh"
#include "casm/monte/events/OccCandidate.hh"
#include "casm/monte/events/OccLocation.hh"

namespace CASM {
namespace monte {

struct OccEvent;
class OccLocation;
class OccSwap;

// /// \brief Typedef of function pointer
// ///
// /// Could be `propose_canonical_event`, `propose_semigrand_canonical_event`,
// or a
// /// similar custom function.
// typedef OccEvent &(*ProposeOccEventFuntionType)(OccEvent &e,
//                                                 OccLocation const &,
//                                                 std::vector<OccSwap> const &,
//                                                 GeneratorType &);

/// \brief Choose a swap type from a list of allowed canonical swap types
template <typename GeneratorType>
OccSwap const &choose_canonical_swap(OccLocation const &occ_location,
                                     std::vector<OccSwap> const &canonical_swap,
                                     GeneratorType &random_number_generator);

/// \brief Propose canonical OccEvent of particular swap type
template <typename GeneratorType>
OccEvent &propose_canonical_event_from_swap(
    OccEvent &e, OccLocation const &occ_location, OccSwap const &swap,
    GeneratorType &random_number_generator);

/// \brief Propose canonical OccEvent from list of swap types
template <typename GeneratorType>
OccEvent &propose_canonical_event(OccEvent &e, OccLocation const &occ_location,
                                  std::vector<OccSwap> const &canonical_swap,
                                  GeneratorType &random_number_generator);

/// \brief Choose a swap type from a list of allowed grand canonical swap types
template <typename GeneratorType>
OccSwap const &choose_semigrand_canonical_swap(
    OccLocation const &occ_location,
    std::vector<OccSwap> const &semigrand_canonical_swap,
    GeneratorType &random_number_generator);

/// \brief Propose grand canonical OccEvent of particular swap type
template <typename GeneratorType>
OccEvent &propose_semigrand_canonical_event_from_swap(
    OccEvent &e, OccLocation const &occ_location, OccSwap const &swap,
    GeneratorType &random_number_generator);

/// \brief Propose grand canonical OccEvent from list of swap types
template <typename GeneratorType>
OccEvent &propose_semigrand_canonical_event(
    OccEvent &e, OccLocation const &occ_location,
    std::vector<OccSwap> const &semigrand_canonical_swap,
    GeneratorType &random_number_generator);

// --- Implementation ---

/// \brief Choose a swap type from a list of allowed canonical swap types
///
/// \param occ_location Contains lookup table with occupant locations for
/// efficient choosing in dilute systems.
/// \param canonical_swap List of allowed swap types (OccSwap). Swap types
/// consists of two pairs (asymmetric unit a, species_index a) and (asymmetric
/// unit b, species_index b) defining what will change, but not which sites.
/// For canonical swaps, the species indices must be the different.
/// \param random_number_generator The random number generator used to
///     stochastically choose the swap, requires
///     `random_number_generator.random_real(maximum_value)`.
///
/// Method:
/// Stochastically chooose which swap type will be chosen. To do so, calculate
/// the cumulative number of swaps of each swap type and use a random number to
/// choose which type occurs.
///
template <typename GeneratorType>
OccSwap const &choose_canonical_swap(OccLocation const &occ_location,
                                     std::vector<OccSwap> const &canonical_swap,
                                     GeneratorType &random_number_generator) {
  // Calculate m_tsum[i]:
  // - The total number of possible events of swap types [0, i)`
  // - The total number of each swap type is
  //   `cand_size(canonical_swap[i].cand_a) *
  //    cand_size(canonical_swap[i].cand_b)`
  // - For example: System with two asym unit sites (asym1, asym2), and two
  //   species (A, B), allowed on both asym unit sites. The number of swap type
  //   (A on asym1 <-> B on asym1) is equal to the number of species A on asym1
  //   times the number of species B on asym1.
  //
  // Notes:
  // - m_tsum[0] is 0.0
  // - m_tsum[canonical_swap.size()] is total number of possible events
  //
  Index tsize = canonical_swap.size();
  static std::vector<double> m_tsum;
  m_tsum.resize(tsize + 1);

  m_tsum[0] = 0.;
  for (Index i = 0; i < tsize; ++i) {
    m_tsum[i + 1] =
        m_tsum[i] +
        ((double)occ_location.cand_size(canonical_swap[i].cand_a)) *
            ((double)occ_location.cand_size(canonical_swap[i].cand_b));
  }

  if (m_tsum.back() == 0.0) {
    throw std::runtime_error(
        "Error in choose_canonical_swap: No events possible.");
  }

  // Choose a random number on [0, m_tsum[canonical_swap.size()])
  // Swap type canonical_swap[i] occurs if the random number is between
  // m_tsum[i] and m_tsum[i+1]
  double rand = random_number_generator.random_real(m_tsum.back());

  for (Index i = 0; i < tsize; ++i) {
    if (rand < m_tsum[i + 1]) {
      return canonical_swap[i];
    }
  }

  throw std::runtime_error("Error in choose_canonical_swap");
}

/// \brief Propose canonical OccEvent of particular swap type
///
/// \param e [out] The OccEvent that will be populated with the proposed event.
/// \param occ_location Contains lookup table with occupant locations for
/// efficient choosing in dilute systems.
/// \param swap: Type of swap, defining what will change, but not which sites.
/// For canonical swaps, the species indices must be the different.
/// \param random_number_generator The random number generator used to
///     stochastically choose the swap, requires
///     `random_number_generator.random_real(maximum_value)`.
///
/// Method:
/// Stochastically choose which two sites (consistent with the given swap type)
/// will be swapped by using the list of sites of each candidate type.
/// Sets e.atom_traj to size 0.
template <typename GeneratorType>
OccEvent &propose_canonical_event_from_swap(
    OccEvent &e, OccLocation const &occ_location, OccSwap const &swap,
    GeneratorType &random_number_generator) {
  e.occ_transform.resize(2);
  e.atom_traj.resize(0);
  e.linear_site_index.resize(2);
  e.new_occ.resize(2);

  OccTransform &transform_a = e.occ_transform[0];
  Mol const &mol_a =
      occ_location.choose_mol(swap.cand_a, random_number_generator);
  transform_a.mol_id = mol_a.id;
  transform_a.l = mol_a.l;
  transform_a.asym = swap.cand_a.asym;
  transform_a.from_species = swap.cand_a.species_index;
  transform_a.to_species = swap.cand_b.species_index;

  OccTransform &transform_b = e.occ_transform[1];
  Mol const &mol_b =
      occ_location.choose_mol(swap.cand_b, random_number_generator);
  transform_b.mol_id = mol_b.id;
  transform_b.l = mol_b.l;
  transform_b.asym = swap.cand_b.asym;
  transform_b.from_species = swap.cand_b.species_index;
  transform_b.to_species = swap.cand_a.species_index;

  for (Index i = 0; i < 2; ++i) {
    OccTransform const &t = e.occ_transform[i];
    e.linear_site_index[i] = t.l;
    e.new_occ[i] = occ_location.convert().occ_index(t.asym, t.to_species);
  }

  return e;
}

/// \brief Propose canonical OccEvent from list of swap types
///
/// \param e [out] The OccEvent that will be populated with the proposed event.
/// \param occ_location Contains lookup table with occupant locations for
/// efficient choosing in dilute systems.
/// \param canonical_swap: List of allowed swap types (OccSwap). Swap types
/// consists of two pairs (asymmetric unit a, species_index a) and (asymmetric
/// unit b, species_index b) defining what will change, but not which sites.
/// For canonical swaps, the species indices must be the different. Do not
/// include reverse swaps (i.e. a->b and b->a).
/// \param random_number_generator The random number generator used to
///     stochastically choose the swap, requires
///     `random_number_generator.random_real(maximum_value)`.
///
/// Method:
/// - First, stochastically chooose which swap type will be chosen:
///   - To do so, calculate the cumulative number of swaps of each
///     swap type and use a random number to choose which type occurs.
/// - Second, stochastically choose which two sites (consistent with the chosen
///   swap type) will be swapped by using the list of sites of each candidate
///   type.
template <typename GeneratorType>
OccEvent &propose_canonical_event(OccEvent &e, OccLocation const &occ_location,
                                  const std::vector<OccSwap> &canonical_swap,
                                  GeneratorType &random_number_generator) {
  auto const &swap = choose_canonical_swap(occ_location, canonical_swap,
                                           random_number_generator);
  return propose_canonical_event_from_swap(e, occ_location, swap,
                                           random_number_generator);
}

/// \brief Choose a swap type from a list of allowed grand canonical swap types
///
/// \param occ_location Contains lookup table with occupant locations for
/// efficient choosing in dilute systems.
/// \param semigrand_canonical_swap List of allowed swap types (OccSwap). Swap
/// types consists of two pairs (asymmetric unit a, species_index a) and
/// (asymmetric unit b, species_index b) defining what will change, but not
/// which sites. For grand canonical swaps, the species indices must be the
/// different and the asymmetric unit index must be the same. Do include reverse
/// swaps (i.e. a->b and b->a). \param random_number_generator The random number
/// generator used to
///     stochastically choose the swap, requires
///     `random_number_generator.random_real(maximum_value)`.
///
///
/// Method:
/// Stochastically chooose which swap type will be chosen. To do so, calculate
/// the cumulative number of swaps of each swap type and use a random number to
/// choose which type occurs.
///
template <typename GeneratorType>
OccSwap const &choose_semigrand_canonical_swap(
    OccLocation const &occ_location,
    std::vector<OccSwap> const &semigrand_canonical_swap,
    GeneratorType &random_number_generator) {
  // Calculate m_tsum[i]:
  // - The total number of possible events of swap types [0, i)`
  // - The total number of each swap type is
  //   `cand_size(canonical_swap[i].cand_a)`
  // - For example: System with two asym unit sites (asym1, asym2), and two
  //   species (A, B), allowed on both asym unit sites. The number of swap type
  //   (A on asym1 <-> B on asym1) is equal to the number of species A on asym1
  //   times the number of species B on asym1.
  //
  // Notes:
  // - m_tsum[0] is 0.0
  // - m_tsum[canonical_swap.size()] is total number of possible events
  //
  Index tsize = semigrand_canonical_swap.size();
  static std::vector<double> m_tsum;
  m_tsum.resize(tsize + 1);

  m_tsum[0] = 0.;
  for (Index i = 0; i < tsize; ++i) {
    m_tsum[i + 1] =
        m_tsum[i] +
        ((double)occ_location.cand_size(semigrand_canonical_swap[i].cand_a));
  }

  if (m_tsum.back() == 0.0) {
    throw std::runtime_error(
        "Error in choose_semigrand_canonical_swap: No events possible.");
  }

  // Choose a random number on [0, m_tsum[grand_canonical_swap.size()])
  // Swap type semigrand_canonical_swap[i] occurs if the random number is
  // between m_tsum[i] and m_tsum[i+1]
  double rand = random_number_generator.random_real(m_tsum.back());

  for (Index i = 0; i < tsize; ++i) {
    if (rand < m_tsum[i + 1]) {
      return semigrand_canonical_swap[i];
    }
  }

  throw std::runtime_error("Error in choose_semigrand_canonical_swap");
}

/// \brief Propose grand canonical OccEvent of particular swap type
///
/// Given the choice of OccSwap, this method stochastically chooses which site
/// the swap occurs on.
/// Sets e.atom_traj to size 0.
template <typename GeneratorType>
OccEvent &propose_semigrand_canonical_event_from_swap(
    OccEvent &e, OccLocation const &occ_location, OccSwap const &swap,
    GeneratorType &random_number_generator) {
  e.occ_transform.resize(1);
  e.atom_traj.resize(0);
  e.linear_site_index.resize(1);
  e.new_occ.resize(1);

  OccTransform &transform = e.occ_transform[0];
  Mol const &mol =
      occ_location.choose_mol(swap.cand_a, random_number_generator);
  transform.mol_id = mol.id;
  transform.l = mol.l;
  transform.asym = swap.cand_a.asym;
  transform.from_species = swap.cand_a.species_index;
  transform.to_species = swap.cand_b.species_index;

  e.linear_site_index[0] = transform.l;
  e.new_occ[0] =
      occ_location.convert().occ_index(transform.asym, transform.to_species);

  return e;
}

/// \brief Propose grand canonical OccEvent from list of swap types
template <typename GeneratorType>
OccEvent &propose_semigrand_canonical_event(
    OccEvent &e, OccLocation const &occ_location,
    std::vector<OccSwap> const &semigrand_canonical_swap,
    GeneratorType &random_number_generator) {
  auto const &swap = choose_semigrand_canonical_swap(
      occ_location, semigrand_canonical_swap, random_number_generator);
  return propose_semigrand_canonical_event_from_swap(e, occ_location, swap,
                                                     random_number_generator);
}

}  // namespace monte
}  // namespace CASM

#endif
