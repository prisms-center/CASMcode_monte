#ifndef CASM_monte_OccEvent
#define CASM_monte_OccEvent

#include <vector>

#include "casm/crystallography/UnitCellCoord.hh"
#include "casm/global/definitions.hh"

namespace CASM {
namespace monte {

/// \brief Track the position of individual atoms, as if no periodic boundaries
struct Atom {
  Eigen::Vector3l
      translation;  ///< Current translation, as if no periodic boundary
  Index n_jumps;
};

/// \brief Represents the occupant on a site
///
/// - May be divisible into components or indivisible
struct Mol {
  Index id;             ///< Location in OccLocation.m_mol
  Index l;              ///< Location in config
  Index asym;           ///< Asym unit index (must be consistent with l)
  Index species_index;  ///< Species type index (must be consistent with
                        ///< config.occ(l))
  std::vector<Index>
      component;  ///< Location of component atom in OccLocation.m_atoms
  Index loc;      ///< Location in OccLocation.m_loc
};

struct OccTransform {
  Index l;             ///< Config occupant that is being transformed
  Index mol_id;        ///< Location in OccLocation.m_mol
  Index asym;          ///< Asym index
  Index from_species;  ///< Species index before transformation
  Index to_species;    ///< Species index after transformation
};

/// Note, for reservoir positions, use l == -1, and mol_id = species_index
struct AtomLocation {
  Index l;         ///< Config occupant that is being transformed
  Index mol_id;    ///< Location in OccLocation.m_mol
  Index mol_comp;  ///< Location in mol.components
};

struct AtomTraj {
  AtomLocation from;
  AtomLocation to;
  Eigen::Vector3l delta_ijk;
};

/// \brief Describes a Monte Carlo event that modifies occupation
struct OccEvent {
  /// \brief Linear site indices, indicating on which sites the occupation will
  ///     be modified
  std::vector<Index> linear_site_index;

  /// \brief Occupant indices, indicating the new occupation index on the sites
  ///     being modified
  std::vector<int> new_occ;

  /// \brief Information used to update occupant tracking information stored in
  ///     OccLocation
  std::vector<OccTransform> occ_transform;

  /// \brief Information used to update occupant tracking information stored in
  ///     OccLocation - use if tracking species trajectories for KMC
  std::vector<AtomTraj> atom_traj;
};

}  // namespace monte
}  // namespace CASM

#endif
