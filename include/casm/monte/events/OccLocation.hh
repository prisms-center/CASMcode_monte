#ifndef CASM_monte_OccLocation
#define CASM_monte_OccLocation

#include <optional>
#include <set>
#include <vector>

#include "casm/crystallography/UnitCellCoord.hh"
#include "casm/global/definitions.hh"
#include "casm/monte/events/OccCandidate.hh"
#include "casm/monte/events/OccEvent.hh"

namespace CASM {
namespace monte {

class Conversions;
struct OccCandidate;
class OccCandidateList;

/// \brief Data structure to store information about atoms that have been
/// added to or removed from the supercell
struct AtomInfo {
  AtomInfo(Atom _atom, Index _species_index, Index _position_index,
           double _time)
      : atom(_atom),
        species_index(_species_index),
        position_index(_position_index),
        time(_time) {}

  Atom atom;
  Index species_index;
  Index position_index;
  double time;
};

/// \brief Stores data to enable efficient proposal and update of occupation
/// mutation
///
/// Note:
/// - In the context of OccLocation, Mol, Atom, OccTransform, OccEvent,
///   OccCandidate, etc. the "asymmetric unit" must be sites that are *both*
///   equivalent by symmetry, and have the same occupation index to species
///   index mapping.
///
/// What data it has:
/// - Input Conversions provides information about conversions between site
///   indices and asymmetric unit indices, species indices and site occupant
///   indices
/// - Input OccCandidateList specifies all unique (asymmetric unit, species
///   index) pairs
/// - `mol` list (type=monte::Mol, shape=(number of mutating sites,) ), stores
///   information about each of the occupants currently in the supercell
///    including site_index (l), asymmetric unit index (asym), species_index.
/// - `loc` list (type=Index, shape=(number of OccCandidate, number of current
///   occupants of that OccCandidate type)), stores the indices in the `mol`
///   list (mol_id) for all occupants of each OccCandidate type
///
/// Choosing events:
/// - `loc` list can be used to choose amongst particular types of occupants
///   (asymmeric unit and specie_index)
///
/// Updating after events occur, use `apply`:
/// - Both `loc`, and `mol` are updated.
///
/// For molecule support:
/// - `species` list (type=Monte::Atom, shape=(number of atom components,)),
///    stores information about individual atom components of molecules,
///    including species_index, initial site, initial molecule component index
/// - `Mol` also store indices of their atom components in the `species` list
class OccLocation {
 public:
  typedef Index size_type;

  OccLocation(const Conversions &_convert,
              const OccCandidateList &_candidate_list,
              bool _update_atoms = false, bool _track_unique_atoms = false,
              bool _save_atom_info = false);

  /// Fill tables with occupation info
  void initialize(Eigen::VectorXi const &occupation,
                  std::optional<double> time = std::nullopt);

  //  /// Update occupation vector and this to reflect that event 'e' occurred
  //  void apply(OccEvent const &e, Eigen::Ref<Eigen::VectorXi> occupation);

  /// Update occupation vector and this to reflect that event 'e' occurred at
  /// specified 'time'
  void apply(OccEvent const &e, Eigen::Ref<Eigen::VectorXi> occupation,
             std::optional<double> time = std::nullopt);

  /// Stochastically choose an occupant of a particular OccCandidate type
  template <typename GeneratorType>
  Mol const &choose_mol(Index cand_index,
                        GeneratorType &random_number_generator) const;

  /// Stochastically choose an occupant of a particular OccCandidate type,
  /// excluding some occupants by `loc`
  template <typename GeneratorType>
  Mol const &choose_mol(Index cand_index, std::set<Index> exclude,
                        GeneratorType &random_number_generator) const;

  /// Stochastically choose an occupant of a particular OccCandidate type
  template <typename GeneratorType>
  Mol const &choose_mol(OccCandidate const &cand,
                        GeneratorType &random_number_generator) const;

  /// Stochastically choose an occupant of a particular OccCandidate type,
  /// excluding some occupants by `loc`
  template <typename GeneratorType>
  Mol const &choose_mol(OccCandidate const &cand, std::set<Index> exclude,
                        GeneratorType &random_number_generator) const;

  /// Total number of mutating sites
  size_type mol_size() const;

  /// Access Mol by id
  Mol &mol(Index mol_id);

  /// Access Mol by id
  Mol const &mol(Index mol_id) const;

  /// Total number of atoms
  size_type atom_size() const;

  /// Access Atom by id
  Atom &atom(Index atom_id);

  /// Access Atom by id
  Atom const &atom(Index atom_id) const;

  /// \brief Return current atom positions in cartesian coordinates, shape=(3,
  /// n_atoms)
  Eigen::MatrixXd atom_positions_cart() const;

  /// \brief Return current atom positions in cartesian coordinates, shape=(3,
  /// n_atoms)
  Eigen::MatrixXd atom_positions_cart_within() const;

  /// Holds *initial* species index for each atom in atom position matrices
  std::vector<Index> const &initial_atom_species_index() const;

  /// Holds *initial* atom position index for each atom in atom position
  /// matrices
  std::vector<Index> const &initial_atom_position_index() const;

  /// Return *current* name for each atom in atom position matrices
  std::vector<std::string> current_atom_names() const;

  /// \brief Return current species index for atoms in atom position matricess
  std::vector<Index> current_atom_species_index() const;

  /// \brief Return current atom position index for atoms in atom position
  /// matricess
  std::vector<Index> current_atom_position_index() const;

  /// \brief Return number of jumps made by each atom
  std::vector<Index> current_atom_n_jumps() const;

  /// \brief Return unique atom id for each atom in atom position matrices
  std::vector<Index> const &unique_atom_id() const;

  /// \brief The initial position, type, and time when atom are added to the
  ///     supercell, stored by unique atom id
  std::map<Index, AtomInfo> const &atom_info_initial() const;

  /// \brief The final position, type, and time when atom are removed from the
  ///     supercell, stored by unique atom id
  std::map<Index, AtomInfo> const &atom_info_final() const;

  /// \brief Clear information about atoms that have been removed from the
  /// supercell
  void clear_atom_info_final();

  /// Access the OccCandidateList
  OccCandidateList const &candidate_list() const;

  /// Total number of mutating sites, of OccCandidate type, specified by index
  size_type cand_size(Index cand_index) const;

  /// Total number of mutating sites, of OccCandidate type
  size_type cand_size(OccCandidate const &cand) const;

  /// Mol.id of a particular OccCandidate type
  Index mol_id(Index cand_index, Index loc) const;

  /// Mol.id of a particular OccCandidate type
  Index mol_id(OccCandidate const &cand, Index loc) const;

  /// Convert from config index to variable site index
  Index l_to_mol_id(Index l) const;

  /// Get Conversions objects
  Conversions const &convert() const;

  /// \brief Replace the Mol at a specified site
  void replace_mol(Index l, int occ, std::vector<Index> const &atom_id,
                   std::vector<Atom> const &atom);

 private:
  Conversions const &m_convert;

  OccCandidateList const &m_candidate_list;

  /// Gives a list of all Mol of the same {asym, species}-type allowed to mutate
  ///   m_loc[cand_index][i] -> m_mol index
  std::vector<std::vector<Index>> m_loc;

  /// Holds Monte::Atom objects
  std::vector<Atom> m_atoms;

  /// Holds *initial* species index for each atom in m_atoms
  std::vector<Index> m_initial_atom_species_index;

  /// Holds *initial* atom position index for each atom in m_atoms
  std::vector<Index> m_initial_atom_position_index;

  /// Holds Mol objects, one for each mutating site in the configuration
  std::vector<Mol> m_mol;

  /// l_to_mol[l] -> Mol.id, m_mol.size() otherwise
  std::vector<Index> m_l_to_mol;

  /// If true, update Atom location during apply
  bool m_update_atoms;

  // -- Track atoms moving to/from the reservoir --

  /// The `atom_id` (positions in `m_atoms`) that have been emptied due to
  /// an atom being removed from the supercell (and not yet filled).
  std::set<Index> m_available_atom_id;

  /// If true, track unique atom ids
  const bool m_track_unique_atoms;

  /// The next unique atom id to assign
  Index m_next_unique_atom_id;

  /// m_unique_atom_id[atom_id] -> unique atom id for current Atom in `m_atoms`
  std::vector<Index> m_unique_atom_id;

  /// If true, save the initial and final atom info
  bool m_save_atom_info;

  /// The initial position, type, and time when atom are added to the supercell,
  /// stored by unique atom id
  std::map<Index, AtomInfo> m_atom_info_initial;

  /// The final position, type, and time when atom are removed from the
  /// supercell, stored by unique atom id
  std::map<Index, AtomInfo> m_atom_info_final;
};

/// --- Implementation ---

/// Stochastically choose an occupant of a particular OccCandidate type
template <typename GeneratorType>
Mol const &OccLocation::choose_mol(
    Index cand_index, GeneratorType &random_number_generator) const {
  return mol(m_loc[cand_index][random_number_generator.random_int(
      m_loc[cand_index].size() - 1)]);
}

/// Stochastically choose an occupant of a particular OccCandidate type,
/// excluding some occupants by `loc`
template <typename GeneratorType>
Mol const &OccLocation::choose_mol(
    Index cand_index, std::set<Index> exclude,
    GeneratorType &random_number_generator) const {
  Index loc;
  do {
    loc = random_number_generator.random_int(m_loc[cand_index].size() - 1);
  } while (exclude.count(loc));
  return mol(m_loc[cand_index][loc]);
}

/// Stochastically choose an occupant of a particular OccCandidate type
template <typename GeneratorType>
Mol const &OccLocation::choose_mol(
    OccCandidate const &cand, GeneratorType &random_number_generator) const {
  return choose_mol(m_candidate_list.index(cand), random_number_generator);
}

/// Stochastically choose an occupant of a particular OccCandidate type,
/// excluding some occupants by `loc`
template <typename GeneratorType>
Mol const &OccLocation::choose_mol(
    OccCandidate const &cand, std::set<Index> exclude,
    GeneratorType &random_number_generator) const {
  return choose_mol(m_candidate_list.index(cand), exclude,
                    random_number_generator);
}

/// Total number of mutating sites
inline OccLocation::size_type OccLocation::mol_size() const {
  return m_mol.size();
}

inline Mol &OccLocation::mol(Index mol_id) { return m_mol[mol_id]; }

inline const Mol &OccLocation::mol(Index mol_id) const { return m_mol[mol_id]; }

/// Total number of atoms
inline OccLocation::size_type OccLocation::atom_size() const {
  return m_atoms.size();
}

/// Access Atom by id
inline Atom &OccLocation::atom(Index atom_id) { return m_atoms[atom_id]; }

/// Access Atom by id
inline Atom const &OccLocation::atom(Index atom_id) const {
  return m_atoms[atom_id];
}

/// Holds *initial* species index for each atom in atom position matrices
inline std::vector<Index> const &OccLocation::initial_atom_species_index()
    const {
  return m_initial_atom_species_index;
}

/// Holds *initial* atom position index for each atom in atom position matrices
inline std::vector<Index> const &OccLocation::initial_atom_position_index()
    const {
  return m_initial_atom_position_index;
}

/// \brief Return current unique atom id for each atom in atom position matrices
inline std::vector<Index> const &OccLocation::unique_atom_id() const {
  return m_unique_atom_id;
}

/// \brief The initial position, type, and time when atom are added to the
///     supercell, stored by unique atom id
///
/// - Key value is unique atom id
/// - Atoms that were in the system when it was initialized have a time
///   of 0.0
inline std::map<Index, AtomInfo> const &OccLocation::atom_info_initial() const {
  return m_atom_info_initial;
}

/// \brief The final position, type, and time when atom are removed from the
///     supercell, stored by unique atom id
///
/// - Key value is unique atom id
inline std::map<Index, AtomInfo> const &OccLocation::atom_info_final() const {
  return m_atom_info_final;
}

/// \brief Clear information about atoms that have been removed from the
/// supercell
///
/// - This also clears the initial information for the removed atoms
inline void OccLocation::clear_atom_info_final() {
  for (auto const &pair : m_atom_info_final) {
    m_atom_info_initial.erase(pair.first);
  }
  m_atom_info_final.clear();
}

/// Access the OccCandidateList
inline OccCandidateList const &OccLocation::candidate_list() const {
  return m_candidate_list;
}

/// Total number of mutating sites, of OccCandidate type, specified by index
inline OccLocation::size_type OccLocation::cand_size(Index cand_index) const {
  return m_loc[cand_index].size();
}

/// Total number of mutating sites, of OccCandidate type
inline OccLocation::size_type OccLocation::cand_size(
    const OccCandidate &cand) const {
  return cand_size(m_candidate_list.index(cand));
}

/// The index into the configuration of a particular mutating site
inline Index OccLocation::mol_id(Index cand_index, Index loc) const {
  return m_loc[cand_index][loc];
}

/// The index into the configuration of a particular mutating site
inline Index OccLocation::mol_id(const OccCandidate &cand, Index loc) const {
  return mol_id(m_candidate_list.index(cand), loc);
}

/// Convert from config index to variable site index
inline Index OccLocation::l_to_mol_id(Index l) const { return m_l_to_mol[l]; }

/// Get Conversions objects
inline Conversions const &OccLocation::convert() const { return m_convert; }

}  // namespace monte
}  // namespace CASM

#endif
