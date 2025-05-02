#include "casm/monte/events/OccLocation.hh"

#include "casm/crystallography/Molecule.hh"
#include "casm/external/MersenneTwister/MersenneTwister.h"
#include "casm/monte/Conversions.hh"
#include "casm/monte/events/OccCandidate.hh"

namespace CASM {
namespace monte {

/// \brief Constructor
///
/// \param _convert Conversions object
/// \param _candidate_list Specifies allowed types of occupants
///     by {asymmetric unit index, species index}
/// \param _update_atoms If true, track atom trajectories when
///     applying OccEvent
/// \param _track_unique_atoms If true, track atoms added to the supercell
///     with unique ids. Requires `update_atoms` to be true.
/// \param _save_atom_info If true, save the initial and final atom positions
///     for atoms added to the supercell or removed from the supercell.
///     Requires `track_unique_atoms` to be true.
OccLocation::OccLocation(const Conversions &_convert,
                         const OccCandidateList &_candidate_list,
                         bool _update_atoms, bool _track_unique_atoms,
                         bool _save_atom_info)
    : m_convert(_convert),
      m_candidate_list(_candidate_list),
      m_loc(_candidate_list.size()),
      m_update_atoms(_update_atoms),
      m_track_unique_atoms(_track_unique_atoms),
      m_save_atom_info(_save_atom_info) {}

/// Fill tables with occupation info
///
/// \param occupation Current occupation vector
/// \param time If time has a value, and `save_atom_info` is true, then the
///     initial atom info will be stored with the given time.
void OccLocation::initialize(Eigen::VectorXi const &occupation,
                             std::optional<double> time) {
  m_mol.clear();
  m_atoms.clear();
  m_l_to_mol.clear();
  for (auto &vec : m_loc) {
    vec.clear();
  }

  Index Nmut = 0;
  for (Index l = 0; l < occupation.size(); ++l) {
    Index asym = m_convert.l_to_asym(l);
    if (m_convert.occ_size(asym) > 1) {
      Nmut++;
    }
  }

  if (m_update_atoms && m_track_unique_atoms) {
    m_available_atom_id.clear();
    m_next_unique_atom_id = 0;
    m_unique_atom_id.clear();
    if (m_save_atom_info) {
      m_atom_info_initial.clear();
      m_atom_info_final.clear();
    }
  }

  m_mol.resize(Nmut);
  m_l_to_mol.reserve(occupation.size());
  Index mol_id = 0;
  for (Index l = 0; l < occupation.size(); ++l) {
    Index asym = m_convert.l_to_asym(l);
    if (m_convert.occ_size(asym) > 1) {
      Index species_index = m_convert.species_index(asym, occupation[l]);
      Index cand_index = m_candidate_list.index(asym, species_index);

      Mol &mol = m_mol[mol_id];
      mol.id = mol_id;
      mol.l = l;
      mol.asym = asym;
      mol.species_index = species_index;
      mol.loc = m_loc[cand_index].size();

      if (m_update_atoms) {
        xtal::Molecule const &molecule =
            m_convert.species_to_mol(species_index);
        int n_atoms = molecule.atoms().size();
        for (Index atom_index = 0; atom_index < n_atoms; ++atom_index) {
          mol.component.push_back(m_atoms.size());
          Atom atom;
          atom.translation = m_convert.l_to_ijk(mol.l);
          atom.n_jumps = 0;
          m_atoms.push_back(atom);

          if (m_track_unique_atoms) {
            m_unique_atom_id.push_back(m_next_unique_atom_id);

            if (m_save_atom_info && time.has_value()) {
              m_atom_info_initial.emplace(
                  m_next_unique_atom_id,
                  AtomInfo(atom, species_index, atom_index, time.value()));
            }
            ++m_next_unique_atom_id;
          }
          m_initial_atom_species_index.push_back(species_index);
          m_initial_atom_position_index.push_back(atom_index);
        }
      }

      m_loc[cand_index].push_back(mol_id);
      m_l_to_mol.push_back(mol_id);
      mol_id++;
    } else {
      m_l_to_mol.push_back(Nmut);
    }
  }
}

///// Update occupation vector and this to reflect that event 'e' occurred
// void OccLocation::apply(const OccEvent &e,
//                         Eigen::Ref<Eigen::VectorXi> occupation) {
//   static std::vector<Index> updating_atoms;
//
//   // copy original Mol.component
//   if (m_update_atoms) {
//     if (updating_atoms.size() < e.atom_traj.size()) {
//       updating_atoms.resize(e.atom_traj.size());
//     }
//     Index i_updating_atom = 0;
//     for (const auto &traj : e.atom_traj) {
//       if (traj.from.l == -1) {
//         // move from reservoir -- create a new atom
//         Atom atom;
//         atom.translation = m_convert.l_to_ijk(traj.to.l);
//         atom.n_jumps = 0;
//         Index species_index = traj.from.mol_id;
//         xtal::Molecule molecule = m_convert.species_to_mol(species_index);
//         Index atom_position_index = traj.from.mol_comp;
//         m_reservoir_mol[species_index].component[atom_position_index] =
//             m_atoms.size();
//         updating_atoms[i_updating_atom] = m_atoms.size();
//         m_atoms.push_back(atom);
//         m_initial_atom_species_index.push_back(species_index);
//         m_initial_atom_position_index.push_back(atom_position_index);
//       } else {  // move from within supercell
//         updating_atoms[i_updating_atom] =
//             m_mol[traj.from.mol_id].component[traj.from.mol_comp];
//       }
//       ++i_updating_atom;
//     }
//   }
//
//   // update Mol and config occupation
//   for (const auto &occ : e.occ_transform) {
//     auto &mol = m_mol[occ.mol_id];
//
//     if (mol.species_index != occ.from_species) {
//       throw std::runtime_error("Error in OccLocation::apply: species
//       mismatch");
//     }
//
//     occupation[mol.l] = m_convert.occ_index(mol.asym, occ.to_species);
//
//     // remove from m_loc
//     Index cand_index = m_candidate_list.index(mol.asym, mol.species_index);
//     Index back = m_loc[cand_index].back();
//     m_loc[cand_index][mol.loc] = back;
//     m_mol[back].loc = mol.loc;
//     m_loc[cand_index].pop_back();
//
//     // set Mol.species index
//     mol.species_index = occ.to_species;
//
//     if (m_update_atoms) {
//       mol.component.resize(m_convert.components_size(mol.species_index));
//     }
//
//     // add to m_loc
//     cand_index = m_candidate_list.index(mol.asym, mol.species_index);
//     mol.loc = m_loc[cand_index].size();
//     m_loc[cand_index].push_back(mol.id);
//   }
//
//   if (m_update_atoms) {
//     Index i_updating_atom = 0;
//     for (const auto &traj : e.atom_traj) {
//       if (traj.to.l != -1) {
//         // move to position in supercell
//         Index atom_id = updating_atoms[i_updating_atom];
//
//         // update Mol.component
//         m_mol[traj.to.mol_id].component[traj.to.mol_comp] = atom_id;
//
//         // update atom translation
//         m_atoms[atom_id].translation += traj.delta_ijk;
//
//         // update number of atom jumps
//         m_atoms[atom_id].n_jumps += 1;
//       }
//       // else {
//       //   // move to reservoir
//       //   // mark explicitly?
//       //   // or know implicitly (because not found in
//       //   m_mol[mol_id]->component)?
//       // }
//       ++i_updating_atom;
//     }
//   }
// }

/// Update occupation vector and this to reflect that event 'e' occurred at
/// specified 'time'
///
/// \param e The event to apply
/// \param occupation Current occupation vector
/// \param time If time has a value, and `save_atom_info` is true, then the
///     initial/final atom info will be stored with the given time.
void OccLocation::apply(OccEvent const &e,
                        Eigen::Ref<Eigen::VectorXi> occupation,
                        std::optional<double> time) {
  // current `atom_id` (position in m_atoms) of updating atoms
  static std::vector<Index> updating_atoms;

  // copy original Mol.component
  if (m_update_atoms) {
    if (updating_atoms.size() < e.atom_traj.size()) {
      updating_atoms.resize(e.atom_traj.size());
    }
    Index i_updating_atom = 0;
    for (const auto &traj : e.atom_traj) {
      if (traj.from.l == -1) {
        // move from reservoir -- will create a new atom
        updating_atoms[i_updating_atom] = -1;
      } else {
        // move from within supercell
        auto &mol = m_mol[traj.from.mol_id];
        Index atom_position_index = traj.from.mol_comp;
        Index atom_id = mol.component[atom_position_index];

        if (m_track_unique_atoms) {
          if (traj.to.l == -1 && m_save_atom_info && time.has_value()) {
            // if move to reservoir, store final atom info using current mol
            // info
            m_atom_info_final.emplace(
                m_unique_atom_id[atom_id],
                AtomInfo(m_atoms[atom_id], mol.species_index,
                         atom_position_index, time.value()));
          }
        }

        updating_atoms[i_updating_atom] = atom_id;
      }
      ++i_updating_atom;
    }
  }

  // update Mol and config occupation
  for (const auto &occ : e.occ_transform) {
    auto &mol = m_mol[occ.mol_id];

    if (mol.species_index != occ.from_species) {
      throw std::runtime_error("Error in OccLocation::apply: species mismatch");
    }

    occupation[mol.l] = m_convert.occ_index(mol.asym, occ.to_species);

    // remove from m_loc
    Index cand_index = m_candidate_list.index(mol.asym, mol.species_index);
    Index back = m_loc[cand_index].back();
    m_loc[cand_index][mol.loc] = back;
    m_mol[back].loc = mol.loc;
    m_loc[cand_index].pop_back();

    // set Mol.species index
    mol.species_index = occ.to_species;

    if (m_update_atoms) {
      mol.component.resize(m_convert.components_size(mol.species_index));
    }

    // add to m_loc
    cand_index = m_candidate_list.index(mol.asym, mol.species_index);
    mol.loc = m_loc[cand_index].size();
    m_loc[cand_index].push_back(mol.id);
  }

  if (m_update_atoms) {
    Index i_updating_atom;

    // Atoms moving to the reservoir
    if (m_track_unique_atoms) {
      i_updating_atom = 0;
      for (const auto &traj : e.atom_traj) {
        if (traj.to.l == -1 && traj.from.l != -1) {
          // move to reservoir
          Index atom_id = updating_atoms[i_updating_atom];

          m_atoms[atom_id].translation[0] = std::numeric_limits<long>::min();
          m_atoms[atom_id].translation[1] = std::numeric_limits<long>::min();
          m_atoms[atom_id].translation[2] = std::numeric_limits<long>::min();
          m_atoms[atom_id].n_jumps = std::numeric_limits<long>::min();
          m_unique_atom_id[atom_id] = -1;
          m_initial_atom_species_index[atom_id] = -1;
          m_initial_atom_position_index[atom_id] = -1;

          m_available_atom_id.insert(atom_id);
        }
        ++i_updating_atom;
      }

      // Atoms moving from the reservoir
      i_updating_atom = 0;
      for (const auto &traj : e.atom_traj) {
        if (traj.to.l != -1 && traj.from.l == -1) {
          // move from reservoir -- create a new atom
          Atom atom;
          atom.translation = m_convert.l_to_ijk(traj.to.l);
          atom.n_jumps = 0;
          Index species_index = traj.to.mol_id;
          xtal::Molecule molecule = m_convert.species_to_mol(species_index);
          Index atom_position_index = traj.to.mol_comp;

          Index atom_id;
          if (m_available_atom_id.size() > 0) {
            // If there is an empty position in `m_atoms`,
            // put the added atom there
            atom_id = *m_available_atom_id.begin();
            m_available_atom_id.erase(m_available_atom_id.begin());

            m_atoms[atom_id] = atom;
            m_unique_atom_id[atom_id] = m_next_unique_atom_id;
            m_initial_atom_species_index[atom_id] = species_index;
            m_initial_atom_position_index[atom_id] = atom_position_index;

          } else {
            // Else, add atom to end of m_atoms
            atom_id = m_atoms.size();
            m_atoms.push_back(atom);
            m_unique_atom_id.push_back(m_next_unique_atom_id);
            m_initial_atom_species_index.push_back(species_index);
            m_initial_atom_position_index.push_back(atom_position_index);
          }

          if (m_save_atom_info && time.has_value()) {
            m_atom_info_initial.emplace(
                m_next_unique_atom_id,
                AtomInfo(atom, species_index, atom_position_index,
                         time.value()));
          }

          // update Mol.component
          m_mol[traj.to.mol_id].component[traj.to.mol_comp] = atom_id;

          ++m_next_unique_atom_id;
        }
        ++i_updating_atom;
      }
    }

    // Atoms moving within the supercell
    i_updating_atom = 0;
    for (const auto &traj : e.atom_traj) {
      if (traj.to.l != -1 && traj.from.l != -1) {
        // move to position in supercell from position in supercell
        Index atom_id = updating_atoms[i_updating_atom];

        // update Mol.component
        m_mol[traj.to.mol_id].component[traj.to.mol_comp] = atom_id;

        // update atom translation
        m_atoms[atom_id].translation += traj.delta_ijk;

        // update number of atom jumps
        m_atoms[atom_id].n_jumps += 1;
      }
      ++i_updating_atom;
    }
  }
}

/// \brief Return current atom positions in cartesian coordinates, shape=(3,
/// n_atoms)
///
/// Notes:
/// - Positions are returned with translations included as if no periodic
/// boundaries
Eigen::MatrixXd OccLocation::atom_positions_cart() const {
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(3, this->atom_size());

  auto const &convert = this->convert();
  Eigen::Matrix3d const &L = convert.lat_column_mat();

  // collect atom name indices
  for (Index i = 0; i < this->mol_size(); ++i) {
    monte::Mol const &mol = this->mol(i);
    xtal::Molecule const &molecule = convert.species_to_mol(mol.species_index);
    Eigen::Vector3d site_basis_cart = convert.l_to_basis_cart(mol.l);
    Index atom_position_index = 0;
    for (Index atom_id : mol.component) {
      R.col(atom_id) = site_basis_cart +
                       molecule.atom(atom_position_index).cart() +
                       L * this->atom(atom_id).translation.cast<double>();
      ++atom_position_index;
    }
  }
  return R;
}

/// \brief Return current atom positions in cartesian coordinates, shape=(3,
/// n_atoms)
///
/// Notes:
/// - Positions are returned within periodic boundaries
Eigen::MatrixXd OccLocation::atom_positions_cart_within() const {
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(3, this->atom_size());

  auto const &convert = this->convert();

  // collect atom name indices
  for (Index i = 0; i < this->mol_size(); ++i) {
    monte::Mol const &mol = this->mol(i);
    xtal::Molecule const &molecule = convert.species_to_mol(mol.species_index);
    Eigen::Vector3d site_cart = convert.l_to_cart(mol.l);
    Index atom_position_index = 0;
    for (Index atom_id : mol.component) {
      R.col(atom_id) = site_cart + molecule.atom(atom_position_index).cart();
      ++atom_position_index;
    }
  }
  return R;
}

/// \brief Return current atom names, in order corresponding to columns
///     of atom_positions_cart matrices
///
/// Notes:
/// - Values are set to "UK" if atom is no longer in supercell
std::vector<std::string> OccLocation::current_atom_names() const {
  std::vector<std::string> _atom_names(this->atom_size(), "UK");

  // collect atom name indices
  for (Index i = 0; i < this->mol_size(); ++i) {
    monte::Mol const &mol = this->mol(i);
    xtal::Molecule const &molecule =
        m_convert.species_to_mol(mol.species_index);
    Index atom_position_index = 0;
    for (Index atom_id : mol.component) {
      _atom_names[atom_id] = molecule.atom(atom_position_index).name();
      ++atom_position_index;
    }
  }
  return _atom_names;
}

/// \brief Return current species index for atoms in atom position matricess
///
/// Notes:
/// - Values are set to -1 if atom is no longer in supercell
std::vector<Index> OccLocation::current_atom_species_index() const {
  std::vector<Index> _atom_species_index(this->atom_size(), -1);

  // collect atom name indices
  for (Index i = 0; i < this->mol_size(); ++i) {
    monte::Mol const &mol = this->mol(i);
    for (Index atom_id : mol.component) {
      _atom_species_index[atom_id] = mol.species_index;
    }
  }
  return _atom_species_index;
}

/// \brief Return current atom position index for atoms in atom position
/// matricess
///
/// Notes:
/// - The atom position index is the index into atoms in the Molecule in which
///   the atom is contained
/// - Values are set to -1 if atom is no longer in supercell
std::vector<Index> OccLocation::current_atom_position_index() const {
  std::vector<Index> _atom_position_index(this->atom_size(), -1);

  // collect atom name indices
  for (Index i = 0; i < this->mol_size(); ++i) {
    monte::Mol const &mol = this->mol(i);
    Index atom_position_index = 0;
    for (Index atom_id : mol.component) {
      _atom_position_index[atom_id] = atom_position_index;
      ++atom_position_index;
    }
  }
  return _atom_position_index;
}

/// \brief Return number of jumps made by each atom
std::vector<Index> OccLocation::current_atom_n_jumps() const {
  std::vector<Index> _atom_n_jumps(this->atom_size(), 0);

  // collect atom n_jumps
  for (Index i = 0; i < this->atom_size(); ++i) {
    _atom_n_jumps[i] = this->atom(i).n_jumps;
  }
  return _atom_n_jumps;
}

/// \brief Replace the Mol at a specified site
///
/// \param l Site index
/// \param occ Occupant index for new occupant
/// \param atom_id Atom id for each component of the new occupant
/// \param atom Atom object for each component of the new occupant
///
/// Notes:
/// - This sets the location of the Mol on site `l`
/// - This may only be used after initialization.
/// - This is intended for use restoring saved states in the case when all
///   Atoms remain in the Supercell (i.e. KMC without moves to or from a
///   reservoir) so that unique_atom_id and atom_id are equivalent.
void OccLocation::replace_mol(Index l, int occ,
                              std::vector<Index> const &atom_id,
                              std::vector<Atom> const &atom) {
  if (m_track_unique_atoms) {
    throw std::runtime_error(
        "Error in OccLocation::replace_mol: "
        "`replace_mol` is not supported when tracking unique atoms");
  }

  Index cand_index;
  Index back;
  auto &mol = m_mol[l_to_mol_id(l)];

  // Unset current Mol location
  cand_index = m_candidate_list.index(mol.asym, mol.species_index);
  back = m_loc[cand_index].back();
  m_loc[cand_index][mol.loc] = back;
  m_mol[back].loc = mol.loc;
  m_loc[cand_index].pop_back();

  // Set new Mol location
  mol.species_index = m_convert.species_index(mol.asym, occ);
  cand_index = m_candidate_list.index(mol.asym, mol.species_index);
  mol.loc = m_loc[cand_index].size();
  m_loc[cand_index].push_back(mol.id);

  // Set Atom
  mol.component.clear();
  auto it = atom.begin();
  auto end = atom.end();
  auto atom_id_it = atom_id.begin();
  for (; it != end; ++atom_id_it, ++it) {
    mol.component.push_back(*atom_id_it);
    m_atoms[*atom_id_it] = *it;
  }
}

}  // namespace monte
}  // namespace CASM
