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
/// \param _update_atoms If true, track species trajectories when
///     applying OccEvent
OccLocation::OccLocation(const Conversions &_convert,
                         const OccCandidateList &_candidate_list,
                         bool _update_atoms)
    : m_convert(_convert),
      m_candidate_list(_candidate_list),
      m_loc(_candidate_list.size()),
      m_update_atoms(_update_atoms) {}

/// Fill tables with occupation info
void OccLocation::initialize(Eigen::VectorXi const &occupation) {
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
        int n_atoms = m_convert.species_to_mol(species_index).atoms().size();
        for (Index atom_index = 0; atom_index < n_atoms; ++atom_index) {
          Atom atom;
          atom.species_index = species_index;
          atom.atom_index = atom_index;
          atom.id = m_atoms.size();
          mol.component.push_back(atom.id);
          atom.delta_ijk = xtal::UnitCell(0, 0, 0);
          atom.bijk_begin = m_convert.l_to_bijk(l);
          atom.species_index_begin = species_index;
          atom.atom_index_begin = atom_index;

          m_atoms.push_back(atom);
        }
      }

      m_loc[cand_index].push_back(mol_id);
      m_l_to_mol.push_back(mol_id);
      mol_id++;
    } else {
      m_l_to_mol.push_back(Nmut);
    }
  }
  if (m_update_atoms) {
    m_tmol = m_mol;
  }
}

/// Stochastically choose an occupant of a particular OccCandidate type
Mol const &OccLocation::choose_mol(Index cand_index, MTRand &mtrand) const {
  return mol(m_loc[cand_index][mtrand.randInt(m_loc[cand_index].size() - 1)]);
}

/// Stochastically choose an occupant of a particular OccCandidate type
Mol const &OccLocation::choose_mol(OccCandidate const &cand,
                                   MTRand &mtrand) const {
  return choose_mol(m_candidate_list.index(cand), mtrand);
}

/// Update occupation vector and this to reflect that event 'e' occurred
void OccLocation::apply(const OccEvent &e, Eigen::VectorXi &occupation) {
  // copy original Mol.component
  if (m_update_atoms) {
    for (const auto &occ : e.occ_transform) {
      m_tmol[occ.mol_id].component = m_mol[occ.mol_id].component;
    }
  }

  // update Mol and config occupation
  for (const auto &occ : e.occ_transform) {
    auto &mol = m_mol[occ.mol_id];

    // set config occupation
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
    for (const auto &traj : e.atom_traj) {
      auto &to_mol = m_mol[traj.to.mol_id];
      auto &from_mol = m_tmol[traj.from.mol_id];
      Index atom_id = from_mol.component[traj.from.mol_comp];

      // update Mol.component
      to_mol.component[traj.to.mol_comp] = atom_id;

      // update atom species_index, atom_index, delta_ijk
      m_atoms[atom_id].species_index = to_mol.species_index;
      m_atoms[atom_id].atom_index = traj.to.mol_comp;
      m_atoms[atom_id].delta_ijk += traj.delta_ijk;
    }
  }
}

/// Total number of mutating sites
OccLocation::size_type OccLocation::mol_size() const { return m_mol.size(); }

Mol &OccLocation::mol(Index mol_id) { return m_mol[mol_id]; }

const Mol &OccLocation::mol(Index mol_id) const { return m_mol[mol_id]; }

/// Total number of atoms
OccLocation::size_type OccLocation::atom_size() const { return m_atoms.size(); }

/// Access Atom by id
Atom &OccLocation::atom(Index atom_id) { return m_atoms[atom_id]; }

/// Access Atom by id
Atom const &OccLocation::atom(Index atom_id) const { return m_atoms[atom_id]; }

/// Access the OccCandidateList
OccCandidateList const &OccLocation::candidate_list() const {
  return m_candidate_list;
}

/// Total number of mutating sites, of OccCandidate type, specified by index
OccLocation::size_type OccLocation::cand_size(Index cand_index) const {
  return m_loc[cand_index].size();
}

/// Total number of mutating sites, of OccCandidate type
OccLocation::size_type OccLocation::cand_size(const OccCandidate &cand) const {
  return cand_size(m_candidate_list.index(cand));
}

/// The index into the configuration of a particular mutating site
Index OccLocation::mol_id(Index cand_index, Index loc) const {
  return m_loc[cand_index][loc];
}

/// The index into the configuration of a particular mutating site
Index OccLocation::mol_id(const OccCandidate &cand, Index loc) const {
  return mol_id(m_candidate_list.index(cand), loc);
}

/// Convert from config index to variable site index
Index OccLocation::l_to_mol_id(Index l) const { return m_l_to_mol[l]; }

/// Get Conversions objects
Conversions const &OccLocation::convert() const { return m_convert; }

}  // namespace monte
}  // namespace CASM
