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
      m_update_atoms(_update_atoms) {
  if (m_update_atoms) {
    m_resevoir_mol.resize(m_convert.species_size());
    for (Index species_index = 0; species_index < m_convert.species_size();
         ++species_index) {
      Mol &mol = m_resevoir_mol[species_index];
      mol.id = species_index;
      mol.l = -1;
      mol.asym = -1;
      mol.species_index = species_index;
      mol.loc = -1;
      int n_atoms = m_convert.species_to_mol(species_index).atoms().size();
      mol.component.resize(n_atoms);
    }
  }
}

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
          mol.component.push_back(m_atoms.size());
          atom.translation = xtal::UnitCell(0, 0, 0);
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
}

/// Update occupation vector and this to reflect that event 'e' occurred
void OccLocation::apply(const OccEvent &e, Eigen::VectorXi &occupation) {
  static std::vector<Index> updating_atoms;

  // copy original Mol.component
  if (m_update_atoms) {
    if (updating_atoms.size() < e.atom_traj.size()) {
      updating_atoms.resize(e.atom_traj.size());
    }
    Index i_updating_atom = 0;
    for (const auto &traj : e.atom_traj) {
      if (traj.from.l == -1) {
        // move from resevoir -- create a new atom
        Atom atom;
        atom.translation = xtal::UnitCell(0, 0, 0);
        m_resevoir_mol[traj.from.mol_id].component[traj.from.mol_comp] =
            m_atoms.size();
        updating_atoms[i_updating_atom] = m_atoms.size();
        m_atoms.push_back(atom);
      } else {  // move from within supercell
        updating_atoms[i_updating_atom] =
            m_mol[traj.from.mol_id].component[traj.from.mol_comp];
      }
      ++i_updating_atom;
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
    Index i_updating_atom = 0;
    for (const auto &traj : e.atom_traj) {
      if (traj.to.l != -1) {
        // move to poisition in supercell
        Index atom_id = updating_atoms[i_updating_atom];

        // update Mol.component
        m_mol[traj.to.mol_id].component[traj.to.mol_comp] = atom_id;

        // update atom translation
        m_atoms[atom_id].translation += traj.delta_ijk;
      }
      // else {
      //   // move to resevoir
      //   // mark explicitly?
      //   // or know implicitly (because not found in
      //   m_mol[mol_id]->component)?
      // }
      ++i_updating_atom;
    }
  }
}

}  // namespace monte
}  // namespace CASM
