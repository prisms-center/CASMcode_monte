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
        xtal::Molecule const &molecule =
            m_convert.species_to_mol(species_index);
        int n_atoms = molecule.atoms().size();
        for (Index atom_index = 0; atom_index < n_atoms; ++atom_index) {
          mol.component.push_back(m_atoms.size());
          Atom atom;
          atom.translation = m_convert.l_to_ijk(mol.l);
          atom.n_jumps = 0;
          m_atoms.push_back(atom);
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
        atom.translation = m_convert.l_to_ijk(traj.to.l);
        atom.n_jumps = 0;
        Index species_index = traj.from.mol_id;
        xtal::Molecule molecule = m_convert.species_to_mol(species_index);
        Index atom_position_index = traj.from.mol_comp;
        m_resevoir_mol[species_index].component[atom_position_index] =
            m_atoms.size();
        updating_atoms[i_updating_atom] = m_atoms.size();
        m_atoms.push_back(atom);
        m_initial_atom_species_index.push_back(species_index);
        m_initial_atom_position_index.push_back(atom_position_index);
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
    Index i_updating_atom = 0;
    for (const auto &traj : e.atom_traj) {
      if (traj.to.l != -1) {
        // move to position in supercell
        Index atom_id = updating_atoms[i_updating_atom];

        // update Mol.component
        m_mol[traj.to.mol_id].component[traj.to.mol_comp] = atom_id;

        // update atom translation
        m_atoms[atom_id].translation += traj.delta_ijk;

        // update number of atom jumps
        m_atoms[atom_id].n_jumps += 1;
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

/// \brief Return current atom positions in cartesian coordinates, shape=(3,
/// n_atoms)
///
/// Notes:
/// - Positions are returned with translations included as if no periodic
/// boundaries
Eigen::MatrixXd OccLocation::atom_positions_cart() const {
  Eigen::MatrixXd R(3, this->atom_size());

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
  Eigen::MatrixXd R(3, this->atom_size());

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

}  // namespace monte
}  // namespace CASM
