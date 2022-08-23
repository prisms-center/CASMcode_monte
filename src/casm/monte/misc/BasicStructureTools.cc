#include "casm/monte/misc/BasicStructureTools.hh"

#include "casm/crystallography/BasicStructure.hh"
#include "casm/misc/algorithm.hh"

// debug
#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/container/stream_io.hh"
#include "casm/casm_io/json/jsonParser.hh"

namespace CASM {
namespace monte {

/// \brief Check if a molecule is contained in a list, in given orientation
bool is_contained_in_this_orientation(
    std::vector<xtal::Molecule> const &molecule_list,
    xtal::Molecule const &molecule, double tol) {
  if (molecule.name().empty()) {
    throw std::runtime_error("Error: molecule has empty name");
  }
  auto it = std::find_if(
      molecule_list.begin(), molecule_list.end(),
      [&](xtal::Molecule const &x) { return x.identical(molecule, tol); });
  if (it != molecule_list.end()) {
    if (it->name() != molecule.name()) {
      throw std::runtime_error(
          "Error: equivalent molecules have different names");
    }
    return true;
  }
  return false;
}

/// \brief Check if a molecule is contained in a list, in any orientation
bool is_contained_in_any_orientation(
    std::vector<xtal::Molecule> const &molecule_list,
    xtal::Molecule const &molecule,
    std::vector<xtal::SymOp> const &factor_group, double tol) {
  for (auto const &op : factor_group) {
    xtal::Molecule tmol = sym::copy_apply(op, molecule);
    if (is_contained_in_this_orientation(molecule_list, tmol, tol)) {
      return true;
    }
  }
  return false;
}

/// \brief Check that all molecules have non-empty names and that
///     symmetrically equivalent molecules have the same name
bool is_valid_molecule_naming(xtal::BasicStructure const &prim,
                              std::vector<xtal::SymOp> const &factor_group) {
  try {
    auto molecule_list = molecule_list_single_orientation(prim, factor_group);
  } catch (std::exception &e) {
    return false;
  }
  return true;
}

/// \brief Generate a list of all molecule orientations in a prim
std::vector<xtal::Molecule> molecule_list_all_orientations(
    xtal::BasicStructure const &prim) {
  double tol = prim.lattice().tol();
  std::vector<xtal::Molecule> molecule_list;
  for (auto const &site : prim.basis()) {
    for (auto const &mol : site.occupant_dof()) {
      if (!is_contained_in_this_orientation(molecule_list, mol, tol)) {
        molecule_list.emplace_back(mol);
      }
    }
  }
  return molecule_list;
}

/// \brief Generate a list of symmetrically unique molecules in a prim
std::vector<xtal::Molecule> molecule_list_single_orientation(
    xtal::BasicStructure const &prim,
    std::vector<xtal::SymOp> const &factor_group) {
  double tol = prim.lattice().tol();
  std::vector<xtal::Molecule> molecule_list;
  for (auto const &site : prim.basis()) {
    for (auto const &mol : site.occupant_dof()) {
      if (!is_contained_in_any_orientation(molecule_list, mol, factor_group,
                                           tol)) {
        molecule_list.emplace_back(mol);
      }
    }
  }
  return molecule_list;
}

/// \brief Generate a list of names of molecules in a prim, using
///     the `prim->unique_names()`
std::vector<std::string> make_orientation_name_list(
    xtal::BasicStructure const &prim) {
  std::vector<xtal::Molecule> molecule_list =
      molecule_list_all_orientations(prim);
  std::vector<std::string> orientation_name_list;
  for (auto const &mol : molecule_list) {
    orientation_name_list.push_back(orientation_name(mol, prim));
  }
  return orientation_name_list;
}

/// \brief Generate a list of names of molecules in a prim, using
///     the `prim->unique_names()`
std::vector<std::string> make_orientation_name_list(
    std::vector<xtal::Molecule> const &molecule_list,
    xtal::BasicStructure const &prim) {
  std::vector<std::string> orientation_name_list;
  for (auto const &mol : molecule_list) {
    orientation_name_list.push_back(orientation_name(mol, prim));
  }
  return orientation_name_list;
}

/// \brief Generate a list of names of symmetrically unique molecules
///     from a list of all molecule orientations
std::vector<std::string> make_chemical_name_list(
    xtal::BasicStructure const &prim,
    std::vector<xtal::SymOp> const &factor_group) {
  std::vector<xtal::Molecule> molecule_list =
      molecule_list_single_orientation(prim, factor_group);
  std::vector<std::string> chemical_name_list;
  for (auto const &mol : molecule_list) {
    chemical_name_list.push_back(mol.name());
  }
  return chemical_name_list;
}

/// \brief Generate a list of unique atom names (sorted)
std::vector<std::string> make_atom_name_list(xtal::BasicStructure const &prim) {
  std::vector<std::string> atom_name_list;

  Index b = 0;
  for (auto const &site : prim.basis()) {
    Index occupant_index = 0;
    for (auto const &mol : site.occupant_dof()) {
      for (auto const &atom : mol.atoms()) {
        Index atom_name_index = find_index(atom_name_list, atom.name());
        if (atom_name_index == atom_name_list.size()) {
          atom_name_index = atom_name_list.size();
          atom_name_list.push_back(atom.name());
        }
      }
      ++occupant_index;
    }

    ++b;
  }
  std::sort(atom_name_list.begin(), atom_name_list.end());
  return atom_name_list;
}

/// \brief Generate a list of molecules in a prim, exclude symmetrically
///     equivalent orientations
std::vector<xtal::Molecule> molecule_list_single_orientation(
    std::vector<xtal::Molecule> const &molecule_list_all_orientations,
    std::vector<xtal::SymOp> const &factor_group, double tol) {
  std::vector<xtal::Molecule> molecule_list;
  for (auto const &mol : molecule_list_all_orientations) {
    if (!is_contained_in_any_orientation(molecule_list, mol, factor_group,
                                         tol)) {
      molecule_list.emplace_back(mol);
    }
  }
  return molecule_list;
}

/// \brief Get the unique name of a specific molecule orientation
std::string orientation_name(xtal::Molecule const &molecule,
                             xtal::BasicStructure const &prim) {
  double tol = prim.lattice().tol();
  if (prim.unique_names().size() != prim.basis().size()) {
    throw std::runtime_error("Error in orientation_name: basis size mismatch");
  }

  auto basis_unique_names_it = prim.unique_names().begin();
  for (auto const &site : prim.basis()) {
    if (basis_unique_names_it->size() != site.occupant_dof().size()) {
      throw std::runtime_error(
          "Error in orientation_name: occupant size mismatch");
    }
    auto occupant_dof_unique_names_it = basis_unique_names_it->begin();
    for (auto const &mol : site.occupant_dof()) {
      if (molecule.identical(mol, tol)) {
        return *occupant_dof_unique_names_it;
      }
      ++occupant_dof_unique_names_it;
    }
    ++basis_unique_names_it;
  }
  throw std::runtime_error(
      "Error in orientation_name: molecule not found in prim");
}

}  // namespace monte
}  // namespace CASM
