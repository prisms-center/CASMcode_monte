#ifndef CASM_monte_misc_BasicStructureTools
#define CASM_monte_misc_BasicStructureTools

#include <string>
#include <vector>

#include "casm/global/definitions.hh"

namespace CASM {

namespace xtal {
class BasicStructure;
class Molecule;
struct SymOp;
class UnitCellCoord;
}  // namespace xtal

namespace monte {

/// \brief Check if a molecule is contained in a list, in given orientation
bool is_contained_in_this_orientation(
    std::vector<xtal::Molecule> const &molecule_list,
    xtal::Molecule const &molecule, double tol);

/// \brief Check if a molecule is contained in a list, in any orientation
bool is_contained_in_any_orientation(
    std::vector<xtal::Molecule> const &molecule_list,
    xtal::Molecule const &molecule,
    std::vector<xtal::SymOp> const &factor_group, double tol);

/// \brief Check that all molecules have non-empty names and that
///     symmetrically equivalent molecules have the same name
bool is_valid_molecule_naming(xtal::BasicStructure const &prim,
                              std::vector<xtal::SymOp> const &factor_group);

/// \brief Generate a list of all molecule orientations in a prim
std::vector<xtal::Molecule> molecule_list_all_orientations(
    xtal::BasicStructure const &prim);

/// \brief Generate a list of molecules in a prim, exclude symmetrically
///     equivalent orientations
std::vector<xtal::Molecule> molecule_list_single_orientation(
    xtal::BasicStructure const &prim,
    std::vector<xtal::SymOp> const &factor_group);

/// \brief Generate a list of symmetrically unique molecules from a
///     list of all molecule orientations
std::vector<xtal::Molecule> molecule_list_single_orientation(
    std::vector<xtal::Molecule> const &molecule_list_all_orientations,
    std::vector<xtal::SymOp> const &factor_group, double tol);

/// \brief Generate a list of names of molecules in a prim, using
///     the `prim->unique_names()`
std::vector<std::string> make_orientation_name_list(
    xtal::BasicStructure const &prim);

/// \brief Generate a list of names of molecules in a prim, using
///     the `prim->unique_names()`
std::vector<std::string> make_orientation_name_list(
    std::vector<xtal::Molecule> const &molecule_list,
    xtal::BasicStructure const &prim);

/// \brief Generate a list of names of symmetrically unique molecules
///     from a list of all molecule orientations
std::vector<std::string> make_chemical_name_list(
    xtal::BasicStructure const &prim,
    std::vector<xtal::SymOp> const &factor_group);

/// \brief Generate a list of unique atom names
std::vector<std::string> make_atom_name_list(xtal::BasicStructure const &prim);

/// \brief Get the unique name of a specific molecule orientation
std::string orientation_name(xtal::Molecule const &molecule,
                             xtal::BasicStructure const &prim);

}  // namespace monte
}  // namespace CASM

#endif
