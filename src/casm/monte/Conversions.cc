#include "casm/monte/Conversions.hh"

#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/BasicStructureTools.hh"
#include "casm/misc/algorithm.hh"
#include "casm/monte/misc/BasicStructureTools.hh"

namespace CASM {
namespace monte {

namespace {

/// Return indices of equivalent basis sites by symmetry and order of occupants
std::vector<Index> make_b_to_asym(
    const xtal::BasicStructure &struc,
    std::vector<xtal::Molecule> const &species_list) {
  std::vector<Index> b_to_asym(struc.basis().size());
  std::set<std::set<Index>> asym_unit = xtal::make_asymmetric_unit(struc);

  // [b][occ] -> species
  auto index_converter = xtal::make_index_converter(struc, species_list);

  // { asym by symmetry, occ->species }
  typedef std::pair<Index, std::vector<Index>> sublat_type;
  typedef Index sublat_index;
  std::map<sublat_type, std::vector<sublat_index>> asym_by_sym_and_occ;

  Index asym_by_symmetry = 0;
  for (auto const &orbit : asym_unit) {
    for (Index b : orbit) {
      sublat_type key = std::make_pair(asym_by_symmetry, index_converter[b]);
      asym_by_sym_and_occ[key].push_back(b);
    }
    ++asym_by_symmetry;
  }

  Index asym = 0;
  for (auto const &pair : asym_by_sym_and_occ) {
    for (Index b : pair.second) {
      b_to_asym[b] = asym;
    }
    ++asym;
  }
  return b_to_asym;
}

/// Return indices of equivalent basis sites
std::vector<Index> make_b_to_asym(const xtal::BasicStructure &struc) {
  return make_b_to_asym(struc, molecule_list_all_orientations(struc));
}

}  // namespace

/// \brief Constructor (uses asymmetric unit determined from prim factor group)
///
/// \param prim The primitive structure
/// \param transformation_matrix_to_super Defines a supercell lattice,
///     S = P * T, where S = supercell lattice column matrix, P = prim lattice
///     column matrix, T = transformation_matrix_to_super.
///
/// This overload uses the prim factor group symmetry to determine the
/// asymmetric unit.
Conversions::Conversions(xtal::BasicStructure const &prim,
                         Eigen::Matrix3l const &transformation_matrix_to_super)
    : Conversions(prim, molecule_list_all_orientations(prim),
                  transformation_matrix_to_super, Eigen::Matrix3l::Identity(),
                  make_b_to_asym(prim)) {}

/// \brief Constructor (user specified asymmetric unit with reduced symmetry)
///
/// \param prim The primitive structure
/// \param transformation_matrix_to_super Defines a supercell lattice,
///     S = P * T, where S = supercell lattice column matrix, P = prim lattice
///     column matrix, T = transformation_matrix_to_super.
/// \param b_to_asym Specifies the asymmetric unit orbit index
///     corresponding to each sublattice in the prim. Asymmetric unit orbit
///     indices are distinct indices `(0, 1, ...)` indicating that sites with
///     the same index map onto each other via symmetry operations *and* have
///     occupants listed in the same order.
///
/// This overload allows specifying lower symmetry than the prim factor group
/// (but same periodicity) to determine the asymmetric unit.
///
Conversions::Conversions(xtal::BasicStructure const &prim,
                         Eigen::Matrix3l const &transformation_matrix_to_super,
                         std::vector<Index> const &b_to_asym)
    : Conversions(prim, molecule_list_all_orientations(prim),
                  transformation_matrix_to_super, Eigen::Matrix3l::Identity(),
                  b_to_asym) {}

/// \brief Constructor (user specified asymmetric unit with reduced
/// translational symmetry)
///
/// \param prim The primitive structure
/// \param species_list Vector of all distinct molecules, including each
///     orientation.
/// \param transformation_matrix_to_super Defines a supercell lattice,
///     S = P * T, where S = supercell lattice column matrix, P = prim lattice
///     column matrix, T = transformation_matrix_to_super.
/// \param unit_transformation_matrix_to_super Defines a sub-supercell lattice,
///     U = P * T, where U = supercell lattice column matrix, P = prim lattice
///     column matrix, T_unit = unit_transformation_matrix_to_super. U must
///     tile into S (i.e. S = U * T', where T' is an integer matrix). Allows
///     specifying an asymmetric unit which does not fit in the primitive cell.
/// \param unitl_to_asym Specifies the asymmetric unit orbit index
///     corresponding to each site in the supercell U. Asymmetric unit orbit
///     indices are distinct indices `(0, 1, ...)` indicating that sites with
///     the same index map onto each other via symmetry operations *and* have
///     occupants listed in the same order.
///
/// This overload allows specifying an asymmetric unit which does not fit in
/// the primitive cell.
///
Conversions::Conversions(
    xtal::BasicStructure const &prim,
    std::vector<xtal::Molecule> const &species_list,
    Eigen::Matrix3l const &transformation_matrix_to_super,
    Eigen::Matrix3l const &unit_transformation_matrix_to_super,
    std::vector<Index> const &unitl_to_asym)
    : m_unitcell_index_converter(transformation_matrix_to_super),
      m_unit_transformation_matrix_to_super(
          unit_transformation_matrix_to_super),
      m_unitl_and_bijk_converter(unit_transformation_matrix_to_super,
                                 prim.basis().size()),
      m_transformation_matrix_to_super(transformation_matrix_to_super),
      m_l_and_bijk_converter(transformation_matrix_to_super,
                             prim.basis().size()),
      m_struc_mol(species_list),
      m_struc_molname(make_orientation_name_list(m_struc_mol, prim)),
      m_unitl_to_asym(unitl_to_asym) {
  m_lat_column_mat = prim.lattice().lat_column_mat();
  for (Index b = 0; b < prim.basis().size(); ++b) {
    m_basis_cart.push_back(prim.basis()[b].const_cart());
    m_basis_frac.push_back(prim.basis()[b].const_frac());
  }

  // find m_Nasym
  m_Nasym =
      *std::max_element(m_unitl_to_asym.begin(), m_unitl_to_asym.end()) + 1;

  // make m_asym_to_unitl & m_asym_to_b
  Index unit_Nsites =
      unit_transformation_matrix_to_super.determinant() * prim.basis().size();
  m_asym_to_unitl.resize(m_Nasym);
  m_asym_to_b.resize(m_Nasym);
  for (Index unitl = 0; unitl < unit_Nsites; ++unitl) {
    Index asym = m_unitl_to_asym[unitl];
    m_asym_to_unitl[asym].insert(unitl);
    m_asym_to_b[asym].insert(unitl_to_b(unitl));
  }

  // make m_occ_to_species and m_species_to_occ

  // [b][occ] -> species
  auto index_converter = xtal::make_index_converter(prim, m_struc_mol);

  // [b][species] -> occ, index_converter[b].size() if not allowed
  std::vector<std::vector<Index>> index_converter_inv;
  for (Index b = 0; b < index_converter.size(); ++b) {
    std::vector<Index> occ_indices(m_struc_mol.size(),
                                   index_converter[b].size());
    Index occ_index = 0;
    for (auto species_index : index_converter[b]) {
      occ_indices[species_index] = occ_index;
      ++occ_index;
    }
    index_converter_inv.push_back(occ_indices);
  }

  m_occ_to_species.resize(m_Nasym);
  m_species_to_occ.resize(m_Nasym);
  for (Index asym = 0; asym < m_Nasym; ++asym) {
    Index b = *(m_asym_to_b[asym].begin());
    m_occ_to_species[asym] = index_converter[b];
    m_species_to_occ[asym] = index_converter_inv[b];
  }
}

Eigen::Matrix3d Conversions::lat_column_mat() const { return m_lat_column_mat; }

Index Conversions::l_size() const {
  return m_l_and_bijk_converter.total_sites();
}

Index Conversions::l_to_b(Index l) const {
  return m_l_and_bijk_converter(l).sublattice();
}

xtal::UnitCell Conversions::l_to_ijk(Index l) const {
  return m_l_and_bijk_converter(l).unitcell();
}

xtal::UnitCellCoord Conversions::l_to_bijk(Index l) const {
  return m_l_and_bijk_converter(l);
}

Index Conversions::l_to_unitl(Index l) const {
  return bijk_to_unitl(l_to_bijk(l));
}

Index Conversions::l_to_asym(Index l) const {
  return m_unitl_to_asym[l_to_unitl(l)];
}

Eigen::Vector3d Conversions::l_to_cart(Index l) const {
  xtal::UnitCellCoord bijk = l_to_bijk(l);
  return m_basis_cart[bijk.sublattice()] +
         lat_column_mat() * bijk.unitcell().cast<double>();
}

Eigen::Vector3d Conversions::l_to_frac(Index l) const {
  xtal::UnitCellCoord bijk = l_to_bijk(l);
  return m_basis_frac[bijk.sublattice()] + bijk.unitcell().cast<double>();
}

Eigen::Vector3d Conversions::l_to_basis_cart(Index l) const {
  return m_basis_cart[l_to_b(l)];
}

Eigen::Vector3d Conversions::l_to_basis_frac(Index l) const {
  return m_basis_frac[l_to_b(l)];
}

Index Conversions::bijk_to_l(xtal::UnitCellCoord const &bijk) const {
  return m_l_and_bijk_converter(bijk);
}
Index Conversions::bijk_to_unitl(xtal::UnitCellCoord const &bijk) const {
  return m_unitl_and_bijk_converter(bijk);
}

Index Conversions::bijk_to_asym(xtal::UnitCellCoord const &bijk) const {
  return l_to_asym(bijk_to_l(bijk));
}

Index Conversions::unitl_size() const {
  return m_unitl_and_bijk_converter.total_sites();
}

Index Conversions::unitl_to_b(Index unitl) const {
  return m_unitl_and_bijk_converter(unitl).sublattice();
}

xtal::UnitCellCoord Conversions::unitl_to_bijk(Index unitl) const {
  return m_unitl_and_bijk_converter(unitl);
}

Index Conversions::unitl_to_asym(Index unitl) const {
  return m_unitl_to_asym[unitl];
}

Index Conversions::asym_size() const { return m_Nasym; }

std::set<Index> const &Conversions::asym_to_b(Index asym) const {
  return m_asym_to_b[asym];
}

std::set<Index> const &Conversions::asym_to_unitl(Index asym) const {
  return m_asym_to_unitl[asym];
}

Eigen::Matrix3l const &Conversions::unit_transformation_matrix_to_super()
    const {
  return m_unit_transformation_matrix_to_super;
}

Eigen::Matrix3l const &Conversions::transformation_matrix_to_super() const {
  return m_transformation_matrix_to_super;
}

xtal::UnitCellIndexConverter const &Conversions::unitcell_index_converter()
    const {
  return m_unitcell_index_converter;
}

xtal::UnitCellCoordIndexConverter const &Conversions::unit_index_converter()
    const {
  return m_unitl_and_bijk_converter;
}

xtal::UnitCellCoordIndexConverter const &Conversions::index_converter() const {
  return m_l_and_bijk_converter;
}

Index Conversions::occ_size(Index asym) const {
  return m_occ_to_species[asym].size();
}

Index Conversions::species_index(Index asym, Index occ_index) const {
  return m_occ_to_species[asym][occ_index];
}

std::vector<xtal::Molecule> const &Conversions::species_list() const {
  return m_struc_mol;
}

Index Conversions::occ_index(Index asym, Index species_index) const {
  // returns occ_size(asym) if species not allowed
  return m_species_to_occ[asym][species_index];
}

bool Conversions::species_allowed(Index asym, Index species_index) const {
  return occ_index(asym, species_index) != occ_size(asym);
}

Index Conversions::species_size() const { return m_struc_mol.size(); }

Index Conversions::species_index(std::string species_name) const {
  return find_index(m_struc_molname, species_name);
}
xtal::Molecule const &Conversions::species_to_mol(Index species_index) const {
  return m_struc_mol[species_index];
}
std::string const &Conversions::species_name(Index species_index) const {
  return m_struc_molname[species_index];
}
Index Conversions::components_size(Index species_index) const {
  return species_to_mol(species_index).size();
}

}  // namespace monte
}  // namespace CASM
