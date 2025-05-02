#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// nlohmann::json binding
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "pybind11_json/pybind11_json.hpp"

// std
#include <random>

// CASM
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/monte/Conversions.hh"
#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/events/OccCandidate.hh"
#include "casm/monte/events/OccEvent.hh"
#include "casm/monte/events/OccEventProposal.hh"
#include "casm/monte/events/OccLocation.hh"
#include "casm/monte/events/io/OccCandidate_json_io.hh"
#include "casm/monte/events/io/OccCandidate_stream_io.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

// used for libcasm.monte:
typedef monte::default_engine_type engine_type;
typedef monte::RandomNumberGenerator<engine_type> generator_type;

std::shared_ptr<monte::Conversions> make_monte_conversions(
    xtal::BasicStructure const &prim,
    Eigen::Matrix3l const &transformation_matrix_to_super) {
  return std::make_shared<monte::Conversions>(prim,
                                              transformation_matrix_to_super);
}

monte::OccCandidateList make_OccCandidateList(
    monte::Conversions const &convert,
    std::optional<std::vector<monte::OccCandidate>> candidates) {
  if (candidates.has_value()) {
    return monte::OccCandidateList(*candidates, convert);
  }
  return monte::OccCandidateList(convert);
}

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<CASM::Index>);
PYBIND11_MAKE_OPAQUE(std::vector<CASM::monte::Atom>);
PYBIND11_MAKE_OPAQUE(std::vector<CASM::monte::AtomTraj>);
PYBIND11_MAKE_OPAQUE(std::vector<CASM::monte::Mol>);
PYBIND11_MAKE_OPAQUE(std::vector<CASM::monte::OccTransform>);
PYBIND11_MAKE_OPAQUE(std::map<CASM::monte::OccSwap, int>);
PYBIND11_MAKE_OPAQUE(std::map<CASM::Index, CASM::monte::AtomInfo>);

PYBIND11_MODULE(_monte_events, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Monte Carlo simulation events

        libcasm.monte.events
        --------------------

        Data structures for representing (kinetic) Monte Carlo events, and methods
        for proposing and applying events.
    )pbdoc";
  py::module::import("libcasm.xtal");

  py::bind_vector<std::vector<int>>(m, "IntVector");
  py::bind_vector<std::vector<Index>>(m, "LongVector");

  py::class_<monte::Conversions, std::shared_ptr<monte::Conversions>>(
      m, "Conversions", R"pbdoc(
    Data structure used for index conversions

    Notes
    -----
    The following shorthand is used for member function names:

    - `l`, :math:`l`: Linear site index in a particular supercell
    - `b`, :math:`b`: :class:`~libcasm.xtal.Prim` sublattice index
    - `unitl`, :math:`l'`: Linear site index in a non-primitive unit cell. When a
      non-primitive unit cell is used to construct a supercell and determines the
      appropriate symmetry for a problem, conversions between :math:`l`, :math:`b`,
      and :math:`l'` may all be useful.
    - `ijk`, :math:`(i,j,k)`: Integer unit cell indices (fractional coordinates with
      respect to the :class:`~libcasm.xtal.Prim` lattice vectors)
    - `bijk`, :math:`(b, i, j, k)`: Integral site coordinates (sublattice index and
      integer unit cell indices)
    - `asym`, :math:`a`: Asymmetric unit orbit index (value is the same for all
      sites which are symmetrically equivalent and have occupants listed in the
      same order)
    - `occ_index`, :math:`s`: Index into occupant list for a particular site
    - `species_index`: Index into the molecule list for a particular
      :class:`~libcasm.xtal.Prim`. If there are orientational variants,
      `species_index` should correspond to `orientation_index`.

    )pbdoc")
      .def(py::init<>(&make_monte_conversions),
           R"pbdoc(
         .. rubric:: Constructor

         Parameters
         ----------
         xtal_prim : libcasm.xtal.Prim
             A :class:`~libcasm.xtal.Prim`

         transformation_matrix_to_super: array_like, shape=(3,3), dtype=int
             The transformation matrix, :math:`T`, relating the superstructure lattice vectors, :math:`S`, to the prim lattice vectors, :math:`P`, according to :math:`S = P T`, where :math:`S` and :math:`P` are shape=(3,3) matrices with lattice vectors as columns.

         )pbdoc",
           py::arg("xtal_prim"), py::arg("transformation_matrix_to_super"))
      //
      .def_static(
          "make_with_custom_asym",
          [](xtal::BasicStructure const &xtal_prim,
             Eigen::Matrix3l const &transformation_matrix_to_super,
             std::vector<Index> const &b_to_asym) {
            return std::make_shared<monte::Conversions>(
                xtal_prim, transformation_matrix_to_super, b_to_asym);
          },
          R"pbdoc(
        Construct a Conversions object with lower symmetry than the :class:`~libcasm.xtal.Prim`.

        Parameters
        ----------
        xtal_prim : libcasm.xtal.Prim
            A :class:`~libcasm.xtal.Prim`

        transformation_matrix_to_super: array_like, shape=(3,3), dtype=int
            The transformation matrix, :math:`T`, relating the superstructure lattice vectors, :math:`S`, to the prim lattice vectors, :math:`P`, according to :math:`S = P T`, where :math:`S` and :math:`P` are shape=(3,3) matrices with lattice vectors as columns.

        b_to_asym: List[int]
            Specifies the asymmetric unit orbit index corresponding to each sublattice in the prim. Asymmetric unit orbit indices are distinct indices `(0, 1, ...)` indicating that sites with the same index map onto each other via symmetry operations and have occupants listed in the same order.

            This option allows specifying lower symmetry than the prim factor group
             (but same periodicity) to determine the asymmetric unit.

        )pbdoc",
          py::arg("xtal_prim"), py::arg("transformation_matrix_to_super"),
          py::arg("b_to_asym"))
      .def_static(
          "make_with_custom_unitcell",
          [](xtal::BasicStructure const &xtal_prim,
             std::vector<xtal::Molecule> const &species_list,
             Eigen::Matrix3l const &transformation_matrix_to_super,
             Eigen::Matrix3l const &unit_transformation_matrix_to_super,
             std::vector<Index> const &unitl_to_asym) {
            return std::make_shared<monte::Conversions>(
                xtal_prim, species_list, transformation_matrix_to_super,
                unit_transformation_matrix_to_super, unitl_to_asym);
          },
          R"pbdoc(
        Construct a Conversions object for a system with an asymmetric unit which does not fit in the primitive cell.

        Parameters
        ----------
        xtal_prim : libcasm.xtal.Prim
            A :class:`~libcasm.xtal.Prim`

        transformation_matrix_to_super: array_like, shape=(3,3), dtype=int
            The transformation matrix, :math:`T`, relating the superstructure lattice vectors, :math:`S`, to the prim lattice vectors, :math:`P`, according to :math:`S = P T`, where :math:`S` and :math:`P` are shape=(3,3) matrices with lattice vectors as columns.

        species_list: List[:class:`~libcasm.xtal.Occupant`]
            List of all distinct :class:`~libcasm.xtal.Occupant`, including each orientation.

        unit_transformation_matrix_to_super: array_like, shape=(3,3), dtype=int
            This defines a sub-supercell lattice, :math:`U = P T_{unit}`, where :math:`U` is the sub-supercell lattice column matrix, :math:`P` is the prim lattice column matrix, :math:`T_{unit}` = unit_transformation_matrix_to_super. The sub-supercell :math:`U` must tile into the supercell :math:`S` (i.e. :math:`S = U \tilde{T}`', where :math:`\tilde{T}` is an integer matrix). This option allows specifying an asymmetric unit which does not fit in the primitive cell.

        unitl_to_asym: List[int]
           This specifies the asymmetric unit orbit index corresponding to each site in the sub-supercell :math:`U`. Asymmetric unit orbit indices are distinct indices `(0, 1, ...)` indicating that sites with the same index map onto each other via symmetry operations and have occupants listed in the same order.

        )pbdoc",
          py::arg("xtal_prim"), py::arg("species_list"),
          py::arg("transformation_matrix_to_super"),
          py::arg("unit_transformation_matrix_to_super"),
          py::arg("unitl_to_asym"))
      .def(
          "lat_column_mat",
          [](monte::Conversions const &conversions) {
            return conversions.lat_column_mat();
          },
          R"pbdoc(
         :class:`~libcasm.xtal.Prim` lattice vectors, as a column vector matrix, :math:`P`.
         )pbdoc")
      .def(
          "l_size",
          [](monte::Conversions const &conversions) {
            return conversions.l_size();
          },
          R"pbdoc(
         Number of sites in the supercell.
         )pbdoc")
      .def(
          "l_to_b",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_b(l);
          },
          R"pbdoc(
        Get the sublattice index, :math:`b`, from the linear site index, :math:`l`.
        )pbdoc",
          py::arg("l"))
      .def(
          "l_to_ijk",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_ijk(l);
          },
          R"pbdoc(
        Get the unit cell indices, :math:`(i,j,k)` from the linear site index, :math:`l`.
        )pbdoc",
          py::arg("l"))
      .def(
          "l_to_bijk",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_bijk(l);
          },
          R"pbdoc(
        Get the integral site coordinates, :math:`(b,i,j,k)` from the linear site index, :math:`l`.
        )pbdoc",
          py::arg("l"))
      .def(
          "l_to_unitl",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_unitl(l);
          },
          R"pbdoc(
        Get the non-primitive unit cell sublattice index, :math:`l'`, from the linear site index, :math:`l`.
        )pbdoc",
          py::arg("l"))
      .def(
          "l_to_asym",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_asym(l);
          },
          R"pbdoc(
        Get the asymmetric unit index, :math:`a`, from the linear site index, :math:`l`.
        )pbdoc",
          py::arg("l"))
      .def(
          "l_to_cart",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_cart(l);
          },
          R"pbdoc(
        Get the Cartesian coordinate, :math:`r_{cart}`, from the linear site index, :math:`l`.
        )pbdoc",
          py::arg("l"))
      .def(
          "l_to_frac",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_frac(l);
          },
          R"pbdoc(
        Get the fractional coordinate, :math:`r_{frac}`, relative to the :class:`~libcasm.xtal.Prim` lattice vectors, :math:`P`, from the linear site index, :math:`l`.
        )pbdoc",
          py::arg("l"))
      .def(
          "l_to_basis_cart",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_basis_cart(l);
          },
          R"pbdoc(
        Get the Cartesian coordinate, :math:`r_{cart}`, in the primitive unit cell, of the sublattice that the linear site index, :math:`l`, belongs to.
        )pbdoc",
          py::arg("l"))
      .def(
          "l_to_basis_frac",
          [](monte::Conversions const &conversions, Index l) {
            return conversions.l_to_basis_frac(l);
          },
          R"pbdoc(
        Get the fractional coordinate, :math:`r_{frac}`, in the primitive unit cell, of the sublattice that the linear site index, :math:`l`, belongs to.
        )pbdoc",
          py::arg("l"))
      .def("bijk_to_l", &monte::Conversions::bijk_to_l,
           R"pbdoc(
        Get the linear site index, :math:`l`, from the integral site coordinates, :math:`(b,i,j,k)`.
        )pbdoc",
           py::arg("bijk"))
      .def("bijk_to_unitl", &monte::Conversions::bijk_to_unitl,
           R"pbdoc(
        Get the non-primitive unit cell sublattice index, :math:`l'`, from the integral site coordinates, :math:`(b,i,j,k)`.
        )pbdoc",
           py::arg("bijk"))
      .def("bijk_to_asym", &monte::Conversions::bijk_to_asym,
           R"pbdoc(
        Get the asymmetric unit index, :math:`a`, from the integral site coordinates, :math:`(b,i,j,k)`.
        )pbdoc",
           py::arg("bijk"))
      .def("unitl_size", &monte::Conversions::unitl_size,
           R"pbdoc(
        Number of sites in the unit cell.
        )pbdoc")
      .def("unitl_to_b", &monte::Conversions::unitl_to_b,
           R"pbdoc(
        Get the sublattice index, :math:`b`, from the non-primitive unit cell sublattice index, :math:`l'`.
        )pbdoc",
           py::arg("unitl"))
      .def("unitl_to_bijk", &monte::Conversions::unitl_to_b,
           R"pbdoc(
        Get the integral site coordinates, :math:`(b,i,j,k)`, from the non-primitive unit cell sublattice index, :math:`l'`.
        )pbdoc",
           py::arg("unitl"))
      .def("unitl_to_asym", &monte::Conversions::unitl_to_b,
           R"pbdoc(
        Get the asymmetric unit index, :math:`a`, from the non-primitive unit cell sublattice index, :math:`l'`.
        )pbdoc",
           py::arg("unitl"))
      .def("asym_size", &monte::Conversions::asym_size,
           R"pbdoc(
        Number of sites in the asymmetric unit.
        )pbdoc")
      .def("asym_to_b", &monte::Conversions::asym_to_b,
           R"pbdoc(
        Get the sublattice index, :math:`b`, from the asymmetric unit index, :math:`a`.
        )pbdoc",
           py::arg("asym"))
      .def("asym_to_unitl", &monte::Conversions::asym_to_unitl,
           R"pbdoc(
        Get the non-primitive unit cell sublattice index, :math:`l'`, from the asymmetric unit index, :math:`a`.
        )pbdoc",
           py::arg("asym"))
      .def("unit_transformation_matrix_to_super",
           &monte::Conversions::unit_transformation_matrix_to_super,
           R"pbdoc(
        Get the possibly non-primitive unit cell transformation matrix. See :func:`~libcasm.monte.Conversions.make_with_custom_unitcell`.
        )pbdoc")
      .def("transformation_matrix_to_super",
           &monte::Conversions::transformation_matrix_to_super,
           R"pbdoc(
        Get the transformation matrix from the prim to the superlattice vectors. See :class:`~libcasm.monte.Conversions`.
        )pbdoc")
      .def("unitcell_index_converter",
           &monte::Conversions::unitcell_index_converter,
           R"pbdoc(
        Get the :class:`~libcasm.xtal.UnitCellIndexConverter` for this supercell.
        )pbdoc")
      .def("unit_site_index_converter",
           &monte::Conversions::unit_index_converter,
           R"pbdoc(
        Get the :class:`~libcasm.xtal.SiteIndexConverter` for the possibly non-primitive unit cell.
        )pbdoc")
      .def("site_index_converter", &monte::Conversions::index_converter,
           R"pbdoc(
        Get the :class:`~libcasm.xtal.SiteIndexConverter` for the supercell.
        )pbdoc")
      .def("occ_size", &monte::Conversions::occ_size,
           R"pbdoc(
        Get the number of occupants allowed on a site by its asymmetric unit index, :math:`a`.
        )pbdoc",
           py::arg("asym"))
      .def(
          "occ_to_species_index",
          [](monte::Conversions const &conversions, Index asym,
             Index occ_index) {
            return conversions.species_index(asym, occ_index);
          },
          R"pbdoc(
        Get the `species_index` of an occupant from the occupant index and asymmetric unit index, :math:`a`, of the site it is occupying.
        )pbdoc",
          py::arg("asym"), py::arg("occ_index"))
      .def(
          "species_to_occ_index",
          [](monte::Conversions const &conversions, Index asym,
             Index species_index) {
            return conversions.occ_index(asym, species_index);
          },
          R"pbdoc(
        Get the `occ_index` of an occupant from the species index and asymmetric unit index, :math:`a`, of the site it is occupying.
        )pbdoc",
          py::arg("asym"), py::arg("species_index"))
      .def("species_allowed", &monte::Conversions::species_allowed,
           R"pbdoc(
        Return True is a species, specified by `species_index`, is allowed on the sites with specified asymmetric unit index, :math:`a`.
        )pbdoc",
           py::arg("asym"), py::arg("species_index"))
      .def("species_size", &monte::Conversions::species_size,
           R"pbdoc(
        The number of species (including orientation variants if applicable).
        )pbdoc")
      .def(
          "species_name_to_index",
          [](monte::Conversions const &conversions, std::string species_name) {
            return conversions.species_index(species_name);
          },
          R"pbdoc(
        Get the `species_index` from the species name.
        )pbdoc",
          py::arg("species_name"))
      .def(
          "species_index_to_occupant",
          [](monte::Conversions const &conversions, Index species_index) {
            return conversions.species_to_mol(species_index);
          },
          R"pbdoc(
        Get the :class:`~libcasm.xtal.Occupant` from the species index.
        )pbdoc",
          py::arg("species_index"))
      .def(
          "species_index_to_name",
          [](monte::Conversions const &conversions, Index species_index) {
            return conversions.species_name(species_index);
          },
          R"pbdoc(
        Get the species name from the `species_index`.
        )pbdoc",
          py::arg("species_index"))
      .def(
          "species_index_to_atoms_size",
          [](monte::Conversions const &conversions, Index species_index) {
            return conversions.species_name(species_index);
          },
          R"pbdoc(
        Get the number of atomic components in an occupant, by `species_index`.
        )pbdoc",
          py::arg("species_index"));

  py::class_<monte::Atom>(m, "Atom", R"pbdoc(
      Track the position of individual atoms, as if no periodic boundaries

      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          .. rubric:: Constructor

          Default constructor only.
          )pbdoc")
      .def_readwrite("translation", &monte::Atom::translation,
                     R"pbdoc(
          np.ndarray(shape=[3,1], dtype=np.int_): Current translation, \
          in fractional coordinates, as if no periodic boundary
          )pbdoc")
      .def_readwrite("n_jumps", &monte::Atom::n_jumps,
                     R"pbdoc(
          int: Current number of jumps
          )pbdoc");

  py::bind_vector<std::vector<monte::Atom>>(m, "AtomVector", R"pbdoc(
    AtomVector is a list[:class:`Atom`]-like object.
    )pbdoc");

  py::class_<monte::AtomInfo>(m, "AtomInfo", R"pbdoc(
      Holds atom position and time information

      )pbdoc")
      .def(py::init<monte::Atom, Index, Index, double>(),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          atom: Atom
              Atom translation and jumps information
          species_index: int
              Species index, as defined by :class:`~libcasm.monte.Conversions`, specifying the Occupant the atom is a component of.
          position_index: int
              Index of atom in Mol.component_id and Occupant.atoms.
          time: Optional[float] = None
              Time when the atom was added or removed
          )pbdoc")
      .def_readwrite("atom", &monte::AtomInfo::atom,
                     R"pbdoc(
          Atom: Atom translation and jumps information
          )pbdoc")
      .def_readwrite("species_index", &monte::AtomInfo::species_index,
                     R"pbdoc(
          int: Species index, as defined by \
          :class:`~libcasm.monte.Conversions`, specifying the Occupant the \
          atom is a component of.
          )pbdoc")
      .def_readwrite("position_index", &monte::AtomInfo::position_index,
                     R"pbdoc(
          int: Index of atom in Mol.component_id and Occupant.atoms.
          )pbdoc")
      .def_readwrite("time", &monte::AtomInfo::time,
                     R"pbdoc(
          float: Time when the atom was added or removed
          )pbdoc");

  py::bind_map<std::map<Index, monte::AtomInfo>>(m, "AtomInfoMap",
                                                 R"pbdoc(
    AtomInfoMap stores :class:`~libcasm.monte.events.AtomInfo` by unique atom
    id.

    Notes
    -----
    AtomInfoMap is a Dict[int, :class:`~libcasm.monte.events.AtomInfo`]-like object.
    )pbdoc",
                                                 py::module_local(false));

  py::class_<monte::Mol>(m, "Mol", R"pbdoc(
      Represents the occupant on a site

      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          .. rubric:: Constructor

          Default constructor only.
          )pbdoc")
      .def_readwrite("id", &monte::Mol::id,
                     R"pbdoc(
          int: Location of molecule in OccLocation mol list.
          )pbdoc")
      .def_readwrite("component_id", &monte::Mol::component,
                     R"pbdoc(
          LongVector: Location of component atoms in OccLocation atom list.
          )pbdoc")
      .def_readwrite("linear_site_index", &monte::Mol::l,
                     R"pbdoc(
          int: Location in configuration occupation vector.
          )pbdoc")
      .def_readwrite("asymmetric_unit_index", &monte::Mol::asym,
                     R"pbdoc(
          int: Current site asymmetric unit index. Must be consistent with `linear_site_index`.
          )pbdoc")
      .def_readwrite("species_index", &monte::Mol::species_index,
                     R"pbdoc(
          int: Species index, as defined by \
          :class:`~libcasm.monte.Conversions`, specifying the Occupant the \
          Mol is representing.
          )pbdoc")
      .def_readwrite("mol_location_index", &monte::Mol::loc,
                     R"pbdoc(
          int: Location in OccLocation mol location list
          )pbdoc");

  py::bind_vector<std::vector<monte::Mol>>(m, "MolVector", R"pbdoc(
    MolVector is a list[:class:`Mol`]-like object.
    )pbdoc");

  py::class_<monte::OccTransform>(m, "OccTransform", R"pbdoc(
      Information used to update :class:`~libcasm.events.OccLocation`

      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          .. rubric:: Constructor

          Default constructor only.
          )pbdoc")
      .def_readwrite("linear_site_index", &monte::OccTransform::l,
                     R"pbdoc(
          int: Location in configuration occupation list being transformed.
          )pbdoc")
      .def_readwrite("mol_id", &monte::OccTransform::mol_id,
                     R"pbdoc(
          int: Location in OccLocation mol list being transformed.
          )pbdoc")
      .def_readwrite("asym", &monte::OccTransform::asym,
                     R"pbdoc(
          int: Asymmetric unit index of site being transformed.
          )pbdoc")
      .def_readwrite("from_species", &monte::OccTransform::from_species,
                     R"pbdoc(
          int: Species index, as defined by \
          :class:`~libcasm.monte.Conversions`, before transformation.
          )pbdoc")
      .def_readwrite("to_species", &monte::OccTransform::to_species,
                     R"pbdoc(
          int: Species index, as defined by \
          :class:`~libcasm.monte.Conversions`, after transformation.
          )pbdoc");

  py::bind_vector<std::vector<monte::OccTransform>>(m, "OccTransformVector",
                                                    R"pbdoc(
    OccTransformVector is a list[:class:`OccTransform`]-like object.
    )pbdoc");

  py::class_<monte::AtomLocation>(m, "AtomLocation", R"pbdoc(
    Specify a specific atom location, on a site, or in a molecule

    )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
        .. rubric:: Constructor

        Default constructor only.
        )pbdoc")
      .def_readwrite("linear_site_index", &monte::AtomLocation::l,
                     R"pbdoc(
        int: Location in configuration occupation list.
        )pbdoc")
      .def_readwrite("mol_id", &monte::AtomLocation::mol_id,
                     R"pbdoc(
        int: Location in OccLocation mol list.
        )pbdoc")
      .def_readwrite("mol_comp", &monte::AtomLocation::mol_comp,
                     R"pbdoc(
        int: Location in Mol components list.
        )pbdoc");

  py::class_<monte::AtomTraj>(m, "AtomTraj", R"pbdoc(
    Specifies a trajectory from one AtomLocation to another

    )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
        .. rubric:: Constructor

        Default constructor only.
        )pbdoc")
      .def_readwrite("from", &monte::AtomTraj::from,
                     R"pbdoc(
        AtomLocation: Initial AtomLocation.
        )pbdoc")
      .def_readwrite("to", &monte::AtomTraj::to,
                     R"pbdoc(
        AtomLocation: Final AtomLocation.
        )pbdoc")
      .def_readwrite("delta_ijk", &monte::AtomTraj::delta_ijk,
                     R"pbdoc(
        np.ndarray(shape=[3,1], dtype=np.int_): Amount to increment Atom \
        translation, in fractional coordinates
        )pbdoc");

  py::bind_vector<std::vector<monte::AtomTraj>>(m, "AtomTrajVector", R"pbdoc(
    AtomTrajVector is a list[:class:`AtomTraj`]-like object.
    )pbdoc");

  py::class_<monte::OccEvent>(m, "OccEvent", R"pbdoc(
      Describes a Monte Carlo event that modifies occupation

      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          .. rubric:: Constructor

          Default constructor only.
          )pbdoc")
      .def_readwrite("linear_site_index", &monte::OccEvent::linear_site_index,
                     R"pbdoc(
          LongVector: Linear site indices, indicating on which sites the occupation \
          will be modified.
          )pbdoc")
      .def_readwrite("new_occ", &monte::OccEvent::new_occ,
                     R"pbdoc(
          IntVector: Occupant indices, indicating the new occupation index on the \
          sites being modified.
          )pbdoc")
      .def_readwrite("occ_transform", &monte::OccEvent::occ_transform,
                     R"pbdoc(
          OccTransformVector: Information used to update occupant tracking \
          information stored in :class:`~libcasm.monte.event.OccLocation`.
          )pbdoc")
      .def_readwrite("atom_traj", &monte::OccEvent::atom_traj,
                     R"pbdoc(
          OccTransformVector: Information used to update occupant location \
          information stored in :class:`~libcasm.monte.event.OccLocation` - \
          use if tracking species trajectories for kinetic Monte Carlo.
          )pbdoc")
      .def("__copy__",
           [](monte::OccEvent const &self) { return monte::OccEvent(self); })
      .def("__deepcopy__", [](monte::OccEvent const &self, py::dict) {
        return monte::OccEvent(self);
      });

  py::class_<monte::OccCandidate>(m, "OccCandidate", R"pbdoc(
    A pair of asymmetric unit index and species index, indicating a type of
    occupant that may be chosen for Monte Carlo events

    )pbdoc")
      .def(py::init<Index, Index>(),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          asymmetric_unit_index: int
              Asymmetric unit index
          species_index: int
              Species index, distinguishing each allowed site occupant, including
              distinct molecular orientations if applicable.
          )pbdoc",
           py::arg("asymmetric_unit_index"), py::arg("species_index"))
      .def_readwrite("asymmetric_unit_index", &monte::OccCandidate::asym,
                     R"pbdoc(
          int: Asymmetric unit index
          )pbdoc")
      .def_readwrite("species_index", &monte::OccCandidate::species_index,
                     R"pbdoc(
          int: Species index, distinguishing each allowed site occupant, including\
          distinct molecular orientations if applicable.
          )pbdoc")
      .def(
          "is_valid",
          [](monte::OccCandidate const &self,
             monte::Conversions const &convert) {
            return is_valid(convert, self);
          },
          R"pbdoc(
          Checks if indices are valid.

          Parameters
          ----------
          convert: :class:`~libcasm.monte.Conversions`
              The `convert` instance provides the number of asymmetric unit sites and
              species.

          Returns
          -------
          result: bool
              True if `asymmetric_unit_index` and `species_index` are valid.
          )pbdoc",
          py::arg("convert"))
      .def(py::self < py::self,
           "Compares as if (asymmetric_unit_index, species_index)")
      .def(py::self <= py::self,
           "Compares as if (asymmetric_unit_index, species_index)")
      .def(py::self > py::self,
           "Compares as if (asymmetric_unit_index, species_index)")
      .def(py::self >= py::self,
           "Compares as if (asymmetric_unit_index, species_index)")
      .def(py::self == py::self,
           "Compares as if (asymmetric_unit_index, species_index)")
      .def(py::self != py::self,
           "Compares as if (asymmetric_unit_index, species_index)")
      .def("__copy__",
           [](monte::OccCandidate const &self) {
             return monte::OccCandidate(self);
           })
      .def("__deepcopy__", [](monte::OccCandidate const &self,
                              py::dict) { return monte::OccCandidate(self); })
      .def(
          "to_dict",
          [](monte::OccCandidate const &self,
             monte::Conversions const &convert) {
            jsonParser json;
            to_json(self, json, convert);
            return static_cast<nlohmann::json>(json);
          },
          R"pbdoc(
         Represent the OccCandidate as a Python dict

         Parameters
         ----------
         convert : :class:`~libcasm.monte.Conversions`
             Provides index conversions

         Returns
         -------
         data : dict
              The OccCandidate as a Python dict
         )pbdoc",
          py::arg("convert"))
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             monte::Conversions const &convert) -> monte::OccCandidate {
            jsonParser json{data};
            return jsonConstructor<monte::OccCandidate>::from_json(json,
                                                                   convert);
          },
          R"pbdoc(
         Construct an OccCandidate from a Python dict

         Parameters
         ----------
         data : dict
             The OccCandidate representation

         convert : :class:`~libcasm.monte.Conversions`
             Provides index conversions

         Returns
         -------
         candidate : :class:`~libcasm.monte.events.OccCandidate`
              The OccCandidate
         )pbdoc",
          py::arg("data"), py::arg("convert"));

  py::class_<monte::OccSwap>(m, "OccSwap", R"pbdoc(
    Represents a Monte Carlo event that swaps occupants

    )pbdoc")
      .def(py::init<const monte::OccCandidate &, const monte::OccCandidate &>(),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          first: :class:`~libcasm.monte.events.OccCandidate`
              The first candidate occupant
          second: :class:`~libcasm.monte.events.OccCandidate`
              The second candidate occupant
          )pbdoc",
           py::arg("first"), py::arg("second"))
      .def_readwrite("first", &monte::OccSwap::cand_a,
                     R"pbdoc(
          :class:`~libcasm.monte.events.OccCandidate`: The first candidate occupant
          )pbdoc")
      .def_readwrite("second", &monte::OccSwap::cand_b,
                     R"pbdoc(
          :class:`~libcasm.monte.events.OccCandidate`: The second candidate occupant
          )pbdoc")
      .def("reverse", &monte::OccSwap::reverse,
           R"pbdoc(
          Transforms self so that `first` and `second` are reversed.
          )pbdoc")
      .def("sort", &monte::OccSwap::sort,
           R"pbdoc(
          Mutates self so that (first, second) <= (second, first).
          )pbdoc")
      .def("sorted", &monte::OccSwap::sorted,
           R"pbdoc(
          OccSwap: Returns the sorted swap.
          )pbdoc")
      .def(
          "is_valid",
          [](monte::OccSwap const &self, monte::Conversions const &convert) {
            return is_valid(convert, self);
          },
          R"pbdoc(
          Checks if `first` and `second` are valid.

          Parameters
          ----------
          convert: :class:`~libcasm.monte.Conversions`
              The `convert` instance provides the number of asymmetric unit sites and
              species.

          Returns
          -------
          result: bool
              True if `first` and `second` are valid.
          )pbdoc",
          py::arg("convert"))
      .def(py::self < py::self, "Compares as if (first, second)")
      .def(py::self <= py::self, "Compares as if (first, second)")
      .def(py::self > py::self, "Compares as if (first, second)")
      .def(py::self >= py::self, "Compares as if (first, second)")
      .def(py::self == py::self, "Compares as if (first, second)")
      .def(py::self != py::self, "Compares as if (first, second)")
      .def("__copy__",
           [](monte::OccSwap const &self) { return monte::OccSwap(self); })
      .def("__deepcopy__", [](monte::OccSwap const &self,
                              py::dict) { return monte::OccSwap(self); })
      .def(
          "to_dict",
          [](monte::OccSwap const &self, monte::Conversions const &convert) {
            jsonParser json;
            to_json(self, json, convert);
            return static_cast<nlohmann::json>(json);
          },
          R"pbdoc(
         Represent the OccSwap as a Python dict

         Parameters
         ----------
         convert : :class:`~libcasm.monte.Conversions`
             Provides index conversions

         Returns
         -------
         data : dict
              The OccSwap as a Python dict
         )pbdoc",
          py::arg("convert"))
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             monte::Conversions const &convert) -> monte::OccSwap {
            jsonParser json{data};
            return jsonConstructor<monte::OccSwap>::from_json(json, convert);
          },
          R"pbdoc(
         Construct an OccSwap from a Python dict

         Parameters
         ----------
         data : dict
             The OccSwap representation

         convert : :class:`~libcasm.monte.Conversions`
             Provides index conversions

         Returns
         -------
         swap : :class:`~libcasm.monte.events.OccSwap`
              The OccSwap
         )pbdoc",
          py::arg("data"), py::arg("convert"));

  py::bind_map<std::map<monte::OccSwap, int>>(m, "OccSwapCountMap",
                                              R"pbdoc(
    OccSwapCountMap stores :class:`~libcasm.monte.events.OccSwap` and the number that will be performed.

    Notes
    -----
    OccSwapCountMap is a Dict[:class:`~libcasm.monte.events.OccSwap`, int]-like object.
    )pbdoc",
                                              py::module_local(false));

  py::class_<monte::MultiOccSwap>(m, "MultiOccSwap", R"pbdoc(
    Represents a Monte Carlo event that performs multiple occupant swaps

    This represents 1 or more :class:`libcasm.monte.events.OccSwap`. It
    does not allow representing cycles.

    )pbdoc")
      .def(py::init<std::map<monte::OccSwap, int> const &>(),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          swaps: libcasm.monte.events.OccSwapCountMap
              The occupant swaps included in the multi-occ swap and how many.
          )pbdoc",
           py::arg("swaps"))
      .def_readonly("swaps", &monte::MultiOccSwap::swaps,
                    R"pbdoc(
          libcasm.monte.events.OccSwapCountMap: The occupant swaps included in \
          the multi-occ swap and how many.
          )pbdoc")
      .def_readonly("total_count", &monte::MultiOccSwap::total_count,
                    R"pbdoc(
          int: The total number of individual swaps
          )pbdoc")
      .def("reverse", &monte::MultiOccSwap::reverse,
           R"pbdoc(
          Transforms self so that all individual swaps are reversed.
          )pbdoc")
      .def("sort", &monte::MultiOccSwap::sort,
           R"pbdoc(
          Mutates self so that it compares less than its reverse.
          )pbdoc")
      .def("sorted", &monte::MultiOccSwap::sorted,
           R"pbdoc(
          MultiOccSwap: Returns the sorted multi-occ swap.
          )pbdoc")
      .def(py::self < py::self, "Compares two MultiOccSwap")
      .def(py::self <= py::self, "Compares two MultiOccSwap")
      .def(py::self > py::self, "Compares two MultiOccSwap")
      .def(py::self >= py::self, "Compares two MultiOccSwap")
      .def(py::self == py::self, "Compares two MultiOccSwap")
      .def(py::self != py::self, "Compares two MultiOccSwap")
      .def("__copy__",
           [](monte::MultiOccSwap const &self) {
             return monte::MultiOccSwap(self);
           })
      .def("__deepcopy__", [](monte::MultiOccSwap const &self,
                              py::dict) { return monte::MultiOccSwap(self); })
      .def(
          "to_dict",
          [](monte::MultiOccSwap const &self,
             monte::Conversions const &convert) {
            jsonParser json;
            to_json(self, json, convert);
            return static_cast<nlohmann::json>(json);
          },
          R"pbdoc(
         Represent the MultiOccSwap as a Python dict

         Parameters
         ----------
         convert : :class:`~libcasm.monte.Conversions`
             Provides index conversions

         Returns
         -------
         data : dict
              The MultiOccSwap as a Python dict
         )pbdoc",
          py::arg("convert"))
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             monte::Conversions const &convert) -> monte::MultiOccSwap {
            jsonParser json{data};
            return jsonConstructor<monte::MultiOccSwap>::from_json(json,
                                                                   convert);
          },
          R"pbdoc(
         Construct a MultiOccSwap from a Python dict

         Parameters
         ----------
         data : dict
             The MultiOccSwap representation

         convert : :class:`~libcasm.monte.Conversions`
             Provides index conversions

         Returns
         -------
         multiswap : :class:`~libcasm.monte.events.MultiOccSwap`
              The MultiOccSwap
         )pbdoc",
          py::arg("data"), py::arg("convert"));

  py::class_<monte::OccCandidateList>(m, "OccCandidateList", R"pbdoc(
    Stores a list of allowed OccCandidate

    )pbdoc")
      .def(py::init<>(&make_OccCandidateList),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          convert: :class:`~libcasm.monte.Conversions`
              The `convert` instance provides the number of asymmetric unit sites and
              species.
          candidates: Optional[list[:class:`~libcasm.monte.events.OccCandidate`]] = None
              A custom list of candidate occupant types for Monte Carlo events. If None,
              then all possible candidates are constructed.
          )pbdoc",
           py::arg("convert"), py::arg("candidates") = std::nullopt)
      .def(
          "index",
          [](monte::OccCandidateList const &self,
             monte::OccCandidate const &cand) { return self.index(cand); },
          R"pbdoc(
          int: Return index of `candidate` in the list, or len(self) if not allowed
          )pbdoc",
          py::arg("candidate"))
      .def(
          "matching_index",
          [](monte::OccCandidateList const &self, Index asym,
             Index species_index) { return self.index(asym, species_index); },
          R"pbdoc(
          int: Return index of `candidate` with matching (asymmetric_unit_index, species_index) in the list, or len(self) if not allowed
          )pbdoc",
          py::arg("asymmetric_unit_index"), py::arg("species_index"))
      .def(
          "__getitem__",
          [](monte::OccCandidateList const &self, Index candidate_index) {
            return self[candidate_index];
          },
          R"pbdoc(
         int: Return index of `candidate` in the list, or len(self) if not allowed
         )pbdoc")
      .def("__len__",
           [](monte::OccCandidateList const &self) { return self.size(); })
      .def(
          "__iter__",  // for x in occ_candidate_list
          [](monte::OccCandidateList const &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<
              0, 1>() /* Essential: keep object alive while iterator exists */)
      .def(
          "to_dict",
          [](monte::OccCandidateList const &self,
             monte::Conversions const &convert) {
            jsonParser json;
            to_json(self, json, convert);
            return static_cast<nlohmann::json>(json);
          },
          R"pbdoc(
          Represent the OccCandidateList as a Python dict

          Includes the possible canonical and semi-grand canonical events
          that can be generated from the candidates.

          Parameters
          ----------
          convert : :class:`~libcasm.monte.Conversions`
              Provides index conversions

          Returns
          -------
          data : dict
              The OccCandidateList as a Python dict
          )pbdoc",
          py::arg("convert"))
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             monte::Conversions const &convert) -> monte::OccCandidateList {
            jsonParser json{data};
            return jsonConstructor<monte::OccCandidateList>::from_json(json,
                                                                       convert);
          },
          R"pbdoc(
          Construct an OccCandidateList from a Python dict

          Parameters
          ----------
          data : dict
              The OccCandidateList representation

          convert : :class:`~libcasm.monte.Conversions`
              Provides index conversions

          Returns
          -------
          candidate_list : :class:`~libcasm.monte.events.OccCandidateList`
              The OccCandidateList
          )pbdoc",
          py::arg("data"), py::arg("convert"));

  m.def("is_allowed_canonical_swap", &monte::allowed_canonical_swap,
        R"pbdoc(
        Check that candidates form an allowed canonical Monte Carlo swap

        Checks that:
        - `first` and `second` are valid
        - the `species_index` are different and allowed on both asymmetric unit sites


        Parameters
        ----------
        convert: :class:`~libcasm.monte.Conversions`
            Provides index conversions
        first: :class:`~libcasm.monte.events.OccCandidate`
            The first candidate occupant
        second: :class:`~libcasm.monte.events.OccCandidate`
            The second candidate occupant

        Returns
        -------
        is_allowed : bool
            True if candidates form an allowed canonical Monte Carlo swap
        )pbdoc",
        py::arg("convert"), py::arg("first"), py::arg("second"));

  m.def("make_canonical_swaps", &monte::make_canonical_swaps,
        R"pbdoc(
        Make all allowed OccSwap for canonical Monte Carlo events

        Parameters
        ----------
        convert: :class:`~libcasm.monte.Conversions`
            Provides index conversions
        occ_candidate_list: :class:`~libcasm.monte.events.OccCandidateList`
            The allowed candidate occupants

        Returns
        -------
        canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for canonical Monte Carlo events. This
            does not allow both forward and reverse swaps to be included.
        )pbdoc",
        py::arg("convert"), py::arg("occ_candidate_list"));

  m.def("is_allowed_semigrand_canonical_swap",
        &monte::allowed_semigrand_canonical_swap,
        R"pbdoc(
        Check that candidates form an allowed semi-grand canonical Monte Carlo swap

        Checks that:
        - `first` and `second` are valid
        - the `asymmetric_unit_index` are the same
        - the `species_index` are different and both allowed on the asymmetric unit site

        Parameters
        ----------
        convert: :class:`~libcasm.monte.Conversions`
            Provides index conversions
        first: :class:`~libcasm.monte.events.OccCandidate`
            The first candidate occupant
        second: :class:`~libcasm.monte.events.OccCandidate`
            The second candidate occupant

        Returns
        -------
        is_allowed : bool
            True if candidates form an allowed semi-grand canonical Monte Carlo swap
        )pbdoc",
        py::arg("convert"), py::arg("first"), py::arg("second"));

  m.def("make_semigrand_canonical_swaps",
        &monte::make_semigrand_canonical_swaps,
        R"pbdoc(
        Make all allowed OccSwap for semi-grand canonical Monte Carlo events

        Parameters
        ----------
        convert: :class:`~libcasm.monte.Conversions`
            Provides index conversions
        occ_candidate_list: :class:`~libcasm.monte.events.OccCandidateList`
            The allowed candidate occupants

        Returns
        -------
        semigrand_canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for semi-grand canonical Monte Carlo events.
            This does include both forward and reverse swaps.
        )pbdoc",
        py::arg("convert"), py::arg("occ_candidate_list"));

  m.def("make_multiswaps", &monte::make_multiswaps,
        R"pbdoc(
        Construct unique MultiOccSwap

        Parameters
        ----------
        single_swaps: list[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap to make MultiOccSwap from. This should
            include both forward and reverse swaps.
        max_total_count: int
            The maximum number of single OccSwap making up the MultiOccSwap

        Returns
        -------
        multiswaps : List[:class:`~libcasm.monte.events.MultiOccSwap`]
            A list of unique MultiOccSwap made up of the `single_swaps`.
            This does include both forward and reverse multi-occ swaps.
        )pbdoc",
        py::arg("single_swaps"), py::arg("max_total_count"));

  m.def("swaps_allowed_per_unitcell", &monte::get_n_allowed_per_unitcell,
        R"pbdoc(
        For semi-grand canonical swaps, get the number of possible events per unit cell

        Parameters
        ----------
        convert: :class:`~libcasm.monte.Conversions`
            Provides index conversions
        semigrand_canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for semi-grand canonical Monte Carlo events.
            This does include both forward and reverse swaps.


        Returns
        -------
        result: int
            Total number of possible swaps per unit cell, using the multiplicity
            for each site asymmetric unit.
        )pbdoc",
        py::arg("convert"), py::arg("occ_candidate_list"));

  py::class_<monte::OccLocation, std::shared_ptr<monte::OccLocation>>(
      m, "OccLocation", R"pbdoc(
    Specify a specific atom location, on a site, or in a molecule

    )pbdoc")
      .def(py::init<const monte::Conversions &, const monte::OccCandidateList &,
                    bool, bool, bool>(),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          convert: :class:`~libcasm.monte.Conversions`
              The `convert` instance provides the number of asymmetric unit sites and
              species.
          occ_candidate_list: :class:`~libcasm.monte.events.OccCandidateList`
              A list of candidate occupant types for Monte Carlo events.
          update_atoms: bool = False
              If True, update atom location information when updating occupation.
              This can be used by kinetic Monte Carlo methods for measuring
              diffusion.
          track_unique_atoms: bool = False
              If True, track unique atom id for each atom in the supercell.
              This can be used by kinetic Monte Carlo methods which allow
              adding / removing atoms to track individual atoms. This requires
              `update_atoms` to be True.
          save_atom_info: bool = False
              If True, save initial and final atom position, type, and time
              information. This can be used by kinetic Monte Carlo
              methods to record the exact deposition / dissolution events. This
              requires `update_atoms` and `track_unique_atoms` to be True.
          )pbdoc",
           py::arg("convert"), py::arg("occ_candidate_list"),
           py::arg("update_atoms") = false,
           py::arg("track_unique_atoms") = false,
           py::arg("save_atom_info") = false)
      .def("initialize", &monte::OccLocation::initialize,
           R"pbdoc(
          Fill tables with current occupation info

          Parameters
          ----------
          occupation: np.ndarray[np.int[n_sites,]]
              The occupation vector to initialize with
          time: Optional[float] = None
              If time has a value, and `save_atom_info` is True, then the
              initial atom info will be stored with the given time.
          )pbdoc",
           py::arg("occupation"), py::arg("time") = std::nullopt)
      .def("apply", &monte::OccLocation::apply,
           R"pbdoc(
          Update occupation vector and this to reflect that `event` occurred.

          Parameters
          ----------
          event: :class:`~libcasm.monte.events.OccEvent`
              The event to apply
          occupation: np.ndarray[np.int[n_sites,]]
              The occupation vector to update
          time: Optional[float] = None
              The time the event occurred (for kinetic Monte Carlo). If time
              has a value, and `save_atom_info` is True, then the
              initial/final atom info will be stored with the given time.
          )pbdoc",
           py::arg("event"), py::arg("occupation"),
           py::arg("time") = std::nullopt)
      .def(
          "choose_mol_by_candidate_index",
          [](monte::OccLocation const &self, Index cand_index,
             generator_type &random_number_generator) {
            return self.choose_mol(cand_index, random_number_generator);
          },
          R"pbdoc(
          Stochastically choose an occupant of a particular OccCandidate type.
          )pbdoc",
          py::arg("candidate_index"), py::arg("random_number_generator"))
      .def(
          "choose_mol",
          [](monte::OccLocation const &self, monte::OccCandidate const &cand,
             generator_type &random_number_generator) {
            return self.choose_mol(cand, random_number_generator);
          },
          R"pbdoc(
          Stochastically choose an occupant of a particular OccCandidate type.
          )pbdoc",
          py::arg("candidate"), py::arg("random_number_generator"))
      .def("mol_size", &monte::OccLocation::mol_size,
           R"pbdoc(
          Total number of mutating sites.
          )pbdoc")
      .def(
          "mol",
          [](monte::OccLocation &self, Index mol_id) {
            return self.mol(mol_id);
          },
          R"pbdoc(
          Access Mol by id (location of molecule in mol list).
          )pbdoc",
          py::arg("mol_id"))
      .def("atom_size", &monte::OccLocation::mol_size,
           R"pbdoc(
          Total number of atoms.
          )pbdoc")
      .def(
          "atom",
          [](monte::OccLocation &self, Index atom_id) {
            return self.atom(atom_id);
          },
          R"pbdoc(
          Access Atom by id (location of atom in atom list).
          )pbdoc",
          py::arg("atom_id"))
      .def("atom_positions_cart", &monte::OccLocation::atom_positions_cart,
           R"pbdoc(
          Return current atom positions in cartesian coordinates, shape=(3, atom_size).
          )pbdoc")
      .def("atom_positions_cart_within",
           &monte::OccLocation::atom_positions_cart_within,
           R"pbdoc(
          Return current atom positions in cartesian coordinates, shape=(3, atom_size).
          )pbdoc")
      .def("unique_atom_id", &monte::OccLocation::unique_atom_id,
           R"pbdoc(
          Return unique atom id for each atom in atom position matrices
          )pbdoc")
      .def("atom_info_initial", &monte::OccLocation::atom_info_initial,
           R"pbdoc(
          Return the initial position, type, and time when atom are added to the
          supercell, stored by unique atom id

          Returns
          -------
          atom_info_initial : dict[int, AtomInfo]
              Dictionary with unique atom id as key and
              :class:`~libcasm.monte.events.AtomInfo` as value, where the
              AtomInfo gives the initial position, type, and time when atoms
              are added to the supercell.
          )pbdoc")
      .def("atom_info_final", &monte::OccLocation::atom_info_final,
           R"pbdoc(
          Return the final position, type, and time when atom are removed from the
          supercell, stored by unique atom id

          Returns
          -------
          atom_info_initial : dict[int, AtomInfo]
              Dictionary with unique atom id as key and
              :class:`~libcasm.monte.events.AtomInfo` as value, where the
              AtomInfo gives the final position, type, and time when atoms
              are removed from the supercell.
          )pbdoc")
      .def("clear_atom_info_final", &monte::OccLocation::clear_atom_info_final,
           R"pbdoc(
          Clear information about atoms that have been removed from the supercell

          This also clears the initial information for the removed atoms.
          )pbdoc")
      .def("initial_atom_species_index",
           &monte::OccLocation::initial_atom_species_index,
           R"pbdoc(
          Holds initial species index for each atom in atom position matrices.
          )pbdoc")
      .def("initial_atom_position_index",
           &monte::OccLocation::initial_atom_position_index,
           R"pbdoc(
          Holds initial atom position index for each atom in atom position matrices.
          )pbdoc")
      .def("current_atom_names", &monte::OccLocation::current_atom_names,
           R"pbdoc(
          Return current name for each atom in atom position matrices.
          )pbdoc")
      .def("current_atom_species_index",
           &monte::OccLocation::current_atom_species_index,
           R"pbdoc(
          Return current species index for atoms in atom position matrices.
          )pbdoc")
      .def("current_atom_position_index",
           &monte::OccLocation::current_atom_position_index,
           R"pbdoc(
          Return current atom position index for atoms in atom position matrices.
          )pbdoc")
      .def("current_atom_n_jumps", &monte::OccLocation::current_atom_n_jumps,
           R"pbdoc(
          Return number of jumps made by each atom.
          )pbdoc")
      .def("candidate_list", &monte::OccLocation::candidate_list,
           R"pbdoc(
          Access the OccCandidateList.
          )pbdoc")
      .def(
          "cand_size_by_candidate_index",
          [](monte::OccLocation const &self, Index cand_index) {
            return self.cand_size(cand_index);
          },
          R"pbdoc(
          Total number of mutating sites, of OccCandidate type, specified by index.
          )pbdoc",
          py::arg("candidate_index"))
      .def(
          "cand_size",
          [](monte::OccLocation const &self, monte::OccCandidate const &cand) {
            return self.cand_size(cand);
          },
          R"pbdoc(
          Total number of mutating sites, of OccCandidate type.
          )pbdoc",
          py::arg("candidate"))
      .def(
          "mol_id_by_candidate_index",
          [](monte::OccLocation const &self, Index cand_index, Index loc) {
            return self.mol_id(cand_index, loc);
          },
          R"pbdoc(
          Mol.id of a particular OccCandidate type, specified by index.
          )pbdoc",
          py::arg("candidate_index"), py::arg("location_index"))
      .def(
          "mol_id",
          [](monte::OccLocation const &self, monte::OccCandidate const &cand,
             Index loc) { return self.mol_id(cand, loc); },
          R"pbdoc(
          Mol.id of a particular OccCandidate type.
          )pbdoc",
          py::arg("candidate"), py::arg("location_index"))
      .def(
          "linear_site_index_to_mol_id",
          [](monte::OccLocation const &self, Index linear_site_index) {
            return self.l_to_mol_id(linear_site_index);
          },
          R"pbdoc(
          Convert from linear site index in configuration to variable site index (mol_id).
          )pbdoc",
          py::arg("linear_site_index"))
      .def("convert", &monte::OccLocation::convert,
           R"pbdoc(
           Get Conversions objects.
           )pbdoc");

  m.def(
      "choose_canonical_swap",
      [](monte::OccLocation const &occ_location,
         std::vector<monte::OccSwap> const &canonical_swaps,
         generator_type &random_number_generator) {
        return monte::choose_canonical_swap(occ_location, canonical_swaps,
                                            random_number_generator);
      },
      R"pbdoc(
        Choose a swap type from a list of allowed canonical swap types

        Parameters
        ----------
        occ_location: :class:`~libcasm.monte.OccLocation`
            Current occupant location list
        canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for canonical Monte Carlo events.
            This should not include both forward and reverse swaps.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        swap: :class:`~libcasm.monte.events.OccSwap`
            Chosen swap type.
        )pbdoc",
      py::arg("occ_location"), py::arg("canonical_swaps"),
      py::arg("random_number_generator"));

  m.def(
      "propose_canonical_event_from_swap",
      [](monte::OccEvent &e, monte::OccLocation const &occ_location,
         monte::OccSwap const &swap, generator_type &random_number_generator) {
        return monte::propose_canonical_event_from_swap(
            e, occ_location, swap, random_number_generator);
      },
      R"pbdoc(
        Propose canonical OccEvent of particular swap type

        Parameters
        ----------
        event: :class:`~libcasm.monte.events.OccEvent`
            Event to update based on the chosen OccSwap.
        occ_location: :class:`~libcasm.monte.events.OccLocation`
            Current occupant location list
        swap : :class:`~libcasm.monte.events.OccSwap`
            Chosen swap type.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        event: :class:`~libcasm.monte.events.OccEvent`
            Updated event based on the chosen swap type and particular event.

        )pbdoc",
      py::arg("event"), py::arg("occ_location"), py::arg("swap"),
      py::arg("random_number_generator"));

  m.def(
      "propose_canonical_event",
      [](monte::OccEvent &e, monte::OccLocation const &occ_location,
         std::vector<monte::OccSwap> const &canonical_swap,
         generator_type &random_number_generator) {
        return monte::propose_canonical_event(e, occ_location, canonical_swap,
                                              random_number_generator);
      },
      R"pbdoc(
        Propose canonical OccEvent from list of swap types

        Parameters
        ----------
        event: :class:`~libcasm.monte.events.OccEvent`
            Event to update based on the chosen OccSwap.
        occ_location: :class:`~libcasm.monte.events.OccLocation`
            Current occupant location list
        canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for canonical Monte Carlo events.
            This should not include both forward and reverse swaps.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        event: :class:`~libcasm.monte.events.OccEvent`
            Updated event based on the chosen swap type and particular event.

        )pbdoc",
      py::arg("event"), py::arg("occ_location"), py::arg("canonical_swaps"),
      py::arg("random_number_generator"));

  m.def(
      "choose_semigrand_canonical_swap",
      [](monte::OccLocation const &occ_location,
         std::vector<monte::OccSwap> const &semigrand_canonical_swaps,
         generator_type &random_number_generator) {
        return monte::choose_semigrand_canonical_swap(
            occ_location, semigrand_canonical_swaps, random_number_generator);
      },
      R"pbdoc(
        Choose a swap type from a list of allowed semi-grand canonical swap types

        Parameters
        ----------
        occ_location: :class:`~libcasm.monte.OccLocation`
            Current occupant location list
        semigrand_canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for semi-grand canonical Monte Carlo events.
            This should include both forward and reverse swaps.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        swap: :class:`~libcasm.monte.events.OccSwap`
            Chosen swap type.
        )pbdoc",
      py::arg("occ_location"), py::arg("semigrand_canonical_swaps"),
      py::arg("random_number_generator"));

  m.def(
      "propose_semigrand_canonical_event_from_swap",
      [](monte::OccEvent &e, monte::OccLocation const &occ_location,
         monte::OccSwap const &swap, generator_type &random_number_generator) {
        return monte::propose_semigrand_canonical_event_from_swap(
            e, occ_location, swap, random_number_generator);
      },
      R"pbdoc(
        Propose semi-grand canonical OccEvent of particular swap type

        Parameters
        ----------
        event: :class:`~libcasm.monte.events.OccEvent`
            Event to update based on the chosen OccSwap.
        occ_location: :class:`~libcasm.monte.events.OccLocation`
            Current occupant location list
        swap : :class:`~libcasm.monte.events.OccSwap`
            Chosen swap type.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        event: :class:`~libcasm.monte.events.OccEvent`
            Updated event based on the chosen swap type and particular event.

        )pbdoc",
      py::arg("event"), py::arg("occ_location"), py::arg("swap"),
      py::arg("random_number_generator"));

  m.def(
      "propose_semigrand_canonical_event",
      [](monte::OccEvent &e, monte::OccLocation const &occ_location,
         std::vector<monte::OccSwap> const &semigrand_canonical_swaps,
         generator_type &random_number_generator) {
        return monte::propose_semigrand_canonical_event(
            e, occ_location, semigrand_canonical_swaps,
            random_number_generator);
      },
      R"pbdoc(
        Propose semi-grand canonical OccEvent from list of swap types

        Parameters
        ----------
        event: :class:`~libcasm.monte.events.OccEvent`
            Event to update.
        occ_location: :class:`~libcasm.monte.events.OccLocation`
            Current occupant location list
        semigrand_canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for semi-grand canonical Monte Carlo events.
            This should include both forward and reverse swaps.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        event: :class:`~libcasm.monte.events.OccEvent`
            Updated event based on the chosen swap type and particular event.

        )pbdoc",
      py::arg("event"), py::arg("occ_location"),
      py::arg("semigrand_canonical_swaps"), py::arg("random_number_generator"));

  // ~~~

  m.def(
      "choose_semigrand_canonical_multiswap",
      [](monte::OccLocation const &occ_location,
         std::vector<monte::MultiOccSwap> const &semigrand_canonical_multiswaps,
         generator_type &random_number_generator) {
        return monte::choose_semigrand_canonical_multiswap(
            occ_location, semigrand_canonical_multiswaps,
            random_number_generator);
      },
      R"pbdoc(
        Choose a swap type from a list of allowed semi-grand canonical swap types

        Parameters
        ----------
        occ_location: :class:`~libcasm.monte.OccLocation`
            Current occupant location list
        semigrand_canonical_multiswaps : List[:class:`~libcasm.monte.events.MultiOccSwap`]
            A list of allowed MultiOccSwap for semi-grand canonical Monte Carlo events.
            This should include both forward and reverse swaps.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        multiswap: :class:`~libcasm.monte.events.MultiOccSwap`
            Chosen multi-occ swap type.
        )pbdoc",
      py::arg("occ_location"), py::arg("semigrand_canonical_multiswaps"),
      py::arg("random_number_generator"));

  m.def(
      "propose_semigrand_canonical_event_from_multiswap",
      [](monte::OccEvent &e, monte::OccLocation const &occ_location,
         monte::MultiOccSwap const &multiswap,
         generator_type &random_number_generator) {
        return monte::propose_semigrand_canonical_event_from_multiswap(
            e, occ_location, multiswap, random_number_generator);
      },
      R"pbdoc(
        Propose semi-grand canonical OccEvent of particular multi-occ swap type

        Parameters
        ----------
        event: :class:`~libcasm.monte.events.OccEvent`
            Event to update based on the chosen OccSwap.
        occ_location: :class:`~libcasm.monte.events.OccLocation`
            Current occupant location list
        multiswap : :class:`~libcasm.monte.events.MultiOccSwap`
            Chosen multi-occ swap type.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        event: :class:`~libcasm.monte.events.OccEvent`
            Updated event based on the chosen multi-occ swap type and particular event.

        )pbdoc",
      py::arg("event"), py::arg("occ_location"), py::arg("multiswap"),
      py::arg("random_number_generator"));

  m.def(
      "propose_semigrand_canonical_multiswap_event",
      [](monte::OccEvent &e, monte::OccLocation const &occ_location,
         std::vector<monte::MultiOccSwap> const &semigrand_canonical_multiswaps,
         generator_type &random_number_generator) {
        return monte::propose_semigrand_canonical_multiswap_event(
            e, occ_location, semigrand_canonical_multiswaps,
            random_number_generator);
      },
      R"pbdoc(
        Propose semi-grand canonical OccEvent from list of multi-occ swap types

        Parameters
        ----------
        event: :class:`~libcasm.monte.events.OccEvent`
            Event to update.
        occ_location: :class:`~libcasm.monte.events.OccLocation`
            Current occupant location list
        semigrand_canonical_multiswaps : List[:class:`~libcasm.monte.events.MultiOccSwap`]
            A list of allowed MultiOccSwap for semi-grand canonical Monte Carlo events.
            This should include both forward and reverse swaps.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        event: :class:`~libcasm.monte.events.OccEvent`
            Updated event based on the chosen multi-occ swap type and particular event.

        )pbdoc",
      py::arg("event"), py::arg("occ_location"),
      py::arg("semigrand_canonical_multiswaps"),
      py::arg("random_number_generator"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
