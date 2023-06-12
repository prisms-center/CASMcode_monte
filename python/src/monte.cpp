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
#include "casm/monte/BasicStatistics.hh"
#include "casm/monte/Conversions.hh"
#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/checks/EquilibrationCheck.hh"
#include "casm/monte/sampling/Sampler.hh"
#include "casm/monte/sampling/SamplingParams.hh"
#include "casm/monte/state/ValueMap.hh"
#include "casm/monte/state/io/json/ValueMap_json_io.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

typedef std::mt19937_64 engine_type;
typedef monte::RandomNumberGenerator<engine_type> generator_type;
typedef std::map<std::string, std::shared_ptr<monte::Sampler>> SamplerMap;
typedef std::map<monte::SamplerComponent, monte::RequestedPrecision>
    RequestedPrecisionMap;
typedef std::map<
    monte::SamplerComponent,
    monte::IndividualConvergenceCheckResult<monte::BasicStatistics>>
    ConvergenceResultMap;
typedef std::map<monte::SamplerComponent,
                 monte::IndividualEquilibrationCheckResult>
    EquilibrationResultMap;

std::shared_ptr<monte::Conversions> make_monte_conversions(
    xtal::BasicStructure const &prim,
    Eigen::Matrix3l const &transformation_matrix_to_super) {
  return std::make_shared<monte::Conversions>(prim,
                                              transformation_matrix_to_super);
};

/// \brief Make a random number engine seeded by std::random_device
std::shared_ptr<engine_type> make_random_number_engine() {
  std::shared_ptr<engine_type> engine = std::make_shared<engine_type>();
  std::random_device device;
  engine->seed(device());
  return engine;
};

/// \brief Make a random number generator that uses the provided engine
///
/// Notes:
/// - If _engine == nullptr, use an engine seeded by std::random_device
generator_type make_random_number_generator(
    std::shared_ptr<engine_type> _engine = std::shared_ptr<engine_type>()) {
  return monte::RandomNumberGenerator(_engine);
};

std::shared_ptr<monte::Sampler> make_sampler(
    std::vector<Index> shape,
    std::optional<std::vector<std::string>> component_names,
    monte::CountType capacity_increment) {
  if (component_names.has_value()) {
    return std::make_shared<monte::Sampler>(shape, *component_names,
                                            capacity_increment);
  } else {
    return std::make_shared<monte::Sampler>(shape, capacity_increment);
  }
}

monte::RequestedPrecision make_requested_precision(std::optional<double> abs,
                                                   std::optional<double> rel) {
  if (abs.has_value() && rel.has_value()) {
    return monte::RequestedPrecision::abs_and_rel(*abs, *rel);
  } else if (abs.has_value()) {
    return monte::RequestedPrecision::abs(*abs);
  } else if (rel.has_value()) {
    return monte::RequestedPrecision::rel(*rel);
  } else {
    throw std::runtime_error(
        "Error constructing RequestedPrecision: one of abs or rel must have a "
        "value");
  }
}

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, bool>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, double>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, Eigen::VectorXd>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, Eigen::MatrixXd>);
PYBIND11_MAKE_OPAQUE(CASMpy::SamplerMap);
PYBIND11_MAKE_OPAQUE(CASMpy::RequestedPrecisionMap);
PYBIND11_MAKE_OPAQUE(CASMpy::ConvergenceResultMap);
PYBIND11_MAKE_OPAQUE(CASMpy::EquilibrationResultMap);

PYBIND11_MODULE(_monte, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Methods for evaluating functions of configurations

        libcasm.monte
        -------------

    )pbdoc";
  py::module::import("libcasm.xtal");

  py::bind_map<std::map<std::string, bool>>(m, "BooleanValueMap");
  py::bind_map<std::map<std::string, double>>(m, "ScalarValueMap");
  py::bind_map<std::map<std::string, Eigen::VectorXd>>(m, "VectorValueMap");
  py::bind_map<std::map<std::string, Eigen::MatrixXd>>(m, "MatrixValueMap");

  py::class_<monte::ValueMap>(m, "ValueMap", R"pbdoc(
      Data structure for holding Monte Carlo data

      Notes
      -----
      Data should not have the same key, even if the values have
      different type. Conversions for input/output are made
      to/from a single combined dict.
      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          ValueMap only has a default constructor
          )pbdoc")
      .def_readwrite("boolean_values", &monte::ValueMap::boolean_values,
                     R"pbdoc(
          A Dict[str, bool]-like object.
          )pbdoc")
      .def_readwrite("scalar_values", &monte::ValueMap::scalar_values,
                     R"pbdoc(
          A Dict[str, float]-like object.
          )pbdoc")
      .def_readwrite("vector_values", &monte::ValueMap::vector_values,
                     R"pbdoc(
          A Dict[str, numpy.ndarray[numpy.float64[m, 1]]]-like object.
          )pbdoc")
      .def_readwrite("matrix_values", &monte::ValueMap::matrix_values,
                     R"pbdoc(
          A Dict[str, numpy.ndarray[numpy.float64[m, n]]]-like object.
          )pbdoc")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) {
            jsonParser json{data};
            monte::ValueMap values;
            from_json(values, json);
            return values;
          },
          "Construct a ValueMap from a Python dict. Types are automatically "
          "checked and items added to the corresponding attribute. Integer "
          "values are converted to floating-point. The presence of other types "
          "(i.e. str) will result in an exception.",
          py::arg("data"))
      .def(
          "to_dict",
          [](monte::ValueMap const &values) {
            jsonParser json;
            to_json(values, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the ValueMap as a Python dict. Items from all attributes "
          "are combined into a single dict");

  m.def("is_mismatched", &monte::is_mismatched,
        "Return true if A and B do not have the same properties");

  m.def("make_incremented_values", &monte::make_incremented_values,
        R"pbdoc(
      Return values[property] + n_increment*increment[property] for each property

      Note
      ----
      Does not change boolean values.
      )pbdoc",
        py::arg("values"), py::arg("increment"), py::arg("n_increment"));

  py::class_<monte::Conversions, std::shared_ptr<monte::Conversions>>(
      m, "Conversions", R"pbdoc(
      Data structure used for index conversions

      Notes
      -----
      The following shorthand is used for member function names:

      - ``l``, :math:`l`: Linear site index in a particular supercell
      - ``b``, :math:`b`: :class:`~libcasm.xtal.Prim` sublattice index
      - ``unitl``, :math:`l'`: Linear site index in a non-primitive unit cell. When a non-primitive unit cell is used to construct a supercell and determines the appropriate symmetry for a problem, conversions between :math:`l`, :math:`b`, and :math:`l'` may all be useful.
      - ``ijk``, :math:`(i,j,k)`: Integer unit cell indices (fractional coordinates with respect to the :class:`~libcasm.xtal.Prim` lattice vectors)
      - ``bijk``, :math:`(b, i, j, k)`: Integral site coordinates (sublattice index and integer unit cell indices)
      - ``asym``, :math:`a`: Asymmetric unit orbit index (value is the same for all sites which are symmetrically equivalent)
      - ``occ_index``, :math:`s`: Index into occupant list for a particular site
      - ``species_index``: Index into the molecule list for a particular :class:`~libcasm.xtal.Prim`. If there are orientational variants, ``species_index`` should correspond to ``orientation_index``.

      )pbdoc")
      .def(py::init<>(&make_monte_conversions),
           R"pbdoc(
           Constructor

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
              Specifies the asymmetric unit orbit index corresponding to each sublattice in the prim. Asymmetric unit orbit indices are distinct indices `(0, 1, ...)` indicating that sites with the same index map onto each other via symmetry operations.

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
             This specifies the asymmetric unit orbit index corresponding to each site in the sub-supercell :math:`U`. Asymmetric unit orbit indices are distinct indices `(0, 1, ...)` indicating that sites with the same index map onto each other via symmetry operations.

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
          Get the ``species_index`` of an occupant from the occupant index and asymmetric unit index, :math:`a`, of the site it is occupying.
          )pbdoc",
          py::arg("asym"), py::arg("occ_index"))
      .def(
          "species_to_occ_index",
          [](monte::Conversions const &conversions, Index asym,
             Index species_index) {
            return conversions.occ_index(asym, species_index);
          },
          R"pbdoc(
          Get the ``occ_index`` of an occupant from the species index and asymmetric unit index, :math:`a`, of the site it is occupying.
          )pbdoc",
          py::arg("asym"), py::arg("species_index"))
      .def("species_allowed", &monte::Conversions::species_allowed,
           R"pbdoc(
          Return True is a species, specified by ``species_index``, is allowed on the sites with specified asymmetric unit index, :math:`a`.
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
          Get the ``species_index`` from the species name.
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
          Get the species name from the ``species_index``.
          )pbdoc",
          py::arg("species_index"))
      .def(
          "species_index_to_atoms_size",
          [](monte::Conversions const &conversions, Index species_index) {
            return conversions.species_name(species_index);
          },
          R"pbdoc(
          Get the number of atomic components in an occupant, by ``species_index``.
          )pbdoc",
          py::arg("species_index"));

  py::class_<engine_type, std::shared_ptr<engine_type>>(m, "RandomNumberEngine",
                                                        R"pbdoc(
      A pseudo-random number engine, using std::MT19937_64
      )pbdoc")
      .def(py::init<>(&make_random_number_engine),
           R"pbdoc(
           Construct a pseudo-random number engine using std::random_device to seed.
           )pbdoc")
      .def(
          "seed",
          [](engine_type &e, engine_type::result_type value) { e.seed(value); },
          R"pbdoc(
          Seed the pseudo-random number engine using a single value.
          )pbdoc")
      .def(
          "seed_seq",
          [](engine_type &e, std::vector<engine_type::result_type> values) {
            std::seed_seq ss(values.begin(), values.end());
            e.seed(ss);
          },
          R"pbdoc(
          Seed the pseudo-random number engine using std::seed_seq initialized with the provided values.
          )pbdoc")
      .def(
          "dump",
          [](engine_type const &e) {
            std::stringstream ss;
            ss << e;
            return ss.str();
          },
          R"pbdoc(
          Dump the current state of the psueudo-random number engine.
          )pbdoc")
      .def(
          "load",
          [](engine_type &e, std::string state) {
            std::stringstream ss(state);
            ss >> e;
          },
          R"pbdoc(
          Load a saved state of the psueudo-random number engine.
          )pbdoc");

  py::class_<generator_type>(m, "RandomNumberGenerator", R"pbdoc(
      A pseudo-random number generator, which uses a shared :class:`~libcasm.monte.RandomNumberEngine` to construct uniformly distributed integer or real-valued numbers.
      )pbdoc")
      .def(py::init<>(&make_random_number_generator),
           R"pbdoc(
          Constructs a pseudo-random number generator using a shared :class:`~libcasm.monte.RandomNumberEngine`.

          Parameters
          ----------
          engine : Optional[:class:`~libcasm.monte.RandomNumberEngine`]
              A :class:`~libcasm.monte.RandomNumberEngine` to use for generating random numbers. If provided, the engine will be shared. If None, then a new :class:`~libcasm.monte.RandomNumberEngine` will be constructed and seeded using std::random_device.
          )pbdoc",
           py::arg("engine") = std::shared_ptr<engine_type>())
      .def(
          "random_int",
          [](generator_type &g, uint64_t maximum_value) {
            return g.random_int(maximum_value);
          },
          R"pbdoc(
          Return uniformly distributed ``uint64`` integer in [0, maximum_value].
          )pbdoc",
          py::arg("maximum_value"))
      .def(
          "random_real",
          [](generator_type &g, double maximum_value) {
            return g.random_real(maximum_value);
          },
          R"pbdoc(
            Return uniformly distributed double floating point value in [0, maximum_value).
          )pbdoc",
          py::arg("maximum_value"))
      .def(
          "engine", [](generator_type const &g) { return g.engine; },
          R"pbdoc(
            Return the internal shared :class:`~libcasm.monte.RandomNumberEngine`.
          )pbdoc");

  py::enum_<monte::SAMPLE_MODE>(m, "SAMPLE_MODE",
                                R"pbdoc(
      Enum specifying sampling modes.
      )pbdoc")
      .value("BY_PASS", monte::SAMPLE_MODE::BY_PASS,
             R"pbdoc(
          Sample by Monte Carlo pass (1 pass = 1 step per supercell site with degrees of freedom):

          .. code-block:: Python

              sample_mode = libcasm.monte.SAMPLE_MODE.BY_PASS


          )pbdoc")
      .value("BY_STEP", monte::SAMPLE_MODE::BY_STEP,
             R"pbdoc(
          Sample by Monte Carlo step (1 step == 1 Metropolis attempt):

          .. code-block:: Python

              sample_mode = libcasm.monte.SAMPLE_MODE.BY_PASS


          )pbdoc")
      .value("BY_TIME", monte::SAMPLE_MODE::BY_TIME,
             R"pbdoc(
          Sample by Monte Carlo time:

          .. code-block:: Python

              sample_mode = libcasm.monte.SAMPLE_MODE.BY_TIME

          )pbdoc")
      .export_values();

  py::enum_<monte::SAMPLE_METHOD>(m, "SAMPLE_METHOD",
                                  R"pbdoc(
      Enum specifying sampling method (linearly or logarithmically spaced samples).
      )pbdoc")
      .value("LINEAR", monte::SAMPLE_METHOD::LINEAR,
             R"pbdoc(
          Linearly spaced samples:

          .. code-block:: Python

              sample_method = libcasm.monte.SAMPLE_METHOD.LINEAR

          )pbdoc")
      .value("LOG", monte::SAMPLE_METHOD::LOG,
             R"pbdoc(
          Logarithmically spaced samples:

          .. code-block:: Python

              sample_method = libcasm.monte.SAMPLE_METHOD.LOG

          )pbdoc")
      .export_values();

  py::class_<monte::SamplingParams>(m, "SamplingParams", R"pbdoc(
      Parameters controlling sampling fixtures
      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          ValueMap only has a default constructor
          )pbdoc")
      .def_readwrite("sample_mode", &monte::SamplingParams::sample_mode,
                     R"pbdoc(
            Sample by pass, step, or time. Default=``SAMPLE_MODE::BY_PASS``.
          )pbdoc")
      .def_readwrite("sample_method", &monte::SamplingParams::sample_mode,
                     R"pbdoc(
            Sample linearly, or logarithmically. Default=``SAMPLE_METHOD::LINEAR``.

            Notes
            -----

            For ``SAMPLE_METHOD::LINEAR``, take the n-th sample when:

            .. code-block:: Python

                sample/pass = round( begin + (period / samples_per_period) * n )

                time = begin + (period / samples_per_period) * n

            The default values are:

            For ``SAMPLE_METHOD::LOG``, take the n-th sample when:

            .. code-block:: Python

                sample/pass = round( begin + period ** ( (n + shift) /
                                     samples_per_period ) )

                time = begin + period ** ( (n + shift) / samples_per_period )

            If ``stochastic_sample_period == true``, then instead of setting the sample
            time / count deterministally, use the sampling period to determine the
            sampling rate and determine the next sample time / count stochastically.
          )pbdoc")
      .def_readwrite("begin", &monte::SamplingParams::begin,
                     R"pbdoc(
            See `sample_method`. Default=``0.0``.
          )pbdoc")
      .def_readwrite("period", &monte::SamplingParams::period,
                     R"pbdoc(
            See `sample_method`. Default=``1.0``.
          )pbdoc")
      .def_readwrite("samples_per_period",
                     &monte::SamplingParams::samples_per_period,
                     R"pbdoc(
            See `sample_method`. Default=``1.0``.
          )pbdoc")
      .def_readwrite("shift", &monte::SamplingParams::shift,
                     R"pbdoc(
            See `sample_method`. Default=``0.0``.
          )pbdoc")
      .def_readwrite("sampler_names", &monte::SamplingParams::sampler_names,
                     R"pbdoc(
            The names of quantities to sample (i.e. sampling function names). Default=``[]``.
          )pbdoc")
      .def_readwrite("do_sample_trajectory",
                     &monte::SamplingParams::do_sample_trajectory,
                     R"pbdoc(
            If true, save the configuration when a sample is taken. Default=``False``.
          )pbdoc")
      .def_readwrite("do_sample_time", &monte::SamplingParams::do_sample_time,
                     R"pbdoc(
            If true, save current time when taking a sample. Default=``False``.
          )pbdoc");

  m.def(
      "scalar_as_vector",
      [](double value) { return Eigen::VectorXd::Constant(1, value); },
      R"pbdoc(
      Return a scalar as a size=1 vector (np.ndarray).
      )pbdoc",
      py::arg("scalar"));

  m.def(
      "matrix_as_vector",
      [](Eigen::MatrixXd const &value) { return value.reshaped(); },
      R"pbdoc(
      Return a vector or matrix as column-major vector (np.ndarray).
      )pbdoc",
      py::arg("matrix"));

  m.def("default_component_names", &monte::default_component_names,
        R"pbdoc(
      Construct default component names based on the shape of a quantity.



      Parameters
      ----------
      shape : List[int]
          The shape of quantity to be sampled

      Returns
      -------
      component_names : List[str]
          The default component names are:

          - shape = [] (scalar) -> {"0"}
          - shape = [n] (vector) -> {"0", "1", ..., "n-1"}
          - shape = [m, n] (matrix) -> {"0,0", "1,0", ..., "m-1,n-1"}

      )pbdoc",
        py::arg("shape"));

  m.def("colmajor_component_names", &monte::colmajor_component_names,
        R"pbdoc(
      Constructs a vector of (row,col) names

      Parameters
      ----------
      n_rows : int
          Number of rows
      n_cols : int
          Number of columns

      Returns
      -------
      component_names : List[str]
          A vector of (row,col) names ["0,0", "1,0", ..., "n_rows-1,n_cols-1"]
      )pbdoc",
        py::arg("n_rows"), py::arg("n_cols"));

  py::class_<monte::Sampler, std::shared_ptr<monte::Sampler>>(m, "Sampler",
                                                              R"pbdoc(
      Sampler stores vector valued samples in a matrix

      Notes
      -----
      For sampling scalars, a size=1 vector is expected. For sampling matrices, column-major order is expected. These can be obtained with :func:`~libcasm.monte.scalar_as_vector(value)` or :func:`~libcasm.monte.matrix_as_vector(value)`.
      )pbdoc")
      .def(py::init<>(&make_sampler),
           R"pbdoc(

          Parameters
          ----------
          shape : List[int]
              The shape of quantity to be sampled. Use ``[]`` for scalar, ``[n]``
              for a vector, and ``[m, n]`` for a matrix.
          component_names : Optional[List[str]] = None
              Names to give to each sampled vector element. If None, components
              are given default names according to the shape:

              - shape = [] (scalar) -> {"0"}
              - shape = [n] (vector) -> {"0", "1", ..., "n-1"}
              - shape = [m, n] (matrix) -> {"0,0", "1,0", ..., "m-1,n-1"}

          capacity_increment : int, default=1000
              How much to resize the underlying matrix by whenever space runs out

          )pbdoc",
           py::arg("shape"), py::arg("component_names") = std::nullopt,
           py::arg("capacity_increment") = 1000)
      .def(
          "append",
          [](monte::Sampler &s, Eigen::VectorXd const &vector) {
            s.push_back(vector);
          },
          R"pbdoc(
            Add a new sample - any shape, unrolled.
          )pbdoc",
          py::arg("vector"))
      .def("set_values", &monte::Sampler::set_values,
           R"pbdoc(
            Set all values directly
          )pbdoc",
           py::arg("values"))
      .def("clear", &monte::Sampler::clear,
           R"pbdoc(
            Clear values - preserves ``n_components``, set ``n_samples`` to 0.
          )pbdoc")
      .def("set_sample_capacity", &monte::Sampler::set_sample_capacity,
           R"pbdoc(
            Conservative resize, to increase capacity for more samples.
          )pbdoc",
           py::arg("sample_capacity"))
      .def("set_capacity_increment", &monte::Sampler::set_capacity_increment,
           R"pbdoc(
            Set capacity increment (used when push_back requires more capacity).
          )pbdoc",
           py::arg("capacity_increment"))
      .def("component_names", &monte::Sampler::component_names,
           R"pbdoc(
            Return sampled vector component names.
          )pbdoc")
      .def("shape", &monte::Sampler::shape,
           R"pbdoc(
            Return sampled quantity shape before unrolling.
          )pbdoc")
      .def("n_components", &monte::Sampler::n_components,
           R"pbdoc(
            Number of components (vector size) of samples.
          )pbdoc")
      .def("n_samples", &monte::Sampler::n_samples,
           R"pbdoc(
            Current number of samples taken.
          )pbdoc")
      .def("sample_capacity", &monte::Sampler::sample_capacity,
           R"pbdoc(
            Current sample capacity.
          )pbdoc")
      .def("values", &monte::Sampler::values,
           R"pbdoc(
            Get sampled values as a matrix of shape=(n_samples, n_components).
          )pbdoc")
      .def("component", &monte::Sampler::component,
           R"pbdoc(
            Get all samples of a particular component (a column of `values()`).
          )pbdoc",
           py::arg("component_index"))
      .def("sample", &monte::Sampler::sample,
           R"pbdoc(
            Get a sample (a row of `values()`).
          )pbdoc",
           py::arg("sample_index"));

  py::bind_map<SamplerMap>(m, "SamplerMap",
                           R"pbdoc(
      SamplerMap stores Sampler by name of the sampled quantity

      Notes
      -----
      SamplerMap is a Dict[str, Sampler]-like object.
      )pbdoc");

  m.def("get_n_samples", monte::get_n_samples,
        R"pbdoc(
        Return the number of samples taken. Assumes the same value for all samplers in the SamplerMap.
      )pbdoc",
        py::arg("samplers"));

  py::class_<monte::SamplerComponent>(m, "SamplerComponent",
                                      R"pbdoc(
        Specify a component of a sampled quantity.

        Notes
        -----
        This data structure is used as a key to specify convergence criteria
        and results for components of sampled quantities.
      )pbdoc")
      .def(py::init<std::string, Index, std::string>(),
           R"pbdoc(

          Parameters
          ----------
          sampler_name : str
              Name of the sampled quantity. Should match keys in a :class:`~libcasm.monte.SamplerMap`.
          component_index : int
              Index into the unrolled vector of a sampled quantity.
          component_name : str
              Name corresponding to the component specified by ``component_index``.

          )pbdoc",
           py::arg("sampler_name"), py::arg("component_index"),
           py::arg("component_name"))
      .def_readwrite("sampler_name", &monte::SamplerComponent::sampler_name)
      .def_readwrite("component_index",
                     &monte::SamplerComponent::component_index)
      .def_readwrite("component_name",
                     &monte::SamplerComponent::component_name);

  py::class_<monte::RequestedPrecision>(m, "RequestedPrecision",
                                        R"pbdoc(
        Specify the requested absolute and/or relative precision for convergence.

        Notes
        -----
        Convergence of both absolute and relative precision may be requested

      )pbdoc")
      .def(py::init<>(&make_requested_precision),
           R"pbdoc(

          Parameters
          ----------
          abs : Optional[float] = None
              If provided, the specified absolute precision level requested, :math:`p_{abs}`.
          rel : Optional[float] = None
              If provided, the specified relative precision level requested, :math:`p_{rel}`.

          )pbdoc",
           py::arg("abs"), py::arg("rel"))
      .def_readwrite("abs_convergence_is_required",
                     &monte::RequestedPrecision::abs_convergence_is_required,
                     R"pbdoc(
                     bool: If True, absolute convergence is required.

                     Convergence of absolute precision of the calculated mean, :math:`p_{calc}`, is requested to the specified absolute precision, :math:`p_{abs}`:

                     .. math::

                         p_{calc} < p_{abs}

                     )pbdoc")
      .def_readwrite("abs_precision", &monte::RequestedPrecision::abs_precision,
                     R"pbdoc(
                    float: Value of required absolute precision of the mean, :math:`p_{abs}`
                    )pbdoc")
      .def_readwrite("rel_convergence_is_required",
                     &monte::RequestedPrecision::rel_convergence_is_required,
                     R"pbdoc(
                     bool: If True, relative convergence is required.

                     Convergence of absolute precision of the calculated mean, :math:`p_{calc}`, is requested to the specified absolute precision, :math:`p_{abs}`:

                     .. math::

                         p_{calc} < p_{abs}

                     )pbdoc")
      .def_readwrite("rel_precision", &monte::RequestedPrecision::rel_precision,
                     R"pbdoc(
                     float: Value of requsted relative precision, :math:`p_{rel}`.
                     )pbdoc");

  py::bind_map<RequestedPrecisionMap>(m, "RequestedPrecisionMap",
                                      R"pbdoc(
      RequestedPrecisionMap stores :class:`~libcasm.monte.RequestedPrecision` with :class:`~libcasm.monte.SamplerComponent` keys.

      Notes
      -----
      RequestedPrecisionMap is a Dict[SamplerComponent, RequestedPrecision]-like object.
      )pbdoc");

  py::class_<monte::IndividualEquilibrationCheckResult>(
      m, "IndividualEquilibrationResult",
      R"pbdoc(
      Equilibration check results for a single SamplerComponent
      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          Default constructor only.
          )pbdoc")
      .def_readwrite(
          "is_equilibrated",
          &monte::IndividualEquilibrationCheckResult::is_equilibrated,
          R"pbdoc(
          True if equilibration check performed and value is equilibrated. Default=``False``.
          )pbdoc")
      .def_readwrite("N_samples_for_equilibration",
                     &monte::IndividualEquilibrationCheckResult::
                         N_samples_for_equilibration,
                     R"pbdoc(
          Number of samples involved in equilibration (for this component only). Default=``0``.

          Note
          ----

          Multiple quantities/components may be requested for equilibration and
          convergence, so statistics are only taken when *all* requested values are
          equilibrated.

          )pbdoc");

  py::bind_map<EquilibrationResultMap>(m, "EquilibrationResultMap",
                                       R"pbdoc(
      EquilibrationResultMap stores IndividualEquilibrationResult by SamplerComponent

      Notes
      -----
      EquilibrationResultMap is a Dict[SamplerComponent, IndividualEquilibrationResult]-like object.
      )pbdoc");

  py::class_<monte::EquilibrationCheckResults>(m, "EquilibrationCheckResults",
                                               R"pbdoc(
      Stores equilibration check results
      )pbdoc")
      //
      .def(py::init<>(),
           R"pbdoc(
          Default constructor only.
          )pbdoc")
      .def_readwrite("all_equilibrated",
                     &monte::EquilibrationCheckResults::all_equilibrated,
                     R"pbdoc(
          True if all required properties are equilibrated to the requested precision.

          Notes
          -----

          - True if completion check finds all required properties are equilibrated
            to the requested precision
          - False otherwise, including if no convergence checks were requested
          )pbdoc")
      .def_readwrite(
          "N_samples_for_all_to_equilibrate",
          &monte::EquilibrationCheckResults::N_samples_for_all_to_equilibrate,
          R"pbdoc(
          How long (how many samples) it took for all requested values to equilibrate; set to 0 if no convergence checks were requested
          )pbdoc")
      .def_readwrite("individual_results",
                     &monte::EquilibrationCheckResults::individual_results,
                     R"pbdoc(
          Results from checking equilibration criteria.
          )pbdoc");

  py::class_<monte::BasicStatistics>(m, "BasicStatistics",
                                     R"pbdoc(
      Basic statistics calculated from samples
      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          Default constructor only.
          )pbdoc")
      .def_readwrite("mean", &monte::BasicStatistics::mean,
                     R"pbdoc(
          Mean of property. Default=``0.0``.
          )pbdoc")
      .def_readwrite("calculated_precision",
                     &monte::BasicStatistics::calculated_precision,
                     R"pbdoc(
          Calculated absolute precision of the mean.
          )pbdoc")
      .def("relative_precision", &monte::get_calculated_relative_precision,
           R"pbdoc(
          Calculated precision as a absolute value of fraction of the mean.
          )pbdoc")
      .def(
          "to_dict",
          [](monte::BasicStatistics const &stats) {
            jsonParser json;
            to_json(stats, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the BasicStatistics as a Python dict.");

  m.def(
      "calc_basic_statistics",
      [](Eigen::VectorXd const &observations, double confidence,
         std::optional<Eigen::VectorXd> const &sample_weight) {
        if (sample_weight.has_value()) {
          return monte::calc_basic_statistics(observations, *sample_weight,
                                              confidence);
        } else {
          static Eigen::VectorXd empty_sample_weight;
          return monte::calc_basic_statistics(observations, empty_sample_weight,
                                              confidence);
        }
      },
      R"pbdoc(

      Parameters
      ----------
      observations : numpy.ndarray[numpy.float64[n_samples, 1]]]
          Values to calculate statistics.
      confidence : float = 0.95
          Confidence level to use for calculated precision of the mean.
      sample_weight : Optional[numpy.ndarray[numpy.float64[n_samples, 1]]]]
          Optional weight to give to each to observation. If sample weights are
          provided, a time-series is constructed from the observations and
          covariances used to estimate the precision of the mean are calculated
          by re-sampling at 10000 equally spaced time intervals.
      )pbdoc",
      py::arg("observations"), py::arg("confidence") = 0.95,
      py::arg("sample_weight") = std::nullopt);

  py::class_<monte::IndividualConvergenceCheckResult<monte::BasicStatistics>>(
      m, "IndividualConvergenceResult",
      R"pbdoc(
      Convergence check results for a single SamplerComponent
      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          Default constructor only.
          )pbdoc")
      .def_readwrite("is_converged",
                     &monte::IndividualConvergenceCheckResult<
                         monte::BasicStatistics>::is_converged,
                     R"pbdoc(
          True if mean is converged to requested precision. Default=``False``.
          )pbdoc")
      .def_readwrite("requested_precision",
                     &monte::IndividualConvergenceCheckResult<
                         monte::BasicStatistics>::requested_precision,
                     R"pbdoc(
          Requested precision of the mean.
          )pbdoc")
      .def_readwrite("stats",
                     &monte::IndividualConvergenceCheckResult<
                         monte::BasicStatistics>::stats,
                     R"pbdoc(
          Calculated statistics.
          )pbdoc");

  py::bind_map<ConvergenceResultMap>(m, "ConvergenceResultMap",
                                     R"pbdoc(
      ConvergenceResultMap stores IndividualConvergenceResult by SamplerComponent

      Notes
      -----
      ConvergenceResultMap is a Dict[SamplerComponent, IndividualConvergenceResult]-like object.
      )pbdoc");

  py::class_<monte::ConvergenceCheckResults<monte::BasicStatistics>>(
      m, "ConvergenceCheckResults",
      R"pbdoc(
      Stores convergence check results
      )pbdoc")
      //
      .def(py::init<>(),
           R"pbdoc(
          Default constructor only.
          )pbdoc")
      .def_readwrite("all_converged",
                     &monte::ConvergenceCheckResults<
                         monte::BasicStatistics>::all_converged,
                     R"pbdoc(
          True if all required properties are converged to the requested precision.

          Notes
          -----

          - True if convergence check finds all required properties are converged
            to the requested precision
          - False otherwise, including if no convergence checks were requested
          )pbdoc")
      .def_readwrite("N_samples_for_statistics",
                     &monte::ConvergenceCheckResults<
                         monte::BasicStatistics>::N_samples_for_statistics,
                     R"pbdoc(
          How many samples were used to get statistics.

          Notes
          -----

          - Set to the total number of samples if no convergence checks were
            requested
          )pbdoc")
      .def_readwrite("individual_results",
                     &monte::ConvergenceCheckResults<
                         monte::BasicStatistics>::individual_results,
                     R"pbdoc(
          Results from checking convergence criteria.
          )pbdoc");

  //
  m.def(
      "convergence_check",
      [](RequestedPrecisionMap const &requested_precision,
         monte::CountType N_samples_for_equilibration,
         std::map<std::string, std::shared_ptr<monte::Sampler>> const &samplers,
         monte::CalcStatisticsFunction<monte::BasicStatistics> calc_statistics_f)
          -> monte::ConvergenceCheckResults<monte::BasicStatistics> {
    return monte::convergence_check(requested_precision,
                                    N_samples_for_equilibration, samplers,
                                    calc_statistics_f);
        }
},
      R"pbdoc(

      Parameters
      ----------
      samplers: :class:`~libcasm.monte.SamplerMap`
          The samplers containing the sampled data.
      requested_precision : :class:`~libcasm.monte.RequestedPrecisionMap`
          The requested precision levels for all :class:`~libcasm.monte.SamplerComponent` that are requested to converge.
      N_samples_for_equilibration : int
          Number of initial samples to exclude from statistics because the system is out of equilibrium.
      calc_statistics_f : function, default=`~libcasm.monte.calc_basic_statistics()`
          Fucntion used to calculate statistics.
      )pbdoc",
      py::arg("samplers"),
      py::arg("requested_precision"),
      py::arg("N_samples_for_equilibration"),
      py::arg("calc_statistics_f"));

//
m.def(
    "weighted_convergence_check",
    [](std::map<std::string, std::shared_ptr<monte::Sampler>> const &samplers,
       std::shared_ptr<monte::Sampler> const &sample_weight,
       RequestedPrecisionMap const &requested_precision,
       monte::CountType N_samples_for_equilibration,
       monte::CalcWeightedStatisticsFunction<monte::BasicStatistics>
           calc_weighted_statistics_f)
        -> monte::ConvergenceCheckResults<monte::BasicStatistics> {
      return monte::convergence_check(
          samplers, *sample_weight, requested_precision,
          N_samples_for_equilibration, calc_weighted_statistics_f);
    },
    R"pbdoc(

      Parameters
      ----------
      samplers: :class:`~libcasm.monte.SamplerMap`
          The samplers containing the sampled data.
      sample_weight : :class:`~libcasm.monte.Sampler`
          Optional weight to give to each to observation.
      requested_precision : :class:`~libcasm.monte.RequestedPrecisionMap`
          The requested precision levels for all :class:`~libcasm.monte.SamplerComponent` that are requested to converge.
      N_samples_for_equilibration : int
          Number of initial samples to exclude from statistics because the system is out of equilibrium.
      calc_weighted_statistics_f : function
          Fucntion used to calculate statistics from weighted observations.
      )pbdoc",
    py::arg("samplers"), py::arg("sample_weight"),
    py::arg("requested_precision"), py::arg("N_samples_for_equilibration"),
    py::arg("calc_statistics_f"));

#ifdef VERSION_INFO
m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
m.attr("__version__") = "dev";
#endif
}
