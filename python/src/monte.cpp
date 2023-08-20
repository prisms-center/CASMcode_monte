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
#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/monte/BasicStatistics.hh"
#include "casm/monte/Conversions.hh"
#include "casm/monte/MethodLog.hh"
#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/checks/EquilibrationCheck.hh"
#include "casm/monte/checks/io/json/CompletionCheck_json_io.hh"
#include "casm/monte/checks/io/json/ConvergenceCheck_json_io.hh"
#include "casm/monte/checks/io/json/CutoffCheck_json_io.hh"
#include "casm/monte/checks/io/json/EquilibrationCheck_json_io.hh"
#include "casm/monte/sampling/Sampler.hh"
#include "casm/monte/sampling/SamplingParams.hh"
#include "casm/monte/sampling/io/json/Sampler_json_io.hh"
#include "casm/monte/sampling/io/json/SamplingParams_json_io.hh"
#include "casm/monte/state/StateSampler.hh"
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
typedef std::map<std::string, monte::StateSamplingFunction>
    StateSamplingFunctionMap;
typedef std::map<monte::SamplerComponent, monte::RequestedPrecision>
    RequestedPrecisionMap;
typedef std::map<
    monte::SamplerComponent,
    monte::IndividualConvergenceCheckResult<monte::BasicStatistics>>
    ConvergenceResultMap;
typedef std::map<monte::SamplerComponent,
                 monte::IndividualEquilibrationCheckResult>
    EquilibrationResultMap;

monte::MethodLog make_MethodLog(std::string logfile_path,
                                std::optional<double> log_frequency) {
  monte::MethodLog method_log;
  method_log.logfile_path = logfile_path;
  method_log.log_frequency = log_frequency;
  method_log.reset();
  return method_log;
}

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

monte::StateSamplingFunction make_state_sampling_function(
    std::string name, std::string description, std::vector<Index> shape,
    std::function<Eigen::VectorXd()> function,
    std::optional<std::vector<std::string>> component_names) {
  if (function == nullptr) {
    throw std::runtime_error(
        "Error constructing StateSamplingFunction: function == nullptr");
  }
  if (!component_names.has_value()) {
    return monte::StateSamplingFunction(name, description, shape, function);
  } else {
    return monte::StateSamplingFunction(name, description, *component_names,
                                        shape, function);
  }
}

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, bool>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, double>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, Eigen::VectorXd>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, Eigen::MatrixXd>);
PYBIND11_MAKE_OPAQUE(CASMpy::SamplerMap);
PYBIND11_MAKE_OPAQUE(CASMpy::StateSamplingFunctionMap);
PYBIND11_MAKE_OPAQUE(CASMpy::RequestedPrecisionMap);
PYBIND11_MAKE_OPAQUE(CASMpy::ConvergenceResultMap);
PYBIND11_MAKE_OPAQUE(CASMpy::EquilibrationResultMap);

PYBIND11_MODULE(_monte, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Building blocks for Monte Carlo simualations

        libcasm.monte
        -------------

        The libcasm-monte is a Python interface to the classes and methods in the CASM::monte namespace of the CASM C++ libraries that are useful building blocks for Monte Carlo simulations.

    )pbdoc";
  py::module::import("libcasm.xtal");

  py::class_<monte::MethodLog>(m, "MethodLog", R"pbdoc(
      Logger for Monte Carlo method status

      )pbdoc")
      .def(py::init<>(&make_MethodLog),
           R"pbdoc(
          Constructor

          Parameters
          ----------
          logfile_path : str
              File location for log output
          log_frequency : Optional[float]
              How often to log method status, in seconds
          )pbdoc",
           py::arg("logfile_path"), py::arg("log_frequency") = std::nullopt)
      .def(
          "logfile_path",
          [](monte::MethodLog const &x) { return x.logfile_path.string(); },
          R"pbdoc(
          File location for log output
          )pbdoc")
      .def(
          "log_frequency",
          [](monte::MethodLog const &x) { return x.log_frequency; },
          R"pbdoc(
          How often to log method status, in seconds
          )pbdoc")
      .def("reset", &monte::MethodLog::reset,
           R"pbdoc(
          Reset log file, creating parent directories as necessary
          )pbdoc");

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
          :class:`~libcasm.monte.BooleanValueMap`: A Dict[str, bool]-like object.
          )pbdoc")
      .def_readwrite("scalar_values", &monte::ValueMap::scalar_values,
                     R"pbdoc(
          :class:`~libcasm.monte.ScalarValueMap`: A Dict[str, float]-like object.
          )pbdoc")
      .def_readwrite("vector_values", &monte::ValueMap::vector_values,
                     R"pbdoc(
          :class:`~libcasm.monte.VectorValueMap`: A Dict[str, numpy.ndarray[numpy.float64[m, 1]]]-like object.
          )pbdoc")
      .def_readwrite("matrix_values", &monte::ValueMap::matrix_values,
                     R"pbdoc(
          :class:`~libcasm.monte.MatrixValueMap`: A Dict[str, numpy.ndarray[numpy.float64[m, n]]]-like object.
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
        R"pbdoc(
        Return true if :class:`~libcasm.monte.ValueMap` do not have the same properties.
        )pbdoc",
        py::arg("A"), py::arg("B"));

  m.def("make_incremented_values", &monte::make_incremented_values,
        R"pbdoc(
      Return values[property] + n_increment*increment[property] for each property

      Notes
      -----
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
          SamplingParams only has a default constructor
          )pbdoc")
      .def_readwrite("sample_mode", &monte::SamplingParams::sample_mode,
                     R"pbdoc(
          SAMPLE_MODE: Sample by pass, step, or time. Default=``SAMPLE_MODE.BY_PASS``.
          )pbdoc")
      .def_readwrite("sample_method", &monte::SamplingParams::sample_mode,
                     R"pbdoc(
          SAMPLE_METHOD: Sample linearly or logarithmically. Default=``SAMPLE_METHOD.LINEAR``.

          Notes
          -----

          For ``SAMPLE_METHOD.LINEAR``, take the n-th sample when:

          .. code-block:: Python

              sample/pass = round( begin + (period / samples_per_period) * n )

              time = begin + (period / samples_per_period) * n

          The default values are:

          For ``SAMPLE_METHOD.LOG``, take the n-th sample when:

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
            float: See `sample_method`. Default=``0.0``.
          )pbdoc")
      .def_readwrite("period", &monte::SamplingParams::period,
                     R"pbdoc(
            float: See `sample_method`. Default=``1.0``.
          )pbdoc")
      .def_readwrite("samples_per_period",
                     &monte::SamplingParams::samples_per_period,
                     R"pbdoc(
            float: See `sample_method`. Default=``1.0``.
          )pbdoc")
      .def_readwrite("shift", &monte::SamplingParams::shift,
                     R"pbdoc(
            float: See `sample_method`. Default=``0.0``.
          )pbdoc")
      .def_readwrite("sampler_names", &monte::SamplingParams::sampler_names,
                     R"pbdoc(
            List[str]: The names of quantities to sample (i.e. sampling function names). Default=``[]``.
          )pbdoc")
      .def_readwrite("do_sample_trajectory",
                     &monte::SamplingParams::do_sample_trajectory,
                     R"pbdoc(
            bool: If true, save the configuration when a sample is taken. Default=``False``.
          )pbdoc")
      .def_readwrite("do_sample_time", &monte::SamplingParams::do_sample_time,
                     R"pbdoc(
            bool: If true, save current time when taking a sample. Default=``False``.
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
      Sampler stores sampled data in a matrix

      Notes
      -----
      - :class:`~libcasm.monte.Sampler` helps sampling by re-sizing the underlying matrix holding data automatically, and it allows accessing particular observations as an unrolled vector or accessing a particular component as a vector to check convergence.
      - Sampler can be used to sample quantities of any dimension (scalar, vector, matrix, etc.) by unrolling values. The standard approach is to use column-major order.
      - For sampling scalars, a size=1 vector is expected. This can be done with the function :func:`~libcasm.monte.scalar_as_vector`.
      - For sampling matrices, column-major order unrolling can be done with the function :func:`~libcasm.monte.matrix_as_vector`.

      )pbdoc")
      .def(py::init<>(&make_sampler),
           R"pbdoc(

          Parameters
          ----------
          shape : List[int]
              The shape of quantity to be sampled. Use ``[]`` for scalar, ``[n]``
              for a length ``n`` vector, ``[m, n]`` for a shape ``(m,n)`` matrix, etc.
          component_names : Optional[List[str]] = None
              Names to give to each sampled vector element. If None, components
              are given default names according column-major unrolling:

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
      SamplerMap stores :class:`~libcasm.monte.Sampler` by name of the sampled quantity

      Notes
      -----
      SamplerMap is a Dict[str, :class:`~libcasm.monte.Sampler`]-like object.
      )pbdoc");

  m.def("get_n_samples", monte::get_n_samples,
        R"pbdoc(
        Return the number of samples taken. Assumes the same value for all samplers in the :class:`~libcasm.monte.SamplerMap`.
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
      .def_readwrite("sampler_name", &monte::SamplerComponent::sampler_name,
                     R"pbdoc(
                      str : Name of the sampled quantity.

                      Should match keys in a :class:`~libcasm.monte.SamplerMap`.
                      )pbdoc")
      .def_readwrite("component_index",
                     &monte::SamplerComponent::component_index,
                     R"pbdoc(
                     int : Index into the unrolled vector of a sampled quantity.
                     )pbdoc")
      .def_readwrite("component_name", &monte::SamplerComponent::component_name,
                     R"pbdoc(
                     str : Name of sampler component specified by ``component_index``.
                     )pbdoc");

  py::class_<monte::StateSamplingFunction>(m, "StateSamplingFunction",
                                           R"pbdoc(
        A function that samples data from a Monte Carlo state

        Example usage:

        .. code-block:: Python

            from libcasm.clexmonte import SemiGrandCanonical
            from libcasm.monte import (
                Sampler, SamplerMap, StateSamplingFunction,
                StateSamplingFunctionMap,
            )

            # ... in Monte Carlo simulation setup ...
            # calculation = SemiGrandCanonical(...)

            # ... create sampling functions ...
            sampling_functions = StateSamplingFunctionMap()

            composition_calculator = get_composition_calculator(
                calculation.system
            )

            def mol_composition_f():
                return composition_calculator.
                    mean_num_each_component(get_occupation(calculation.state))

            f = monte.StateSamplingFunction(
                name="mol_composition",
                description="Mol composition per unit cell",
                shape=[len(composition_calculator.components())],
                function=mol_composition_f,
                component_names=composition_calculator.components(),
            )
            sampling_functions[f.name] = f

            # ... create samplers to hold data ...
            samplers = SamplerMap()
            for name, f in sampling_functions.items():
                samplers[name] = Sampler(
                    shape=f.shape,
                    component_names=f.component_names,
                )

            # ... in Monte Carlo simulation ...
            # ... sample data ...
            for name, f in sampling_functions.items():
                samplers[name].append(f())


        Notes
        -----
        - Typically this holds a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the current state of the simulation.
        - StateSamplingFunction can be used to sample quantities of any dimension (scalar, vector, matrix, etc.) by unrolling values. The standard approach is to use column-major order.
        - For sampling scalars, a size=1 vector is expected. This can be done with the function :func:`~libcasm.monte.scalar_as_vector`.
        - For sampling matrices, column-major order unrolling can be done with the function :func:`~libcasm.monte.matrix_as_vector`.
        - Data sampled by a StateSamplingFunction can be stored in a :class:`~libcasm.monte.Sampler`.
        - A call operator exists (:func:`~libcasm.monte.StateSamplingFunction.__call__`) to call the function held by :class:`~libcasm.monte.StateSamplingFunction`.
        )pbdoc")
      .def(py::init<>(&make_state_sampling_function),
           R"pbdoc(

          Parameters
          ----------
          name : str
              Name of the sampled quantity.

          description : str
              Description of the function.

          component_index : int
              Index into the unrolled vector of a sampled quantity.

          shape : List[int]
              Shape of quantity, with column-major unrolling

              Scalar: [], Vector: [n], Matrix: [m, n], etc.

          function : function
              A function with 0 arguments that returns an array of the proper size sampling the current state. Typically this is a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the current state of the simulation.

          component_names : Optional[List[str]]
              A name for each component of the resulting vector.

              Can be strings representing an indices (i.e "0", "1", "2", etc.) or can be a descriptive string (i.e. "Mg", "Va", "O", etc.). If None, indices for column-major ordering are used (i.e. "0,0", "1,0", ..., "m-1,n-1")

          )pbdoc",
           py::arg("name"), py::arg("description"), py::arg("shape"),
           py::arg("function"), py::arg("component_names"))
      .def_readwrite("name", &monte::StateSamplingFunction::name,
                     R"pbdoc(
          str : Name of the sampled quantity.
          )pbdoc")
      .def_readwrite("description", &monte::StateSamplingFunction::description,
                     R"pbdoc(
          str : Description of the function.
          )pbdoc")
      .def_readwrite("shape", &monte::StateSamplingFunction::shape,
                     R"pbdoc(
          List[int] : Shape of quantity, with column-major unrolling.

          Scalar: [], Vector: [n], Matrix: [m, n], etc.
          )pbdoc")
      .def_readwrite("component_names",
                     &monte::StateSamplingFunction::component_names,
                     R"pbdoc(
          List[str] : A name for each component of the resulting vector.

          Can be strings representing an indices (i.e "0", "1", "2", etc.) or can be a descriptive string (i.e. "Mg", "Va", "O", etc.). If the sampled quantity is an unrolled matrix, indices for column-major ordering are typical (i.e. "0,0", "1,0", ..., "m-1,n-1").
          )pbdoc")
      .def_readwrite("function", &monte::StateSamplingFunction::function,
                     R"pbdoc(
          function : The function to be evaluated.

          A function with 0 arguments that returns an array of the proper size sampling the current state. Typically this is a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the current state of the simulation.
          )pbdoc")
      .def(
          "__call__",
          [](monte::StateSamplingFunction const &f) -> Eigen::VectorXd {
            return f();
          },
          R"pbdoc(
          Evaluates the state sampling function

          Equivalent to calling :py::attr:`~libcasm.monte.StateSamplingFunction.function`.
          )pbdoc");

  py::bind_map<StateSamplingFunctionMap>(m, "StateSamplingFunctionMap",
                                         R"pbdoc(
      StateSamplingFunctionMap stores :class:`~libcasm.monte.StateSamplingFunction` by name of the sampled quantity.

      Notes
      -----
      StateSamplingFunctionMap is a Dict[str, :class:`~libcasm.monte.StateSamplingFunction`]-like object.
      )pbdoc");

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
                     )pbdoc")
      .def(
          "to_dict",
          [](monte::RequestedPrecision const &x) {
            jsonParser json;
            to_json(x, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent RequestedPrecision as a Python dict.")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) {
            jsonParser json{data};
            monte::RequestedPrecision x;
            from_json(x, json);
            return x;
          },
          "Construct RequestedPrecision from a Python dict.", py::arg("data"));

  py::bind_map<RequestedPrecisionMap>(m, "RequestedPrecisionMap",
                                      R"pbdoc(
      RequestedPrecisionMap stores :class:`~libcasm.monte.RequestedPrecision` with :class:`~libcasm.monte.SamplerComponent` keys.

      Notes
      -----
      RequestedPrecisionMap is a Dict[:class:`~libcasm.monte.SamplerComponent`, :class:`~libcasm.monte.RequestedPrecision`]-like object.
      )pbdoc");

  py::class_<monte::IndividualEquilibrationCheckResult>(
      m, "IndividualEquilibrationResult",
      R"pbdoc(
      Equilibration check results for a single :class:`~libcasm.monte.SamplerComponent`
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

          )pbdoc")
      .def(
          "to_dict",
          [](monte::IndividualEquilibrationCheckResult const &x) {
            jsonParser json;
            to_json(x, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent IndividualEquilibrationResult as a Python dict.");

  py::bind_map<EquilibrationResultMap>(m, "EquilibrationResultMap",
                                       R"pbdoc(
      EquilibrationResultMap stores :class:`~libcasm.monte.IndividualEquilibrationResult` by :class:`~libcasm.monte.SamplerComponent`

      Notes
      -----
      EquilibrationResultMap is a Dict[:class:`~libcasm.monte.SamplerComponent`, :class:`~libcasm.monte.IndividualEquilibrationResult`]-like object.
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
          )pbdoc")
      .def(
          "to_dict",
          [](monte::EquilibrationCheckResults const &x) {
            jsonParser json;
            to_json(x, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent EquilibrationCheckResults as a Python dict.");

  m.def("default_equilibration_check", &monte::default_equilibration_check,
        R"pbdoc(
      Check if a range of observations have equilibrated

      Notes
      -----
      This uses the following algorithm, based on Van de Walle and Asta, Modelling Simul. Mater. Sci. Eng. 10 (2002) 521–538.

      Partition observations into three ranges:

        - equilibriation stage:  [0, start1)
        - first partition:  [start1, start2)
        - second partition: [start2, N)

      where N is observations.size(), start1 and start 2 are indices into
      observations such 0 <= start1 < start2 <= N, the number of elements in
      the first and second partition are the same (within 1).

      The calculation is considered equilibrated at start1 if the mean of the
      elements in the first and second partition are approximately equal to the
      desired precsion: (std::abs(mean1 - mean2) < prec).

      Additionally, the value start1 is incremented as much as needed to ensure
      that the equilibriation stage has observations on either side of the total
      mean.

      If all observations are approximately equal, then:

      - is_equilibrated = true
      - N_samples_for_equilibration = 0

      If the equilibration conditions are met, the result contains:

      - is_equilibrated = true
      - N_samples_for_equilibration = start1

      If the equilibration conditions are not met, the result contains:

      - is_equilibrated = false
      - N_samples_for_equilibration = <undefined>

      Is samples are weighted, the following change is made:

      Use:

          weighted_observation(i) = sample_weight[i] * observation(i) * N / W

      where:

          W = sum_i sample_weight[i]

      The same weight_factor N/W applies for all properties.


      Parameters
      ----------
      observations : array_like
          A 1d array of observations. Should include all samples.
      sample_weight : array_like
          Sample weights associated with observations. May have size 0, in which case the observations are treated as being equally weighted and no resampling is performed, or have the same size as `observations`.
      requested_precision : :class:`~libcasm.monte.RequestedPrecisionMap`
          The requested precision level for convergence.

      Returns
      -------
      results : :class:`~libcasm.monte.IndividualEquilibrationResult`
          The equilibration check results.
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

  py::class_<monte::BasicStatisticsCalculator>(m, "BasicStatisticsCalculator",
                                               R"pbdoc(
      Basic statistics calculator
      )pbdoc")
      .def(py::init<double, Index, Index>(),
           R"pbdoc(
          Constructor

          Parameters
          ----------
          confidence : float = 0.95
              Confidence level to use for calculated precision of the mean.
          weighted_observations_method : int = 1
              Method used to estimate precision in the sample mean when observations
              are weighted (i.e. N-fold way method). Options are:

              1) Calculate weighted sample variance directly from weighted samples and only autocorrelation factor (1+rho)/(1-rho) from resampled observations
              2) Calculate all statistics from resampled observations
          n_resamples : int = 10000
              Number of resampled observations to make for autocovariance estimation when observations are weighted.

          )pbdoc",
           py::arg("confidence") = 0.95,
           py::arg("weighted_observations_method") = 1,
           py::arg("n_resamples") = 10000)
      .def_readwrite("confidence",
                     &monte::BasicStatisticsCalculator::confidence,
                     R"pbdoc(
          float : Confidence level used to calculate error interval.
          )pbdoc")
      .def_readwrite("weighted_observations_method",
                     &monte::BasicStatisticsCalculator::method,
                     R"pbdoc(
            int : Method used to estimate precision in the sample mean when observations
            are weighted.

            Options are:

            1) Calculate weighted sample variance directly from weighted samples and only autocorrelation factor (1+rho)/(1-rho) from resampled observations
            2) Calculate all statistics from resampled observations
          )pbdoc")
      .def_readwrite("n_resamples",
                     &monte::BasicStatisticsCalculator::n_resamples,
                     R"pbdoc(
          int : Number of resampled observations to make for autocovariance estimation when observations are weighted.
          )pbdoc")
      .def(
          "__call__",
          [](monte::BasicStatisticsCalculator const &f,
             Eigen::VectorXd const &observations,
             Eigen::VectorXd const &sample_weight) {
            return f(observations, sample_weight);
          },
          R"pbdoc(
          Calculate statistics for a range of weighted observations

          The method used to estimate precision in the sample mean when observations are weighted (i.e. N-fold way method) depends on the parameter :py:attr:`~libcasm.monte.weighted_observations_method`.

          Parameters
          ----------
          observations : array_like
              A 1d array of observations. Should only include samples after the calculation has equilibrated.
          sample_weight : array_like
              Sample weights associated with observations. May have size 0, in which case the observations are treated as being equally weighted and no resampling is performed, or have the same size as `observations`.

          Returns
          -------
          stats : :class:`~libcasm.monte.BasicStatistics`
              Calculated statistics.

          )pbdoc")
      .def(
          "to_dict",
          [](monte::BasicStatisticsCalculator const &f) {
            jsonParser json;
            json["confidence"] = f.confidence;
            json["weighted_observations_method"] = f.method;
            json["n_resamples"] = f.n_resamples;
            return static_cast<nlohmann::json>(json);
          },
          "Represent the BasicStatistics as a Python dict.")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) {
            jsonParser json{data};

            double confidence = 0.95;
            json.get_if(confidence, "confidence");

            Index weighted_observations_method = 1;
            json.get_if(weighted_observations_method,
                        "weighted_observations_method");

            Index n_resamples = 10000;
            json.get_if(n_resamples, "n_resamples");

            return monte::BasicStatisticsCalculator(
                confidence, weighted_observations_method, n_resamples);
          },
          "Construct a BasicStatisticsCalculator from a Python dict.",
          py::arg("data"));

  py::class_<monte::IndividualConvergenceCheckResult<monte::BasicStatistics>>(
      m, "IndividualConvergenceResult",
      R"pbdoc(
      Convergence check results for a single :class:`~libcasm.monte.SamplerComponent`
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
          )pbdoc")
      .def(
          "to_dict",
          [](monte::IndividualConvergenceCheckResult<
              monte::BasicStatistics> const &x) {
            jsonParser json;
            to_json(x, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the individual convergence check results as a Python "
          "dict.");

  py::bind_map<ConvergenceResultMap>(m, "ConvergenceResultMap",
                                     R"pbdoc(
      ConvergenceResultMap stores :class:`~libcasm.monte.IndividualConvergenceResult` by :class:`~libcasm.monte.SamplerComponent`

      Notes
      -----
      ConvergenceResultMap is a Dict[:class:`~libcasm.monte.SamplerComponent`, :class:`~libcasm.monte.IndividualConvergenceResult`]-like object.
      )pbdoc");

  py::class_<monte::ConvergenceCheckResults<monte::BasicStatistics>>(
      m, "ConvergenceCheckResults",
      R"pbdoc(
      Stores convergence check results
      )pbdoc")
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
          )pbdoc")
      .def(
          "to_dict",
          [](monte::ConvergenceCheckResults<monte::BasicStatistics> const &x) {
            jsonParser json;
            to_json(x, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the convergence check results as a Python dict.");

  m.def(
      "component_convergence_check",
      [](monte::Sampler const &sampler,
         std::shared_ptr<monte::Sampler> const &sample_weight,
         monte::SamplerComponent const &key,
         monte::RequestedPrecision const &requested_precision,
         monte::CountType N_samples_for_statistics,
         monte::CalcStatisticsFunction<monte::BasicStatistics>
             calc_statistics_f)
          -> monte::IndividualConvergenceCheckResult<monte::BasicStatistics> {
        return monte::component_convergence_check(
            sampler, *sample_weight, key, requested_precision,
            N_samples_for_statistics, calc_statistics_f);
      },
      R"pbdoc(
        Check convergence of an individual sampler component

        Parameters
        ----------
        sampler: :class:`~libcasm.monte.Sampler`
            The sampler containing the sampled data.
        sample_weight : :class:`~libcasm.monte.Sampler`
            Optional weight to give to each to observation.
        key : :class:`~libcasm.monte.SamplerComponent`
            Specifies the component of sampler being checked for convergence.
        requested_precision : :class:`~libcasm.monte.RequestedPrecisionMap`
            The requested precision level for convergence.
        N_samples_for_statistics : int
            The number of tail samples from `sampler` to include in statistics.
        calc_statistics_f : function
            Fucntion used to calculate :class:`~libcasm.monte.BasicStatistics`. For example, an instance of :class:`~libcasm.monte.BasicStatistics`.
        )pbdoc",
      py::arg("sampler"), py::arg("sample_weight"), py::arg("key"),
      py::arg("requested_precision"), py::arg("N_samples_for_statistics"),
      py::arg("calc_statistics_f"));

  m.def(
      "convergence_check",
      [](std::map<std::string, std::shared_ptr<monte::Sampler>> const &samplers,
         std::shared_ptr<monte::Sampler> const &sample_weight,
         RequestedPrecisionMap const &requested_precision,
         monte::CountType N_samples_for_equilibration,
         monte::CalcStatisticsFunction<monte::BasicStatistics>
             calc_statistics_f)
          -> monte::ConvergenceCheckResults<monte::BasicStatistics> {
        return monte::convergence_check(
            samplers, *sample_weight, requested_precision,
            N_samples_for_equilibration, calc_statistics_f);
      },
      R"pbdoc(
        Check convergence of all requested sampler components

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
        calc_statistics_f : function
            Fucntion used to calculate :class:`~libcasm.monte.BasicStatistics`. For example, an instance of :class:`~libcasm.monte.BasicStatistics`.
        )pbdoc",
      py::arg("samplers"), py::arg("sample_weight"),
      py::arg("requested_precision"), py::arg("N_samples_for_equilibration"),
      py::arg("calc_statistics_f"));

  py::class_<monte::CutoffCheckParams>(m, "CutoffCheckParams",
                                       R"pbdoc(
        Completion check parameters that don't depend on the sampled values.

        Notes
        -----
        - A Monte Carlo simulation does not stop before all minimums are met
        - A Monte Carlo simulation does stop when any maximum is met

        )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
           Constructor
           )pbdoc")
      .def_readwrite("min_count", &monte::CutoffCheckParams::min_count,
                     R"pbdoc(
                     Optional[int]: Minimum number of steps or passes.
                     )pbdoc")
      .def_readwrite("min_time", &monte::CutoffCheckParams::min_time,
                     R"pbdoc(
                     Optional[float]: Minimum simulated time.
                     )pbdoc")
      .def_readwrite("min_sample", &monte::CutoffCheckParams::min_sample,
                     R"pbdoc(
                     Optional[int]: Minimum number of samples.
                     )pbdoc")
      .def_readwrite("min_time", &monte::CutoffCheckParams::min_clocktime,
                     R"pbdoc(
                     Optional[float]: Minimum elapsed clocktime.
                     )pbdoc")
      .def_readwrite("max_count", &monte::CutoffCheckParams::max_count,
                     R"pbdoc(
                     Optional[int]: Maximum number of steps or passes.
                     )pbdoc")
      .def_readwrite("max_time", &monte::CutoffCheckParams::max_time,
                     R"pbdoc(
                     Optional[float]: Maximum simulated time.
                     )pbdoc")
      .def_readwrite("max_sample", &monte::CutoffCheckParams::max_sample,
                     R"pbdoc(
                     Optional[int]: Maximum number of samples.
                     )pbdoc")
      .def_readwrite("max_time", &monte::CutoffCheckParams::max_clocktime,
                     R"pbdoc(
                     Optional[float]: Maximum elapsed clocktime.
                     )pbdoc")
      .def(
          "to_dict",
          [](monte::CutoffCheckParams const &x) {
            jsonParser json;
            to_json(x, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent CutoffCheckParams as a Python dict.")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) {
            jsonParser json{data};
            InputParser<monte::CutoffCheckParams> parser(json);
            std::runtime_error error_if_invalid{
                "Error in libcasm.monte.CutoffCheckParams.from_dict"};
            report_and_throw_if_invalid(parser, CASM::log(), error_if_invalid);
            return std::move(*parser.value);
          },
          "Construct CutoffCheckParams from a Python dict.");

  m.def("all_minimums_met", &monte::all_minimums_met,
        R"pbdoc(
      Check if all cutoff check minimums have been met

      Parameters
      ----------
      cutoff_params : :class:`~libcasm.monte.CutoffCheckParams`
          Cutoff check parameters
      count : Optional[int]
          Number of steps or passes
      time : Optional[float]
          Simulated time
      n_samples : int
          Number of samples taken
      clocktime : float
          Elapsed clock time.

      Returns
      -------
      result : bool
          If all cutoff check minimums have been met, return True, else False.
      )pbdoc",
        py::arg("cutoff_params"), py::arg("count"), py::arg("time"),
        py::arg("n_samples"), py::arg("clocktime"));

  m.def("any_maximum_met", &monte::any_maximum_met,
        R"pbdoc(
      Check if any cutoff check maximum has been met

      Parameters
      ----------
      cutoff_params : :class:`~libcasm.monte.CutoffCheckParams`
          Cutoff check parameters
      count : Optional[int]
          Number of steps or passes
      time : Optional[float]
          Simulated time
      n_samples : int
          Number of samples taken
      clocktime : float
          Elapsed clock time.

      Returns
      -------
      result : bool
          If any cutoff check maximum has been met, return True, else False.
      )pbdoc",
        py::arg("cutoff_params"), py::arg("count"), py::arg("time"),
        py::arg("n_samples"), py::arg("clocktime"));

  py::class_<monte::CompletionCheckParams<monte::BasicStatistics>>(
      m, "CompletionCheckParams",
      R"pbdoc(
      Parameters that determine if a simulation is complete
      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          Default constructor only.
          )pbdoc")
      .def_readwrite(
          "cutoff_params",
          &monte::CompletionCheckParams<monte::BasicStatistics>::cutoff_params,
          R"pbdoc(
          :class:`~libcasm.monte.CutoffCheckParams`: Cutoff check parameters
          )pbdoc")
      .def_readwrite("equilibration_check_f",
                     &monte::CompletionCheckParams<
                         monte::BasicStatistics>::equilibration_check_f,
                     R"pbdoc(
                     function: Function that performs equilibration checking.

                     A function, such as :func:`~libcasm.monte.default_equilibration_check`, with signature f(array_like observations, array_like sample_weight,
                     :class:`~libcasm.monte.RequestedPrecision` requested_precision) -> :class:`~libcasm.monte.IndividualEquilibrationResult`.
                     )pbdoc")
      .def_readwrite("calc_statistics_f",
                     &monte::CompletionCheckParams<
                         monte::BasicStatistics>::calc_statistics_f,
                     R"pbdoc(
                     function: Function to calculate statistics.

                     A function, such as an instance of  :class:`~libcasm.monte.BasicStatisticsCalculator`, with signature f(array_like observations, array_like sample_weight) -> :class:`~libcasm.monte.BasicStatistics`.
                     )pbdoc")
      .def_readwrite("requested_precision",
                     &monte::CompletionCheckParams<
                         monte::BasicStatistics>::requested_precision,
                     R"pbdoc(
                     :class:`~libcasm.monte.RequestedPrecisionMap`: Requested precision for convergence of sampler components.

                     A Dict[:class:`~libcasm.monte.SamplerComponent`, :class:`~libcasm.monte.RequestedPrecision`]-like object that specifies convergence criteria.
                     )pbdoc")
      .def_readwrite(
          "log_spacing",
          &monte::CompletionCheckParams<monte::BasicStatistics>::log_spacing,
          R"pbdoc(
          bool: If True, use logirithmic spacing for completiong checking; else use  linear spacing.

          The default value is False, for linear spacing between completion checks. For linear spacing, the n-th check will be taken when:

          .. code-block:: Python

              sample = round( check_begin + (check_period / checks_per_period) * n )

          For "log" spacing, the n-th check will be taken when:

          .. code-block:: Python

              sample = round( check_begin + check_period ^ ( (n + check_shift) /
                              checks_per_period ) )

          )pbdoc")
      .def_readwrite(
          "check_begin",
          &monte::CompletionCheckParams<monte::BasicStatistics>::check_begin,
          R"pbdoc(
                     float: Completion check beginning. Default =`0.0`.
                     )pbdoc")
      .def_readwrite(
          "check_period",
          &monte::CompletionCheckParams<monte::BasicStatistics>::check_period,
          R"pbdoc(
                     float: Completion check period. Default =`10.0`.
                     )pbdoc")
      .def_readwrite("checks_per_period",
                     &monte::CompletionCheckParams<
                         monte::BasicStatistics>::checks_per_period,
                     R"pbdoc(
                     float: Completion checks per period. Default =`1.0`.
                     )pbdoc")
      .def_readwrite(
          "check_shift",
          &monte::CompletionCheckParams<monte::BasicStatistics>::check_shift,
          R"pbdoc(
                     float: Completion check shift. Default =`1.0`.
                     )pbdoc")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             StateSamplingFunctionMap const &sampling_functions) {
            jsonParser json{data};
            InputParser<monte::CompletionCheckParams<monte::BasicStatistics>>
                parser(json, sampling_functions);
            std::runtime_error error_if_invalid{
                "Error in libcasm.monte.CompletionCheckParams.from_dict"};
            report_and_throw_if_invalid(parser, CASM::log(), error_if_invalid);
            return std::move(*parser.value);
          },
          "Construct a CompletionCheckParams from a Python dict.",
          py::arg("data"), py::arg("sampling_functions"));

  py::class_<monte::CompletionCheckResults<monte::BasicStatistics>>(
      m, "CompletionCheckResults",
      R"pbdoc(
      Results of completion checks
      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          Default constructor only.
          )pbdoc")
      .def_readwrite(
          "params",
          &monte::CompletionCheckResults<monte::BasicStatistics>::params,
          R"pbdoc(
                     :class:`~libcasm.monte.CompletionCheckParams`: Completion check parameters
                     )pbdoc")
      .def_readwrite(
          "count",
          &monte::CompletionCheckResults<monte::BasicStatistics>::count,
          R"pbdoc(
                     Optional[int]: Number of steps or passes
                     )pbdoc")
      .def_readwrite(
          "time", &monte::CompletionCheckResults<monte::BasicStatistics>::time,
          R"pbdoc(
                     Optional[int]: Simulated time
                     )pbdoc")
      .def_readwrite(
          "clocktime",
          &monte::CompletionCheckResults<monte::BasicStatistics>::clocktime,
          R"pbdoc(
           float: Elapsed clock time
           )pbdoc")
      .def_readwrite(
          "n_samples",
          &monte::CompletionCheckResults<monte::BasicStatistics>::n_samples,
          R"pbdoc(
          int: Number of samples taken
          )pbdoc")
      .def_readwrite("has_all_minimums_met",
                     &monte::CompletionCheckResults<
                         monte::BasicStatistics>::has_all_minimums_met,
                     R"pbdoc(
           bool: True if all cutoff check minimums have been met
           )pbdoc")
      .def_readwrite("has_any_maximum_met",
                     &monte::CompletionCheckResults<
                         monte::BasicStatistics>::has_any_maximum_met,
                     R"pbdoc(
           bool: True if any cutoff check maximums have been met
           )pbdoc")
      .def_readwrite(
          "n_samples_at_convergence_check",
          &monte::CompletionCheckResults<
              monte::BasicStatistics>::n_samples_at_convergence_check,
          R"pbdoc(
           Optional[int]: Number of samples when the converence check was performed
           )pbdoc")
      .def_readwrite("equilibration_check_results",
                     &monte::CompletionCheckResults<
                         monte::BasicStatistics>::equilibration_check_results,
                     R"pbdoc(
           :class:`~libcasm.monte.EquilibrationCheckResults`: Results of equilibration check
           )pbdoc")
      .def_readwrite("convergence_check_results",
                     &monte::CompletionCheckResults<
                         monte::BasicStatistics>::convergence_check_results,
                     R"pbdoc(
           :class:`~libcasm.monte.ConvergenceCheckResults`: Results of convergence check
           )pbdoc")
      .def_readwrite(
          "is_complete",
          &monte::CompletionCheckResults<monte::BasicStatistics>::is_complete,
          R"pbdoc(
           bool: Outcome of the completion check
           )pbdoc")
      .def(
          "partial_reset",
          &monte::CompletionCheckResults<monte::BasicStatistics>::partial_reset,
          R"pbdoc(
          Reset for step by step updates

          Reset most values, but not:

          - params
          - n_samples_at_convergence_check
          - equilibration_check_results
          - convergence_check_results

          Parameters
          ----------
          count : Optional[int] = None
              Number of steps or passes to reset to
          time : Optional[float] = None
              Simulated time to reset to
          clocktime : float = 0.0
              Elapsed clocktime to reset to
          n_samples : int = 0
              Number of samples to reset to
          )pbdoc",
          py::arg("count") = std::nullopt, py::arg("time") = std::nullopt,
          py::arg("clocktime") = 0.0, py::arg("n_samples") = 0)
      .def("full_reset",
           &monte::CompletionCheckResults<monte::BasicStatistics>::full_reset,
           R"pbdoc(
          Reset for next run

          Reset all values except:

          - params

          Parameters
          ----------
          count : Optional[int] = None
              Number of steps or passes to reset to
          time : Optional[float] = None
              Simulated time to reset to
          n_samples : int = 0
              Number of samples to reset to
          )pbdoc",
           py::arg("count") = std::nullopt, py::arg("time") = std::nullopt,
           py::arg("n_samples") = 0)
      .def(
          "to_dict",
          [](monte::CompletionCheckResults<monte::BasicStatistics> const &x) {
            jsonParser json;
            to_json(x, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the CompletionCheckResults as a Python dict.");

  py::class_<monte::CompletionCheck<monte::BasicStatistics>>(m,
                                                             "CompletionCheck",
                                                             R"pbdoc(
      Implements completion checks
      )pbdoc")
      .def(py::init<monte::CompletionCheckParams<monte::BasicStatistics>>(),
           R"pbdoc(
          Constructor

          Parameters
          ----------
          params : :class:`~libcasm.monte.CompletionCheckParams`
              Data struture holding completion check parameters.
          )pbdoc")
      .def("reset", &monte::CompletionCheck<monte::BasicStatistics>::reset,
           R"pbdoc(
          Reset CompletionCheck for next run
          )pbdoc")
      .def("results", &monte::CompletionCheck<monte::BasicStatistics>::results,
           R"pbdoc(
          Get detailed results of the last check
          )pbdoc")
      .def(
          "count_check",
          [](monte::CompletionCheck<monte::BasicStatistics> &x,
             std::map<std::string, std::shared_ptr<monte::Sampler>> const
                 &samplers,
             monte::Sampler const &sample_weight, monte::CountType count,
             monte::MethodLog &method_log) {
            return x.is_complete(samplers, sample_weight, count,
                                 method_log.log);
          },
          R"pbdoc(
          Perform count based completion check

          Parameters
          ----------
          samplers: :class:`~libcasm.monte.SamplerMap`
              The samplers containing the sampled data.
          sample_weight : :class:`~libcasm.monte.Sampler`
              Sample weights associated with observations. May have 0 samples, in which case the obsservations are treated as being equally weighted, otherwise must match the number of samples made by each sampler in `samplers`.
          count : int
              Number of steps or passes
          method_log : :class:`~libcasm.monte.MethodLog`
              The method log specifies where to write status updates and internally tracks the elapsed clock time.

          Returns
          -------
          is_complete : bool
              True if complete, False otherwise
          )pbdoc",
          py::arg("samplers"), py::arg("sample_weight"), py::arg("count"),
          py::arg("method_log"))
      .def(
          "__call__",
          [](monte::CompletionCheck<monte::BasicStatistics> &x,
             std::map<std::string, std::shared_ptr<monte::Sampler>> const
                 &samplers,
             monte::Sampler const &sample_weight,
             std::optional<monte::CountType> count,
             std::optional<monte::TimeType> time,
             monte::MethodLog &method_log) {
            if (count.has_value()) {
              if (time.has_value()) {
                return x.is_complete(samplers, sample_weight, *count, *time,
                                     method_log.log);
              } else {
                return x.is_complete(samplers, sample_weight, *count,
                                     method_log.log);
              }
            } else {
              if (time.has_value()) {
                return x.is_complete(samplers, sample_weight, *time,
                                     method_log.log);
              } else {
                return x.is_complete(samplers, sample_weight, method_log.log);
              }
            }
          },
          R"pbdoc(
          Perform completion check, with optional count- or time-based cutoff checks

          Parameters
          ----------
          samplers: :class:`~libcasm.monte.SamplerMap`
              The samplers containing the sampled data.
          sample_weight : :class:`~libcasm.monte.Sampler`
              Sample weights associated with observations. May have 0 samples, in which case the obsservations are treated as being equally weighted, otherwise must match the number of samples made by each sampler in `samplers`.
          count : Optional[int]
              Number of steps or passes
          time : Optional[float]
              Simulated time
          method_log : :class:`~libcasm.monte.MethodLog`
              The method log specifies where to write status updates and internally tracks the elapsed clock time.

          Returns
          -------
          is_complete : bool
              True if complete, False otherwise
          )pbdoc",
          py::arg("samplers"), py::arg("sample_weight"), py::arg("count"),
          py::arg("time"), py::arg("method_log"))
      .def(
          "count_and_time_check",
          [](monte::CompletionCheck<monte::BasicStatistics> &x,
             std::map<std::string, std::shared_ptr<monte::Sampler>> const
                 &samplers,
             monte::Sampler const &sample_weight, monte::CountType count,
             monte::TimeType time, monte::MethodLog &method_log) {
            return x.is_complete(samplers, sample_weight, count, time,
                                 method_log.log);
          },
          R"pbdoc(
          Perform completion check, with count- and time-based cutoff checks

          Parameters
          ----------
          samplers: :class:`~libcasm.monte.SamplerMap`
              The samplers containing the sampled data.
          sample_weight : :class:`~libcasm.monte.Sampler`
              Sample weights associated with observations. May have 0 samples, in which case the obsservations are treated as being equally weighted, otherwise must match the number of samples made by each sampler in `samplers`.
          count : int
              Number of steps or passes
          time : float
              Simulated time
          method_log : :class:`~libcasm.monte.MethodLog`
              The method log specifies where to write status updates and internally tracks the elapsed clock time.

          Returns
          -------
          is_complete : bool
              True if complete, False otherwise
          )pbdoc",
          py::arg("samplers"), py::arg("sample_weight"), py::arg("count"),
          py::arg("time"), py::arg("method_log"))
      .def(
          "time_check",
          [](monte::CompletionCheck<monte::BasicStatistics> &x,
             std::map<std::string, std::shared_ptr<monte::Sampler>> const
                 &samplers,
             monte::Sampler const &sample_weight, monte::CountType time,
             monte::MethodLog &method_log) {
            return x.is_complete(samplers, sample_weight, time, method_log.log);
          },
          R"pbdoc(
          Perform completion check, with time-based cutoff checks

          Parameters
          ----------
          samplers: :class:`~libcasm.monte.SamplerMap`
              The samplers containing the sampled data.
          sample_weight : :class:`~libcasm.monte.Sampler`
              Sample weights associated with observations. May have 0 samples, in which case the observations are treated as being equally weighted, otherwise must match the number of samples made by each sampler in `samplers`.
          time : float
              Simulated time
          method_log : :class:`~libcasm.monte.MethodLog`
              The method log specifies where to write status updates and internally tracks the elapsed clock time.

          Returns
          -------
          is_complete : bool
              True if complete, False otherwise
          )pbdoc",
          py::arg("samplers"), py::arg("sample_weight"), py::arg("time"),
          py::arg("method_log"))
      .def(
          "check",
          [](monte::CompletionCheck<monte::BasicStatistics> &x,
             std::map<std::string, std::shared_ptr<monte::Sampler>> const
                 &samplers,
             monte::Sampler const &sample_weight,
             monte::MethodLog &method_log) {
            return x.is_complete(samplers, sample_weight, method_log.log);
          },
          R"pbdoc(
          Perform completion check, without count- or time-based cutoff checks

          Parameters
          ----------
          samplers: :class:`~libcasm.monte.SamplerMap`
              The samplers containing the sampled data.
          sample_weight : :class:`~libcasm.monte.Sampler`
              Sample weights associated with observations. May have 0 samples, in which case the obsservations are treated as being equally weighted, otherwise must match the number of samples made by each sampler in `samplers`.
          method_log : :class:`~libcasm.monte.MethodLog`
              The method log specifies where to write status updates and internally tracks the elapsed clock time.

          Returns
          -------
          is_complete : bool
              True if complete, False otherwise
          )pbdoc",
          py::arg("samplers"), py::arg("sample_weight"), py::arg("method_log"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
