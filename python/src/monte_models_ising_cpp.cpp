#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// nlohmann::json binding
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "pybind11_json/pybind11_json.hpp"

// CASM
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/monte/calculators/basic_semigrand_canonical.hh"
#include "casm/monte/io/json/ValueMap_json_io.hh"
#include "casm/monte/models/ising_eigen.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;
using namespace CASM::monte;
using namespace CASM::monte::models::ising_eigen;

// used for libcasm.monte:
typedef std::mt19937_64 engine_type;
typedef RandomNumberGenerator<engine_type> generator_type;
typedef BasicStatistics statistics_type;
typedef IsingConfiguration configuration_type;
typedef IsingState state_type;

typedef IsingSemiGrandCanonicalEventGenerator<engine_type>
    sgc_event_generator_type;
typedef IsingSemiGrandCanonicalSystem sgc_system_type;

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<CASM::Index>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, bool>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, double>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, Eigen::VectorXd>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, Eigen::MatrixXd>);
PYBIND11_MAKE_OPAQUE(CASM::monte::SamplerMap);
PYBIND11_MAKE_OPAQUE(CASM::monte::StateSamplingFunctionMap);
PYBIND11_MAKE_OPAQUE(CASM::monte::jsonStateSamplingFunctionMap);
PYBIND11_MAKE_OPAQUE(CASM::monte::RequestedPrecisionMap);
PYBIND11_MAKE_OPAQUE(
    CASM::monte::ConvergenceResultMap<CASM::monte::BasicStatistics>);
PYBIND11_MAKE_OPAQUE(CASM::monte::EquilibrationResultMap);

PYBIND11_MODULE(_monte_models_ising_cpp, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Basic Ising model

        libcasm.monte.models._ising_cpp
        --------------------------------------------------------------

        An Ising model, with implementation in C++.
    )pbdoc";
  py::module::import("libcasm.monte");
  py::module::import("libcasm.monte.events");

  py::class_<configuration_type>(m, "IsingConfiguration", R"pbdoc(
      Ising model configuration, using a Eigen::VectorXi

      )pbdoc")
      .def(py::init<Eigen::VectorXi, int>(),
           R"pbdoc(
          Constructor

          Parameters
          ----------
          shape: array_like, dtype=int
              The shape (i.e. [row, cols]) of the configuration supercell. Currently
              only 2d configurations are supported.
          fill_value: int = 1
              The default occupation value. Should be +1 or -1.
          )pbdoc",
           py::arg("shape"), py::arg("fill_value") = 1)
      .def_readwrite("shape", &configuration_type::shape,
                     R"pbdoc(
          np.ndarray(shape=[2], dtype=int): \
          The shape (i.e. [row, cols]) of the configuration supercell.
          )pbdoc")
      .def_readonly("n_sites", &configuration_type::n_sites,
                    R"pbdoc(
          int: Total number of sites in the supercell
          )pbdoc")
      .def_readonly("n_variable_sites", &configuration_type::n_variable_sites,
                    R"pbdoc(
          int: Number of variable sites in the supercell
          )pbdoc")
      .def_readonly("n_unitcells", &configuration_type::n_unitcells,
                    R"pbdoc(
          int: \
          Number of unitcells in the supercell, which is equal to n_sites.
          )pbdoc")
      .def("occupation", &configuration_type::occupation,
           py::return_value_policy::reference_internal,
           R"pbdoc(
          np.ndarray(dtype=int): \
          Get the current site occupation, as a linear array of values, as const reference.
          )pbdoc")
      .def("set_occupation", &configuration_type::set_occupation,
           R"pbdoc(
          Set the current site occupation, as a linear array of values.
          )pbdoc",
           py::arg("occupation"))
      .def("occ", &configuration_type::occ,
           R"pbdoc(
          int: Get the current occupation of one site.
          )pbdoc",
           py::arg("linear_site_index"))
      .def("set_occ", &configuration_type::set_occ,
           R"pbdoc(
          Set the current occupation of one site.
          )pbdoc",
           py::arg("linear_site_index"), py::arg("new_occ"))
      .def("within", &configuration_type::within,
           R"pbdoc(
          int: Get the periodic equivalent within the array, of a given multi_index\
          value, `index`, along a given dimension, `dim`.
          )pbdoc",
           py::arg("index"), py::arg("dim"))
      .def("from_linear_site_index",
           &configuration_type::from_linear_site_index,
           R"pbdoc(
          np.ndarray(dtype=int): Return `multi_index` (i.e. `[row, col]` indices) \
          corresponding to unrolled `linear_site_index`.
          )pbdoc",
           py::arg("linear_site_index"))
      .def(
          "to_linear_site_index",
          [](configuration_type const &self,
             Eigen::Ref<Eigen::VectorXi const> multi_index) -> Index {
            return self.to_linear_site_index(multi_index);
          },
          R"pbdoc(
          int: Return unrolled `linear_site_index` from a given `multi_index`, \
          i.e. `[row, col]`.
          )pbdoc",
          py::arg("multi_index"))
      .def(
          "to_dict",
          [](configuration_type const &self) {
            jsonParser json;
            to_json(self, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the IsingConfiguration as a Python dict. "
          "Items from all attributes are combined into a single dict")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) {
            configuration_type config;
            jsonParser json{data};
            from_json(config, json);
            return config;
          },
          "Construct IsingConfiguration from a Python dict.", py::arg("data"))
      .def("__copy__",
           [](configuration_type const &self) {
             return configuration_type(self);
           })
      .def("__deepcopy__", [](configuration_type const &self, py::dict) {
        return configuration_type(self);
      });

  py::class_<state_type>(m, "IsingState", R"pbdoc(
      Ising model state, including configuration and conditions

      )pbdoc")
      .def(py::init<configuration_type, monte::ValueMap, monte::ValueMap>(),
           R"pbdoc(
          Constructor

          Parameters
          ----------
          configuration: IsingConfiguration
              Monte Carlo configuration.
          conditions: :class:`~libcasm.monte.ValueMap`
              Thermodynamic conditions.
          properties: :class:`~libcasm.monte.ValueMap`
              Properties of the Monte Carlo configuration, if applicable.
          )pbdoc",
           py::arg("configuration"), py::arg("conditions"),
           py::arg("properties") = monte::ValueMap())
      .def_readwrite("configuration", &state_type::configuration,
                     R"pbdoc(
          IsingConfiguration: Monte Carlo configuration
          )pbdoc")
      .def_readwrite("conditions", &state_type::conditions,
                     R"pbdoc(
          :class:`~libcasm.monte.ValueMap`: Thermodynamic conditions.
          )pbdoc")
      .def_readwrite("properties", &state_type::properties,
                     R"pbdoc(
          :class:`~libcasm.monte.ValueMap`: \
          Properties of the Monte Carlo configuration, if applicable.
          )pbdoc")
      .def(
          "to_dict",
          [](state_type const &self) {
            jsonParser json;
            to_json(self.configuration, json["configuration"]);
            to_json(self.conditions, json["conditions"]);
            to_json(self.properties, json["properties"]);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the IsingState as a Python dict. "
          "Items from all attributes are combined into a single dict")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) {
            jsonParser json{data};

            configuration_type config;
            from_json(config, json["configuration"]);

            monte::ValueMap conditions;
            from_json(conditions, json["conditions"]);

            monte::ValueMap properties;
            from_json(properties, json["properties"]);

            state_type state(config, conditions);
            state.properties = properties;

            return state;
          },
          "Construct IsingState from a Python dict.", py::arg("data"))
      .def("__copy__", [](state_type const &self) { return state_type(self); })
      .def("__deepcopy__",
           [](state_type const &self, py::dict) { return state_type(self); });

  py::class_<sgc_event_generator_type>(m,
                                       "IsingSemiGrandCanonicalEventGenerator",
                                       R"pbdoc(
      Propose and apply semi-grand canonical Ising model events

      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          Constructor
           )pbdoc")
      .def("set_state", &sgc_event_generator_type::set_state,
           R"pbdoc(
          Set the current Monte Carlo state used to propose and apply events.
          )pbdoc",
           py::arg("state"))
      .def("propose", &sgc_event_generator_type::propose,
           py::return_value_policy::reference_internal,
           R"pbdoc(
          Propose a semi-grand canonical event (1 site to flip signs)

          Parameters
          ----------
          random_number_generator: class:`~libcasm.monte.RandomNumberGenerator`
              The random number generator used to propose events.

          Returns
          -------
          occ_event: class:`~libcasm.monte.events.OccEvent`
              The proposed Monte Carlo event.
          )pbdoc",
           py::arg("random_number_generator"))
      .def("apply", &sgc_event_generator_type::apply,
           R"pbdoc(
          Apply an event

          Parameters
          ----------
          occ_event: class:`~libcasm.monte.events.OccEvent`
              The Monte Carlo event to apply to the current state.

          Returns
          -------
          occ_event: class:`~libcasm.monte.events.OccEvent`
              The proposed Monte Carlo event.
          )pbdoc",
           py::arg("occ_event"))
      .def("__copy__",
           [](sgc_event_generator_type const &self) {
             return sgc_event_generator_type(self);
           })
      .def("__deepcopy__", [](sgc_event_generator_type const &self, py::dict) {
        return sgc_event_generator_type(self);
      });

  py::class_<IsingFormationEnergy>(m, "IsingFormationEnergy",
                                   R"pbdoc(
      Calculates formation energy for the Ising model

      Currently implements Ising model on square lattice. Could add other lattice types or
      anisotropic bond energies.

      )pbdoc")
      .def(py::init<double, int, bool, state_type const *>(), R"pbdoc(
          Constructor

          Parameters
          ----------
          J: float = 1.0
              Ising model interaction energy.
          lattice_type: int = 1
              Lattice type. One of:

              - 1: 2-dimensional square lattice, using IsingConfiguration

          use_nlist: bool = True
              Optionally create a list of neighbors for each site ahead of time.
          state: Optional[IsingState] = None
              The Monte Carlo state to calculate the formation energy

          )pbdoc",
           py::arg("J") = 1.0, py::arg("lattice_type") = 1,
           py::arg("use_nlist") = true, py::arg("state") = nullptr)
      .def("set_state", &IsingFormationEnergy::set_state,
           R"pbdoc(
          Set the state the formation energy is calculated for.
          )pbdoc")
      .def("per_supercell", &IsingFormationEnergy::per_supercell,
           R"pbdoc(
          Calculates Ising model formation energy (per supercell)
          )pbdoc")
      .def("per_unitcell", &IsingFormationEnergy::per_unitcell,
           R"pbdoc(
          Calculates Ising model formation energy (per unitcell)
          )pbdoc")
      .def("occ_delta_per_supercell",
           &IsingFormationEnergy::occ_delta_per_supercell,
           R"pbdoc(
          Calculate the change in Ising model energy due to changing 1 or more sites

          Parameters
          ----------
          linear_site_index: LongVector
            Linear site indices for sites that are flipped
          new_occ: IntVector
              New value on each site.

          Returns
          -------
          dE: float
              The change in the per_supercell formation energy (energy per supercell).
          )pbdoc",
           py::arg("linear_site_index"), py::arg("new_occ"))
      .def("__copy__",
           [](IsingFormationEnergy const &self) {
             return IsingFormationEnergy(self);
           })
      .def("__deepcopy__", [](IsingFormationEnergy const &self, py::dict) {
        return IsingFormationEnergy(self);
      });

  py::class_<IsingParamComposition>(m, "IsingParamComposition",
                                    R"pbdoc(
      Calculate parametric composition of IsingConfiguration

      )pbdoc")
      .def(py::init<state_type const *>(), R"pbdoc(
          Constructor

          Parameters
          ----------
          state: Optional[IsingState] = None
              The Monte Carlo state to calculate the formation energy

          )pbdoc",
           py::arg("state") = nullptr)
      .def("set_state", &IsingParamComposition::set_state,
           R"pbdoc(
          Set the state the parametric composition is calculated for.
          )pbdoc")
      .def("n_independent_compositions",
           &IsingParamComposition::n_independent_compositions,
           R"pbdoc(
           int: Return the number of independent compositions (size of composition vector)
           )pbdoc")
      .def("per_supercell", &IsingParamComposition::per_supercell,
           R"pbdoc(
          Calculates parametric composition (per supercell, n_unitcells * x)
          )pbdoc")
      .def("per_unitcell", &IsingParamComposition::per_unitcell,
           R"pbdoc(
          Calculates parametric composition (per unitcell, x)
          )pbdoc")
      .def("occ_delta_per_supercell",
           &IsingParamComposition::occ_delta_per_supercell,
           R"pbdoc(
          Calculate the change in parametric composition due to changing 1 or more sites

          Parameters
          ----------
          linear_site_index: LongVector
            Linear site indices for sites that are flipped
          new_occ: IntVector
              New value on each site.

          Returns
          -------
          Ndx: np.ndarray[dtype=float]
              The change in parametric composition (per_supercell, , n_unitcells * dx).
          )pbdoc",
           py::arg("linear_site_index"), py::arg("new_occ"))
      .def("__copy__",
           [](IsingParamComposition const &self) {
             return IsingParamComposition(self);
           })
      .def("__deepcopy__", [](IsingParamComposition const &self, py::dict) {
        return IsingParamComposition(self);
      });

  py::class_<IsingSemiGrandCanonicalSystem,
             std::shared_ptr<IsingSemiGrandCanonicalSystem>>(
      m, "IsingSemiGrandCanonicalSystem",
      R"pbdoc(
      Holds methods and data for calculating Ising model system properties
      for semi-grand canonical Monte Carlo calculations.

      )pbdoc")
      .def(py::init<IsingFormationEnergy, IsingParamComposition>(),
           R"pbdoc(
          Constructor

          Parameters
          ----------
          formation_energy_calculator: IsingFormationEnergy
              Formation energy calculator
          param_composition_calculator: IsingParamComposition
              Parametric composition calculator

          )pbdoc",
           py::arg("formation_energy_calculator"),
           py::arg("param_composition_calculator"))
      .def_readwrite("formation_energy_calculator",
                     &sgc_system_type::formation_energy_calculator,
                     R"pbdoc(
          IsingFormationEnergy: Get the parameterized formation energy calculator.
          )pbdoc")
      .def_readwrite("param_composition_calculator",
                     &sgc_system_type::param_composition_calculator,
                     R"pbdoc(
          IsingFormationEnergy: Get the parameterized parametric composition calculator.
          )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
