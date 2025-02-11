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
#include "casm/monte/ising_cpp/basic_semigrand_canonical.hh"
#include "casm/monte/ising_cpp/model.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

// used for libcasm.monte:
typedef monte::default_engine_type engine_type;
typedef monte::RandomNumberGenerator<engine_type> generator_type;
typedef monte::BasicStatistics statistics_type;

// used for ising_cpp:
using namespace CASM::monte::ising_cpp;
typedef IsingState state_type;
typedef IsingFormationEnergy formation_energy_f_type;
typedef IsingParamComposition param_composition_f_type;

// semi-grand canonical
namespace sgc {

using namespace CASM::monte::ising_cpp::basic_semigrand_canonical;

typedef SemiGrandCanonicalConditions conditions_type;
typedef IsingSystem system_type;
typedef SemiGrandCanonicalPotential<system_type> potential_type;
typedef SemiGrandCanonicalData data_type;
typedef SemiGrandCanonicalEventGenerator<engine_type> event_generator_type;
typedef SemiGrandCanonicalCalculator<system_type, event_generator_type>
    calculator_type;

}  // namespace sgc

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
// #include "opaque_types.cc"
PYBIND11_MAKE_OPAQUE(CASM::monte::SamplerMap);
PYBIND11_MAKE_OPAQUE(CASM::monte::StateSamplingFunctionMap);
PYBIND11_MAKE_OPAQUE(CASM::monte::jsonStateSamplingFunctionMap);
PYBIND11_MAKE_OPAQUE(CASM::monte::RequestedPrecisionMap);
PYBIND11_MAKE_OPAQUE(
    CASM::monte::ConvergenceResultMap<CASM::monte::BasicStatistics>);
PYBIND11_MAKE_OPAQUE(CASM::monte::EquilibrationResultMap);

PYBIND11_MODULE(_monte_ising_cpp_semigrand_canonical, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Ising model semi-grand canonical Monte Carlo

        libcasm.monte.ising_cpp._monte_calculators_sgc_ising_cpp
        ----------------------------------------------------------

        An Ising model semi-grand canonical Monte Carlo calculator, with implementation
        in C++
    )pbdoc";
  py::module::import("libcasm.monte");
  py::module::import("libcasm.monte.events");
  py::module::import("libcasm.monte.ising_cpp");
  py::module::import("libcasm.monte.sampling");

  // #include "local_bindings.cc"

  py::class_<sgc::conditions_type, std::shared_ptr<sgc::conditions_type>>(
      m, "SemiGrandCanonicalConditions",
      R"pbdoc(
      Semi-grand canonical ensemble thermodynamic conditions.

      )pbdoc")
      .def(py::init<double, Eigen::Ref<Eigen::VectorXd const>>(),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          temperature: float
              The temperature, :math:`T`.
          exchange_potential: np.ndarray
              The semi-grand canonical exchange potential, conjugate to the
              parametric composition that will be calculated by the `param_composition_calculator`
              of the system under consideration.
          )pbdoc",
           py::arg("temperature"), py::arg("exchange_potential"))
      .def_readwrite("temperature", &sgc::conditions_type::temperature,
                     R"pbdoc(
          float: The temperature, :math:`T`.
          )pbdoc")
      .def_readwrite("exchange_potential",
                     &sgc::conditions_type::exchange_potential,
                     R"pbdoc(
          np.ndarray: The semi-grand canonical exchange potential.
          )pbdoc")
      .def(
          "to_values",
          [](sgc::conditions_type const &self) { return self.to_values(); },
          "Represent the SemiGrandCanonicalConditions as a Python dict. "
          "Items from all attributes are combined into a single dict")
      .def_static(
          "from_values",
          [](monte::ValueMap const &values) {
            return sgc::conditions_type::from_values(values);
          },
          "Construct SemiGrandCanonicalConditions from a ValueMap.",
          py::arg("values"))
      .def(
          "to_dict",
          [](sgc::conditions_type const &self) {
            jsonParser json;
            to_json(self, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the SemiGrandCanonicalConditions as a Python dict. "
          "Items from all attributes are combined into a single dict")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) {
            sgc::conditions_type conditions;
            jsonParser json{data};
            from_json(conditions, json);
            return conditions;
          },
          "Construct SemiGrandCanonicalConditions from a Python dict.",
          py::arg("data"))
      .def("__copy__",
           [](sgc::conditions_type const &self) {
             return sgc::conditions_type(self);
           })
      .def("__deepcopy__", [](sgc::conditions_type const &self, py::dict) {
        return sgc::conditions_type(self);
      });

  py::class_<sgc::potential_type>(m, "SemiGrandCanonicalPotential", R"pbdoc(
      Calculates the semi-grand canonical energy and changes in energy

      )pbdoc")
      .def(py::init<std::shared_ptr<sgc::system_type>>(), R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          system: libcasm.monte.ising_cpp.IsingSystem
              Holds parameterized formation energy and parametric composition
              calculators, without specifying at a particular state.
          )pbdoc",
           py::arg("system"))
      .def("set_state", &sgc::potential_type::set_state,
           R"pbdoc(
          Set the state the potential is calculated for.

          Parameters
          ----------
          state: libcasm.monte.ising_cpp.IsingState
              The state the potential is calculated for.
          conditions: SemiGrandCanonicalConditions
              The conditions the potential is calculated for.
          )pbdoc")
      .def("per_supercell", &sgc::potential_type::per_supercell,
           R"pbdoc(
          Calculates semi-grand canonical energy (per supercell)
          )pbdoc")
      .def("per_unitcell", &sgc::potential_type::per_unitcell,
           R"pbdoc(
          Calculates semi-grand canonical energy (per unitcell)
          )pbdoc")
      .def(
          "occ_delta_per_supercell",
          [](sgc::potential_type const &potential,
             std::vector<Index> const &linear_site_index,
             std::vector<int> const &new_occ) {
            return potential.occ_delta_per_supercell(linear_site_index,
                                                     new_occ);
          },
          R"pbdoc(
          Calculate the change in semi-grand canonical energy (per_supercell) due to \
          changing 1 or more sites

          Notes
          -----
          This differs from `occ_event_delta_per_supercell` only in the arguments.

          Parameters
          ----------
          linear_site_index: LongVector
            Linear site indices for sites that are flipped
          new_occ: IntVector
              New value on each site.

          Returns
          -------
          dE: float
              The change in the per_supercell semi-grand canonical energy.
          )pbdoc",
          py::arg("linear_site_index"), py::arg("new_occ"))
      .def(
          "occ_event_delta_per_supercell",
          [](sgc::potential_type const &potential,
             monte::OccEvent const &occ_event) {
            return potential.occ_delta_per_supercell(occ_event);
          },
          R"pbdoc(
          Calculate the change in semi-grand canonical energy (per_supercell) due to \
          an event

          Notes
          -----
          This differs from `occ_delta_per_supercell` only in the arguments.

          Parameters
          ----------
          occ_event: libcasm.monte.events.OccEvent
              Event proposed

          Returns
          -------
          dE: float
              The change in the per_supercell semi-grand canonical energy.
          )pbdoc",
          py::arg("occ_event"))
      .def("__copy__",
           [](sgc::potential_type const &self) {
             return sgc::potential_type(self);
           })
      .def("__deepcopy__", [](sgc::potential_type const &self, py::dict) {
        return sgc::potential_type(self);
      });

  py::class_<sgc::data_type, std::shared_ptr<sgc::data_type>>(
      m, "SemiGrandCanonicalData", R"pbdoc(
      Holds semi-grand canonical Metropolis Monte Carlo run data and results

      )pbdoc")
      .def(py::init<
               monte::StateSamplingFunctionMap const &,
               monte::jsonStateSamplingFunctionMap const &, monte::CountType,
               monte::CompletionCheckParams<monte::BasicStatistics> const &>(),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          sampling_functions: libcasm.monte.sampling.StateSamplingFunctionMap
              The sampling functions to use
          json_sampling_functions: libcasm.monte.sampling.jsonStateSamplingFunctionMap
              The JSON sampling functions to use
          n_steps_per_pass: int
              Number of steps per pass
          completion_check_param: libcasm.monte.sampling.CompletionCheckParams
              The completion check parameters.

          )pbdoc",
           py::arg("sampling_functions"), py::arg("json_sampling_functions"),
           py::arg("n_steps_per_pass"), py::arg("completion_check_params"))
      .def_readwrite("sampling_functions", &sgc::data_type::sampling_functions,
                     R"pbdoc(
          The sampling functions.
          )pbdoc")
      .def_readwrite("samplers", &sgc::data_type::samplers,
                     R"pbdoc(
          Holds sampled data.
          )pbdoc")
      .def_readwrite("json_sampling_functions",
                     &sgc::data_type::json_sampling_functions,
                     R"pbdoc(
          The JSON sampling functions.
          )pbdoc")
      .def_readwrite("json_samplers", &sgc::data_type::json_samplers,
                     R"pbdoc(
          Holds sampled JSON data.
          )pbdoc")
      .def_readwrite("sample_weight", &sgc::data_type::sample_weight,
                     R"pbdoc(
          Sample weights

          Sample weights may remain empty (unweighted). Included for compatibility
          with statistics calculators.
          )pbdoc")
      .def_readwrite("n_pass", &sgc::data_type::n_pass,
                     R"pbdoc(
          int: Number of passes. One pass is equal to one Monte Carlo step \
          per variable site in the configuration.
          )pbdoc")
      .def_readwrite("n_steps_per_pass", &sgc::data_type::n_steps_per_pass,
                     R"pbdoc(
          Number of steps per pass.
          )pbdoc")
      .def_readwrite("n_accept", &sgc::data_type::n_accept,
                     R"pbdoc(
          Number of accepted Monte Carlo steps
          )pbdoc")
      .def_readwrite("n_reject", &sgc::data_type::n_reject,
                     R"pbdoc(
          Number of rejected Monte Carlo steps
          )pbdoc")
      .def_readwrite("completion_check", &sgc::data_type::completion_check,
                     R"pbdoc(
          The Monte Carlo run completion checker
          )pbdoc")
      .def("acceptance_rate", &sgc::data_type::acceptance_rate,
           R"pbdoc(
          The fraction of Monte Carlo steps accepted.
          )pbdoc")
      .def("rejection_rate", &sgc::data_type::rejection_rate,
           R"pbdoc(
          The fraction of Monte Carlo steps rejected.
          )pbdoc")
      .def("reset", &sgc::data_type::reset,
           R"pbdoc(
          Reset attributes set during `run`.
          )pbdoc")
      .def(
          "to_dict",
          [](sgc::data_type const &self) {
            jsonParser json;
            to_json(self, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the SemiGrandCanonicalConditions as a Python dict. "
          "Items from all attributes are combined into a single dict")
      .def("__copy__",
           [](std::shared_ptr<sgc::data_type> const &self) {
             return std::make_shared<sgc::data_type>(*self);
           })
      .def("__deepcopy__",
           [](std::shared_ptr<sgc::data_type> const &self, py::dict) {
             return std::make_shared<sgc::data_type>(*self);
           });

  py::class_<sgc::event_generator_type>(m, "SemiGrandCanonicalEventGenerator",
                                        R"pbdoc(
      Propose and apply semi-grand canonical Ising model events

      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          .. rubric:: Constructor

          Default constructor only.
           )pbdoc")
      .def("set_state", &sgc::event_generator_type::set_state,
           R"pbdoc(
          Set the current Monte Carlo state used to propose and apply events.
          )pbdoc",
           py::arg("state"))
      .def("propose", &sgc::event_generator_type::propose,
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
      .def("apply", &sgc::event_generator_type::apply,
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
           [](sgc::event_generator_type const &self) {
             return sgc::event_generator_type(self);
           })
      .def("__deepcopy__", [](sgc::event_generator_type const &self, py::dict) {
        return sgc::event_generator_type(self);
      });

  py::class_<sgc::calculator_type, std::shared_ptr<sgc::calculator_type>>(
      m, "SemiGrandCanonicalCalculator", R"pbdoc(
      A semi-grand canonical Monte Carlo calculator

      )pbdoc")
      .def(py::init<std::shared_ptr<sgc::system_type>>(), R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          system: libcasm.monte.ising_cpp.IsingSystem
              Holds parameterized formation energy and parametric composition
              calculators, without specifying at a particular state.
          )pbdoc",
           py::arg("system"))
      .def_readonly("system", &sgc::calculator_type::system,
                    R"pbdoc(
          Holds parameterized formation energy and parametric composition
          calculators, without specifying at a particular state.
          )pbdoc")
      .def_readonly("state", &sgc::calculator_type::state,
                    py::return_value_policy::reference_internal,
                    R"pbdoc(
          A reference to the current Monte Carlo state
          )pbdoc")
      .def_readonly("conditions", &sgc::calculator_type::conditions,
                    R"pbdoc(
          The current thermodynamic conditions, set in `run` method.
          )pbdoc")
      .def_readonly("potential", &sgc::calculator_type::potential,
                    R"pbdoc(
          The semi-grand canonical energy calculator, set in `run` method to calculate using the current state.
          )pbdoc")
      .def_readonly("formation_energy_calculator",
                    &sgc::calculator_type::formation_energy_calculator,
                    py::return_value_policy::reference_internal,
                    R"pbdoc(
          A reference to the formation energy calculator
          )pbdoc")
      .def_readonly("param_composition_calculator",
                    &sgc::calculator_type::param_composition_calculator,
                    py::return_value_policy::reference_internal,
                    R"pbdoc(
          A reference to the parametric composition calculator
          )pbdoc")
      .def_readonly("data", &sgc::calculator_type::data,
                    R"pbdoc(
          Monte Carlo run data (samplers, completion_check, n_pass, etc.). Constructed
          by the `run` method.
          )pbdoc")
      .def(
          "default_sampling_functions",
          [](std::shared_ptr<sgc::calculator_type> mc_calculator) {
            monte::StateSamplingFunctionMap sampling_functions;
            std::vector<monte::StateSamplingFunction> fv{
                make_parametric_composition_f(mc_calculator),
                make_formation_energy_f(mc_calculator),
                make_potential_energy_f(mc_calculator)};
            for (auto const &f : fv) {
              sampling_functions.emplace(f.name, f);
            }
            return sampling_functions;
          },
          R"pbdoc(
          Get sampling functions for this Monte Carlo calculator.

          Includes:

          - "param_composition": Samples the current parametric composition,
            :math:`x`, using the system's
            :class:`~libcasm.monte.ising_cpp.IsingParamComposition`
            calculator.
          - "formation_energy": Sample the current formation energy, per
            unitcell, using the system's
            :class:`~libcasm.monte.ising_cpp.IsingFormationEnergy`
            calculator.
          - "potential_energy": Sample the current semi-grand canonical energy,
            using the this Monte Carlo calculator's
            :class:`SemiGrandCanonicalPotential` calculator.

          )pbdoc")
      .def(
          "default_json_sampling_functions",
          [](std::shared_ptr<sgc::calculator_type> mc_calculator) {
            monte::jsonStateSamplingFunctionMap json_sampling_functions;
            std::vector<monte::jsonStateSamplingFunction> fv{
                make_configuration_json_f(mc_calculator)};
            for (auto const &f : fv) {
              json_sampling_functions.emplace(f.name, f);
            }
            return json_sampling_functions;
          },
          R"pbdoc(
          Get default JSON sampling functions.

          Includes:

          - "configuration": Samples the current configuration

          )pbdoc")
      .def(
          "run",
          [](sgc::calculator_type &mc_calculator, state_type &state,
             monte::StateSamplingFunctionMap const &sampling_functions,
             monte::jsonStateSamplingFunctionMap const &json_sampling_functions,
             monte::CompletionCheckParams<monte::BasicStatistics> const
                 &completion_check_params,
             sgc::event_generator_type const &event_generator,
             int sample_period, std::optional<monte::MethodLog> method_log,
             std::shared_ptr<engine_type> random_engine,
             std::optional<std::function<void(sgc::calculator_type const &,
                                              monte::MethodLog &)>>
                 write_status_f) {
            if (!write_status_f.has_value() || !write_status_f.value()) {
              using namespace CASM::monte::ising_cpp::basic_semigrand_canonical;
              write_status_f = default_write_status<sgc::calculator_type>;
            }
            mc_calculator.run(state, sampling_functions,
                              json_sampling_functions, completion_check_params,
                              event_generator, sample_period, method_log,
                              random_engine, *write_status_f);
          },
          R"pbdoc(
          Run a semi-grand canonical calculation at a single thermodynamic state

          Notes
          -----
          On completion, results can be obtained from the :attr:`data` attribute.

          Parameters
          ----------
          state: libcasm.monte.ising_cpp.IsingState
              Initial Monte Carlo state, including configuration and conditions. Is
              modified by the method.
          sampling_functions: libcasm.monte.sampling.StateSamplingFunctionMap
              The sampling functions to use
          json_sampling_functions: libcasm.monte.sampling.jsonStateSamplingFunctionMap
              The JSON sampling functions to use
          completion_check_params: libcasm.monte.sampling.CompletionCheckParams
              Controls when the run finishes
          event_generator: libcasm.monte.sampling.jsonStateSamplingFunctionMap
              An event generator which proposes new events and applies accepted events.
          sample_period: int = 1
              Number of passes per sample. One pass is one Monte Carlo step per site
              with variable occupation.
          method_log: libcasm.monte.MethodLog
              Method log, for writing status updates. If None, default writes to
              a "status.json" file in the current working directory every 10 minutes.
          random_engine: libcasm.monte.RandomNumberEngine
              Random number engine. Default constructs a new engine.
          write_status_f: Optional[Callable] = None
              Function with signature

              .. code-block:: Python

                  def f(
                      mc_calculator: SemiGrandCanonicalCalculator,
                      method_log: MethodLog,
                  ) -> None:
                      ...

              accepting `self` as the first argument, that writes status updates,
              after a new sample has been taken and due according to
              ``method_log.log_frequency()``. Default uses :func:`default_write_status`,
              which writes the current completion check results to
              ``method_log.logfile_path()`` and prints a summary of the current state
              and sampled data to stdout.
          )pbdoc",
          py::arg("state"), py::arg("sampling_functions"),
          py::arg("json_sampling_functions"),
          py::arg("completion_check_params"), py::arg("event_generator"),
          py::arg("sample_period") = 1, py::arg("method_log") = std::nullopt,
          py::arg("random_engine") = nullptr,
          py::arg("write_status_f") = std::nullopt);

  m.def(
      "default_write_status",
      &CASM::monte::ising_cpp::basic_semigrand_canonical::default_write_status<
          sgc::calculator_type>,
      R"pbdoc(
        Write status to log file and screen

        Parameters
        ----------
        mc_calculator: Any
            The Monte Carlo calculator to write status for.
        method_log: MethodLog
            Method log, for writing status updates.
        )pbdoc",
      py::arg("mc_calculator"), py::arg("method_log"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
