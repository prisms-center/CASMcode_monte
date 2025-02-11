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
#include "casm/monte/BasicStatistics.hh"
#include "casm/monte/MethodLog.hh"
#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/events/OccEvent.hh"
#include "casm/monte/methods/basic_occupation_metropolis.hh"
#include "casm/monte/sampling/StateSamplingFunction.hh"

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

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
PYBIND11_MAKE_OPAQUE(CASM::monte::SamplerMap);
PYBIND11_MAKE_OPAQUE(CASM::monte::jsonSamplerMap);
PYBIND11_MAKE_OPAQUE(CASM::monte::StateSamplingFunctionMap);

PYBIND11_MODULE(_monte_methods, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Monte Carlo simulation methods

        libcasm.monte.methods._monte_methods
        ------------------------------------

        Data structures and methods implementing Monte Carlo methods
    )pbdoc";
  py::module::import("libcasm.xtal");
  py::module::import("libcasm.monte");
  py::module::import("libcasm.monte.events");
  py::module::import("libcasm.monte.sampling");

  py::class_<monte::methods::BasicOccupationMetropolisData<statistics_type>>(
      m, "BasicOccupationMetropolisData", R"pbdoc(
      Holds basic occupation Metropolis Monte Carlo run data and results

      )pbdoc")
      .def(py::init<monte::StateSamplingFunctionMap const &,
                    monte::jsonStateSamplingFunctionMap const &,
                    monte::CountType,
                    monte::CompletionCheckParams<statistics_type> const &>(),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          sampling_functions: libcasm.monte.sampling.StateSamplingFunctionMap
              The sampling functions to use
          json_sampling_functions: libcasm.monte.sampling.jsonStateSamplingFunctionMap
              The json sampling functions to use
          n_steps_per_pass: int
              Number of steps per pass.  One pass is equal to one Monte Carlo step
              per variable site in the configuration.
          completion_check_params: libcasm.monte.sampling.CompletionCheckParams
              Controls when the run finishes
          )pbdoc",
           py::arg("sampling_functions"), py::arg("json_sampling_functions"),
           py::arg("n_steps_per_pass"), py::arg("completion_check_params"))
      .def_readwrite("completion_check",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::completion_check,
                     R"pbdoc(
          libcasm.monte.sampling.CompletionCheck: The completion checker used during the Monte Carlo run
          )pbdoc")
      .def_readwrite("sampling_functions",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::sampling_functions,
                     R"pbdoc(
          libcasm.monte.sampling.StateSamplingFunctionMap: The sampling functions to use
          )pbdoc")
      .def_readwrite("samplers",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::samplers,
                     R"pbdoc(
          libcasm.monte.sampling.SamplerMap: Holds sampled data.
          )pbdoc")
      .def_readwrite("json_sampling_functions",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::json_sampling_functions,
                     R"pbdoc(
          libcasm.monte.sampling.jsonStateSamplingFunctionMap: The JSON sampling functions to use
          )pbdoc")
      .def_readwrite("json_samplers",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::json_samplers,
                     R"pbdoc(
          libcasm.monte.sampling.jsonSamplerMap: Holds JSON sampled data
          )pbdoc")
      .def_readwrite("sample_weight",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::sample_weight,
                     R"pbdoc(
          libcasm.monte.sampling.Sampler: Sample weights remain empty (unweighted)
          )pbdoc")
      .def_readwrite("n_pass",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::n_pass,
                     R"pbdoc(
          int: Number of passes. One pass is equal to one Monte Carlo step \
          per variable site in the configuration.
          )pbdoc")
      .def_readwrite("n_steps_per_pass",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::n_steps_per_pass,
                     R"pbdoc(
          int: Number of steps per pass.  One pass is equal to one Monte Carlo \
          step per variable site in the configuration.
          )pbdoc")
      .def_readwrite("n_accept",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::n_accept,
                     R"pbdoc(
          int: Number of accepted Monte Carlo steps.
          )pbdoc")
      .def_readwrite("n_reject",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::n_reject,
                     R"pbdoc(
          int: Number of rejected Monte Carlo steps.
          )pbdoc")
      .def("acceptance_rate",
           &monte::methods::BasicOccupationMetropolisData<
               statistics_type>::acceptance_rate,
           R"pbdoc(
          float: Monte Carlo step acceptance rate.
          )pbdoc")
      .def("rejection_rate",
           &monte::methods::BasicOccupationMetropolisData<
               statistics_type>::rejection_rate,
           R"pbdoc(
          float: Monte Carlo step rejection rate.
          )pbdoc")
      .def(
          "to_dict",
          [](monte::methods::BasicOccupationMetropolisData<
              statistics_type> const &data) {
            jsonParser json;
            to_json(data, json);
            return static_cast<nlohmann::json>(json);
          },
          R"pbdoc(
          Represent a summary of the BasicOccupationMetropolisData as a Python dict.

          Includes:

          - "completion_check_results": :class:`~libcasm.monte.sampling.CompletionCheckResults`
          - "n_pass": int
          - "n_steps_per_pass": int
          - "n_accept": int
          - "n_reject": int
          - "acceptance_rate": float
          - "rejection_rate": float

          Does not include individual samples or weights.

          )pbdoc")
      .def("__copy__",
           [](monte::methods::BasicOccupationMetropolisData<
               statistics_type> const &self) {
             return monte::methods::BasicOccupationMetropolisData<
                 statistics_type>(self);
           })
      .def("__deepcopy__", [](monte::methods::BasicOccupationMetropolisData<
                                  statistics_type> const &self,
                              py::dict) {
        return monte::methods::BasicOccupationMetropolisData<statistics_type>(
            self);
      });

  m.def(
      "basic_occupation_metropolis",
      [](monte::methods::BasicOccupationMetropolisData<statistics_type> &data,
         double temperature,
         std::function<double(monte::OccEvent const &)>
             potential_occ_delta_per_supercell_f,
         std::function<monte::OccEvent const &(generator_type &)>
             propose_event_f,
         std::function<void(monte::OccEvent const &)> apply_event_f,
         int sample_period, std::optional<monte::MethodLog> method_log,
         std::shared_ptr<engine_type> random_engine,
         std::function<void(monte::methods::BasicOccupationMetropolisData<
                                statistics_type> const &,
                            monte::MethodLog &)>
             write_status_f) -> void {
        if (!write_status_f) {
          write_status_f =
              monte::methods::default_write_status<statistics_type>;
        }
        monte::methods::basic_occupation_metropolis(
            data, temperature, potential_occ_delta_per_supercell_f,
            propose_event_f, apply_event_f, sample_period, method_log,
            random_engine, write_status_f);
      },
      R"pbdoc(
        Run a basic occupation Metropolis Monte Carlo simulation

        Parameters
        ----------
        data: :class:`~libcasm.monte.methods.BasicOccupationMetropolisData`
            Holds basic occupation Metropolis Monte Carlo run data and
            results when finished.
        temperature: float
            The temperature used for the Metropolis algorithm.
        potential_occ_delta_per_supercell_f: function
            A function with signature ``def (e: OccEvent) -> float``
            that calculates the change in the potential due to a proposed
            occupation event.
        propose_event_f: function
            A function with signature
            ``def f(rng: RandomNumberGenerator) -> OccEvent`` that
            proposes an event of type :class:`~libcasm.monte.events.OccEvent`
            based on the current state and a random number generator.
        apply_event_f: function
            A function with signature ``def f(e: OccEvent) -> None``, which
            applies an accepted event to update the current state.
        sample_period: int = 1
            Number of passes per sample. One pass is one Monte Carlo step per
            site with variable occupation.
        method_log: Optional[:class:`~libcasm.monte.MethodLog`] = None
            Method log, for writing status updates. If None, default writes
            to "status.json" every 10 minutes.
        random_engine: Optional[:class:`~libcasm.monte.RandomNumberEngine`] = None
            Random number engine. Default constructs a new engine.
        write_status_f: Optional[function] = None
            Function with signature
            ``def f(data: BasicOccupationMetropolisData, method_log: MethodLog) -> None``
            that writes status updates, after a new sample has been taken and
            is due according to ``method_log.log_frequency()``. Default writes
            the current completion check results to ``method_log.logfile_path()``
            and prints a summary to stdout.

        )pbdoc",
      py::arg("data"), py::arg("temperature"),
      py::arg("potential_occ_delta_per_supercell_f"),
      py::arg("propose_event_f"), py::arg("apply_event_f"),
      py::arg("sample_period") = 1, py::arg("method_log") = std::nullopt,
      py::arg("random_engine") = nullptr, py::arg("write_status_f") = nullptr);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
