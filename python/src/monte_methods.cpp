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
#include "casm/monte/state/StateSampler.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

// used for libcasm.monte:
typedef std::mt19937_64 engine_type;
typedef monte::RandomNumberGenerator<engine_type> generator_type;
typedef monte::BasicStatistics statistics_type;

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
PYBIND11_MAKE_OPAQUE(CASM::monte::SamplerMap);
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

  py::class_<monte::methods::BasicOccupationMetropolisData<statistics_type>>(
      m, "BasicOccupationMetropolisData", R"pbdoc(
      Holds basic occupation Metropolis Monte Carlo run data and results

      )pbdoc")
      .def(
          py::init<monte::CompletionCheckParams<statistics_type> const &,
                   monte::StateSamplingFunctionMap const &, monte::CountType>(),
          R"pbdoc(
          Constructor

          Parameters
          ----------
          completion_check_params: :class:`~libcasm.monte.CompletionCheckParams`
              Controls when the run finishes
          sampling_functions: :class:`~libcasm.monte.StateSamplingFunctionMap`
              The sampling functions to use
          n_steps_per_pass: int
              Number of steps per pass.  One pass is equal to one Monte Carlo step
              per variable site in the configuration.
          )pbdoc",
          py::arg("completion_check_params"), py::arg("sampling_functions"),
          py::arg("n_steps_per_pass"))
      .def_readwrite("completion_check",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::completion_check,
                     R"pbdoc(
          :class:`~libcasm.monte.CompletionCheck`: \
          The completion checker used during the Monte Carlo run
          )pbdoc")
      .def_readwrite("samplers",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::samplers,
                     R"pbdoc(
          :class:`~libcasm.monte.SamplerMap`: Holds sampled data
          )pbdoc")
      .def_readwrite("sample_weight",
                     &monte::methods::BasicOccupationMetropolisData<
                         statistics_type>::sample_weight,
                     R"pbdoc(
          :class:`~libcasm.monte.Sampler`: Sample weights remain empty (unweighted)
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
          int: Number of rejepted Monte Carlo steps.
          )pbdoc");

  m.def(
      "basic_occupation_metropolis",
      [](double temperature,
         std::function<double(monte::OccEvent const &)>
             potential_occ_delta_extensive_value_f,
         std::function<monte::OccEvent const &(generator_type &)>
             propose_event_f,
         std::function<void(monte::OccEvent const &)> apply_event_f,
         monte::StateSamplingFunctionMap const &sampling_functions,
         monte::CountType n_steps_per_pass,
         monte::CompletionCheckParams<statistics_type> const
             &completion_check_params,
         int sample_period, std::optional<monte::MethodLog> method_log,
         std::shared_ptr<engine_type> random_engine,
         std::function<void(monte::methods::BasicOccupationMetropolisData<
                                statistics_type> const &,
                            monte::MethodLog &)>
             write_status_f)
          -> monte::methods::BasicOccupationMetropolisData<statistics_type> {
        return monte::methods::basic_occupation_metropolis(
            temperature, potential_occ_delta_extensive_value_f, propose_event_f,
            apply_event_f, sampling_functions, n_steps_per_pass,
            completion_check_params, sample_period, method_log, random_engine,
            write_status_f);
      },

      //    [](double temperature,
      //         std::function<double(monte::OccEvent const &)>
      //             potential_occ_delta_extensive_value_f,
      //         std::function<monte::OccEvent const &(generator_type &)>
      //             propose_event_f,
      //         std::function<void(monte::OccEvent const &)> apply_event_f,
      //         monte::StateSamplingFunctionMap const &sampling_functions,
      //         monte::CountType n_steps_per_pass,
      //         monte::CompletionCheckParams<statistics_type> const
      //             &completion_check_params,
      //         int sample_period, std::optional<monte::MethodLog> method_log,
      //         std::shared_ptr<engine_type> random_engine,
      //         std::function<void(monte::methods::BasicOccupationMetropolisData<
      //                                statistics_type> const &,
      //                            monte::MethodLog &)>
      //             write_status_f)
      //          ->
      //          monte::methods::BasicOccupationMetropolisData<statistics_type>
      //          {
      //        return monte::methods::basic_occupation_metropolis(
      //            temperature, potential_occ_delta_extensive_value_f,
      //            propose_event_f, apply_event_f, sampling_functions,
      //            n_steps_per_pass, completion_check_params, sample_period,
      //            method_log, random_engine, write_status_f);
      //      },
      R"pbdoc(
        Run a basic occupation Metropolis Monte Carlo simulation

        Parameters
        ----------
        temperature: float
            The temperature used for the Metropolis algorithm.
        potential_occ_delta_extensive_value_f: function
            A function with signature ``def (occ_event: OccEvent) -> float``
            that calculates the change in the potential due to a proposed
            occupation event.
        propose_event_f: function
            A function with signature
            ``def f(rng: RandomNumberGenerator) -> OccEvent const &`` that
            proposes an event of type :class:`~libcasm.monte.events.OccEvent`
            based on the current state and a random number generator.
        apply_event_f: function
            A function with ``def f(OccEvent const &) -> None``), which
            applies an accepted event to update the current state.
        sampling_functions: :class:`~libcasm.monte.StateSamplingFunctionMap`
            The sampling functions to use
        n_steps_per_pass: int
            Number of steps per pass.
        completion_check_params: :class:`~libcasm.monte.CompletionCheckParams`
            Controls when the run finishes
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
            the current completion check results to
            ``method_log.logfile_path()`` and prints a summary of the to stdout.

        Returns
        -------
        data: :class:`~libcasm.monte.methods.BasicOccupationMetropolisData`
            Monte Carlo run data and results.

        )pbdoc",
      py::arg("temperature"), py::arg("potential_occ_delta_extensive_value_f"),
      py::arg("propose_event_f"), py::arg("apply_event_f"),
      py::arg("sampling_functions"), py::arg("n_steps_per_pass"),
      py::arg("completion_check_params"), py::arg("sample_period") = 1,
      py::arg("method_log") = std::nullopt, py::arg("random_engine") = nullptr,
      py::arg("write_status_f") =
          std::function<void(monte::methods::BasicOccupationMetropolisData<
                                 statistics_type> const &,
                             monte::MethodLog &)>(
              monte::methods::default_write_status<statistics_type>));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
