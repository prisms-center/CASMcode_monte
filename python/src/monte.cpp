#include <pybind11/eigen.h>
#include <pybind11/functional.h>
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
#include "casm/monte/MethodLog.hh"
#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/ValueMap.hh"
#include "casm/monte/definitions.hh"
#include "casm/monte/io/json/ValueMap_json_io.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

// used for libcasm.monte:
typedef monte::default_engine_type engine_type;
typedef monte::RandomNumberGenerator<engine_type> generator_type;

monte::ValueMap make_ValueMap(std::optional<nlohmann::json> data) {
  if (!data.has_value()) {
    data = nlohmann::json{};
  }
  jsonParser json{static_cast<const nlohmann::json &>(*data)};
  monte::ValueMap values;
  from_json(values, json);
  return values;
}

monte::MethodLog make_MethodLog(std::optional<std::string> logfile_path,
                                std::optional<double> log_frequency) {
  monte::MethodLog method_log;
  if (logfile_path.has_value()) {
    method_log.logfile_path = logfile_path.value();
    method_log.log_frequency = log_frequency;
    method_log.reset();
  } else {
    method_log.log_frequency = log_frequency;
    method_log.reset_to_stdout();
  }
  return method_log;
}

/// \brief Make a random number engine seeded by std::random_device
std::shared_ptr<engine_type> make_random_number_engine() {
  std::shared_ptr<engine_type> engine = std::make_shared<engine_type>();
  std::random_device device;
  engine->seed(device());
  return engine;
}

/// \brief Make a random number generator that uses the provided engine
///
/// Notes:
/// - If _engine == nullptr, use an engine seeded by std::random_device
generator_type make_random_number_generator(
    std::shared_ptr<engine_type> _engine = std::shared_ptr<engine_type>()) {
  return monte::RandomNumberGenerator(_engine);
}

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

// #include "opaque_types.cc"
PYBIND11_MAKE_OPAQUE(std::map<std::string, bool>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, double>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, Eigen::VectorXd>);
PYBIND11_MAKE_OPAQUE(std::map<std::string, Eigen::MatrixXd>);

PYBIND11_MODULE(_monte, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Building blocks for Monte Carlo simualations

        libcasm.monte
        -------------

        The libcasm-monte is a Python interface to the classes and methods in the
        CASM::monte namespace of the CASM C++ libraries that are useful building blocks
        for Monte Carlo simulations.

    )pbdoc";
  py::module::import("libcasm.xtal");

  // #include "local_bindings.cc"
  py::bind_map<std::map<std::string, bool>>(m, "BooleanValueMap");
  py::bind_map<std::map<std::string, double>>(m, "ScalarValueMap");
  py::bind_map<std::map<std::string, Eigen::VectorXd>>(m, "VectorValueMap");
  py::bind_map<std::map<std::string, Eigen::MatrixXd>>(m, "MatrixValueMap");

  py::class_<monte::MethodLog>(m, "MethodLog", R"pbdoc(
      Logger for Monte Carlo method status

      )pbdoc")
      .def(py::init<>(&make_MethodLog),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          logfile_path : Optional[str]
              File location for log output. If None, log to stdout.
          log_frequency : Optional[float]
              How often to log method status, in seconds
          )pbdoc",
           py::arg("logfile_path") = std::nullopt,
           py::arg("log_frequency") = std::nullopt)
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
          )pbdoc")
      .def("reset_to_stdout", &monte::MethodLog::reset_to_stdout,
           R"pbdoc(
          Reset to print to stdout.
          )pbdoc")
      //
      .def(
          "section",
          [](monte::MethodLog &x, std::string what, bool show_clock) {
            x.log.indent() << "-- " << what << " -- ";
            if (show_clock) {
              x.log << "Time: " << x.log.time_s() << " (s)" << std::endl;
            }
            x.log << std::endl;
          },
          R"pbdoc(
          Print a nicely formatted section header, optionally with the current timer
          value.
          )pbdoc",
          py::arg("what"), py::arg("show_clock") = false)
      // verbosity controlled printing
      .def(
          "set_quiet",
          [](monte::MethodLog &x) { return x.log.set_verbosity(Log::quiet); },
          R"pbdoc(
          Set to "quiet" printing mode. The "standard", "verbose", and "debug" sections
          will not be printed.
          )pbdoc")
      .def(
          "set_standard",
          [](monte::MethodLog &x) {
            return x.log.set_verbosity(Log::standard);
          },
          R"pbdoc(
          Set to "standard" printing mode. The "verbose" and "debug" sections will not
          be printed.
          )pbdoc")
      .def(
          "set_verbose",
          [](monte::MethodLog &x) { return x.log.set_verbosity(Log::verbose); },
          R"pbdoc(
          Set to "verbose" printing mode. Only "debug" sections will not be printed.
          )pbdoc")
      .def(
          "set_debug",
          [](monte::MethodLog &x) { return x.log.set_verbosity(Log::debug); },
          R"pbdoc(
          Set to "debug" printing mode. All sections will be printed.
          )pbdoc")
      .def(
          "begin_section_print_always",
          [](monte::MethodLog &x) { return x.log.begin_section<Log::none>(); },
          R"pbdoc(
          Begin a section which always prints, for every `verbosity_level`.
          )pbdoc")
      .def(
          "begin_section_print_if_quiet",
          [](monte::MethodLog &x) {
            return x.log.begin_section<Log::standard>();
          },
          R"pbdoc(
          Begin a section which only prints if `verbosity_level` >= "quiet".
          )pbdoc")
      .def(
          "begin_section_print_if_standard",
          [](monte::MethodLog &x) {
            return x.log.begin_section<Log::standard>();
          },
          R"pbdoc(
          Begin a section which only prints if `verbosity_level` >= "standard".
          )pbdoc")
      .def(
          "begin_section_print_if_verbose",
          [](monte::MethodLog &x) {
            return x.log.begin_section<Log::verbose>();
          },
          R"pbdoc(
          Begin a section which only prints if `verbosity_level` >= "verbose".
          )pbdoc")
      .def(
          "begin_section_print_if_debug",
          [](monte::MethodLog &x) { return x.log.begin_section<Log::debug>(); },
          R"pbdoc(
          Begin a section which only prints if `verbosity_level` >= "debug".
          )pbdoc")
      .def(
          "end_section",
          [](monte::MethodLog &x) { return x.log.end_section(); },
          R"pbdoc(
          End the current section.
          )pbdoc")
      // clock
      .def(
          "restart_clock",
          [](monte::MethodLog &x) { return x.log.restart_clock(); },
          R"pbdoc(
          Restart internal timer.
          )pbdoc")
      .def(
          "time_s", [](monte::MethodLog const &x) { return x.log.time_s(); },
          R"pbdoc(
          Time in seconds since construction or `restart_clock`.
          )pbdoc")
      .def(
          "begin_lap", [](monte::MethodLog &x) { return x.log.begin_lap(); },
          R"pbdoc(
          Begin a new lap.
          )pbdoc")
      .def(
          "lap_time",
          [](monte::MethodLog const &x) { return x.log.lap_time(); },
          R"pbdoc(
          Time in seconds since `begin_lap`.
          )pbdoc")
      .def(
          "show_clock", [](monte::MethodLog &x) { return x.log.show_clock(); },
          R"pbdoc(
          Show `time_s` as a part of section headings.
          )pbdoc")
      .def(
          "hide_clock", [](monte::MethodLog &x) { return x.log.hide_clock(); },
          R"pbdoc(
          Do not show time as a part of section headings.
          )pbdoc")
      // printing
      .def(
          "print",
          [](monte::MethodLog &x, std::string const &text) {
            x.log.indent() << text;
          },
          R"pbdoc(
          Print with indent.
          )pbdoc",
          py::arg("text"))
      .def(
          "set_paragraph_width",
          [](monte::MethodLog &x, int width) { x.log.set_width(width); },
          R"pbdoc(
          Set paragraph width.
          )pbdoc")
      .def(
          "paragraph_width", [](monte::MethodLog const &x) { x.log.width(); },
          R"pbdoc(
          Return paragraph width.
          )pbdoc")
      .def(
          "set_paragraph_justification",
          [](monte::MethodLog &x, std::string justification_type) {
            // enum class JustificationType { Left, Right, Center, Full };
            if (justification_type == "left") {
              x.log.set_justification(JustificationType::Left);
            } else if (justification_type == "right") {
              x.log.set_justification(JustificationType::Right);
            } else if (justification_type == "center") {
              x.log.set_justification(JustificationType::Center);
            } else if (justification_type == "full") {
              x.log.set_justification(JustificationType::Full);
            } else {
              throw std::runtime_error("Invalid justification_type");
            }
          },
          R"pbdoc(
          Set paragraph justification type. One of "left", "right", "center" or "full".
          )pbdoc",
          py::arg("justification_type"))
      .def(
          "paragraph_justification",
          [](monte::MethodLog &x) -> std::string {
            // enum class JustificationType { Left, Right, Center, Full };
            if (x.log.justification() == JustificationType::Left) {
              return std::string("left");
            }
            if (x.log.justification() == JustificationType::Right) {
              return std::string("right");
            }
            if (x.log.justification() == JustificationType::Center) {
              return std::string("center");
            }
            if (x.log.justification() == JustificationType::Full) {
              return std::string("full");
            }
            throw std::runtime_error("Error in paragraph_justification");
          },
          R"pbdoc(
          Return paragraph justification type. One of "left", "right", "center" or
          "full".
          )pbdoc")
      .def(
          "paragraph",
          [](monte::MethodLog &x, std::string const &text) {
            x.log.paragraph(text);
          },
          R"pbdoc(
          Print with indent, line wrapping, and justification.
          )pbdoc",
          py::arg("text"))
      .def(
          "verbatim",
          [](monte::MethodLog &x, std::string const &text,
             bool indent_first_line) {
            x.log.verbatim(text, indent_first_line);
          },
          R"pbdoc(
          Print with indent and line wrapping, without justification.
          )pbdoc",
          py::arg("text"), py::arg("indent_first_line") = true)
      .def(
          "set_initial_indent_space",
          [](monte::MethodLog &x, int initial_indent_space) {
            x.log.set_initial_indent_space(initial_indent_space);
          },
          R"pbdoc(
          Set an initial number of indent spaces, before applying indent levels.
          )pbdoc",
          py::arg("initial_indent_space"))
      .def(
          "set_indent_space",
          [](monte::MethodLog &x, int indent_space) {
            x.log.set_indent_space(indent_space);
          },
          R"pbdoc(
          Set the number of spaces per indent level.
          )pbdoc",
          py::arg("indent_space"))
      .def(
          "increase_indent",
          [](monte::MethodLog &x) { x.log.increase_indent(); },
          R"pbdoc(
          Increase indent level.
          )pbdoc")
      .def(
          "decrease_indent",
          [](monte::MethodLog &x) { x.log.decrease_indent(); },
          R"pbdoc(
          Decrease indent level.
          )pbdoc")
      .def(
          "indent_str", [](monte::MethodLog &x) { x.log.indent_str(); },
          R"pbdoc(
          The current indent string.
          )pbdoc");

  py::class_<monte::ValueMap>(m, "ValueMap", R"pbdoc(
      Data structure for holding Monte Carlo data

      Notes
      -----
      Data should not have the same key, even if the values have
      different type. Conversions for input/output are made
      to/from a single combined dict.
      )pbdoc")
      .def(py::init<>(&make_ValueMap),
           R"pbdoc(
          .. rubric:: Constructor

          Notes
          -----

          - The constructor is equivalent to :func:`ValueMap.from_dict`, except
            that it also accepts ``None``.

          Parameters
          ----------
          data: Optional[dict] = None
              A dict with keys of type `str` and boolean, scalar, vector, or
              matrix values.
          )pbdoc",
           py::arg("data") = std::nullopt)
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
      .def("is_mismatched", &monte::is_mismatched,
           R"pbdoc(
            Return true if :class:`~libcasm.monte.ValueMap` do not have the same properties.
            )pbdoc",
           py::arg("other"))
      .def("make_incremented_values", &monte::make_incremented_values,
           R"pbdoc(
            Return self[property] + n_increment*increment[property] for each property

            Notes
            -----
            Does not change boolean values.
            )pbdoc",
           py::arg("increment"), py::arg("n_increment"))
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
          "are combined into a single dict")
      .def("__copy__",
           [](monte::ValueMap const &self) { return monte::ValueMap(self); })
      .def("__deepcopy__", [](monte::ValueMap const &self, py::dict) {
        return monte::ValueMap(self);
      });

  py::class_<engine_type, std::shared_ptr<engine_type>>(m, "RandomNumberEngine",
                                                        R"pbdoc(
      A pseudo-random number engine, using std::MT19937_64
      )pbdoc")
      .def(py::init<>(&make_random_number_engine),
           R"pbdoc(
           .. rubric:: Constructor

           Default constructor only. Constructs a pseudo-random number engine
           using std::random_device to seed.
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
          .. rubric:: Constructor

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

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
