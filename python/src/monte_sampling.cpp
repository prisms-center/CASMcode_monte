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
#include "casm/monte/BasicStatistics.hh"
#include "casm/monte/MethodLog.hh"
#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/ValueMap.hh"
#include "casm/monte/checks/EquilibrationCheck.hh"
#include "casm/monte/checks/io/json/CompletionCheck_json_io.hh"
#include "casm/monte/checks/io/json/ConvergenceCheck_json_io.hh"
#include "casm/monte/checks/io/json/CutoffCheck_json_io.hh"
#include "casm/monte/checks/io/json/EquilibrationCheck_json_io.hh"
#include "casm/monte/definitions.hh"
#include "casm/monte/io/json/ValueMap_json_io.hh"
#include "casm/monte/run_management/StateSampler.hh"
#include "casm/monte/sampling/Sampler.hh"
#include "casm/monte/sampling/SamplingParams.hh"
#include "casm/monte/sampling/StateSamplingFunction.hh"
#include "casm/monte/sampling/io/json/Sampler_json_io.hh"
#include "casm/monte/sampling/io/json/SamplingParams_json_io.hh"

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

monte::jsonStateSamplingFunction make_json_state_sampling_function(
    std::string name, std::string description,
    std::function<py::object()> function) {
  if (function == nullptr) {
    throw std::runtime_error(
        "Error constructing jsonStateSamplingFunction: function == nullptr");
  }
  return monte::jsonStateSamplingFunction(
      name, description, [=]() -> jsonParser {
        nlohmann::json j = function();
        return jsonParser(static_cast<nlohmann::json const &>(j));
      });
}

monte::CompletionCheckParams<statistics_type> make_completion_check_params(
    std::optional<monte::RequestedPrecisionMap> requested_precision =
        std::nullopt,
    std::optional<monte::CutoffCheckParams> cutoff_params = std::nullopt,
    monte::CalcStatisticsFunction<statistics_type> calc_statistics_f = nullptr,
    monte::EquilibrationCheckFunction equilibration_check_f = nullptr,
    bool log_spacing = false,
    std::optional<monte::CountType> check_begin = std::nullopt,
    std::optional<monte::CountType> check_period = std::nullopt,
    std::optional<double> check_base = std::nullopt,
    std::optional<double> check_shift = std::nullopt,
    std::optional<monte::CountType> check_period_max = std::nullopt) {
  monte::CompletionCheckParams<statistics_type> result;
  if (!cutoff_params.has_value()) {
    cutoff_params = monte::CutoffCheckParams();
  }
  if (!equilibration_check_f) {
    equilibration_check_f = monte::default_equilibration_check;
  }
  if (!calc_statistics_f) {
    calc_statistics_f = monte::BasicStatisticsCalculator();
  }
  if (!requested_precision.has_value()) {
    requested_precision = monte::RequestedPrecisionMap();
  }

  if (!log_spacing) {
    result.check_begin = 100;
    result.check_period = 100;
  } else {
    result.check_begin = 0;
    result.check_base = 10.0;
    result.check_shift = 2.0;
    result.check_period_max = 10000;
  }

  result.cutoff_params = cutoff_params.value();
  result.equilibration_check_f = equilibration_check_f;
  result.calc_statistics_f = calc_statistics_f;
  result.requested_precision = requested_precision.value();
  result.log_spacing = log_spacing;

  if (check_begin.has_value()) {
    result.check_begin = check_begin.value();
  }
  if (check_period.has_value()) {
    result.check_period = check_period.value();
  }
  if (check_base.has_value()) {
    result.check_base = check_base.value();
  }
  if (check_shift.has_value()) {
    result.check_shift = check_shift.value();
  }
  if (check_period_max.has_value()) {
    result.check_period_max = check_period_max.value();
  }
  return result;
}

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

PYBIND11_MODULE(_monte_sampling, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Monte Carlo sampling and convergence checking

        libcasm.monte.sampling
        ----------------------

        Data structures and methods for Monte Carlo sampling and convergence checking.

    )pbdoc";
  py::module::import("libcasm.monte");

  py::bind_map<monte::SamplerMap>(m, "SamplerMap",
                                  R"pbdoc(
    SamplerMap stores :class:`~libcasm.monte.Sampler` by name of the sampled quantity

    Notes
    -----
    SamplerMap is a Dict[str, :class:`~libcasm.monte.Sampler`]-like object.
    )pbdoc",
                                  py::module_local(false));

  py::bind_map<monte::StateSamplingFunctionMap>(m, "StateSamplingFunctionMap",
                                                R"pbdoc(
    StateSamplingFunctionMap stores :class:`~libcasm.monte.StateSamplingFunction` by name of the sampled quantity.

    Notes
    -----
    StateSamplingFunctionMap is a Dict[str, :class:`~libcasm.monte.StateSamplingFunction`]-like object.
    )pbdoc",
                                                py::module_local(false));

  py::bind_map<monte::jsonStateSamplingFunctionMap>(
      m, "jsonStateSamplingFunctionMap",
      R"pbdoc(
    jsonStateSamplingFunctionMap stores :class:`~libcasm.monte.jsonStateSamplingFunction` by name of the sampled quantity.

    Notes
    -----
    jsonStateSamplingFunctionMap is a Dict[str, :class:`~libcasm.monte.jsonStateSamplingFunction`]-like object.
    )pbdoc",
      py::module_local(false));

  py::bind_map<monte::RequestedPrecisionMap>(m, "RequestedPrecisionMap",
                                             R"pbdoc(
    RequestedPrecisionMap stores :class:`~libcasm.monte.RequestedPrecision` with :class:`~libcasm.monte.SamplerComponent` keys.

    Notes
    -----
    RequestedPrecisionMap is a Dict[:class:`~libcasm.monte.SamplerComponent`, :class:`~libcasm.monte.RequestedPrecision`]-like object.
    )pbdoc",
                                             py::module_local(false));

  py::bind_map<monte::EquilibrationResultMap>(m, "EquilibrationResultMap",
                                              R"pbdoc(
    EquilibrationResultMap stores :class:`~libcasm.monte.IndividualEquilibrationResult` by :class:`~libcasm.monte.SamplerComponent`

    Notes
    -----
    EquilibrationResultMap is a Dict[:class:`~libcasm.monte.SamplerComponent`, :class:`~libcasm.monte.IndividualEquilibrationResult`]-like object.
    )pbdoc",
                                              py::module_local(false));

  py::bind_map<monte::ConvergenceResultMap<statistics_type>>(
      m, "ConvergenceResultMap",
      R"pbdoc(
    ConvergenceResultMap stores :class:`~libcasm.monte.IndividualConvergenceResult` by :class:`~libcasm.monte.SamplerComponent`

    Notes
    -----
    ConvergenceResultMap is a Dict[:class:`~libcasm.monte.SamplerComponent`, :class:`~libcasm.monte.IndividualConvergenceResult`]-like object.
    )pbdoc",
      py::module_local(false));

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
          .. rubric:: Constructor

          Default constructor only.
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
      .def_readwrite("begin", &monte::SamplingParams::begin, R"pbdoc(
          float: See `sample_method`. Default=``0.0``.
          )pbdoc")
      .def_readwrite("period", &monte::SamplingParams::period, R"pbdoc(
          float: See `sample_method`. Default=``1.0``.
          )pbdoc")
      .def_readwrite("samples_per_period",
                     &monte::SamplingParams::samples_per_period, R"pbdoc(
          float: See `sample_method`. Default=``1.0``.
          )pbdoc")
      .def_readwrite("shift", &monte::SamplingParams::shift, R"pbdoc(
          float: See `sample_method`. Default=``0.0``.
          )pbdoc")
      .def_readwrite("sampler_names", &monte::SamplingParams::sampler_names,
                     R"pbdoc(
          List[str]: The names of quantities to sample (i.e. sampling function \
          names). Default=``[]``.
          )pbdoc")
      .def_readwrite("do_sample_trajectory",
                     &monte::SamplingParams::do_sample_trajectory, R"pbdoc(
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
      [](Eigen::MatrixXd const &value) -> Eigen::VectorXd {
        return value.reshaped();
      },
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

          .. rubric:: Constructor

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

          .. rubric:: Constructor

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
            # mc_calculator = SemiGrandCanonical(...)

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

          .. rubric:: Constructor

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

          component_names : Optional[List[str]] = None
              A name for each component of the resulting vector.

              Can be strings representing an indices (i.e "0", "1", "2", etc.) or can be a descriptive string (i.e. "Mg", "Va", "O", etc.). If None, indices for column-major ordering are used (i.e. "0,0", "1,0", ..., "m-1,n-1")

          )pbdoc",
           py::arg("name"), py::arg("description"), py::arg("shape"),
           py::arg("function"), py::arg("component_names") = std::nullopt)
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

  py::class_<monte::jsonStateSamplingFunction>(m, "jsonStateSamplingFunction",
                                               R"pbdoc(
        A function that samples JSON data from a Monte Carlo state

        Example usage:

        .. code-block:: Python

            from libcasm.clexmonte import SemiGrandCanonical
            from libcasm.monte import (
                Sampler, SamplerMap, StateSamplingFunction,
                StateSamplingFunctionMap, jsonStateSamplingFunctionMap
            )

            # ... in Monte Carlo simulation setup ...
            # calculation = SemiGrandCanonical(...)

            # ... create sampling functions ...
            json_sampling_functions = jsonStateSamplingFunctionMap()

            def configuration_f():
                return calculation.state.configuration.to_dict()

            f = monte.jsonStateSamplingFunction(
                name="configuration",
                description="Configuration values",
                function=configuration_json_f,
            )
            sampling_functions[f.name] = f

            # ... create samplers to hold data ...
            json_sampled_data = jsonSampledDataMap()
            for name, f in json_sampling_functions.items():
                json_sampled_data[name] = []

            # ... in Monte Carlo simulation ...
            # ... sample JSON data ...
            for name, f in json_sampling_functions.items():
                json_sampled_data[name].append(f())


        Notes
        -----
        - Typically this holds a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the current state of the simulation.
        - jsonStateSamplingFunction can be used to sample quantities not easily converted to scalar, vector, matrix, etc.
        - Data sampled by a jsonStateSamplingFunction can be stored in a :class:`~libcasm.monte.jsonSampledDataMap`.
        - A call operator exists (:func:`~libcasm.monte.jsonStateSamplingFunction.__call__`) to call the function held by :class:`~libcasm.monte.jsonStateSamplingFunction`.
        )pbdoc")
      .def(py::init<>(&make_json_state_sampling_function),
           R"pbdoc(

          .. rubric:: Constructor

          Parameters
          ----------
          name : str
              Name of the sampled quantity.

          description : str
              Description of the function.

          component_index : int
              Index into the unrolled vector of a sampled quantity.

          function : function
              A function with 0 arguments that samples the current state and returns a Python object that is convertible to JSON. Typically this is a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the current state of the simulation.

          )pbdoc",
           py::arg("name"), py::arg("description"), py::arg("function"))
      .def_readwrite("name", &monte::jsonStateSamplingFunction::name,
                     R"pbdoc(
          str : Name of the sampled quantity.
          )pbdoc")
      .def_readwrite("description",
                     &monte::jsonStateSamplingFunction::description,
                     R"pbdoc(
          str : Description of the function.
          )pbdoc")
      .def_readwrite("function", &monte::jsonStateSamplingFunction::function,
                     R"pbdoc(
          function : The function to be evaluated.

          A function with 0 arguments that returns an array of the proper size sampling the current state. Typically this is a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the current state of the simulation.
          )pbdoc")
      .def(
          "__call__",
          [](monte::jsonStateSamplingFunction const &f) -> jsonParser {
            return f();
          },
          R"pbdoc(
          Evaluates the JSON state sampling function

          Equivalent to calling :py::attr:`~libcasm.monte.jsonStateSamplingFunction.function`.
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

          .. rubric:: Constructor

          Parameters
          ----------
          abs : Optional[float] = None
              If provided, the specified absolute precision level requested, :math:`p_{abs}`.
          rel : Optional[float] = None
              If provided, the specified relative precision level requested, :math:`p_{rel}`.

          )pbdoc",
           py::arg("abs") = std::nullopt, py::arg("rel") = std::nullopt)
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

  py::class_<monte::IndividualEquilibrationCheckResult>(
      m, "IndividualEquilibrationResult",
      R"pbdoc(
      Equilibration check results for a single :class:`~libcasm.monte.SamplerComponent`
      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          .. rubric:: Constructor

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

  py::class_<monte::EquilibrationCheckResults>(m, "EquilibrationCheckResults",
                                               R"pbdoc(
      Stores equilibration check results
      )pbdoc")
      //
      .def(py::init<>(),
           R"pbdoc(
          .. rubric:: Constructor

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
      requested_precision : :class:`~libcasm.monte.RequestedPrecision`
          The requested precision level for convergence.

      Returns
      -------
      results : :class:`~libcasm.monte.IndividualEquilibrationResult`
          The equilibration check results.
      )pbdoc");

  /// BasicStatistics implementation ~~~~~~~~~~~~~~~~~~~~~~~~~

  py::class_<monte::BasicStatistics>(m, "BasicStatistics",
                                     R"pbdoc(
      Basic statistics calculated from samples
      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          .. rubric:: Constructor

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

      This is a callable class, which calculates :class:`~libcasm.monte.BasicStatistics`
      from a series of observations, and optionally sample weights.

      .. rubric:: Special Methods

      The call operator is equivalent to :func:`~libcasm.monte.BasicStatisticsCalculator.calculate`.


      )pbdoc")
      .def(py::init<double, Index, Index>(),
           R"pbdoc(

          .. rubric:: Constructor

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
          "calculate",
          [](monte::BasicStatisticsCalculator const &f,
             Eigen::VectorXd const &observations,
             Eigen::VectorXd const &sample_weight) {
            return f(observations, sample_weight);
          },
          R"pbdoc(
          Calculate statistics for a range of weighted observations

          The method used to estimate precision in the sample mean when observations are
          weighted (i.e. N-fold way method) depends on the parameter
          :py:attr:`~libcasm.monte.weighted_observations_method`.

          .. rubric:: Case 1: No sample weights

          The calculated precision is estimated as

          .. math::

              \hat{\gamma}_k &= \sum^{N-k}_i\left( X_i - \bar{X} \right) \left( X_{i+k} - \bar{X} \right)

              \gamma_k = \gamma_0 \rho^{-|k|}

              \sigma^2 = \gamma_0 \left(\frac{1+\rho}{1-\rho}\right)



          Parameters
          ----------
          observations : array_like
              A 1d array of observations. Should only include samples after the calculation
              has equilibrated.
          sample_weight : array_like
              Sample weights associated with observations. May have size 0, in which case the
              observations are treated as being equally weighted and no resampling is
              performed, or have the same size as `observations`.

          Returns
          -------
          stats : :class:`~libcasm.monte.BasicStatistics`
              Calculated statistics.

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

  /// Equilibration, convergence, completion checks ~~~~~~~~~~~~~~~~~~~~~~~~~
  /// - templated by statistics_type
  ///
  /// Should work for another statistics_type if there is a statistics
  /// calculating function with signature:
  /// - f(Eigen::VectorXd const &observations,
  ///     Eigen::VectorXd const &sample_weight) -> statistics_type
  /// and the following are implemented:
  /// - double get_calculated_precision(statistics_type const &stats);
  /// - double get_calculated_relative_precision(statistics_type const &stats);
  /// - void to_json(statistics_type const &stats, jsonParser &json);
  /// - void append_statistics_to_json_arrays(
  ///     std::optional<statistics_type> const &stats, jsonParser &json);

  py::class_<monte::IndividualConvergenceCheckResult<statistics_type>>(
      m, "IndividualConvergenceResult",
      R"pbdoc(
      Convergence check results for a single :class:`~libcasm.monte.SamplerComponent`
      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          .. rubric:: Constructor

          Default constructor only.
          )pbdoc")
      .def_readwrite("is_converged",
                     &monte::IndividualConvergenceCheckResult<
                         statistics_type>::is_converged,
                     R"pbdoc(
          True if mean is converged to requested precision. Default=``False``.
          )pbdoc")
      .def_readwrite("requested_precision",
                     &monte::IndividualConvergenceCheckResult<
                         statistics_type>::requested_precision,
                     R"pbdoc(
          Requested precision of the mean.
          )pbdoc")
      .def_readwrite(
          "stats",
          &monte::IndividualConvergenceCheckResult<statistics_type>::stats,
          R"pbdoc(
          Calculated statistics.
          )pbdoc")
      .def(
          "to_dict",
          [](monte::IndividualConvergenceCheckResult<statistics_type> const
                 &x) {
            jsonParser json;
            to_json(x, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the individual convergence check results as a Python "
          "dict.");

  py::class_<monte::ConvergenceCheckResults<statistics_type>>(
      m, "ConvergenceCheckResults",
      R"pbdoc(
      Stores convergence check results
      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          .. rubric:: Constructor

          Default constructor only.
          )pbdoc")
      .def_readwrite(
          "all_converged",
          &monte::ConvergenceCheckResults<statistics_type>::all_converged,
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
                         statistics_type>::N_samples_for_statistics,
                     R"pbdoc(
          How many samples were used to get statistics.

          Notes
          -----

          - Set to the total number of samples if no convergence checks were
            requested
          )pbdoc")
      .def_readwrite(
          "individual_results",
          &monte::ConvergenceCheckResults<statistics_type>::individual_results,
          R"pbdoc(
          Results from checking convergence criteria.
          )pbdoc")
      .def(
          "to_dict",
          [](monte::ConvergenceCheckResults<statistics_type> const &x) {
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
         monte::CalcStatisticsFunction<statistics_type> calc_statistics_f)
          -> monte::IndividualConvergenceCheckResult<statistics_type> {
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
            Fucntion used to calculate :class:`~libcasm.monte.BasicStatistics`. For example, an instance of :class:`~libcasm.monte.BasicStatisticsCalculator`.
        )pbdoc",
      py::arg("sampler"), py::arg("sample_weight"), py::arg("key"),
      py::arg("requested_precision"), py::arg("N_samples_for_statistics"),
      py::arg("calc_statistics_f"));

  m.def(
      "convergence_check",
      [](std::map<std::string, std::shared_ptr<monte::Sampler>> const &samplers,
         std::shared_ptr<monte::Sampler> const &sample_weight,
         monte::RequestedPrecisionMap const &requested_precision,
         monte::CountType N_samples_for_equilibration,
         monte::CalcStatisticsFunction<statistics_type> calc_statistics_f)
          -> monte::ConvergenceCheckResults<statistics_type> {
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
           .. rubric:: Constructor

          Default constructor only
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
      .def_readwrite("min_clocktime", &monte::CutoffCheckParams::min_clocktime,
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
      .def_readwrite("max_clocktime", &monte::CutoffCheckParams::max_clocktime,
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

  py::class_<monte::CompletionCheckParams<statistics_type>>(
      m, "CompletionCheckParams",
      R"pbdoc(
      Parameters that determine if a simulation is complete

      CompletionCheckParams allow:

      - setting the requested precision for convergence of sampled data
      - setting cutoff parameters, forcing the simulation to keep running to meet certain
        minimums (number of steps or passes, number of samples, amount of simulated time
        or elapsed clocktime), or stop when certain maximums are met
      - controlling when completion checks are performed
      - customizing the method used to calculate statistics
      - customizing the method used to check for equilibration.

      )pbdoc")
      .def(py::init<>(&make_completion_check_params), R"pbdoc(

          .. rubric:: Constructor

          Parameters
          ----------
          requested_precision : Optional[:class:`~libcasm.monte.RequestedPrecisionMap`] = None
              Requested precision for convergence of sampler components. When all components
              reach the requested precision, and all `cutoff_params` minimums are met,
              then the completion check returns True, indicating the Monte Carlo simulation
              is complete.
          cutoff_params: Optional[:class:`~libcasm.monte.CutoffCheckParams`] = None,
              Cutoff check parameters allow setting limits on the Monte Carlo simulation to
              prevent calculations from stopping too soon or running too long. If None, no
              cutoffs are applied.
          calc_statistics_f: Optional[Callable] = None,
              A function for calculating :class:`~libcasm.monte.BasicStatistics` from
              sampled data, with signature:

              .. code-block:: Python

                  def calc_statistics_f(
                      observations: np.ndarray,
                      sample_weight: np.ndarray,
                  ) -> libcasm.monte.BasicStatistics:
                      ...

              If None, the default is :class:`~libcasm.monte.BasicStatisticsCalculator`.
          equilibration_check_f: Optional[Callable] = None,
              A function for checking equilibration of sampled data, with signature:

              .. code-block:: Python

                  def equilibration_check_f(
                      observations: np.ndarray,
                      sample_weight: np.ndarray,
                      requested_precision: libcasm.monte.RequestedPrecision,
                  ) -> libcasm.monte.IndividualEquilibrationResult:
                      ...

              If None, the default is :class:`~libcasm.monte.default_equilibration_check`.
          log_spacing: bool = False
              If True, use logarithmic spacing for completion checking; else use linear
              spacing. For linear spacing, the n-th check will be taken when:

              .. code-block:: Python

                  sample = check_begin + check_period * n

              For logarithmic spacing, the n-th check will be taken when:

              .. code-block:: Python

                  sample = check_begin + round( check_base ** (n + check_shift) )

              However, if sample(n) - sample(n-1) > `check_period_max`, then subsequent
              samples are taken every `check_period_max` samples.

              For linear spacing, the default is to check for completion after `100`,
              `200`, `300`, etc. samples are taken.

              For log spacing, the default is to check for completion after `100`,
              `1000`, `10000`, `20000`, `30000`, etc. samples are taken (note the
              effect of the default ``check_period_max=10000``).

              The default value is False, for linear spacing.
          check_begin:  Optional[int] = None
              The earliest sample to begin completion checking. Default is 100 for linear
              spacing and 0 for log spacing.
          check_period:  Optional[int] = None
              The linear completion checking period. Default is 100.
          check_base: Optional[float] = None
              The logarithmic completion checking base. Default is 10.
          check_shift: Optional[float] = None
              The shift for the logarithmic spacing exponent. Default is 2.
          check_period_max: Optional[int] = None
              The maximum check spacing for logarithmic check spacing. Default is 10000.

          )pbdoc",
           py::arg("requested_precision") = std::nullopt,
           py::arg("cutoff_params") = std::nullopt,
           py::arg("calc_statistics_f") = nullptr,
           py::arg("equilibration_check_f") = nullptr,
           py::arg("log_spacing") = false,
           py::arg("check_begin") = std::nullopt,
           py::arg("check_period") = std::nullopt,
           py::arg("check_base") = std::nullopt,
           py::arg("check_shift") = std::nullopt,
           py::arg("check_period_max") = std::nullopt)
      .def_readwrite(
          "cutoff_params",
          &monte::CompletionCheckParams<statistics_type>::cutoff_params,
          R"pbdoc(
          :class:`~libcasm.monte.CutoffCheckParams`: Cutoff check parameters
          )pbdoc")
      .def_readwrite(
          "equilibration_check_f",
          &monte::CompletionCheckParams<statistics_type>::equilibration_check_f,
          R"pbdoc(
          function: Function that performs equilibration checking.

          A function, such as :func:`~libcasm.monte.default_equilibration_check`, with
          signature f(array_like observations, array_like sample_weight,
          :class:`~libcasm.monte.RequestedPrecision` requested_precision) ->
          :class:`~libcasm.monte.IndividualEquilibrationResult`.
          )pbdoc")
      .def_readwrite(
          "calc_statistics_f",
          &monte::CompletionCheckParams<statistics_type>::calc_statistics_f,
          R"pbdoc(
          function: Function to calculate statistics.

          A function, such as an instance of
          :class:`~libcasm.monte.BasicStatisticsCalculator`, with signature
          f(array_like observations, array_like sample_weight) ->
          :class:`~libcasm.monte.BasicStatistics`.
          )pbdoc")
      .def_readwrite(
          "requested_precision",
          &monte::CompletionCheckParams<statistics_type>::requested_precision,
          R"pbdoc(
          :class:`~libcasm.monte.RequestedPrecisionMap`: Requested precision for \
          convergence of sampler components.

          A Dict[:class:`~libcasm.monte.SamplerComponent`,
          :class:`~libcasm.monte.RequestedPrecision`]-like object that specifies
          convergence criteria.
          )pbdoc")
      .def_readwrite(
          "log_spacing",
          &monte::CompletionCheckParams<statistics_type>::log_spacing,
          R"pbdoc(
          bool: If True, use logirithmic spacing for completiong checking; else \
          use linear spacing.

          The default value is False, for linear spacing between completion checks.
          For linear spacing, the n-th check will be taken when:

          .. code-block:: Python

              sample = check_begin + check_period * n

          For log spacing, the n-th check will be taken when:

          .. code-block:: Python

              sample = check_begin + round( check_base ** (n + check_shift) )

          However, if sample(n) - sample(n-1) > `check_period_max`, then subsequent
          samples are taken every `check_period_max` samples.

          For linear spacing, the default is to check for completion after `100`,
          `200`, `300`, etc. samples are taken.

          For log spacing, the default is to check for completion after `100`,
          `1000`, `10000`, `20000`, `30000`, etc. samples are taken.

          )pbdoc")
      .def_readwrite(
          "check_begin",
          &monte::CompletionCheckParams<statistics_type>::check_begin,
          R"pbdoc(
          int: Earliest number of samples to begin completion checking. Default=``100``.
          )pbdoc")
      .def_readwrite(
          "check_period",
          &monte::CompletionCheckParams<statistics_type>::check_period,
          R"pbdoc(
          int: The linear completion checking period. Default=``100``.
          )pbdoc")
      .def_readwrite("check_base",
                     &monte::CompletionCheckParams<statistics_type>::check_base,
                     R"pbdoc(
          float: The logarithmic completion checking base. Default=``10``.
          )pbdoc")
      .def_readwrite(
          "check_shift",
          &monte::CompletionCheckParams<statistics_type>::check_shift,
          R"pbdoc(
          float: The shift for the logarithmic spacing exponent. Default=``2``.
          )pbdoc")
      .def_readwrite(
          "check_period_max",
          &monte::CompletionCheckParams<statistics_type>::check_period_max,
          R"pbdoc(
          float: The maximum check spacing for logarithmic check spacing. \
          Default=``10000``.
          )pbdoc")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             monte::StateSamplingFunctionMap const &sampling_functions) {
            jsonParser json{data};
            InputParser<monte::CompletionCheckParams<statistics_type>> parser(
                json, sampling_functions);
            std::runtime_error error_if_invalid{
                "Error in libcasm.monte.CompletionCheckParams.from_dict"};
            report_and_throw_if_invalid(parser, CASM::log(), error_if_invalid);
            return std::move(*parser.value);
          },
          "Construct a CompletionCheckParams from a Python dict.",
          py::arg("data"), py::arg("sampling_functions"));

  py::class_<monte::CompletionCheckResults<statistics_type>>(
      m, "CompletionCheckResults",
      R"pbdoc(
      Results of completion checks
      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          .. rubric:: Constructor

          Default constructor only.
          )pbdoc")
      .def_readwrite("params",
                     &monte::CompletionCheckResults<statistics_type>::params,
                     R"pbdoc(
          :class:`~libcasm.monte.CompletionCheckParams`: Completion check parameters
          )pbdoc")
      .def_readwrite("count",
                     &monte::CompletionCheckResults<statistics_type>::count,
                     R"pbdoc(
                     Optional[int]: Number of steps or passes
                     )pbdoc")
      .def_readwrite("time",
                     &monte::CompletionCheckResults<statistics_type>::time,
                     R"pbdoc(
          Optional[int]: Simulated time
          )pbdoc")
      .def_readwrite("clocktime",
                     &monte::CompletionCheckResults<statistics_type>::clocktime,
                     R"pbdoc(
          float: Elapsed clock time
          )pbdoc")
      .def_readwrite("n_samples",
                     &monte::CompletionCheckResults<statistics_type>::n_samples,
                     R"pbdoc(
          int: Number of samples taken
          )pbdoc")
      .def_readwrite(
          "has_all_minimums_met",
          &monte::CompletionCheckResults<statistics_type>::has_all_minimums_met,
          R"pbdoc(
          bool: True if all cutoff check minimums have been met
          )pbdoc")
      .def_readwrite(
          "has_any_maximum_met",
          &monte::CompletionCheckResults<statistics_type>::has_any_maximum_met,
          R"pbdoc(
          bool: True if any cutoff check maximums have been met
          )pbdoc")
      .def_readwrite("n_samples_at_convergence_check",
                     &monte::CompletionCheckResults<
                         statistics_type>::n_samples_at_convergence_check,
                     R"pbdoc(
          Optional[int]: Number of samples when the converence check was performed
          )pbdoc")
      .def_readwrite("equilibration_check_results",
                     &monte::CompletionCheckResults<
                         statistics_type>::equilibration_check_results,
                     R"pbdoc(
          :class:`~libcasm.monte.EquilibrationCheckResults`: Results of equilibration check
          )pbdoc")
      .def_readwrite("convergence_check_results",
                     &monte::CompletionCheckResults<
                         statistics_type>::convergence_check_results,
                     R"pbdoc(
          :class:`~libcasm.monte.ConvergenceCheckResults`: Results of convergence check
          )pbdoc")
      .def_readwrite(
          "is_complete",
          &monte::CompletionCheckResults<statistics_type>::is_complete,
          R"pbdoc(
          bool: Outcome of the completion check
          )pbdoc")
      .def("partial_reset",
           &monte::CompletionCheckResults<statistics_type>::partial_reset,
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
           &monte::CompletionCheckResults<statistics_type>::full_reset,
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
          [](monte::CompletionCheckResults<statistics_type> const &x) {
            jsonParser json;
            to_json(x, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the CompletionCheckResults as a Python dict.");

  py::class_<monte::CompletionCheck<statistics_type>>(m, "CompletionCheck",
                                                      R"pbdoc(
      Implements completion checks
      )pbdoc")
      .def(py::init<monte::CompletionCheckParams<statistics_type>>(),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          params : :class:`~libcasm.monte.CompletionCheckParams`
              Data struture holding completion check parameters.
          )pbdoc")
      .def("reset", &monte::CompletionCheck<statistics_type>::reset,
           R"pbdoc(
          Reset CompletionCheck for next run
          )pbdoc")
      .def("params", &monte::CompletionCheck<statistics_type>::params,
           R"pbdoc(
          Get CompletionCheckParams
          )pbdoc")
      .def("results", &monte::CompletionCheck<statistics_type>::results,
           R"pbdoc(
          Get detailed results of the last check
          )pbdoc")
      .def(
          "count_check",
          [](monte::CompletionCheck<statistics_type> &x,
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
              Sample weights associated with observations. May have 0 samples, in which
              case the obsservations are treated as being equally weighted, otherwise
              must match the number of samples made by each sampler in `samplers`.
          count : int
              Number of steps or passes
          method_log : :class:`~libcasm.monte.MethodLog`
              The method log specifies where to write status updates and internally
              tracks the elapsed clock time.

          Returns
          -------
          is_complete : bool
              True if complete, False otherwise
          )pbdoc",
          py::arg("samplers"), py::arg("sample_weight"), py::arg("count"),
          py::arg("method_log"))
      .def(
          "__call__",
          [](monte::CompletionCheck<statistics_type> &x,
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
              Sample weights associated with observations. May have 0 samples, in which
              case the obsservations are treated as being equally weighted, otherwise
              must match the number of samples made by each sampler in `samplers`.
          count : Optional[int]
              Number of steps or passes
          time : Optional[float]
              Simulated time
          method_log : :class:`~libcasm.monte.MethodLog`
              The method log specifies where to write status updates and internally
              tracks the elapsed clock time.

          Returns
          -------
          is_complete : bool
              True if complete, False otherwise
          )pbdoc",
          py::arg("samplers"), py::arg("sample_weight"), py::arg("count"),
          py::arg("time"), py::arg("method_log"))
      .def(
          "count_and_time_check",
          [](monte::CompletionCheck<statistics_type> &x,
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
              Sample weights associated with observations. May have 0 samples, in which
              case the obsservations are treated as being equally weighted, otherwise
              must match the number of samples made by each sampler in `samplers`.
          count : int
              Number of steps or passes
          time : float
              Simulated time
          method_log : :class:`~libcasm.monte.MethodLog`
              The method log specifies where to write status updates and internally
              tracks the elapsed clock time.

          Returns
          -------
          is_complete : bool
              True if complete, False otherwise
          )pbdoc",
          py::arg("samplers"), py::arg("sample_weight"), py::arg("count"),
          py::arg("time"), py::arg("method_log"))
      .def(
          "time_check",
          [](monte::CompletionCheck<statistics_type> &x,
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
              Sample weights associated with observations. May have 0 samples, in which
              case the observations are treated as being equally weighted, otherwise
              must match the number of samples made by each sampler in `samplers`.
          time : float
              Simulated time
          method_log : :class:`~libcasm.monte.MethodLog`
              The method log specifies where to write status updates and internally
              tracks the elapsed clock time.

          Returns
          -------
          is_complete : bool
              True if complete, False otherwise
          )pbdoc",
          py::arg("samplers"), py::arg("sample_weight"), py::arg("time"),
          py::arg("method_log"))
      .def(
          "check",
          [](monte::CompletionCheck<statistics_type> &x,
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
              Sample weights associated with observations. May have 0 samples, in which
              case the obsservations are treated as being equally weighted, otherwise
              must match the number of samples made by each sampler in `samplers`.
          method_log : :class:`~libcasm.monte.MethodLog`
              The method log specifies where to write status updates and internally
              tracks the elapsed clock time.

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
