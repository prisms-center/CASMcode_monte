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
#include "casm/monte/checks/EquilibrationCheck.hh"
#include "casm/monte/checks/io/json/CompletionCheck_json_io.hh"
#include "casm/monte/checks/io/json/ConvergenceCheck_json_io.hh"
#include "casm/monte/checks/io/json/CutoffCheck_json_io.hh"
#include "casm/monte/checks/io/json/EquilibrationCheck_json_io.hh"
#include "casm/monte/definitions.hh"
#include "casm/monte/io/json/ValueMap_json_io.hh"
#include "casm/monte/sampling/HistogramFunction.hh"
#include "casm/monte/sampling/Sampler.hh"
#include "casm/monte/sampling/SamplingParams.hh"
#include "casm/monte/sampling/SelectedEventData.hh"
#include "casm/monte/sampling/StateSamplingFunction.hh"
#include "casm/monte/sampling/io/json/Sampler_json_io.hh"
#include "casm/monte/sampling/io/json/SamplingParams_json_io.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

typedef monte::BasicStatistics statistics_type;

monte::SamplingParams make_sampling_params(
    std::vector<std::string> sampler_names,
    std::vector<std::string> json_sampler_names, monte::SAMPLE_MODE sample_mode,
    monte::SAMPLE_METHOD sample_method, double period,
    std::optional<double> begin, double base, double shift,
    std::optional<std::function<double(monte::CountType)>> custom_sample_at,
    bool stochastic_sample_period, bool do_sample_trajectory,
    bool do_sample_time) {
  if (!begin.has_value()) {
    if (sample_method == monte::SAMPLE_METHOD::LINEAR) {
      begin = period;
    } else if (sample_method == monte::SAMPLE_METHOD::LOG) {
      begin = 0.0;
    } else if (sample_method == monte::SAMPLE_METHOD::CUSTOM) {
      begin = 0.0;
    } else {
      throw std::runtime_error(
          "Error in make_sampling_params: Invalid sample_method");
    }
  }
  if (sample_method == monte::SAMPLE_METHOD::CUSTOM &&
      !custom_sample_at.has_value()) {
    throw std::runtime_error(
        "Error in make_sampling_params: "
        "sample_method==SAMPLE_METHOD::CUSTOM and "
        "!custom_sample_at.has_value()");
  }
  monte::SamplingParams s;
  s.sampler_names = sampler_names;
  s.json_sampler_names = json_sampler_names;
  s.sample_mode = sample_mode;
  s.sample_method = sample_method;
  s.period = period;
  s.begin = begin.value();
  s.base = base;
  s.shift = shift;
  if (custom_sample_at.has_value()) {
    s.custom_sample_at = custom_sample_at.value();
  }
  s.stochastic_sample_period = stochastic_sample_period;
  s.do_sample_trajectory = do_sample_trajectory;
  s.do_sample_time = do_sample_time;
  return s;
}

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

monte::HistogramFunction<Eigen::VectorXi> make_vector_int_histogram_function(
    std::string name, std::string description, std::vector<Index> shape,
    std::function<Eigen::VectorXi()> function,
    std::optional<std::vector<std::string>> component_names, Index max_size) {
  if (function == nullptr) {
    throw std::runtime_error(
        "Error constructing VectorIntHistogramFunction: function == nullptr");
  }
  if (!component_names.has_value()) {
    return monte::HistogramFunction<Eigen::VectorXi>(name, description, shape,
                                                     function, max_size);
  } else {
    return monte::HistogramFunction<Eigen::VectorXi>(
        name, description, *component_names, shape, function, max_size);
  }
}

monte::HistogramFunction<Eigen::VectorXd> make_vector_float_histogram_function(
    std::string name, std::string description, std::vector<Index> shape,
    std::function<Eigen::VectorXd()> function,
    std::optional<std::vector<std::string>> component_names, Index max_size,
    double tol) {
  if (function == nullptr) {
    throw std::runtime_error(
        "Error constructing VectorFloatHistogramFunction: function == nullptr");
  }
  if (!component_names.has_value()) {
    return monte::HistogramFunction<Eigen::VectorXd>(name, description, shape,
                                                     function, max_size, tol);
  } else {
    return monte::HistogramFunction<Eigen::VectorXd>(
        name, description, *component_names, shape, function, max_size, tol);
  }
}

monte::PartitionedHistogramFunction<double> make_partitioned_histogram_function(
    std::string name, std::string description, std::function<double()> function,
    std::vector<std::string> partition_names,
    std::function<int()> get_partition, bool is_log, double initial_begin,
    double bin_width, Index max_size) {
  if (function == nullptr) {
    throw std::runtime_error(
        "Error constructing PartitionedHistogramFunction: function == nullptr");
  }
  if (get_partition == nullptr) {
    throw std::runtime_error(
        "Error constructing PartitionedHistogramFunction: "
        "get_partition_function == nullptr");
  }
  return monte::PartitionedHistogramFunction<double>(
      name, description, function, partition_names, get_partition, is_log,
      initial_begin, bin_width, max_size);
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
    calc_statistics_f = monte::default_statistics_calculator<statistics_type>();
  }
  if (!requested_precision.has_value()) {
    requested_precision = monte::RequestedPrecisionMap();
  }

  if (!log_spacing) {
    if (!check_period.has_value()) {
      check_period = 100;
    }
    result.check_begin = check_period.value();
    result.check_period = check_period.value();
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

monte::CutoffCheckParams make_cutoff_check_params(
    std::optional<monte::CountType> min_count,
    std::optional<monte::CountType> max_count,
    std::optional<monte::TimeType> min_time,
    std::optional<monte::TimeType> max_time,
    std::optional<monte::CountType> min_sample,
    std::optional<monte::CountType> max_sample,
    std::optional<monte::TimeType> min_clocktime,
    std::optional<monte::TimeType> max_clocktime) {
  monte::CutoffCheckParams params;
  params.min_count = min_count;
  params.max_count = max_count;
  params.min_time = min_time;
  params.max_time = max_time;
  params.min_sample = min_sample;
  params.max_sample = max_sample;
  params.min_clocktime = min_clocktime;
  params.max_clocktime = max_clocktime;
  return params;
}

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
// #include "opaque_types.cc"
PYBIND11_MAKE_OPAQUE(CASM::monte::SamplerMap);
PYBIND11_MAKE_OPAQUE(CASM::monte::jsonSamplerMap);
PYBIND11_MAKE_OPAQUE(CASM::monte::StateSamplingFunctionMap);
PYBIND11_MAKE_OPAQUE(CASM::monte::jsonStateSamplingFunctionMap);
PYBIND11_MAKE_OPAQUE(
    std::map<std::string, CASM::monte::HistogramFunction<Eigen::VectorXi>>);
PYBIND11_MAKE_OPAQUE(
    std::map<std::string, CASM::monte::HistogramFunction<Eigen::VectorXd>>);
PYBIND11_MAKE_OPAQUE(
    std::map<std::string, CASM::monte::PartitionedHistogramFunction<double>>);
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
    SamplerMap stores :class:`~libcasm.monte.sampling.Sampler` by name of the sampled quantity

    Notes
    -----
    SamplerMap is a Dict[str, :class:`~libcasm.monte.sampling.Sampler`]-like object.
    )pbdoc",
                                  py::module_local(false));

  py::bind_map<monte::jsonSamplerMap>(m, "jsonSamplerMap",
                                      R"pbdoc(
    SamplerMap stores :class:`~libcasm.monte.sampling.jsonSampler` by name of the sampled quantity

    Notes
    -----
    jsonSamplerMap is a Dict[str, :class:`~libcasm.monte.sampling.jsonSampler`]-like object.
    )pbdoc",
                                      py::module_local(false));

  py::bind_map<monte::StateSamplingFunctionMap>(m, "StateSamplingFunctionMap",
                                                R"pbdoc(
    StateSamplingFunctionMap stores :class:`~libcasm.monte.sampling.StateSamplingFunction` by name of the sampled quantity.

    Notes
    -----
    StateSamplingFunctionMap is a Dict[str, :class:`~libcasm.monte.sampling.StateSamplingFunction`]-like object.
    )pbdoc",
                                                py::module_local(false));

  py::bind_map<monte::jsonStateSamplingFunctionMap>(
      m, "jsonStateSamplingFunctionMap",
      R"pbdoc(
    jsonStateSamplingFunctionMap stores :class:`~libcasm.monte.sampling.jsonStateSamplingFunction` by name of the sampled quantity.

    Notes
    -----
    jsonStateSamplingFunctionMap is a Dict[str, :class:`~libcasm.monte.sampling.jsonStateSamplingFunction`]-like object.
    )pbdoc",
      py::module_local(false));

  py::bind_map<monte::RequestedPrecisionMap>(m, "RequestedPrecisionMap",
                                             R"pbdoc(
    RequestedPrecisionMap stores :class:`~libcasm.monte.sampling.RequestedPrecision` with :class:`~libcasm.monte.sampling.SamplerComponent` keys.

    Notes
    -----
    RequestedPrecisionMap is a Dict[:class:`~libcasm.monte.sampling.SamplerComponent`, :class:`~libcasm.monte.sampling.RequestedPrecision`]-like object.
    )pbdoc",
                                             py::module_local(false));

  py::bind_map<monte::EquilibrationResultMap>(m, "EquilibrationResultMap",
                                              R"pbdoc(
    EquilibrationResultMap stores :class:`~libcasm.monte.sampling.IndividualEquilibrationResult` by :class:`~libcasm.monte.sampling.SamplerComponent`

    Notes
    -----
    EquilibrationResultMap is a Dict[:class:`~libcasm.monte.sampling.SamplerComponent`, :class:`~libcasm.monte.sampling.IndividualEquilibrationResult`]-like object.
    )pbdoc",
                                              py::module_local(false));

  py::bind_map<monte::ConvergenceResultMap<statistics_type>>(
      m, "ConvergenceResultMap",
      R"pbdoc(
    ConvergenceResultMap stores :class:`~libcasm.monte.sampling.IndividualConvergenceResult` by :class:`~libcasm.monte.sampling.SamplerComponent`

    Notes
    -----
    ConvergenceResultMap is a Dict[:class:`~libcasm.monte.sampling.SamplerComponent`, :class:`~libcasm.monte.sampling.IndividualConvergenceResult`]-like object.
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

              from libcasm.monte.sampling import SAMPLE_MODE
              sample_mode = SAMPLE_MODE.BY_PASS


          )pbdoc")
      .value("BY_STEP", monte::SAMPLE_MODE::BY_STEP,
             R"pbdoc(
          Sample by Monte Carlo step (i.e. one Metropolis proposal or one KMC event):

          .. code-block:: Python

              from libcasm.monte.sampling import SAMPLE_MODE
              sample_mode = libcasm.monte.sampling.SAMPLE_MODE.BY_PASS


          )pbdoc")
      .value("BY_TIME", monte::SAMPLE_MODE::BY_TIME,
             R"pbdoc(
          Sample by Monte Carlo time:

          .. code-block:: Python

              from libcasm.monte.sampling import SAMPLE_MODE
              sample_mode = SAMPLE_MODE.BY_TIME

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

              from libcasm.monte.sampling import SAMPLE_METHOD
              sample_method = SAMPLE_METHOD.LINEAR

          )pbdoc")
      .value("LOG", monte::SAMPLE_METHOD::LOG,
             R"pbdoc(
          Logarithmically spaced samples:

          .. code-block:: Python

              from libcasm.monte.sampling import SAMPLE_METHOD
              sample_method = libcasm.monte.SAMPLE_METHOD.LOG

          )pbdoc")
      .value("CUSTOM", monte::SAMPLE_METHOD::CUSTOM,
             R"pbdoc(
          Use a custom function to specify sample spacing:

          .. code-block:: Python

              from libcasm.monte.sampling import SAMPLE_METHOD
              sample_method = libcasm.monte.SAMPLE_METHOD.CUSTOM

          )pbdoc")
      .export_values();

  py::class_<monte::SamplingParams>(m, "SamplingParams", R"pbdoc(
      Parameters controlling sampling fixtures
      )pbdoc")
      .def(py::init<>(&make_sampling_params),
           R"pbdoc(

          .. rubric:: Sample mode

          Sampling can be requested in three modes: by Monte Carlo step, pass,
          or simulated time (if applicable). A Monte Carlo pass is defined
          as performing a number Monte Carlo steps equal to the number of
          supercell sites with degrees of freedom. It is called sampling by
          count if the sampling is performed by step or pass, and sampling by
          time if the sampling is performed by simulated time.

          .. rubric:: Sample spacing

          Sample spacing can be either linear or logarithmic. Generally linear
          sampling is used for calculating statistics, but logarithmic sampling
          can be useful for understanding system dynamics, especially in kinetic
          Monte Carlo calculations.

          For linear sample spacing, the n-th sample (n=0,1,2,...) is taken
          when the step/pass count or simulated time is equal to:

          .. code-block:: Python

              count = round( begin + period * n )

              time = begin + period * n

          For logarithmic sample spacing, the n-th sample is taken when:

          .. code-block:: Python

              count = round( begin + base ** ( n + shift)

              time = begin + base ** ( n + shift) )


          .. rubric:: Custom sample spacing

          It is also possible to provide a custom sample spacing function,
          using :py:attr:`SAMPLE_METHOD.CUSTOM`. A function must be provided
          which returns a non-decreasing series of values indicating the
          count/time at which a sample should be taken. It must have the
          signature:

          .. code-block:: Python

              def custom_sample_at(n: int) -> float:
                  ...

          where `n` is the sample index (n=0,1,2,...).

          .. rubric:: Stochastic sample spacing

          Occasionally it is useful to take samples at stochastically spaced
          intervals instead of deterministic intervals, for instance to
          understand systems with highly correlated events.

          If ``stochastic_sample_period == true``, then instead of setting the
          sample count/time deterministically, samples are taken
          probabilistically based on mean sample rate (samples per count/time).
          For linear sample spacing, the mean sample rate is calculated as
          :math:`1/p`, where :math:`p` is the sample `period`. For logarithmic
          sample spacing the mean sample rate is calculated as
          :math:`1/\left(ln(b)b^{n+s}\right)`, where `b` is the `base`
          and `s` is the `shift`. For custom sample spacing, the mean
          sample rate is calculated as :math:`1/\left(f(n+1) - f(n)\right)`,
          where :math:`f(n)` is the custom sample count/time for the n-th
          sample.

          For sampling by count: If the mean sample rate is 1.0 or
          greater, a sample is taken every step or pass. If the mean sample
          rate is less than 1.0, a sample is taken at each step/pass
          with probability equal to the mean sample rate.

          For sampling by time: The time until the next sample is calculated
          as a Poisson process with mean rate :math:`r` using :math:`-ln(R)/r`,
          where :math:`R` is a random number in :math:`[0, 1.0)`.

          .. rubric:: Constructor

          Parameters
          ----------
          sampler_names: list[str] = []
              List of sampling functions to call when a sample is taken.
          json_sampler_names: list[str] = []
              List of JSON sampling functions to call when a sample is taken.
          sample_mode: :class:`SAMPLE_MODE` = :py:attr:`SAMPLE_MODE.BY_PASS`
              Sample by pass, step, or time.
          sample_method: :class:`SAMPLE_METHOD` = :py:attr:`SAMPLE_METHOD.LINEAR`
              Sample with linear, logarithmic, or custom spacing.
          period: float = 1.0
              Sample spacing parameter.
          begin: Optional[float] = None
              Sample spacing parameter. If None, uses `period` with
              :py:attr:`SAMPLE_METHOD.LINEAR`, and uses 0.0 with
              :py:attr:`SAMPLE_METHOD.LOG`.
          base: float = math.pow(10.0, 1.0/10.0)
              Base for log sample spacing.
          shift: float = 1.0
              Log sample spacing parameter.
          custom_sample_at: Optional[Callable] = None
              A custom sample spacing function, which must have the signature
              ``def custom_sample_at(n: int) -> float``, used with
              :py:attr:`SAMPLE_METHOD.CUSTOM`.
          stochastic_sample_period: bool = False
              If true, the sample period is stochastically chosen based on
              the mean sample rate.
          do_sample_trajectory: bool = False
              If true, save the configuration when a sample is taken.
          do_sample_time: bool = False
              If true, save current time when taking a sample, if applicable.

          )pbdoc",
           py::arg("sampler_names") = std::vector<std::string>(),
           py::arg("json_sampler_names") = std::vector<std::string>(),
           py::arg("sample_mode") = monte::SAMPLE_MODE::BY_PASS,
           py::arg("sample_method") = monte::SAMPLE_METHOD::LINEAR,
           py::arg("period") = 1.0, py::arg("begin") = std::nullopt,
           py::arg("base") = std::pow(10.0, 1.0 / 10.0), py::arg("shift") = 0.0,
           py::arg("custom_sample_at") = std::nullopt,
           py::arg("stochastic_sample_period") = false,
           py::arg("do_sample_trajectory") = false,
           py::arg("do_sample_time") = false)
      .def_readwrite("sample_mode", &monte::SamplingParams::sample_mode,
                     R"pbdoc(
          SAMPLE_MODE: Sample by pass, step, or time.

          The default value is :py:attr:`SAMPLE_MODE.BY_PASS`.
          )pbdoc")
      .def_readwrite("sample_method", &monte::SamplingParams::sample_mode,
                     R"pbdoc(
          SAMPLE_METHOD: Sample with linear, logarithmic, or custom spacing.

          The default value is :py:attr:`SAMPLE_METHOD.LINEAR`.
          )pbdoc")
      .def_readwrite("begin", &monte::SamplingParams::begin, R"pbdoc(
          float: Sample spacing `begin` parameter.
          )pbdoc")
      .def_readwrite("period", &monte::SamplingParams::period, R"pbdoc(
          float: Sample spacing `period` parameter.
          )pbdoc")
      .def_readwrite("base", &monte::SamplingParams::base,
                     R"pbdoc(
          float: Sample spacing `base` parameter (logarithmic sample spacing only).
          )pbdoc")
      .def_readwrite("shift", &monte::SamplingParams::shift, R"pbdoc(
          float: Sample spacing `shift` parameter (logarithmic sample spacing only).
          )pbdoc")
      .def_readwrite("sampler_names", &monte::SamplingParams::sampler_names,
                     R"pbdoc(
          List[str]: Get or set (as a copy) the names of quantities to sample
          (i.e. sampling function names).

          Note that this is a property that either (i) gets a copy of the
          list, or (ii) sets the entire list. Doing `x.sampler_names.append(y)`
          or `x.sampler_names += [y, z]` will not modify the SamplingParams
          object. Instead, use `x.append_to_sampler_names(y)`,
          `x.remove_from_sampler_names(y)`, or `x.extend_sampler_names([y, z])`.
          )pbdoc")
      .def(
          "append_to_sampler_names",
          [](monte::SamplingParams &s, std::string name) {
            s.sampler_names.push_back(name);
          },
          R"pbdoc(
          Append a name to `sampler_names`.
          )pbdoc",
          py::arg("name"))
      .def(
          "remove_from_sampler_names",
          [](monte::SamplingParams &s, std::string name) {
            if (auto it = std::find(s.sampler_names.begin(),
                                    s.sampler_names.end(), name);
                it != s.sampler_names.end()) {
              s.sampler_names.erase(it);
            }
          },
          R"pbdoc(
          Remove a name from `sampler_names`.
          )pbdoc",
          py::arg("name"))
      .def(
          "extend_sampler_names",
          [](monte::SamplingParams &s, std::vector<std::string> names) {
            s.sampler_names.insert(s.sampler_names.end(), names.begin(),
                                   names.end());
          },
          R"pbdoc(
          Append multiple names to `sampler_names`.
          )pbdoc",
          py::arg("names"))
      .def_readwrite("json_sampler_names",
                     &monte::SamplingParams::json_sampler_names,
                     R"pbdoc(
          List[str]: Get or set (as a copy) the names of JSON quantities to sample
          (i.e. json sampling function names).

          Note that this is a property that either (i) gets a copy of the
          list, or (ii) sets the entire list. Doing `x.json_sampler_names.append(y)`
          or `x.json_sampler_names += [y, z]` will not modify the SamplingParams
          object. Instead, use `x.append_to_json_sampler_names(y)`,
          `x.remove_from_json_sampler_names(y)`, or
          `x.extend_json_sampler_names([y, z])`.
          )pbdoc")
      .def(
          "append_to_json_sampler_names",
          [](monte::SamplingParams &s, std::string name) {
            s.json_sampler_names.push_back(name);
          },
          R"pbdoc(
          Append a name to `json_sampler_names`.
          )pbdoc",
          py::arg("name"))
      .def(
          "remove_from_json_sampler_names",
          [](monte::SamplingParams &s, std::string name) {
            if (auto it = std::find(s.json_sampler_names.begin(),
                                    s.json_sampler_names.end(), name);
                it != s.json_sampler_names.end()) {
              s.json_sampler_names.erase(it);
            }
          },
          R"pbdoc(
          Remove a name from `json_sampler_names`.
          )pbdoc",
          py::arg("name"))
      .def(
          "extend_json_sampler_names",
          [](monte::SamplingParams &s, std::vector<std::string> names) {
            s.json_sampler_names.insert(s.json_sampler_names.end(),
                                        names.begin(), names.end());
          },
          R"pbdoc(
          Append multiple names to `json_sampler_names`.
          )pbdoc",
          py::arg("names"))
      .def_readwrite("stochastic_sample_period",
                     &monte::SamplingParams::stochastic_sample_period, R"pbdoc(
          bool: If true, the sample period is stochastically chosen based on \
          the mean sample rate.
          )pbdoc")
      .def_readwrite("do_sample_trajectory",
                     &monte::SamplingParams::do_sample_trajectory, R"pbdoc(
            bool: If true, save the configuration when a sample is taken.
          )pbdoc")
      .def_readwrite("do_sample_time", &monte::SamplingParams::do_sample_time,
                     R"pbdoc(
            bool: If true, save current time when taking a sample, if \
            applicable.
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
      - :class:`~libcasm.monte.sampling.Sampler` helps sampling by re-sizing the underlying matrix holding data automatically, and it allows accessing particular observations as an unrolled vector or accessing a particular component as a vector to check convergence.
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

  py::class_<monte::jsonSampler, std::shared_ptr<monte::jsonSampler>>(
      m, "jsonSampler",
      R"pbdoc(
      Sampler stores sampled dict-like data for JSON output

      Notes
      -----
      - jsonSampler can be used to sample quantities that do not need to be
        checked for convergence directly and are not well represented by an array.

      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(

          .. rubric:: Constructor

          Default constructor only.

          )pbdoc")
      .def(
          "append",
          [](monte::jsonSampler &s, const nlohmann::json &data) {
            s.values.push_back(jsonParser(data));
          },
          R"pbdoc(
          Add a new sample.
          )pbdoc",
          py::arg("data"))
      .def(
          "set_values",
          [](monte::jsonSampler &s, std::vector<nlohmann::json> const &data) {
            s.values.clear();
            for (auto const &value : data) {
              s.values.push_back(jsonParser(value));
            }
          },
          R"pbdoc(
           Set all values directly
           )pbdoc",
          py::arg("values"))
      .def(
          "clear", [](monte::jsonSampler &s) { s.values.clear(); },
          R"pbdoc(
          Clear values.
          )pbdoc")
      .def(
          "set_sample_capacity",
          [](monte::jsonSampler &s, monte::CountType sample_capacity) {
            s.values.reserve(sample_capacity);
          },
          R"pbdoc(
          Conservative resize, to increase capacity for more samples.
          )pbdoc",
          py::arg("sample_capacity"))
      .def(
          "n_samples", [](monte::jsonSampler const &s) { s.values.size(); },
          R"pbdoc(
            Current number of samples taken.
          )pbdoc")
      .def(
          "sample_capacity",
          [](monte::jsonSampler const &s) { s.values.capacity(); },
          R"pbdoc(
            Current sample capacity.
          )pbdoc")
      .def(
          "values",
          [](monte::jsonSampler const &s) -> std::vector<nlohmann::json> {
            std::vector<nlohmann::json> list;
            for (auto const &value : s.values) {
              list.push_back(value);
            }
            return list;
          },
          R"pbdoc(
          Get sampled values as a const reference.
          )pbdoc")
      .def(
          "sample",
          [](monte::jsonSampler const &s, monte::CountType sample_index)
              -> nlohmann::json const & { return s.values.at(sample_index); },
          py::return_value_policy::reference_internal,
          R"pbdoc(
          Get a sampled value as a const reference.
          )pbdoc",
          py::arg("sample_index"))
      .def("__len__",
           [](monte::jsonSampler const &s) { return s.values.size(); })
      .def(
          "__iter__",  // for x in occ_candidate_list
          [](monte::jsonSampler const &s) {
            return py::make_iterator(s.values.begin(), s.values.end());
          },
          py::keep_alive<
              0, 1>() /* Essential: keep object alive while iterator exists */)
      .def(
          "to_list",
          [](monte::jsonSampler const &s) -> std::vector<nlohmann::json> {
            std::vector<nlohmann::json> list;
            for (auto const &value : s.values) {
              list.push_back(value);
            }
            return list;
          },
          R"pbdoc(
          Represent the jsonSampler values as a list of dict

          Returns
          -------
          data : list[dict]
              The jsonSampler values as a list of dict
          )pbdoc");

  m.def("get_n_samples", monte::get_n_samples,
        R"pbdoc(
        Return the number of samples taken. Assumes the same value for all samplers in the :class:`~libcasm.monte.sampling.SamplerMap`.
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
              Name of the sampled quantity. Should match keys in a :class:`~libcasm.monte.sampling.SamplerMap`.
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

                      Should match keys in a :class:`~libcasm.monte.sampling.SamplerMap`.
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
        - Data sampled by a StateSamplingFunction can be stored in a :class:`~libcasm.monte.sampling.Sampler`.
        - A call operator exists (:func:`~libcasm.monte.StateSamplingFunction.__call__`) to call the function held by :class:`~libcasm.monte.sampling.StateSamplingFunction`.
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

          Equivalent to calling :py::attr:`~libcasm.monte.sampling.StateSamplingFunction.function`.
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
            json_samplers = jsonSamplerMap()
            for name, f in json_sampling_functions.items():
                json_samplers[name] = []

            # ... in Monte Carlo simulation ...
            # ... sample JSON data ...
            for name, f in json_sampling_functions.items():
                json_samplers[name].append(f())


        Notes
        -----
        - Typically this holds a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the current state of the simulation.
        - jsonStateSamplingFunction can be used to sample quantities not easily converted to scalar, vector, matrix, etc.
        - Data sampled by a jsonStateSamplingFunction can be stored in a :class:`~libcasm.monte.sampling.jsonSamplerMap`.
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

          Equivalent to calling :py::attr:`~libcasm.monte.sampling.jsonStateSamplingFunction.function`.
          )pbdoc");

  py::class_<monte::HistogramFunction<Eigen::VectorXi>>(
      m, "VectorIntHistogramFunction",
      R"pbdoc(
      A function that returns a integer vector value to add to a histogram

      Notes
      -----
      - Typically this holds a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the calculator's event data.
      - VectorIntHistogramFunction can be used to return quantities of any dimension (scalar, vector, matrix, etc.) by unrolling values. The standard approach is to use column-major order.
      - For sampling scalars, a size=1 vector is expected. This can be done with the function :func:`~libcasm.monte.scalar_as_vector`.
      - For sampling matrices, column-major order unrolling can be done with the function :func:`~libcasm.monte.matrix_as_vector`.
      - Data returned by a VectorIntHistogramFunction can be stored in a :class:`~libcasm.monte.sampling.DiscreteVectorIntHistogram`.
      - A call operator exists (:func:`~libcasm.monte.VectorIntHistogramFunction.__call__`) to call the function held by :class:`~libcasm.monte.sampling.VectorIntHistogramFunction`.
      )pbdoc")
      .def(py::init<>(&make_vector_int_histogram_function),
           R"pbdoc(

          .. rubric:: Constructor

          Parameters
          ----------
          name : str
              Name of the sampled quantity.

          description : str
              Description of the function.

          shape : List[int]
              Shape of quantity, with column-major unrolling

              Scalar: [], Vector: [n], Matrix: [m, n], etc.

          function : function
              A function with 0 arguments that returns an array of the proper size sampling the current state. Typically this is a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the current state of the simulation.

          component_names : Optional[List[str]] = None
              A name for each component of the resulting vector.

              Can be strings representing an indices (i.e "0", "1", "2", etc.) or can be a descriptive string (i.e. "Mg", "Va", "O", etc.). If None, indices for column-major ordering are used (i.e. "0,0", "1,0", ..., "m-1,n-1")

          max_size : int = 10000
              Maximum number of bins to create. If adding an additional data
              point would cause the number of bins to exceed `max_size`, the
              count / weight is instead added to the `out_of_range_count` of the
              :class:`~libcasm.monte.sampling.DiscreteVectorIntHistogram`.

          )pbdoc",
           py::arg("name"), py::arg("description"), py::arg("shape"),
           py::arg("function"), py::arg("component_names") = std::nullopt,
           py::arg("max_size") = 10000)
      .def_readwrite("name", &monte::HistogramFunction<Eigen::VectorXi>::name,
                     R"pbdoc(
          str : Name of the quantity.
          )pbdoc")
      .def_readwrite("description",
                     &monte::HistogramFunction<Eigen::VectorXi>::description,
                     R"pbdoc(
          str : Description of the function.
          )pbdoc")
      .def_readwrite("shape", &monte::HistogramFunction<Eigen::VectorXi>::shape,
                     R"pbdoc(
          List[int] : Shape of quantity, with column-major unrolling.

          Scalar: [], Vector: [n], Matrix: [m, n], etc.
          )pbdoc")
      .def_readwrite(
          "component_names",
          &monte::HistogramFunction<Eigen::VectorXi>::component_names,
          R"pbdoc(
          List[str] : A name for each component of the resulting vector.

          Can be strings representing an indices (i.e "0", "1", "2", etc.) or can be a descriptive string (i.e. "Mg", "Va", "O", etc.). If the sampled quantity is an unrolled matrix, indices for column-major ordering are typical (i.e. "0,0", "1,0", ..., "m-1,n-1").
          )pbdoc")
      .def_readwrite("function",
                     &monte::HistogramFunction<Eigen::VectorXi>::function,
                     R"pbdoc(
          function : The function to be evaluated.

          A function with 0 arguments that returns an array of the proper size sampling the current state. Typically this is a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the current state of the simulation.
          )pbdoc")
      .def_readwrite("max_size",
                     &monte::HistogramFunction<Eigen::VectorXi>::max_size,
                     R"pbdoc(
          int: Maximum number of bins in the histogram.
          )pbdoc")
      .def(
          "__call__",
          [](monte::HistogramFunction<Eigen::VectorXi> const &f)
              -> Eigen::VectorXi { return f(); },
          R"pbdoc(
          Evaluates the function

          Equivalent to calling :py::attr:`~libcasm.monte.sampling.VectorIntHistogramFunction.function`.
          )pbdoc");

  py::bind_map<
      std::map<std::string, monte::HistogramFunction<Eigen::VectorXi>>>(
      m, "VectorIntHistogramFunctionMap",
      R"pbdoc(
      VectorIntHistogramFunctionMap stores :class:`~libcasm.monte.sampling.VectorIntHistogramFunction` by name of the sampled quantity

      Notes
      -----
      VectorIntHistogramFunctionMap is a Dict[str, :class:`~libcasm.monte.sampling.VectorIntHistogramFunction`]-like object.
      )pbdoc",
      py::module_local(false));

  py::class_<monte::HistogramFunction<Eigen::VectorXd>>(
      m, "VectorFloatHistogramFunction",
      R"pbdoc(
      A function that returns a floating-point vector value to add to a histogram

      Notes
      -----
      - Typically this holds a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the calculator's event data.
      - VectorFloatHistogramFunction can be used to return quantities of any dimension (scalar, vector, matrix, etc.) by unrolling values. The standard approach is to use column-major order.
      - For sampling scalars, a size=1 vector is expected. This can be done with the function :func:`~libcasm.monte.scalar_as_vector`.
      - For sampling matrices, column-major order unrolling can be done with the function :func:`~libcasm.monte.matrix_as_vector`.
      - Data returned by a VectorFloatHistogramFunction can be stored in a :class:`~libcasm.monte.sampling.DiscreteVectorIntHistogram`.
      - A call operator exists (:func:`~libcasm.monte.VectorFloatHistogramFunction.__call__`) to call the function held by :class:`~libcasm.monte.sampling.VectorFloatHistogramFunction`.
      )pbdoc")
      .def(py::init<>(&make_vector_float_histogram_function),
           R"pbdoc(

          .. rubric:: Constructor

          Parameters
          ----------
          name : str
              Name of the sampled quantity.

          description : str
              Description of the function.

          shape : List[int]
              Shape of quantity, with column-major unrolling

              Scalar: [], Vector: [n], Matrix: [m, n], etc.

          function : function
              A function with 0 arguments that returns an array of the proper size sampling the current state. Typically this is a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the current state of the simulation.

          component_names : Optional[List[str]] = None
              A name for each component of the resulting vector.

              Can be strings representing an indices (i.e "0", "1", "2", etc.) or can be a descriptive string (i.e. "Mg", "Va", "O", etc.). If None, indices for column-major ordering are used (i.e. "0,0", "1,0", ..., "m-1,n-1")

          max_size : int = 10000
              Maximum number of bins to create. If adding an additional data
              point would cause the number of bins to exceed `max_size`, the
              count / weight is instead added to the `out_of_range_count` of the
              :class:`~libcasm.monte.sampling.DiscreteVectorFloatHistogram`.

          tol : float = :data:`~libcasm.casmglobal.TOL`
              Tolerance for floating point comparisons used when determining counts
              for the histogram.
          )pbdoc",
           py::arg("name"), py::arg("description"), py::arg("shape"),
           py::arg("function"), py::arg("component_names") = std::nullopt,
           py::arg("max_size") = 10000, py::arg("tol") = CASM::TOL)
      .def_readwrite("name", &monte::HistogramFunction<Eigen::VectorXd>::name,
                     R"pbdoc(
          str : Name of the quantity.
          )pbdoc")
      .def_readwrite("description",
                     &monte::HistogramFunction<Eigen::VectorXd>::description,
                     R"pbdoc(
          str : Description of the function.
          )pbdoc")
      .def_readwrite("shape", &monte::HistogramFunction<Eigen::VectorXd>::shape,
                     R"pbdoc(
          List[int] : Shape of quantity, with column-major unrolling.

          Scalar: [], Vector: [n], Matrix: [m, n], etc.
          )pbdoc")
      .def_readwrite(
          "component_names",
          &monte::HistogramFunction<Eigen::VectorXd>::component_names,
          R"pbdoc(
          List[str] : A name for each component of the resulting vector.

          Can be strings representing an indices (i.e "0", "1", "2", etc.) or can be a descriptive string (i.e. "Mg", "Va", "O", etc.). If the sampled quantity is an unrolled matrix, indices for column-major ordering are typical (i.e. "0,0", "1,0", ..., "m-1,n-1").
          )pbdoc")
      .def_readwrite("function",
                     &monte::HistogramFunction<Eigen::VectorXd>::function,
                     R"pbdoc(
          function : The function to be evaluated.

          A function with 0 arguments that returns an array of the proper size
          sampling the current state. Typically this is a lambda function that
          has been given a reference or pointer to a Monte Carlo calculation
          object so that it can access the current state of the simulation.
          )pbdoc")
      .def_readwrite("max_size",
                     &monte::HistogramFunction<Eigen::VectorXd>::max_size,
                     R"pbdoc(
          int: Maximum number of bins in the histogram.
          )pbdoc")
      .def_readwrite("tol", &monte::HistogramFunction<Eigen::VectorXd>::tol,
                     R"pbdoc(
          float: Tolerance for floating point comparisons used when determining
          counts for the histogram.
          )pbdoc")
      .def(
          "__call__",
          [](monte::HistogramFunction<Eigen::VectorXd> const &f)
              -> Eigen::VectorXd { return f(); },
          R"pbdoc(
          Evaluates the function

          Equivalent to calling :py::attr:`~libcasm.monte.sampling.VectorFloatHistogramFunction.function`.
          )pbdoc");

  py::bind_map<
      std::map<std::string, monte::HistogramFunction<Eigen::VectorXd>>>(
      m, "VectorFloatHistogramFunctionMap",
      R"pbdoc(
      VectorFloatHistogramFunctionMap stores :class:`~libcasm.monte.sampling.VectorFloatHistogramFunction` by name of the sampled quantity

      Notes
      -----
      VectorFloatHistogramFunctionMap is a Dict[str, :class:`~libcasm.monte.sampling.VectorFloatHistogramFunction`]-like object.
      )pbdoc",
      py::module_local(false));

  py::class_<monte::PartitionedHistogramFunction<double>>(
      m, "PartitionedHistogramFunction",
      R"pbdoc(
      A function that returns a floating-point value to add to one of mutually
      exclusive histograms

      A common use case is to collect a histogram of event data, by the type
      of event selected. This function outputs a scalar value and a partition
      index, which is used to determine which histogram to add the value to.

      Notes
      -----
      - Typically this holds a lambda function that has been given a reference or pointer to a Monte Carlo calculation object so that it can access the calculator's event data.
      - PartitionedHistogramFunction can be used to return a scalar floating-point quantity (i.e. event rate) and a partition index (i.e. event type index).
      - Data returned by a PartitionedHistogramFunction can be stored in a :class:`~libcasm.monte.sampling.PartitionedHistogram1D`.
      - A call operator exists (:func:`~libcasm.monte.PartitionedHistogramFunction.__call__`) to evaluate the stored function.
      - A :func:`~libcasm.monte.PartitionedHistogramFunction.partition` function returns the partition index of the individual histogram the value should be added to.
      )pbdoc")
      .def(py::init<>(&make_partitioned_histogram_function),
           R"pbdoc(

          .. rubric:: Constructor

          Parameters
          ----------
          name : str
              Name of the sampled quantity.
          description : str
              Description of the function.
          function : function
              A function with 0 arguments that returns a float. Typically this
              is a lambda function that has been given a reference or pointer to
              a Monte Carlo calculation object so that it can access the last
              selected event state and the Monte Carlo state before the event
              occurs.
          partition_names : List[str]
              A name for each partition.
          get_partition_function : function
              A function with 0 arguments that returns an int. Typically this
              is a lambda function that has been given a reference or pointer
              to a Monte Carlo calculation object so that it can access the
              last selected event state and the Monte Carlo state before the
              event occurs.
          is_log : bool = False
              True if bin coordinate spacing is log-scaled; False otherwise.
          initial_begin : float = 0.0
              Initial `begin` coordinate, specifying the beginning of the range
              for the first bin. The bin number for a particular value is
              calculated as `(value - begin) / bin_width`, so the range for
              bin `i` is [begin, begin + i*bin_width). Coordinates are adjusted
              to fit the data encountered by starting `begin` at
              `initial_begin` and adjusting it as necessary by multiples of
              `bin_width`.
          bin_width : float = 1.0
              Bin width.
          max_size : int = 10000
              Maximum number of bins to create. If adding an additional data
              point would cause the number of bins to exceed `max_size`, the
              count / weight is instead added to the `out_of_range_count` of the
              :class:`~libcasm.monte.sampling.PartitionedHistogram1D`.
          )pbdoc",
           py::arg("name"), py::arg("description"), py::arg("function"),
           py::arg("partition_names"), py::arg("get_partition_function"),
           py::arg("is_log") = false, py::arg("initial_begin") = 0.0,
           py::arg("bin_width") = 1.0, py::arg("max_size") = 10000)
      .def_readwrite("name", &monte::PartitionedHistogramFunction<double>::name,
                     R"pbdoc(
          str : Name of the quantity.
          )pbdoc")
      .def_readwrite("description",
                     &monte::PartitionedHistogramFunction<double>::description,
                     R"pbdoc(
          str : Description of the function.
          )pbdoc")
      .def_readwrite("function",
                     &monte::PartitionedHistogramFunction<double>::function,
                     R"pbdoc(
          function : The function to be evaluated.

          A function with 0 arguments that returns a float. Typically this is a
          lambda function that has been given a reference or pointer to a Monte
          Carlo calculation object so that it can access the last selected
          event state and the Monte Carlo state before the event occurs.
          )pbdoc")
      .def_readwrite(
          "get_partition_function",
          &monte::PartitionedHistogramFunction<double>::get_partition,
          R"pbdoc(
          function : The function to be evaluated.

          A function with 0 arguments that returns an int. Typically this is a
          lambda function that has been given a reference or pointer to a Monte
          Carlo calculation object so that it can access the last selected
          event state and the Monte Carlo state before the event occurs.
          )pbdoc")
      .def_readwrite("is_log",
                     &monte::PartitionedHistogramFunction<double>::is_log,
                     R"pbdoc(
          bool : True if bin coordinate spacing is log-scaled; False otherwise.
          )pbdoc")
      .def_readwrite(
          "initial_begin",
          &monte::PartitionedHistogramFunction<double>::initial_begin,
          R"pbdoc(
          float : Initial `begin` coordinate, specifying the beginning of the
          range for the first bin.

          The bin number for a particular value is calculated as
          `(value - begin) / bin_width`, so the range for bin `i` is
          [begin, begin + i*bin_width). Coordinates are adjusted to fit
          the data encountered by starting `begin` at `initial_begin` and
          adjusting it as necessary by multiples of `bin_width`.
          )pbdoc")
      .def_readwrite("bin_width",
                     &monte::PartitionedHistogramFunction<double>::bin_width,
                     R"pbdoc(
          float : Bin width.
          )pbdoc")
      .def_readwrite("max_size",
                     &monte::PartitionedHistogramFunction<double>::max_size,
                     R"pbdoc(
          int : Maximum number of bins to create.

          If adding an additional data point would cause the number of bins to
          exceed `max_size`, the count / weight is instead added to the
          `out_of_range_count` of the :class:`PartitionedHistogram1D`.
          )pbdoc")
      .def(
          "__call__",
          [](monte::PartitionedHistogramFunction<double> const &f) -> double {
            return f();
          },
          R"pbdoc(
          Evaluates the function

          Equivalent to calling
          :py::attr:`~libcasm.monte.sampling.PartitionedHistogramFunction.function`.
          )pbdoc")
      .def(
          "partition",
          [](monte::PartitionedHistogramFunction<double> const &f) -> int {
            return f.get_partition();
          },
          R"pbdoc(
          Evaluates `get_partition_function`

          Equivalent to calling
          :py::attr:`~libcasm.monte.sampling.PartitionedHistogramFunction.get_partition_function`.
          )pbdoc");

  py::bind_map<
      std::map<std::string, monte::PartitionedHistogramFunction<double>>>(
      m, "PartitionedHistogramFunctionMap",
      R"pbdoc(
      PartitionedHistogramFunctionMap stores :class:`~libcasm.monte.sampling.PartitionedHistogramFunction` by name of the sampled quantity

      Notes
      -----
      PartitionedHistogramFunction is a Dict[str, :class:`~libcasm.monte.sampling.PartitionedHistogramFunction`]-like object.
      )pbdoc",
      py::module_local(false));

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
      requested_precision : libcasm.monte.sampling.RequestedPrecision
          The requested precision level for convergence.

      Returns
      -------
      results : libcasm.monte.sampling.IndividualEquilibrationResult
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
          stats : libcasm.monte.sampling.BasicStatistics
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
          stats : libcasm.monte.sampling.BasicStatistics
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
      Convergence check results for a single :class:`~libcasm.monte.sampling.SamplerComponent`
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
        sampler: libcasm.monte.sampling.Sampler
            The sampler containing the sampled data.
        sample_weight : libcasm.monte.sampling.Sampler
            Optional weight to give to each to observation.
        key : libcasm.monte.sampling.SamplerComponent
            Specifies the component of sampler being checked for convergence.
        requested_precision : libcasm.monte.sampling.RequestedPrecisionMap
            The requested precision level for convergence.
        N_samples_for_statistics : int
            The number of tail samples from `sampler` to include in statistics.
        calc_statistics_f : function
            Fucntion used to calculate :class:`~libcasm.monte.sampling.BasicStatistics`. For example, an instance of :class:`~libcasm.monte.sampling.BasicStatisticsCalculator`.
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
        samplers: libcasm.monte.sampling.SamplerMap
            The samplers containing the sampled data.
        sample_weight : libcasm.monte.sampling.Sampler
            Optional weight to give to each to observation.
        requested_precision : libcasm.monte.sampling.RequestedPrecisionMap
            The requested precision levels for all :class:`~libcasm.monte.sampling.SamplerComponent` that are requested to converge.
        N_samples_for_equilibration : int
            Number of initial samples to exclude from statistics because the system is out of equilibrium.
        calc_statistics_f : function
            Fucntion used to calculate :class:`~libcasm.monte.sampling.BasicStatistics`. For example, an instance of :class:`~libcasm.monte.sampling.BasicStatistics`.
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
      .def(py::init<>(&make_cutoff_check_params), R"pbdoc(

          .. rubric:: Constructor

          Parameters
          ----------
          min_count: Optional[int] = None
              Minimum number of steps or passes.
          max_count: Optional[int] = None
              Maximum number of steps or passes.
          min_time: Optional[float] = None
              Minimum simulated time, if applicable.
          max_time: Optional[float] = None
              Maximum simulated time, if applicable.
          min_sample: Optional[int] = None
              Minimum number of samples.
          max_sample: Optional[int] = None
              Maximum number of samples.
          min_clocktime: Optional[float] = None
              Minimum elapsed clocktime.
          max_clocktime: Optional[float] = None
              Maximum elapsed clocktime.

          )pbdoc",
           py::arg("min_count") = std::nullopt,
           py::arg("max_count") = std::nullopt,
           py::arg("min_time") = std::nullopt,
           py::arg("max_time") = std::nullopt,
           py::arg("min_sample") = std::nullopt,
           py::arg("max_sample") = std::nullopt,
           py::arg("min_clocktime") = std::nullopt,
           py::arg("max_clocktime") = std::nullopt)
      .def_readwrite("min_count", &monte::CutoffCheckParams::min_count,
                     R"pbdoc(
                     Optional[int]: Minimum number of steps or passes.
                     )pbdoc")
      .def_readwrite("min_time", &monte::CutoffCheckParams::min_time,
                     R"pbdoc(
                     Optional[float]: Minimum simulated time, if applicable.
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
                     Optional[float]: Maximum simulated time, if applicable.
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
      cutoff_params : libcasm.monte.sampling.CutoffCheckParams
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
      cutoff_params : libcasm.monte.sampling.CutoffCheckParams
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
          requested_precision : Optional[libcasm.monte.sampling.RequestedPrecisionMap] = None
              Requested precision for convergence of sampler components. When all components
              reach the requested precision, and all `cutoff_params` minimums are met,
              then the completion check returns True, indicating the Monte Carlo simulation
              is complete.
          cutoff_params: Optional[libcasm.monte.sampling.CutoffCheckParams] = None,
              Cutoff check parameters allow setting limits on the Monte Carlo simulation to
              prevent calculations from stopping too soon or running too long. If None, no
              cutoffs are applied.
          calc_statistics_f: Optional[Callable] = None,
              A function for calculating :class:`~libcasm.monte.sampling.BasicStatistics` from
              sampled data, with signature:

              .. code-block:: Python

                  def calc_statistics_f(
                      observations: np.ndarray,
                      sample_weight: np.ndarray,
                  ) -> libcasm.monte.BasicStatistics:
                      ...

              If None, the default is :class:`~libcasm.monte.sampling.BasicStatisticsCalculator`.
          equilibration_check_f: Optional[Callable] = None,
              A function for checking equilibration of sampled data, with signature:

              .. code-block:: Python

                  def equilibration_check_f(
                      observations: np.ndarray,
                      sample_weight: np.ndarray,
                      requested_precision: libcasm.monte.RequestedPrecision,
                  ) -> libcasm.monte.IndividualEquilibrationResult:
                      ...

              If None, the default is :class:`~libcasm.monte.sampling.default_equilibration_check`.
          log_spacing: bool = False
              If True, use logarithmic spacing for completion checking; else use linear
              spacing. For linear spacing, the n-th check (n=0,1,2,...) will be taken when:

              .. code-block:: Python

                  sample = check_begin + check_period * n

              For logarithmic spacing, the n-th check will be taken when:

              .. code-block:: Python

                  sample = check_begin + round( check_base ** (n + check_shift) )

              However, if check(n) - check(n-1) > `check_period_max`, then subsequent
              checks are made every `check_period_max` samples.

              For linear spacing, the default is to check for completion after `100`,
              `200`, `300`, etc. samples are taken.

              For log spacing, the default is to check for completion after `100`,
              `1000`, `10000`, `20000`, `30000`, etc. samples are taken (note the
              effect of the default ``check_period_max=10000``).

              The default value is False, for linear spacing.
          check_period:  Optional[int] = None
              The linear completion checking period. Default is 100.
          check_begin:  Optional[int] = None
              The earliest sample to begin completion checking. Default is
              `check_period` for linear spacing and 0 for log spacing.
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
          :class:`~libcasm.monte.sampling.CutoffCheckParams`: Cutoff check parameters
          )pbdoc")
      .def_readwrite(
          "equilibration_check_f",
          &monte::CompletionCheckParams<statistics_type>::equilibration_check_f,
          R"pbdoc(
          function: Function that performs equilibration checking.

          A function, such as :func:`~libcasm.monte.default_equilibration_check`, with
          signature f(array_like observations, array_like sample_weight,
          :class:`~libcasm.monte.sampling.RequestedPrecision` requested_precision) ->
          :class:`~libcasm.monte.sampling.IndividualEquilibrationResult`.
          )pbdoc")
      .def_readwrite(
          "calc_statistics_f",
          &monte::CompletionCheckParams<statistics_type>::calc_statistics_f,
          R"pbdoc(
          function: Function to calculate statistics.

          A function, such as an instance of
          :class:`~libcasm.monte.sampling.BasicStatisticsCalculator`, with signature
          f(array_like observations, array_like sample_weight) ->
          :class:`~libcasm.monte.sampling.BasicStatistics`.
          )pbdoc")
      .def_readwrite(
          "requested_precision",
          &monte::CompletionCheckParams<statistics_type>::requested_precision,
          R"pbdoc(
          :class:`~libcasm.monte.sampling.RequestedPrecisionMap`: Requested precision for \
          convergence of sampler components.

          A Dict[:class:`~libcasm.monte.sampling.SamplerComponent`,
          :class:`~libcasm.monte.sampling.RequestedPrecision`]-like object that specifies
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
          :class:`~libcasm.monte.sampling.CompletionCheckParams`: Completion check parameters
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
          :class:`~libcasm.monte.sampling.EquilibrationCheckResults`: Results of equilibration check
          )pbdoc")
      .def_readwrite("convergence_check_results",
                     &monte::CompletionCheckResults<
                         statistics_type>::convergence_check_results,
                     R"pbdoc(
          :class:`~libcasm.monte.sampling.ConvergenceCheckResults`: Results of convergence check
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
          clocktime : float = 0.0
              Elapsed clocktime to reset to
          n_samples : int = 0
              Number of samples to reset to
          )pbdoc",
           py::arg("count") = std::nullopt, py::arg("time") = std::nullopt,
           py::arg("clocktime") = 0.0, py::arg("n_samples") = 0)
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
          params : libcasm.monte.sampling.CompletionCheckParams
              Data structure holding completion check parameters.
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
          samplers: libcasm.monte.sampling.SamplerMap
              The samplers containing the sampled data.
          sample_weight : libcasm.monte.sampling.Sampler
              Sample weights associated with observations. May have 0 samples, in which
              case the obsservations are treated as being equally weighted, otherwise
              must match the number of samples made by each sampler in `samplers`.
          count : int
              Number of steps or passes
          method_log : libcasm.monte.MethodLog
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
          samplers: libcasm.monte.sampling.SamplerMap
              The samplers containing the sampled data.
          sample_weight : libcasm.monte.sampling.Sampler
              Sample weights associated with observations. May have 0 samples, in which
              case the obsservations are treated as being equally weighted, otherwise
              must match the number of samples made by each sampler in `samplers`.
          count : Optional[int]
              Number of steps or passes
          time : Optional[float]
              Simulated time
          method_log : libcasm.monte.MethodLog
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
          samplers: libcasm.monte.sampling.SamplerMap
              The samplers containing the sampled data.
          sample_weight : libcasm.monte.sampling.Sampler
              Sample weights associated with observations. May have 0 samples, in which
              case the obsservations are treated as being equally weighted, otherwise
              must match the number of samples made by each sampler in `samplers`.
          count : int
              Number of steps or passes
          time : float
              Simulated time
          method_log : libcasm.monte.MethodLog
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
          samplers: libcasm.monte.sampling.SamplerMap
              The samplers containing the sampled data.
          sample_weight : libcasm.monte.sampling.Sampler
              Sample weights associated with observations. May have 0 samples, in which
              case the observations are treated as being equally weighted, otherwise
              must match the number of samples made by each sampler in `samplers`.
          time : float
              Simulated time
          method_log : libcasm.monte.MethodLog
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
          samplers: libcasm.monte.sampling.SamplerMap
              The samplers containing the sampled data.
          sample_weight : libcasm.monte.sampling.Sampler
              Sample weights associated with observations. May have 0 samples, in which
              case the obsservations are treated as being equally weighted, otherwise
              must match the number of samples made by each sampler in `samplers`.
          method_log : libcasm.monte.MethodLog
              The method log specifies where to write status updates and internally
              tracks the elapsed clock time.

          Returns
          -------
          is_complete : bool
              True if complete, False otherwise
          )pbdoc",
          py::arg("samplers"), py::arg("sample_weight"), py::arg("method_log"));

  // -- SelectedEventData and related -----

  py::class_<monte::CorrelationsDataParams>(m, "CorrelationsDataParams",
                                            R"pbdoc(
      Parameters for collecting hop correlations data (not basis function
      correlations)
      )pbdoc")
      .def(py::init<Index, Index, bool, bool>(),
           R"pbdoc(
          .. rubric:: Constructor

          Parameters
          ----------
          jumps_per_position_sample: int = 1
              Every `jumps_per_position_sample` steps of an individual atom, store
              its position.
          max_n_position_samples: int = 100
              The maximum number of positions to store for each atom.
          output_incomplete_samples: bool = False
              If false, only output data when all atoms have jumped the necessary
              number of times. If true, output matrices with 0.0 values for atoms that
              have not jumped enough times to be sampled.
          stop_run_when_complete: bool = False
              If true, stop the run when the maximum number of positions have been
              sampled for all atoms. If false, continue running until the standard
              completion check is met, but do not collect any more position samples.
          )pbdoc",
           py::arg("jumps_per_position_sample") = 1,
           py::arg("max_n_position_samples") = 100,
           py::arg("output_incomplete_samples") = false,
           py::arg("stop_run_when_complete") = false)
      .def_readwrite("jumps_per_position_sample",
                     &monte::CorrelationsDataParams::jumps_per_position_sample,
                     R"pbdoc(
           int: Every `jumps_per_position_sample` steps of an individual atom, store
           its position in this object
           )pbdoc")
      .def_readwrite("max_n_position_samples",
                     &monte::CorrelationsDataParams::max_n_position_samples,
                     R"pbdoc(
           int: The maximum number of positions to store for each atom.
           )pbdoc")
      .def_readwrite("output_incomplete_samples",
                     &monte::CorrelationsDataParams::output_incomplete_samples,
                     R"pbdoc(
           bool: Controls whether to output incomplete samples

           If False, only output data when all atoms have jumped the necessary
           number of times. If True, output matrices with 0.0 values for atoms that
           have not jumped enough times to be sampled.
           )pbdoc")
      .def_readwrite("stop_run_when_complete",
                     &monte::CorrelationsDataParams::stop_run_when_complete,
                     R"pbdoc(
           bool: Controls whether to stop the run when the maximum number of positions
           have been sampled for all atoms.

           If True, stop the run when the maximum number of positions have been
           sampled for all atoms. If False, continue running until the standard
           completion check is met, but do not collect any more position samples.
           )pbdoc");

  py::class_<monte::CorrelationsData>(m, "CorrelationsData",
                                      R"pbdoc(
        Hop correlations data (not basis function correlations)

        Atom positions are stored every `jumps_per_position_sample` jumps, along
        with the (step, pass, sample, time) at which the atom jumped. The
        Cartesian coordinates of the atoms are stored as if the atom was
        jumping in a system without periodic boundaries. For displacements, the
        positions at two different times must be subtracted. By storing the
        number of samples taken when each atom jumped, the user can restrict
        correlation factor calculations to use atom positions taken after the
        system has equilibrated, as determined by the
        :class:`libcasm.monte.sampling.EquilibrationCheckResults`.

        )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
            .. rubric:: Constructor

            Default constructor only.
            )pbdoc")
      .def_readonly("jumps_per_position_sample",
                    &monte::CorrelationsData::jumps_per_position_sample,
                    R"pbdoc(
             int: Every `jumps_per_position_sample` steps of an individual
             atom, store its position in this object )pbdoc")
      .def_readonly("max_n_position_samples",
                    &monte::CorrelationsData::max_n_position_samples,
                    R"pbdoc(
             int: The maximum number of positions to store for each atom.
             )pbdoc")
      .def_readonly("output_incomplete_samples",
                    &monte::CorrelationsData::output_incomplete_samples,
                    R"pbdoc(
             bool: Controls whether to output incomplete samples

             If False, only output data when all atoms have jumped the
             necessary number of times. If True, output matrices with 0.0
             values for atoms that have not jumped enough times to be sampled.
             )pbdoc")
      .def_readonly("stop_run_when_complete",
                    &monte::CorrelationsData::stop_run_when_complete,
                    R"pbdoc(
             bool: Controls whether to stop the run when the maximum number of
             positions have been sampled for all atoms.

             If True, stop the run when the maximum number of positions have
             been sampled for all atoms. If False, continue running until the
             standard completion check is met, but do not collect any more
             position samples. )pbdoc")
      .def_readonly("n_position_samples",
                    &monte::CorrelationsData::n_position_samples,
                    R"pbdoc(
          list[int]: For each atom, the number of positions stored in this object.
          )pbdoc")
      .def_readonly("n_complete_samples",
                    &monte::CorrelationsData::n_complete_samples,
                    R"pbdoc(
          int: Number of position samples completed for all atoms.
          )pbdoc")
      .def_readonly("step", &monte::CorrelationsData::step,
                    R"pbdoc(
          numpy.ndarray[numpy.int[max_n_position_samples,n_atoms]]: The
          value `step[i_sample, i_atom]` is the number of steps completed
          (resets to 0 every pass) when the `i_atom`-th atom jumped the
          `i_sample * jumps_per_position_sample`-th time.
          )pbdoc")
      .def_readonly("pass", &monte::CorrelationsData::pass,
                    R"pbdoc(
          numpy.ndarray[numpy.int[max_n_position_samples,n_atoms]]: The
          value `pass[i_sample, i_atom]` is the number of passes completed when
          the `i_atom`-th atom jumped the
          `i_sample * jumps_per_position_sample`-th time.
          )pbdoc")
      .def_readonly("sample", &monte::CorrelationsData::sample,
                    R"pbdoc(
          numpy.ndarray[numpy.int[max_n_position_samples,n_atoms]]: The
          value `sample[i_sample, i_atom]` is the number of samples taken when
          the `i_atom`-th atom jumped the
          `i_sample * jumps_per_position_sample`-th time.
          )pbdoc")
      .def_readonly("time", &monte::CorrelationsData::time,
                    R"pbdoc(
          numpy.ndarray[numpy.float[max_n_position_samples,n_atoms]]: The
          value `time[i_sample, i_atom]` is the simulated time when the
          `i_atom`-th atom jumped the `i_sample * jumps_per_position_sample`-th
          time.
          )pbdoc")
      .def_readonly("atom_positions_cart",
                    &monte::CorrelationsData::atom_positions_cart,
                    R"pbdoc(
          list[numpy.ndarray[numpy.float[3,n_atoms]]]: The array
          `atom_positions_cart[i_sample]` contains the Cartesian coordinates,
          as columns, of each atom (as if periodic boundaries did not exist)
          after the `i_sample * jumps_per_position_sample`-th jump.
          )pbdoc")
      .def(
          "equilibrated_samples",
          [](monte::CorrelationsData const &self,
             Index N_samples_for_all_to_equilibrate) {
            std::vector<int> result;
            if (self.n_complete_samples >= self.sample.rows()) {
              throw std::runtime_error(
                  "Error in "
                  "monte.sampling.CorrelationsData.equilibrated_samples: "
                  "n_complete_samples >= sample.rows()");
            }
            for (Index j = 0; j < self.n_complete_samples; ++j) {
              int max_sample = self.sample.row(j).maxCoeff();
              if (max_sample >= N_samples_for_all_to_equilibrate) {
                result.push_back(j);
              }
            }
            return result;
          },
          R"pbdoc(
          Return the indices of samples taken after the system has equilibrated

          Parameter
          ---------
          N_samples_for_all_to_equilibrate: int
              The long it took (how many samples) for the system to equilibrate.

          Returns
          -------
          indices: list[int]
              The indices, j, such that
              `samples[j][i_atom] >= N_samples_for_all_to_equilibrate` for all
              `i_atom`.
          )pbdoc")
      .def("initialize", &monte::CorrelationsData::initialize,
           R"pbdoc(
          Initialize for a new run

          Parameters
          ----------
          n_atoms: int
              The number of atoms in the simulation supercell.
          jumps_per_position_sample: int
              Every `jumps_per_position_sample` steps of an individual atom,
              its position will be stored in Cartesian coordinates (as if
              periodic boundaries did not exist).
          max_n_position_samples: int
              The maximum number of positions to store for each atom.
          output_incomplete_samples: bool
              If false, when representing this object as a Python dict, only
              output data for the number of samples for which all atoms have
              jumped the necessary number of times. If true, output matrices
              with 0.0 values for atoms that have not jumped enough times to be
              sampled.
          )pbdoc")
      .def("insert", &monte::CorrelationsData::insert, R"pbdoc(
          Insert a new position sample for an atom, if the atom has jumped
          the necessary number of times

          Parameters
          ----------
          atom_id: int
              Atom index (corresponds to columns of arrays).
          n_jumps: int
              Number of times the atom has jumped.
          position_cart: numpy.ndarray[numpy.float[3]]
              Cartesian coordinates of the atom after the jump.
          step: int
              Number of steps completed when the atom jumped.
          pass: int
              Number of passes completed when the atom jumped.
          sample: int
              Number of samples taken when the atom jumped.
          time: float
              Simulated time when the atom jumped.
          )pbdoc");

  py::class_<monte::DiscreteVectorIntHistogram>(m, "DiscreteVectorIntHistogram",
                                                R"pbdoc(
      Data structure for holding a histogram of discrete integer vector values
      )pbdoc");

  py::class_<monte::DiscreteVectorFloatHistogram>(
      m, "DiscreteVectorFloatHistogram",
      R"pbdoc(
      Data structure for holding a histogram of discrete floating-point vector values
      )pbdoc");

  py::class_<monte::Histogram1D>(m, "Histogram1D",
                                 R"pbdoc(
      Data structure for holding a 1D histogram
      )pbdoc");

  py::class_<monte::PartitionedHistogram1D>(m, "PartitionedHistogram1D",
                                            R"pbdoc(
      Data structure for holding 1 or more 1D histograms of related data (i.e.
      event rates by event type)
      )pbdoc");

  py::class_<monte::SelectedEventDataFunctions>(m, "SelectedEventDataFunctions",
                                                R"pbdoc(
        Holds functions that return selected event data
        )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
            .. rubric:: Constructor

            Default constructor only.
            )pbdoc");

  py::class_<monte::SelectedEventDataParams>(m, "SelectedEventDataParams",
                                             R"pbdoc(
        Parameters controlling selected events data collection
        )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
            .. rubric:: Constructor

            Default constructor only.
            )pbdoc");

  py::class_<monte::SelectedEventData>(m, "SelectedEventData",
                                       R"pbdoc(
        Holds data collected about selected events
        )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
            .. rubric:: Constructor

            Default constructor only.
            )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
