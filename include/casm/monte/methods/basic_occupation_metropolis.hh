/// A basic data structure and main loop implementation
/// for occupation Metropolis Monte Carlo

#ifndef CASM_monte_methods_basic_occupation_metropolis
#define CASM_monte_methods_basic_occupation_metropolis

#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/checks/io/json/CompletionCheck_json_io.hh"
#include "casm/monte/definitions.hh"
#include "casm/monte/methods/metropolis.hh"
#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {
namespace methods {

/// \brief Holds basic occupation Metropolis Monte Carlo run data and results
template <typename StatisticsType>
struct BasicOccupationMetropolisData {
  typedef StatisticsType statistics_type;

  /// \brief Constructor
  ///
  /// \param _sampling_functions The sampling functions to use
  /// \param _json_sampling_functions The JSON sampling functions to use
  /// \param n_steps_per_pass Number of steps per pass.
  /// \param completion_check_params Controls when the run finishes
  BasicOccupationMetropolisData(
      StateSamplingFunctionMap const &_sampling_functions,
      jsonStateSamplingFunctionMap const &_json_sampling_functions,
      CountType _n_steps_per_pass,
      CompletionCheckParams<statistics_type> const &_completion_check_params)
      : sampling_functions(_sampling_functions),
        json_sampling_functions(_json_sampling_functions),
        sample_weight({}),
        n_steps_per_pass(_n_steps_per_pass),
        completion_check(_completion_check_params) {
    for (auto const &pair : sampling_functions) {
      auto const &f = pair.second;
      this->samplers.emplace(
          f.name, std::make_shared<monte::Sampler>(f.shape, f.component_names));
    }
    for (auto const &pair : json_sampling_functions) {
      auto const &f = pair.second;
      this->json_samplers.emplace(f.name, jsonSampler());
    }

    n_pass = 0;
    n_accept = 0;
    n_reject = 0;
  }

  /// \brief The sampling functions to use
  StateSamplingFunctionMap sampling_functions;

  /// \brief Holds sampled data
  SamplerMap samplers;

  /// \brief The json sampling functions to use
  jsonStateSamplingFunctionMap json_sampling_functions;

  /// \brief Holds sampled JSON data
  jsonSamplerMap json_samplers;

  /// \brief Sample weights
  ///
  /// Sample weights may remain empty (unweighted). Included for compatibility
  /// with statistics ising_cpp.
  Sampler sample_weight;

  /// \brief Number of passes. One pass is equal to one Monte Carlo step
  ///     per variable site in the configuration.
  CountType n_pass;

  /// \brief Number of steps per pass.
  CountType n_steps_per_pass;

  /// \brief Number of accepted Monte Carlo steps
  BigCountType n_accept;

  /// \brief Number of rejected Monte Carlo steps
  BigCountType n_reject;

  /// \brief The Monte Carlo run completion checker
  CompletionCheck<statistics_type> completion_check;

  double acceptance_rate() const {
    double _n_accept = static_cast<double>(this->n_accept);
    double _n_reject = static_cast<double>(this->n_reject);
    double _total = _n_accept + _n_reject;
    return _n_accept / _total;
  }

  double rejection_rate() const {
    double _n_accept = static_cast<double>(this->n_accept);
    double _n_reject = static_cast<double>(this->n_reject);
    double _total = _n_accept + _n_reject;
    return _n_reject / _total;
  }

  /// \brief Reset attributes set during `run`
  void reset() {
    for (auto &pair : samplers) {
      pair.second->clear();
    }
    for (auto &pair : json_samplers) {
      pair.second.values.clear();
    }
    sample_weight.clear();
    n_pass = 0;
    n_accept = 0;
    n_reject = 0;
    completion_check.reset();
  }
};

template <typename StatisticsType>
jsonParser &to_json(BasicOccupationMetropolisData<StatisticsType> const &data,
                    jsonParser &json);

/// \brief Write nothing to run status logfile and nothing to stream
template <typename StatisticsType>
void write_no_status(BasicOccupationMetropolisData<StatisticsType> const &data,
                     MethodLog &method_log);

/// \brief Write run status logfile and stream
template <typename StatisticsType>
void default_write_status(
    BasicOccupationMetropolisData<StatisticsType> const &data,
    MethodLog &method_log);

/// \brief Write run status to stream (#Passes, Steps/Second, etc.)
template <typename StatisticsType>
void default_write_run_status(
    BasicOccupationMetropolisData<StatisticsType> const &data,
    MethodLog &method_log, std::ostream &sout);

/// \brief Write latest completion check results to log file and screen
template <typename StatisticsType>
void default_write_completion_check_status(
    BasicOccupationMetropolisData<StatisticsType> const &data,
    std::ostream &sout);

/// \brief Write completion check results to log file and restart lap timer
template <typename StatisticsType>
void default_finish_write_status(
    BasicOccupationMetropolisData<StatisticsType> const &data,
    MethodLog &method_log);

/// \brief Run an occupation metropolis Monte Carlo calculation
template <typename StatisticsType, typename EngineType,
          typename PotentialOccDeltaExtensiveValueF, typename ProposeEventF,
          typename ApplyEventF,
          typename WriteStatusF =
              void (*)(BasicOccupationMetropolisData<StatisticsType> const &,
                       MethodLog &)>
void basic_occupation_metropolis(
    BasicOccupationMetropolisData<StatisticsType> &data, double temperature,
    PotentialOccDeltaExtensiveValueF potential_occ_delta_per_supercell_f,
    ProposeEventF propose_event_f, ApplyEventF apply_event_f,
    int sample_period = 1, std::optional<MethodLog> method_log = std::nullopt,
    std::shared_ptr<EngineType> random_engine = nullptr,
    WriteStatusF write_status_f = default_write_status<StatisticsType>);

// --- Implementation ---

template <typename StatisticsType>
jsonParser &to_json(BasicOccupationMetropolisData<StatisticsType> const &data,
                    jsonParser &json) {
  json.put_obj();
  to_json(data.completion_check.results(), json["completion_check_results"]);
  to_json(data.n_pass, json["n_pass"]);
  to_json(data.n_steps_per_pass, json["n_steps_per_pass"]);
  to_json(static_cast<long>(data.n_accept), json["n_accept"]);
  to_json(static_cast<long>(data.n_reject), json["n_reject"]);

  to_json(data.acceptance_rate(), json["acceptance_rate"]);
  to_json(data.rejection_rate(), json["rejection_rate"]);
  return json;
}

/// \brief Write nothing to run status logfile and nothing to stream
template <typename StatisticsType>
void write_no_status(BasicOccupationMetropolisData<StatisticsType> const &data,
                     MethodLog &method_log) {
  method_log.log.begin_lap();
  return;
}

/// \brief Write run status logfile and stream
///
/// Is equivalent to:
/// \code
/// std::ostream &sout = std::cout;
/// default_write_run_status(data, method_log, sout);
/// default_write_completion_check_status(data, sout);
/// default_finish_write_status(data, method_log);
/// \endcode
///
/// \param data The Monte Carlo simulation data
/// \param method_log The logger
template <typename StatisticsType>
void default_write_status(
    BasicOccupationMetropolisData<StatisticsType> const &data,
    MethodLog &method_log) {
  std::ostream &sout = std::cout;
  default_write_run_status(data, method_log, sout);
  default_write_completion_check_status(data, sout);
  default_finish_write_status(data, method_log);
}

/// \brief Write run status to stream (#Passes, Steps/Second, etc.)
///
/// \param data The Monte Carlo simulation data
/// \param method_log The logger
/// \param sout The output stream
///
/// Output is:
///
///     Passes={n_pass}, Samples={n_samples}, ClockTime(s)={time_s}, \
///     Steps/Second={steps / time_s}, Seconds/Step={time_s / steps}
///
template <typename StatisticsType>
void default_write_run_status(
    BasicOccupationMetropolisData<StatisticsType> const &data,
    MethodLog &method_log, std::ostream &sout) {
  // ### write status ###
  auto const &n_pass = data.n_pass;
  auto const &n_samples = get_n_samples(data.samplers);

  // ## Print passes, simulated and clock time
  double steps =
      static_cast<double>(n_pass) * static_cast<double>(data.n_steps_per_pass);
  double time_s = method_log.log.time_s();

  sout << "Passes=" << n_pass << ", ";
  sout << "Samples=" << n_samples << ", ";
  sout << "ClockTime(s)=" << time_s << ", ";
  sout << "Steps/Second=" << steps / time_s << ", ";
  sout << "Seconds/Step=" << time_s / steps << std::endl;
}

/// \brief Write completion check results to log file and restart lap timer
///
/// \param data The Monte Carlo simulation data
/// \param method_log The logger
template <typename StatisticsType>
void default_finish_write_status(
    BasicOccupationMetropolisData<StatisticsType> const &data,
    MethodLog &method_log) {
  // Things to do when finished
  auto const &completion_check = data.completion_check;
  auto const &results = completion_check.results();

  method_log.reset();
  jsonParser json;
  to_json(results, json);
  method_log.log << json << std::endl;
  method_log.log.begin_lap();
}

/// \brief Write latest completion check results to log file and screen
///
/// \param data The Monte Carlo simulation data
/// \param method_log The logger
/// \param sout The output stream
template <typename StatisticsType>
void default_write_completion_check_status(
    BasicOccupationMetropolisData<StatisticsType> const &data,
    std::ostream &sout) {
  auto const &completion_check = data.completion_check;
  auto const &results = completion_check.results();

  // ## Print AllEquilibrated=? status
  bool all_equilibrated = results.equilibration_check_results.all_equilibrated;
  sout << "  ";
  sout << "AllEquilibrated=" << all_equilibrated << std::endl;
  if (!all_equilibrated) {
    return;
  }

  // ## Print AllConverted=? status
  bool all_converged = results.convergence_check_results.all_converged;
  sout << "  ";
  sout << "AllConverged=" << all_converged << std::endl;
  if (all_converged) {
    return;
  }

  // ## Print individual requested convergence status
  auto const &converge_results =
      results.convergence_check_results.individual_results;
  for (auto const &pair : completion_check.params().requested_precision) {
    auto const &key = pair.first;
    auto const &req = pair.second;
    auto const &stats = converge_results.at(key).stats;
    double calc_abs_prec = stats.calculated_precision;
    double mean = stats.mean;
    double calc_rel_prec = std::fabs(calc_abs_prec / mean);
    if (req.abs_convergence_is_required) {
      sout << "  - " << key.sampler_name << "(" << key.component_index << "): "
           << "mean=" << mean << ", "
           << "abs_prec=" << calc_abs_prec << " "
           << "< "
           << "requested=" << req.abs_precision << " "
           << "== " << bool(calc_abs_prec < req.abs_precision) << std::endl;
    }
    if (req.rel_convergence_is_required) {
      sout << "  - " << key.sampler_name << "(" << key.component_index << "): "
           << "mean=" << mean << ", "
           << "rel_prec=" << calc_rel_prec << " "
           << "< "
           << "requested=" << req.rel_precision << " "
           << "== " << bool(calc_rel_prec < req.rel_precision) << std::endl;
    }
  }

  return;
}

/// \brief Run an occupation metropolis Monte Carlo calculation
///
/// \param data Holds basic occupation Metropolis Monte Carlo run data and
///     results
/// \param temperature The temperature used for the Metropolis algorithm.
/// \param potential_occ_delta_per_supercell_f A function which calculates
///     the change in the potential due to a proposed occupation event. The
///     expected signature is ``double f(OccEvent const &)``.
/// \param propose_event_f A function, which proposes an event (of type
///     `OccEvent`) based on the current state and a random number generator.
///     The expected signature is
///     ``OccEvent const &(RandomNumberGenerator<EngineType> &)``.
/// \param apply_event_f A function, which applies an accepted event to update
///     the current state. The expected signature is
///     ``void(OccEvent const &)``.
/// \param sample_period Number of passes per sample. One pass is one Monte
///     Carlo step per site with variable occupation.
/// \param method_log Method log, for writing status updates. If None, default
///     writes to "status.json" every 10 minutes.
/// \param random_engine Random number engine. Default constructs a new
///     engine.
/// \param write_status_f Function that writes status updates, after a new
///     sample has been taken and is due according to
///     `method_log->log_frequency`. Default writes the
///     current completion check results to `method_log->logfile_path` and
///     prints a summary of the current state and sampled data to stdout.
///     The expected signature is
///     ``void(BasicOccupationMetropolisData<StatisticsType> &, MethodLog &)``.
/// \return Simulation results, including sampled data, completion check
///     results, etc.
template <typename StatisticsType, typename EngineType,
          typename PotentialOccDeltaExtensiveValueF, typename ProposeEventF,
          typename ApplyEventF, typename WriteStatusF>
void basic_occupation_metropolis(
    BasicOccupationMetropolisData<StatisticsType> &data, double temperature,
    PotentialOccDeltaExtensiveValueF potential_occ_delta_per_supercell_f,
    ProposeEventF propose_event_f, ApplyEventF apply_event_f, int sample_period,
    std::optional<MethodLog> method_log,
    std::shared_ptr<EngineType> random_engine, WriteStatusF write_status_f) {
  // ### Setup ####
  double beta = 1.0 / (KB * temperature);

  // # construct RandomNumberGenerator
  RandomNumberGenerator<EngineType> random_number_generator(random_engine);

  // # method log also tracks elapsed clocktime
  if (!method_log.has_value()) {
    method_log = MethodLog();
    method_log->logfile_path = fs::current_path() / "status.json";
    method_log->log_frequency = 600.0;
  }
  method_log->log.restart_clock();
  method_log->log.begin_lap();

  // # used in main loop
  Index n_pass_next_sample = sample_period;
  CountType n_step = 0;
  double delta_potential_energy;

  // ### Main loop ####
  while (!data.completion_check.is_complete(data.samplers, data.sample_weight,
                                            data.n_pass, method_log->log)) {
    // Propose an event
    OccEvent const &event = propose_event_f(random_number_generator);

    // Calculate change in potential energy (per_supercell) due to event
    delta_potential_energy = potential_occ_delta_per_supercell_f(event);

    // Accept and apply or reject event
    if (metropolis_acceptance(delta_potential_energy, beta,
                              random_number_generator)) {
      data.n_accept++;
      apply_event_f(event);
    } else {
      data.n_reject++;
    }

    // Increment count
    n_step++;
    if (n_step == data.n_steps_per_pass) {
      n_step = 0;
      data.n_pass += 1;
    }

    // Sample data, if a sample is due by count
    if (data.n_pass == n_pass_next_sample) {
      n_pass_next_sample += sample_period;
      for (auto const &pair : data.sampling_functions) {
        auto const &f = pair.second;
        data.samplers.at(f.name)->push_back(f());
      }
      for (auto const &pair : data.json_sampling_functions) {
        auto const &f = pair.second;
        data.json_samplers.at(f.name).values.push_back(f());
      }
      // # write status if due
      if (method_log->log_frequency.has_value() &&
          method_log->log.lap_time() >= method_log->log_frequency.value()) {
        write_status_f(data, *method_log);
      }
    }
  }

  write_status_f(data, *method_log);
}

}  // namespace methods
}  // namespace monte
}  // namespace CASM

#endif
