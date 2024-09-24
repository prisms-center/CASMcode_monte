#ifndef CASM_monte_ising_cpp_complete_semigrand_canonical
#define CASM_monte_ising_cpp_complete_semigrand_canonical

/// This file contains an example semi-grand canonical calculator
///
/// Notes:
/// - This is equivalent to `basic_semigrand_canonical`, but it
///   includes a data structure and the main loop in the `run`
///   function to show the entire method in one file.
/// - The `basic_semigrand_canonical` calculator uses the
///   basic_occupation_metropolis data structure and method.

#include "casm/casm_io/container/stream_io.hh"
#include "casm/global/definitions.hh"
#include "casm/global/eigen.hh"
#include "casm/monte/BasicStatistics.hh"
#include "casm/monte/MethodLog.hh"
#include "casm/monte/ValueMap.hh"
#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/checks/io/json/CompletionCheck_json_io.hh"
#include "casm/monte/events/OccEvent.hh"
#include "casm/monte/io/json/ValueMap_json_io.hh"
#include "casm/monte/ising_cpp/model.hh"
#include "casm/monte/methods/metropolis.hh"
#include "casm/monte/sampling/Sampler.hh"

namespace CASM {
namespace monte {
namespace ising_cpp {
namespace complete_semigrand_canonical {

/// \brief Semi-grand canonical ensemble thermodynamic conditions
class SemiGrandCanonicalConditions {
 public:
  /// \brief Default constructor
  SemiGrandCanonicalConditions() {}

  /// \brief Constructor
  SemiGrandCanonicalConditions(
      double _temperature,
      Eigen::Ref<Eigen::VectorXd const> _exchange_potential)
      : temperature(_temperature), exchange_potential(_exchange_potential) {}

  /// \brief The temperature, \f$T\f$.
  double temperature;

  /// \brief The semi-grand canonical exchange potential
  ///
  /// The semi-grand canonical exchange potential, conjugate to the
  /// arametric composition that will be calculated by the
  /// `param_composition_calculator` of the system under consideration.
  Eigen::VectorXd exchange_potential;

  static SemiGrandCanonicalConditions from_values(ValueMap const &values);
  ValueMap to_values() const;
};

inline SemiGrandCanonicalConditions SemiGrandCanonicalConditions::from_values(
    ValueMap const &values) {
  if (!values.scalar_values.count("temperature")) {
    throw std::runtime_error("Missing required condition: \"temperature\"");
  }
  if (!values.vector_values.count("exchange_potential")) {
    throw std::runtime_error(
        "Missing required condition: \"exchange_potential\"");
  }
  return SemiGrandCanonicalConditions(
      values.scalar_values.at("temperature"),
      values.vector_values.at("exchange_potential"));
}

inline ValueMap SemiGrandCanonicalConditions::to_values() const {
  ValueMap values;
  values.scalar_values["temperature"] = this->temperature;
  values.vector_values["exchange_potential"] = this->exchange_potential;
  return values;
}

inline void from_json(SemiGrandCanonicalConditions &conditions,
                      jsonParser const &json) {
  ValueMap values;
  from_json(values, json);
  conditions = SemiGrandCanonicalConditions::from_values(values);
}

inline jsonParser &to_json(SemiGrandCanonicalConditions const &conditions,
                           jsonParser &json) {
  ValueMap values = conditions.to_values();
  json.put_obj();
  to_json(values, json);
  return json;
}

/// \brief Calculates the semi-grand canonical energy and changes in energy
///
/// Implements the (per_supercell) semi-grand canonical energy:
///
/// \code
/// double E_sgc = E_formation - n_unitcells *
/// (exchange_potential.dot(param_composition)); \endcode
template <typename SystemType>
class SemiGrandCanonicalPotential {
 public:
  typedef SystemType system_type;
  typedef typename system_type::state_type state_type;
  typedef typename system_type::formation_energy_f_type formation_energy_f_type;
  typedef
      typename system_type::param_composition_f_type param_composition_f_type;

  SemiGrandCanonicalPotential(std::shared_ptr<system_type> _system)
      : system(throw_if_null(_system,
                             "Error constructing SemiGrandCanonicalPotential: "
                             "_system==nullptr")),
        state(nullptr),
        conditions(nullptr),
        formation_energy_calculator(system->formation_energy_calculator),
        param_composition_calculator(system->param_composition_calculator) {}

  /// \brief Holds parameterized ising_cpp, without specifying at a particular
  /// state
  std::shared_ptr<system_type> system;

  /// \brief The current state during the calculation
  state_type const *state;

  /// \brief The current thermodynamic conditions during the calculation
  std::shared_ptr<SemiGrandCanonicalConditions> conditions;

  /// \brief The formation energy calculator, set to calculate using the current
  /// state
  formation_energy_f_type formation_energy_calculator;

  /// \brief The parametric composition calculator, set to calculate using the
  /// current
  ///     state
  ///
  /// This is expected to calculate the compositions conjugate to the
  /// the exchange potentials provided by
  /// `state->conditions.exchange_potential`.
  param_composition_f_type param_composition_calculator;

  /// \brief Set the current Monte Carlo state
  void set_state(state_type const *_state,
                 std::shared_ptr<SemiGrandCanonicalConditions> _conditions) {
    this->state = throw_if_null(
        _state,
        "Error in SemiGrandCanonicalPotential::set_state: _state==nullptr");
    this->conditions =
        throw_if_null(_conditions,
                      "Error in SemiGrandCanonicalPotential::set_state: "
                      "_conditions is nullptr");
    this->formation_energy_calculator.set_state(_state);
    this->param_composition_calculator.set_state(_state);
  }

  /// \brief Calculates semi-grand canonical energy (per supercell)
  double per_supercell() {
    return this->formation_energy_calculator.per_supercell() -
           this->conditions->exchange_potential.dot(
               this->param_composition_calculator.per_supercell());
  }

  /// \brief Calculates semi-grand canonical energy (per unit cell)
  double per_unitcell() {
    return this->per_supercell() / this->state->configuration.n_unitcells;
  }

  /// \brief Calculates the change in semi-grand canonical energy (per
  /// supercell)
  double occ_delta_per_supercell(std::vector<Index> const &linear_site_index,
                                 std::vector<int> const &new_occ) const {
    // de_potential = e_potential_final - e_potential_init
    //   = (e_formation_final - n_unitcells * mu @ x_final) -
    //     (e_formation_init - n_unitcells * mu @ x_init)
    //   = de_formation - n_unitcells * mu * dx

    double dE_f = this->formation_energy_calculator.occ_delta_per_supercell(
        linear_site_index, new_occ);
    auto const &mu_exchange = this->conditions->exchange_potential;
    Eigen::VectorXd Ndx =
        this->param_composition_calculator.occ_delta_per_supercell(
            linear_site_index, new_occ);
    return dE_f - mu_exchange.dot(Ndx);
  }

  /// \brief Calculates the change in semi-grand canonical energy (per
  /// supercell)
  double occ_delta_per_supercell(OccEvent const &e) const {
    return occ_delta_per_supercell(e.linear_site_index, e.new_occ);
  }
};

/// \brief Holds semi-grand canonical Metropolis Monte Carlo run data and
/// results
struct SemiGrandCanonicalData {
  typedef BasicStatistics statistics_type;

  /// \brief Constructor
  ///
  /// \param sampling_functions The sampling functions to use
  /// \param json_sampling_functions The JSON sampling functions to use
  /// \param n_steps_per_pass Number of steps per pass.
  /// \param completion_check_params Controls when the run finishes
  SemiGrandCanonicalData(
      StateSamplingFunctionMap const &_sampling_functions,
      jsonStateSamplingFunctionMap const &_json_sampling_functions,
      CountType _n_steps_per_pass,
      CompletionCheckParams<statistics_type> const &completion_check_params)
      : sampling_functions(_sampling_functions),
        json_sampling_functions(_json_sampling_functions),
        sample_weight({}),
        n_steps_per_pass(_n_steps_per_pass),
        completion_check(completion_check_params) {
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
  /// Sample weights remain empty (unweighted). Included for compatibility
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

inline jsonParser &to_json(SemiGrandCanonicalData const &data,
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
///
/// \param mc_calculator The Monte Carlo calculator to (not) write status for.
/// \param method_log The logger
template <typename SemiGrandCanonicalCalculatorType>
void write_no_status(SemiGrandCanonicalCalculatorType const &mc_calculator,
                     MethodLog &method_log) {
  method_log.log.begin_lap();
  return;
}

/// \brief Write status to log file and std::cout
///
/// \param mc_calculator The Monte Carlo calculator to write status for.
/// \param method_log The logger
template <typename SemiGrandCanonicalCalculatorType>
void default_write_status(SemiGrandCanonicalCalculatorType const &mc_calculator,
                          MethodLog &method_log) {
  std::ostream &sout = std::cout;
  auto const &data = *mc_calculator.data;
  {
    // ### write status ###
    auto const &n_pass = data.n_pass;
    auto const &n_samples = get_n_samples(data.samplers);

    // ## Print passes, simulated and clock time
    double steps = static_cast<double>(n_pass) *
                   static_cast<double>(data.n_steps_per_pass);
    double time_s = method_log.log.time_s();

    sout << "Passes=" << n_pass << ", ";
    sout << "Samples=" << n_samples << ", ";
    sout << "ClockTime(s)=" << time_s << ", ";
    sout << "Steps/Second=" << steps / time_s << ", ";
    sout << "Seconds/Step=" << time_s / steps << std::endl;
  }

  {
    // ## Print current property status
    auto const &param_composition_calculator =
        *mc_calculator.param_composition_calculator;
    auto const &formation_energy_calculator =
        *mc_calculator.formation_energy_calculator;
    Eigen::VectorXd param_composition =
        param_composition_calculator.per_unitcell();
    double formation_energy = formation_energy_calculator.per_unitcell();
    sout << "  ";
    sout << "ParametricComposition=" << param_composition.transpose() << ", ";
    sout << "FormationEnergy=" << formation_energy << std::endl;
  }

  auto finish = [&]() {
    // Things to do when finished
    auto const &completion_check = data.completion_check;
    auto const &results = completion_check.results();

    method_log.reset();
    jsonParser json;
    to_json(results, json);
    method_log.log << json << std::endl;
    method_log.log.begin_lap();
  };

  {
    auto const &completion_check = data.completion_check;
    auto const &results = completion_check.results();

    // ## Print AllEquilibrated=? status
    bool all_equilibrated =
        results.equilibration_check_results.all_equilibrated;
    sout << "  ";
    sout << "AllEquilibrated=" << all_equilibrated << std::endl;
    if (!all_equilibrated) {
      finish();
      return;
    }

    // ## Print AllConverted=? status
    bool all_converged = results.convergence_check_results.all_converged;
    sout << "  ";
    sout << "AllConverged=" << all_converged << std::endl;
    if (all_converged) {
      finish();
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
        sout << "  - " << key.sampler_name << "(" << key.component_index
             << "): "
             << "mean=" << mean << ", "
             << "abs_prec=" << calc_abs_prec << " "
             << "< "
             << "requested=" << req.abs_precision << " "
             << "== " << bool(calc_abs_prec < req.abs_precision) << std::endl;
      }
      if (req.rel_convergence_is_required) {
        sout << "  - " << key.sampler_name << "(" << key.component_index
             << "): "
             << "mean=" << mean << ", "
             << "rel_prec=" << calc_rel_prec << " "
             << "< "
             << "requested=" << req.rel_precision << " "
             << "== " << bool(calc_rel_prec < req.rel_precision) << std::endl;
      }
    }
  }

  finish();
  return;
}

/// \brief Propose and apply semi-grand canonical Ising model events
template <typename EngineType>
class SemiGrandCanonicalEventGenerator {
 public:
  typedef IsingState state_type;
  typedef EngineType engine_type;
  typedef RandomNumberGenerator<engine_type> random_number_generator_type;

  /// \brief Constructor
  SemiGrandCanonicalEventGenerator()
      : state(nullptr), m_max_linear_site_index(0) {
    occ_event.linear_site_index.clear();
    occ_event.linear_site_index.push_back(0);
    occ_event.new_occ.clear();
    occ_event.new_occ.push_back(1);
  }

  /// \brief The current state for which events are proposed and applied. Can be
  ///     nullptr, but must be set for use.
  state_type *state;

  /// \brief The current proposed event
  OccEvent occ_event;

 private:
  Index m_max_linear_site_index;

 public:
  /// \brief Set the current Monte Carlo state and occupant locations
  ///
  /// \param _state The current state for which events are proposed and applied.
  ///     Throws if nullptr.
  void set_state(state_type *_state) {
    this->state =
        throw_if_null(_state,
                      "Error in SemiGrandCanonicalEventGenerator::set_state: "
                      "_state==nullptr");

    m_max_linear_site_index = this->state->configuration.n_sites - 1;
  }

  /// \brief Propose a Monte Carlo occupation event, by setting this->occ_event
  OccEvent const &propose(
      random_number_generator_type &random_number_generator) {
    this->occ_event.linear_site_index[0] =
        random_number_generator.random_int(m_max_linear_site_index);
    this->occ_event.new_occ[0] =
        -this->state->configuration.occ(this->occ_event.linear_site_index[0]);
    return this->occ_event;
  }

  /// \brief Update the occupation of the current state, using this->occ_event
  void apply(OccEvent const &e) {
    this->state->configuration.set_occ(e.linear_site_index[0], e.new_occ[0]);
  }
};

/// \brief A semi-grand canonical Monte Carlo calculator
template <typename SystemType, typename EventGeneratorType>
class SemiGrandCanonicalCalculator {
 public:
  typedef SystemType system_type;
  typedef typename system_type::state_type state_type;
  typedef typename system_type::formation_energy_f_type formation_energy_f_type;
  typedef
      typename system_type::param_composition_f_type param_composition_f_type;

  typedef SemiGrandCanonicalPotential<system_type> potential_type;

  typedef EventGeneratorType event_generator_type;
  typedef typename event_generator_type::engine_type engine_type;
  typedef typename event_generator_type::random_number_generator_type
      random_number_generator_type;

  /// \brief Constructor
  ///
  /// \param _system System containing formation_energy_calculator and
  ///     param_composition_calculator. May not be null.
  SemiGrandCanonicalCalculator(std::shared_ptr<system_type> _system)
      : system(throw_if_null(_system,
                             "Error constructing SemiGrandCanonicalCalculator: "
                             "_system==nullptr")),
        state(nullptr),
        conditions(nullptr),
        potential(_system),
        formation_energy_calculator(&potential.formation_energy_calculator),
        param_composition_calculator(&potential.param_composition_calculator) {}

  /// \brief Holds parameterized ising_cpp, without specifying at a particular
  /// state
  std::shared_ptr<system_type> system;

  /// \brief The current state during the calculation
  ///
  /// Set in `run` method
  state_type *state;

  /// \brief The current thermodynamic conditions during the calculation
  std::shared_ptr<SemiGrandCanonicalConditions> conditions;

  /// \brief The semi-grand canonical energy calculator
  potential_type potential;

  /// \brief The formation energy calculator, set to calculate using the current
  /// state
  formation_energy_f_type *formation_energy_calculator;

  /// \brief The parametric composition calculator, set to calculate using the
  ///     current state
  ///
  /// This is expected to calculate the compositions conjugate to the
  /// the exchange potentials provided by
  /// `state->conditions.exchange_potential`.
  param_composition_f_type *param_composition_calculator;

  /// \brief Monte Carlo run data (samplers, completion_check, n_pass, etc.)
  ///
  /// Constructed at beginning of `run` method
  std::shared_ptr<SemiGrandCanonicalData> data;

  /// \brief Run a semi-grand canonical calculation at a single thermodynamic
  /// state
  ///
  /// \param state Initial Monte Carlo state, including configuration and
  ///     conditions.
  /// \param sampling_functions The sampling functions to use
  /// \param completion_check_params Controls when the run finishes
  /// \param event_generator An event generator which proposes new events and
  ///     applies accepted events.
  /// \param sample_period Number of passes per sample. One pass is one Monte
  /// Carlo
  ///     step per site with variable occupation.
  /// \param method_log Method log, for writing status updates. If None, default
  ///     writes to "status.json" every 10 minutes.
  /// \param random_engine Random number engine. Default constructs a new
  ///     engine.
  /// \param write_status_f Function with signature
  ///     ``void f(SemiGrandCanonicalCalculatorType const &mc_calculator,
  ///     MethodLog &method_log)`` accepting *this as the first argument, that
  ///     writes status updates, after a new sample has been taken and due
  ///     according to
  ///     ``method_log->log_frequency``. Default writes the current
  ///     completion check results to `method_log->logfile_path` and
  ///     prints a summary of the current state and sampled data to stdout.
  /// \return Simulation results, including sampled data, completion check
  ///     results, etc.
  template <typename WriteStatusF = void (*)(SemiGrandCanonicalData const &,
                                             MethodLog &)>
  void run(
      state_type &state, StateSamplingFunctionMap const &sampling_functions,
      jsonStateSamplingFunctionMap const &json_sampling_functions,
      CompletionCheckParams<BasicStatistics> const &completion_check_params,
      event_generator_type event_generator, int sample_period = 1,
      std::optional<MethodLog> method_log = std::nullopt,
      std::shared_ptr<engine_type> random_engine = nullptr,
      WriteStatusF write_status_f = default_write_status) {
    // ### Setup ####

    // set state & conditions
    this->state = &state;
    this->conditions = std::make_shared<SemiGrandCanonicalConditions>(
        SemiGrandCanonicalConditions::from_values(this->state->conditions));
    double temperature = this->conditions->temperature;
    CountType n_steps_per_pass = this->state->configuration.n_sites;

    // set potential, pointers to other ising_cpp, dpotential method
    this->potential.set_state(this->state, this->conditions);
    auto dpotential_f = [=](OccEvent const &e) {
      return this->potential.occ_delta_per_supercell(e);
    };

    // set event generator, propose and apply methods
    event_generator.set_state(this->state);
    auto propose_event_f =
        [&](random_number_generator_type &rng) -> OccEvent const & {
      return event_generator.propose(rng);
    };
    auto apply_event_f = [&](OccEvent const &e) -> void {
      return event_generator.apply(e);
    };

    // construct Monte Carlo data structure
    this->data = std::make_shared<SemiGrandCanonicalData>(
        sampling_functions, json_sampling_functions, n_steps_per_pass,
        completion_check_params);

    // ### Setup next steps ####  (equal to basic_occupation_metropolis)
    auto &data = *this->data;
    double beta = 1.0 / (KB * temperature);

    // # construct RandomNumberGenerator
    RandomNumberGenerator<engine_type> random_number_generator(random_engine);

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
    bool accept;
    double delta_potential_energy;

    // ### Main loop ####
    while (!data.completion_check.is_complete(data.samplers, data.sample_weight,
                                              data.n_pass, method_log->log)) {
      // Propose an event
      OccEvent const &event = propose_event_f(random_number_generator);

      // Calculate change in potential energy (per_supercell) due to event
      delta_potential_energy = dpotential_f(event);

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
        for (auto const &pair : sampling_functions) {
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
          write_status_f(*this, *method_log);
        }
      }
    }

    write_status_f(*this, *method_log);
  }
};

/// \brief Returns a parametric composition sampling function
///
/// The sampling function "parametric_composition" gets the
/// parametric composition from:
/// \code
/// mc_calculator->param_composition_calculator->per_unitcell()
/// \endcode
///
/// \tparam CalculatorType A Monte Carlo calculator type
/// \param mc_calculator A Monte Carlo calculator
/// \return A vector StateSamplingFunction with name "param_composition" and
///     component_names=["0", "1", ...]
template <typename CalculatorType>
StateSamplingFunction make_parametric_composition_f(
    std::shared_ptr<CalculatorType> mc_calculator) {
  if (mc_calculator == nullptr) {
    throw std::runtime_error(
        "Error in parametric_composition sampling function: "
        "mc_calculator == nullptr");
  }
  std::string name = "param_composition";
  std::string description = "Parametric composition";
  std::vector<Index> shape;
  shape.push_back(mc_calculator->system->param_composition_calculator
                      .n_independent_compositions());
  auto f = [mc_calculator]() -> Eigen::VectorXd {
    if (mc_calculator == nullptr) {
      throw std::runtime_error(
          "Error in parametric_composition sampling function: "
          "mc_calculator == nullptr");
    }
    if (mc_calculator->param_composition_calculator == nullptr) {
      throw std::runtime_error(
          "Error in parametric_composition sampling function: "
          "mc_calculator->param_composition_calculator == nullptr");
    }
    if (mc_calculator->param_composition_calculator->state == nullptr) {
      throw std::runtime_error(
          "Error in parametric_composition sampling function: "
          "mc_calculator->param_composition_calculator->state == nullptr");
    }
    return mc_calculator->param_composition_calculator->per_unitcell();
  };
  return StateSamplingFunction(name, description, shape, f);
}  // namespace basic_semigrand_canonical

/// \brief Returns a formation energy sampling function
///
/// The sampling function "formation_energy" gets the formation energy from:
/// \code
/// mc_calculator->formation_energy_calculator->per_unitcell()
/// \endcode
///
/// \tparam CalculatorType A Monte Carlo calculator type
/// \param mc_calculator A Monte Carlo calculator
/// \return A scalar StateSamplingFunction with name "formation_energy"
template <typename CalculatorType>
StateSamplingFunction make_formation_energy_f(
    std::shared_ptr<CalculatorType> mc_calculator) {
  std::string name = "formation_energy";
  std::string description = "Intensive formation energy";
  std::vector<Index> shape = {};  // scalar
  auto f = [mc_calculator]() -> Eigen::VectorXd {
    if (mc_calculator == nullptr) {
      throw std::runtime_error(
          "Error in formation_energy sampling function: "
          "mc_calculator == nullptr");
    }
    if (mc_calculator->formation_energy_calculator == nullptr) {
      throw std::runtime_error(
          "Error in formation_energy sampling function: "
          "mc_calculator->formation_energy_calculator == nullptr");
    }
    if (mc_calculator->formation_energy_calculator->state == nullptr) {
      throw std::runtime_error(
          "Error in formation_energy sampling function: "
          "mc_calculator->formation_energy_calculator->state == nullptr");
    }
    Eigen::VectorXd v(1);
    v(0) = mc_calculator->formation_energy_calculator->per_unitcell();
    return v;
  };
  return StateSamplingFunction(name, description, shape, f);
}

/// \brief Returns a potential energy sampling function
///
/// The sampling function "potential_energy" gets the formation energy from:
/// \code
/// mc_calculator->potential.per_unitcell()
/// \endcode
///
/// \tparam CalculatorType A Monte Carlo calculator type
/// \param mc_calculator A Monte Carlo calculator
/// \return A scalar StateSamplingFunction with name "potential_energy"
template <typename CalculatorType>
StateSamplingFunction make_potential_energy_f(
    std::shared_ptr<CalculatorType> mc_calculator) {
  std::string name = "potential_energy";
  std::string description = "Intensive potential energy";
  std::vector<Index> shape = {};  // scalar
  auto f = [mc_calculator]() -> Eigen::VectorXd {
    if (mc_calculator == nullptr) {
      throw std::runtime_error(
          "Error in formation_energy sampling function: "
          "mc_calculator == nullptr");
    }
    if (mc_calculator->potential.state == nullptr) {
      throw std::runtime_error(
          "Error in formation_energy sampling function: "
          "mc_calculator->potential.state == nullptr");
    }
    Eigen::VectorXd v(1);
    v(0) = mc_calculator->potential.per_unitcell();
    return v;
  };
  return StateSamplingFunction(name, description, shape, f);
}

/// \brief Returns a configuration sampling function
///
/// The sampling function "configuration" gets the configuration,
/// as JSON, from:
/// \code
/// jsonParser json;
/// to_json(mc_calculator->state->configuration, json);
/// return json;
/// \endcode
///
/// \tparam CalculatorType A Monte Carlo calculator type
/// \param mc_calculator A Monte Carlo calculator
/// \return A jsonStateSamplingFunction with name "configuration"
template <typename CalculatorType>
jsonStateSamplingFunction make_configuration_json_f(
    std::shared_ptr<CalculatorType> mc_calculator) {
  std::string name = "configuration";
  std::string description = "Configuration values";
  std::vector<Index> shape = {};  // scalar
  auto f = [mc_calculator]() -> jsonParser {
    if (mc_calculator == nullptr) {
      throw std::runtime_error(
          "Error in formation_energy sampling function: "
          "mc_calculator == nullptr");
    }
    if (mc_calculator->potential.state == nullptr) {
      throw std::runtime_error(
          "Error in formation_energy sampling function: "
          "mc_calculator->potential.state == nullptr");
    }
    jsonParser json;
    to_json(mc_calculator->state->configuration, json);
    return json;
  };
  return jsonStateSamplingFunction(name, description, f);
}

}  // namespace complete_semigrand_canonical
}  // namespace ising_cpp
}  // namespace monte
}  // namespace CASM

#endif
