#ifndef CASM_monte_ising_cpp_basic_semigrand_canonical
#define CASM_monte_ising_cpp_basic_semigrand_canonical

/// This file contains an example semi-grand canonical calculator
///
/// Notes:
/// - This is equivalent to `complete_semigrand_canonical`, but it
///   makes use of the basic_occupation_metropolis data structure
///   and method.
/// - The `complete_semigrand_canonical` calculator includes a
///   data structure and the main loop in the `run`
///   function to show the entire method in one file.

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
#include "casm/monte/methods/basic_occupation_metropolis.hh"
#include "casm/monte/methods/metropolis.hh"
#include "casm/monte/sampling/Sampler.hh"
#include "casm/monte/sampling/StateSamplingFunction.hh"

namespace CASM {
namespace monte {
namespace ising_cpp {
namespace basic_semigrand_canonical {

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

  /// \brief Constructor
  ///
  /// \param _system System containing formation_energy_calculator and
  ///     param_composition_calculator. May not be null.
  /// \param _state State to calculate. May be null, in which case `set_state`
  ///     must be called before using this to calculate values.
  SemiGrandCanonicalPotential(std::shared_ptr<system_type> _system)
      : system(throw_if_null(_system,
                             "Error constructing SemiGrandCanonicalPotential: "
                             "_system==nullptr")),
        state(nullptr),
        conditions(nullptr),
        formation_energy_calculator(system->formation_energy_calculator),
        param_composition_calculator(system->param_composition_calculator) {}

  /// \brief Holds parameterized ising_cpp, without specifying at a particular
  ///     state
  std::shared_ptr<system_type> system;

  /// \brief The current state during the calculation
  state_type const *state;

  /// \brief The current thermodynamic conditions during the calculation
  std::shared_ptr<SemiGrandCanonicalConditions> conditions;

  /// \brief The formation energy calculator, set to calculate using the current
  ///     state
  formation_energy_f_type formation_energy_calculator;

  /// \brief The parametric composition calculator, set to calculate using the
  ///     current state
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
        "Error in SemiGrandCanonicalPotential::set_state: _state is nullptr");
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
struct SemiGrandCanonicalData
    : public methods::BasicOccupationMetropolisData<monte::BasicStatistics> {
  /// \brief Constructor
  ///
  /// \param _sampling_functions The sampling functions to use
  /// \param _json_sampling_functions The JSON sampling functions to use
  /// \param n_steps_per_pass Number of steps per pass.
  /// \param completion_check_params Controls when the run finishes
  SemiGrandCanonicalData(
      StateSamplingFunctionMap const &_sampling_functions,
      jsonStateSamplingFunctionMap const &_json_sampling_functions,
      CountType _n_steps_per_pass,
      CompletionCheckParams<monte::BasicStatistics> const
          &completion_check_params)
      : methods::BasicOccupationMetropolisData<monte::BasicStatistics>(
            _sampling_functions, _json_sampling_functions, _n_steps_per_pass,
            completion_check_params) {}
};

inline jsonParser &to_json(SemiGrandCanonicalData const &data,
                           jsonParser &json) {
  typedef methods::BasicOccupationMetropolisData<monte::BasicStatistics>
      base_type;
  return to_json(static_cast<base_type const &>(data), json);
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
  default_write_run_status(*mc_calculator.data, method_log, sout);

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

  default_write_completion_check_status(*mc_calculator.data, sout);
  default_finish_write_status(*mc_calculator.data, method_log);
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

  /// \brief The current thermodynamic conditions
  ///
  /// Set in `run` method
  std::shared_ptr<SemiGrandCanonicalConditions> conditions;

  /// \brief The semi-grand canonical energy calculator
  potential_type potential;

  /// \brief The formation energy calculator, set to calculate using the current
  /// state
  formation_energy_f_type *formation_energy_calculator;

  /// \brief The parametric composition calculator, set to calculate using the
  /// current state
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
  /// \param json_sampling_functions The JSON sampling functions to use
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
  ///     according to ``method_log->log_frequency``. Default writes the
  ///     current completion check results to `method_log->logfile_path` and
  ///     prints a summary of the current state and sampled data to stdout.
  /// \return Data structure containing simulation results, including sampled
  /// data,
  ///     completion check results, etc.
  template <typename WriteStatusF>
  std::shared_ptr<SemiGrandCanonicalData> run(
      state_type &state, StateSamplingFunctionMap const &sampling_functions,
      jsonStateSamplingFunctionMap const &json_sampling_functions,
      CompletionCheckParams<BasicStatistics> const &completion_check_params,
      event_generator_type event_generator, int sample_period = 1,
      std::optional<MethodLog> method_log = std::nullopt,
      std::shared_ptr<engine_type> random_engine = nullptr,
      WriteStatusF write_status_f =
          default_write_status<SemiGrandCanonicalCalculator>) {
    // ### Setup ####

    // set state & conditions
    this->state = &state;
    this->conditions = std::make_shared<SemiGrandCanonicalConditions>(
        SemiGrandCanonicalConditions::from_values(this->state->conditions));
    double temperature = this->conditions->temperature;
    CountType n_steps_per_pass = this->state->configuration.n_variable_sites;

    // set potential, pointers to other ising_cpp, define dpotential method
    this->potential.set_state(this->state, this->conditions);
    auto dpotential_f = [=](OccEvent const &e) {
      return this->potential.occ_delta_per_supercell(e);
    };

    // set event generator state, define propose and apply methods
    event_generator.set_state(this->state);
    auto propose_event_f =
        [&](random_number_generator_type &rng) -> OccEvent const & {
      return event_generator.propose(rng);
    };
    auto apply_event_f = [&](OccEvent const &e) -> void {
      event_generator.apply(e);
    };

    // define write status method
    auto _write_status_f =
        [=](methods::BasicOccupationMetropolisData<monte::BasicStatistics> const
                &data,
            MethodLog &method_log) {
          // data parameter get used in `write_status_f` via this->
          write_status_f(*this, method_log);
        };

    // construct Monte Carlo data structure
    this->data = std::make_shared<SemiGrandCanonicalData>(
        sampling_functions, json_sampling_functions, n_steps_per_pass,
        completion_check_params);

    // ### Main loop ####
    methods::basic_occupation_metropolis(
        *this->data, temperature, dpotential_f, propose_event_f, apply_event_f,
        sample_period, method_log, random_engine, _write_status_f);

    return this->data;
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

}  // namespace basic_semigrand_canonical
}  // namespace ising_cpp
}  // namespace monte
}  // namespace CASM

#endif
