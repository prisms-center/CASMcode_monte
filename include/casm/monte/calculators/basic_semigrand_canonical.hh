/// This file contains an example semi-grand canonical calculator
///
/// Notes:
/// - This is equivalent to `complete_semigrand_canonical`, but it
///   demonstrates how how to make use of the
///   basic_occupation_metropolis data structure and method.
/// - The `complete_semigrand_canonical` calculator includes a
///   data structure and the main loop in the `run`
///   function to show the entire method in one file.

#include "casm/casm_io/container/stream_io.hh"
#include "casm/global/definitions.hh"
#include "casm/global/eigen.hh"
#include "casm/monte/BasicStatistics.hh"
#include "casm/monte/MethodLog.hh"
#include "casm/monte/checks/CompletionCheck.hh"
#include "casm/monte/checks/io/json/CompletionCheck_json_io.hh"
#include "casm/monte/methods/basic_occupation_metropolis.hh"
#include "casm/monte/methods/metropolis.hh"
#include "casm/monte/sampling/Sampler.hh"
#include "casm/monte/state/StateSampler.hh"
#include "casm/monte/state/ValueMap.hh"
#include "casm/monte/state/io/json/ValueMap_json_io.hh"

namespace CASM {
namespace monte {
namespace calculators {
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
      : temperature(_temperature),
        beta(1.0 / (KB * temperature)),
        exchange_potential(_exchange_potential) {}

  /// \brief The temperature, \f$T\f$.
  double temperature;

  /// \brief The reciprocal temperature, \f$\beta = 1/(k_B T)\f$
  double beta;

  /// \brief The semi-grand canonical exchange potential
  ///
  /// The semi-grand canonical exchange potential, conjugate to the
  /// arametric composition that will be calculated by the
  /// `composition_calculator` of the system under consideration.
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
/// Implements the (extensive) semi-grand canonical energy:
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
  typedef typename system_type::composition_f_type composition_f_type;

  SemiGrandCanonicalPotential(std::shared_ptr<system_type> _system,
                              state_type const *_state = nullptr)
      : system(_system),
        state(nullptr),
        formation_energy_calculator(system->formation_energy_calculator),
        composition_calculator(system->composition_calculator) {
    if (_state != nullptr) {
      this->set_state(_state);
    }
  }

  /// \brief Holds parameterized calculators, without specifying at a particular
  /// state
  std::shared_ptr<system_type> system;

  /// \brief The current state during the calculation
  state_type const *state;

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
  composition_f_type composition_calculator;

  /// \brief Set the current Monte Carlo state
  void set_state(state_type const *_state) {
    if (_state == nullptr) {
      throw std::runtime_error("Error: state is nullptr");
    }
    this->state = _state;
    this->formation_energy_calculator.set_state(_state);
    this->composition_calculator.set_state(_state);
  }

  /// \brief Calculates semi-grand canonical energy (per supercell)
  double extensive_value() {
    return this->formation_energy_calculator.extensive_value() -
           this->state->conditions.exchange_potential.dot(
               this->composition_calculator.extensive_value());
  }

  /// \brief Calculates semi-grand canonical energy (per unit cell)
  double intensive_value() {
    return this->extensive_value() / this->state->configuration.n_unitcells;
  }

  /// \brief Calculates the change in semi-grand canonical energy (per
  /// supercell)
  double occ_delta_extensive_value(std::vector<Index> const &linear_site_index,
                                   std::vector<int> const &new_occ) const {
    // de_potential = e_potential_final - e_potential_init
    //   = (e_formation_final - n_unitcells * mu @ x_final) -
    //     (e_formation_init - n_unitcells * mu @ x_init)
    //   = de_formation - n_unitcells * mu * dx

    double dE_f = this->formation_energy_calculator.occ_delta_extensive_value(
        linear_site_index, new_occ);
    auto const &mu_exchange = this->state->conditions.exchange_potential;
    Eigen::VectorXd Ndx =
        this->composition_calculator.occ_delta_extensive_value(
            linear_site_index, new_occ);
    double dE_pot = dE_f - mu_exchange.dot(Ndx);
    // return dE_f - mu_exchange.dot(Ndx);
    return dE_pot;
  }

  /// \brief Calculates the change in semi-grand canonical energy (per
  /// supercell)
  double occ_delta_extensive_value(OccEvent const &e) const {
    return occ_delta_extensive_value(e.linear_site_index, e.new_occ);
  }
};

/// \brief Holds semi-grand canonical Metropolis Monte Carlo run data and results
struct SemiGrandCanonicalData
    : public methods::BasicOccupationMetropolisData<monte::BasicStatistics> {

  /// \brief Constructor
  ///
  /// \param sampling_functions The sampling functions to use
  /// \param n_steps_per_pass Number of steps per pass.
  /// \param completion_check_params Controls when the run finishes
  SemiGrandCanonicalData(StateSamplingFunctionMap const &sampling_functions,
                         CountType _n_steps_per_pass,
                         CompletionCheckParams<monte::BasicStatistics> const
                             &completion_check_params)
      : methods::BasicOccupationMetropolisData<monte::BasicStatistics>(
            sampling_functions, _n_steps_per_pass, completion_check_params) {}
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
void write_no_status(
    SemiGrandCanonicalCalculatorType const &mc_calculator, MethodLog &method_log) {
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
  auto const &composition_calculator = *mc_calculator.composition_calculator;
  auto const &formation_energy_calculator =
      *mc_calculator.formation_energy_calculator;
  Eigen::VectorXd param_composition = composition_calculator.intensive_value();
  double formation_energy = formation_energy_calculator.intensive_value();
  sout << "  ";
  sout << "ParametricComposition=" << param_composition.transpose() << ", ";
  sout << "FormationEnergy=" << formation_energy << std::endl;

  default_write_completion_check_status(*mc_calculator.data, sout);
  default_finish_write_status(*mc_calculator.data, method_log);
}

/// \brief A semi-grand canonical Monte Carlo calculator
template <typename SystemType, typename EventGeneratorType>
class SemiGrandCanonicalCalculator {
 public:
  typedef SystemType system_type;
  typedef typename system_type::state_type state_type;
  typedef typename system_type::formation_energy_f_type formation_energy_f_type;
  typedef typename system_type::composition_f_type composition_f_type;

  typedef SemiGrandCanonicalPotential<system_type> potential_type;

  typedef EventGeneratorType event_generator_type;
  typedef typename event_generator_type::engine_type engine_type;
  typedef typename event_generator_type::random_number_generator_type
      random_number_generator_type;

  /// \brief Constructor
  SemiGrandCanonicalCalculator(std::shared_ptr<system_type> _system)
      : system(_system),
        state(nullptr),
        potential(_system),
        formation_energy_calculator(&potential.formation_energy_calculator),
        composition_calculator(&potential.composition_calculator) {}

  /// \brief Holds parameterized calculators, without specifying at a particular
  /// state
  std::shared_ptr<system_type> system;

  /// \brief The current state during the calculation
  state_type *state;

  /// \brief The semi-grand canonical energy calculator
  potential_type potential;

  /// \brief The formation energy calculator, set to calculate using the current
  /// state
  formation_energy_f_type *formation_energy_calculator;

  /// \brief The parametric composition calculator, set to calculate using the
  /// current
  ///     state
  ///
  /// This is expected to calculate the compositions conjugate to the
  /// the exchange potentials provided by
  /// `state->conditions.exchange_potential`.
  composition_f_type *composition_calculator;

  /// \brief Monte Carlo run data (samplers, completion_check, n_pass, etc.)
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
  template <typename WriteStatusF>
  void run(
      state_type &state, StateSamplingFunctionMap const &sampling_functions,
      CompletionCheckParams<BasicStatistics> const &completion_check_params,
      event_generator_type event_generator, int sample_period = 1,
      std::optional<MethodLog> method_log = std::nullopt,
      std::shared_ptr<engine_type> random_engine = nullptr,
      WriteStatusF write_status_f =
          default_write_status<SemiGrandCanonicalCalculator>) {
    // ### Setup ####

    // set state
    this->state = &state;
    double temperature = this->state->conditions.temperature;
    CountType n_steps_per_pass = this->state->configuration.n_sites;

    // set potential, pointers to other calculators, dpotential method
    this->potential.set_state(this->state);
    auto dpotential_f = [=](OccEvent const &e) {
      return this->potential.occ_delta_extensive_value(e);
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

    // set write status method
    auto _write_status_f =
        [=](methods::BasicOccupationMetropolisData<monte::BasicStatistics> const &data,
            MethodLog &method_log) {
          // data parameter get used in `write_status_f` via this->
          write_status_f(*this, method_log);
        };

    // construct Monte Carlo data structure
    this->data = std::make_shared<SemiGrandCanonicalData>(
        sampling_functions, n_steps_per_pass, completion_check_params);

    // ### Main loop ####
    methods::basic_occupation_metropolis(
        *this->data, temperature, dpotential_f, propose_event_f, apply_event_f,
        sample_period, method_log, random_engine, _write_status_f);
  }
};

/// \brief Returns a parametric composition sampling function
///
/// The sampling function "parametric_composition" gets the
/// parametric composition from:
/// \code
/// mc_calculator->composition_calculator->intensive_value()
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
  shape.push_back(mc_calculator->system->composition_calculator
                      .n_independent_compositions());
  auto f = [mc_calculator]() -> Eigen::VectorXd {
    if (mc_calculator == nullptr) {
      throw std::runtime_error(
          "Error in parametric_composition sampling function: "
          "mc_calculator == nullptr");
    }
    if (mc_calculator->composition_calculator == nullptr) {
      throw std::runtime_error(
          "Error in parametric_composition sampling function: "
          "mc_calculator->composition_calculator == nullptr");
    }
    if (mc_calculator->composition_calculator->state == nullptr) {
      throw std::runtime_error(
          "Error in parametric_composition sampling function: "
          "mc_calculator->composition_calculator->state == nullptr");
    }
    return mc_calculator->composition_calculator->intensive_value();
  };
  return StateSamplingFunction(name, description, shape, f);
}  // namespace basic_semigrand_canonical

/// \brief Returns a formation energy sampling function
///
/// The sampling function "formation_energy" gets the formation energy from:
/// \code
/// mc_calculator->formation_energy_calculator->intensive_value()
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
    v(0) = mc_calculator->formation_energy_calculator->intensive_value();
    return v;
  };
  return StateSamplingFunction(name, description, shape, f);
}

/// \brief Returns a potential energy sampling function
///
/// The sampling function "potential_energy" gets the formation energy from:
/// \code
/// mc_calculator->potential.intensive_value()
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
    v(0) = mc_calculator->potential.intensive_value();
    return v;
  };
  return StateSamplingFunction(name, description, shape, f);
}

}  // namespace basic_semigrand_canonical
}  // namespace calculators
}  // namespace monte
}  // namespace CASM