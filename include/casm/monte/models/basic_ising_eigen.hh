#include "casm/casm_io/container/json_io.hh"
#include "casm/global/definitions.hh"
#include "casm/global/eigen.hh"
#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/events/OccEvent.hh"
#include "casm/monte/state/ValueMap.hh"

namespace CASM {
namespace monte {
namespace models {
namespace basic_ising_eigen {

/// \brief Ising model configuration, using an Eigen::VectorXi
///
/// Simple configuration supports single site unit cells and 2d supercells
/// without off-diagonal transformation matrix components.
class IsingConfiguration {
 public:
  IsingConfiguration() : IsingConfiguration(Eigen::VectorXi::Zero(2), 1){};

  IsingConfiguration(Eigen::Ref<Eigen::VectorXi const> _shape,
                     int fill_value = 1)
      : shape(_shape) {
    if (this->shape.size() != 2) {
      throw std::runtime_error("IsingConfiguration only supports 2d");
    }
    m_occupation =
        Eigen::VectorXi::Constant(this->shape[0] * this->shape[1], fill_value);
    this->n_sites = m_occupation.size();
    this->n_unitcells = m_occupation.size();
  }

  /// \brief Dimensions of the supercell, i.e. [10, 10] for a 10x10 2D supercell
  Eigen::VectorXi shape;

  /// \brief Number of sites in the supercell
  Index n_sites;

  /// \brief Number of unitcells in the supercell, which is equal to n_sites.
  Index n_unitcells;

 private:
  Eigen::VectorXi m_occupation;

 public:
  /// \brief Get the current occupation (as const reference)
  Eigen::VectorXi const &occupation() const { return m_occupation; }

  /// \brief Set the current occupation, without changing supercell shape/size
  void set_occupation(Eigen::Ref<Eigen::VectorXi const> occupation) {
    if (m_occupation.size() != occupation.size()) {
      throw std::runtime_error("Error in set_occupation: size mismatch");
    }
    m_occupation = occupation;
  }

  /// \brief Get the current occupation of one site
  int occ(Index linear_site_index) const {
    return m_occupation[linear_site_index];
  }

  /// \brief Set the current occupation of one site
  void set_occ(Index linear_site_index, int new_occ) {
    m_occupation[linear_site_index] = new_occ;
  }

  /// \brief Get index for periodic equivalent within the array
  Index within(Index index, int dim) const {
    Index result = index % this->shape[dim];
    if (result < 0) {
      result += this->shape[dim];
    }
    return result;
  }

  /// \brief Column-major unrolling index to Eigen::VectorXi of indices
  Eigen::VectorXi from_linear_site_index(Index linear_site_index) const {
    if (this->shape.size() == 2) {
      Eigen::VectorXi multi_index(2);
      multi_index[0] = linear_site_index % this->shape[0];
      multi_index[1] = linear_site_index / this->shape[0];
      return multi_index;
    }
    throw std::runtime_error("IsingConfiguration only supports 2d");
  };

  /// \brief Eigen::VectorXi of indices to column-major unrolling index
  Index to_linear_site_index(Eigen::VectorXi const &multi_index) const {
    if (this->shape.size() == 2) {
      return this->shape[0] * multi_index[1] + multi_index[0];
    }
    throw std::runtime_error("IsingConfiguration only supports 2d");
  }

  /// \brief 2d indices to column-major unrolling index
  Index to_linear_site_index(Index row, Index col) const {
    if (this->shape.size() == 2) {
      Eigen::VectorXi multi_index(2);
      multi_index << row, col;
      return to_linear_site_index(multi_index);
    }
    throw std::runtime_error("IsingConfiguration only supports 2d");
  }
};

/// \brief Construct IsingConfiguration from JSON
inline void from_json(IsingConfiguration &config, jsonParser const &json) {
  if (!json.contains("shape")) {
    throw std::runtime_error(
        "Error reading IsingConfiguration from JSON: no 'shape'");
  }
  Eigen::VectorXi shape;
  from_json(shape, json["shape"]);

  if (!json.contains("occupation")) {
    throw std::runtime_error(
        "Error reading IsingConfiguration from JSON: no 'occupation'");
  }
  Eigen::VectorXi occupation;
  from_json(occupation, json["occupation"]);

  config = IsingConfiguration(shape);
  config.set_occupation(occupation);
}

/// \brief Write IsingConfiguration to JSON
inline jsonParser &to_json(IsingConfiguration const &config, jsonParser &json) {
  json.put_obj();
  to_json_array(config.shape, json["shape"]);
  to_json_array(config.occupation(), json["occupation"]);
  return json;
}

/// \brief Ising state, including configuration and conditions
template <typename ConditionsType>
class IsingState {
 public:
  typedef ConditionsType conditions_type;

  IsingState(IsingConfiguration _configuration, ConditionsType _conditions)
      : configuration(_configuration), conditions(_conditions) {}

  /// \brief Current Monte Carlo configuration
  IsingConfiguration configuration;

  /// \brief Current thermodynamic conditions
  conditions_type conditions;

  /// \brief Current calculated properties, if applicable
  ValueMap properties;
};

/// \brief Propose and apply semi-grand canonical Ising model events
template <typename ConditionsType, typename EngineType>
class IsingSemiGrandCanonicalEventGenerator {
 public:
  typedef ConditionsType conditions_type;
  typedef IsingState<conditions_type> state_type;
  typedef EngineType engine_type;
  typedef RandomNumberGenerator<engine_type> random_number_generator_type;

  /// \brief Constructor
  ///
  /// \param _state The current state for which events are proposed and applied.
  /// Can be
  ///     nullptr, but must be set for use.
  /// \param _occ_location Some event generators require the use of OccLocation
  /// for
  ///     proposing and applying events. May be nullptr. If provided, this
  ///     should be used. Not used for this event generator.
  IsingSemiGrandCanonicalEventGenerator(state_type *_state = nullptr,
                                        OccLocation *_occ_location = nullptr)
      : state(nullptr), occ_location(nullptr), m_max_linear_site_index(0) {
    occ_event.linear_site_index.clear();
    occ_event.linear_site_index.push_back(0);
    occ_event.new_occ.clear();
    occ_event.new_occ.push_back(1);

    if (_state != nullptr) {
      set_state(_state, _occ_location);
    }
  }

  /// \brief The current state for which events are proposed and applied. Can be
  ///     nullptr, but must be set for use.
  state_type *state;

  /// \brief Some event generators require the use of OccLocation for
  ///     proposing and applying events.  Not used for this event generator.
  OccLocation *occ_location;

  /// \brief The current proposed event
  OccEvent occ_event;

 private:
  Index m_max_linear_site_index;

 public:
  /// \brief Set the current Monte Carlo state and occupant locations
  ///
  /// \param _state The current state for which events are proposed and applied.
  /// Throws
  ///     if nullptr.
  /// \param _occ_location Some event generators require the use of OccLocation
  /// for
  ///     proposing and applying events.  Not used for this event generator.
  void set_state(state_type *_state, OccLocation *_occ_location = nullptr) {
    if (_state == nullptr) {
      throw std::runtime_error(
          "Error: IsingSemiGrandCanonicalEventGenerator::set_state with "
          "nullptr");
    }
    this->state = _state;

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

/// \brief Calculates formation energy for the Ising model
///
/// Implements PropertyCalculatorType protocol.
///
/// Currently implements Ising model on square lattice. Could add other lattice
/// types or anisotropic bond energies.
///
template <typename ConditionsType>
class IsingFormationEnergy {
 public:
  typedef ConditionsType conditions_type;
  typedef IsingState<conditions_type> state_type;

  IsingFormationEnergy(double _J = 1.0, int _lattice_type = 1,
                       bool _use_nlist = true,
                       state_type const *_state = nullptr)
      : J(_J),
        lattice_type(_lattice_type),
        state(nullptr),
        m_use_nlist(_use_nlist) {
    if (this->lattice_type != 1) {
      throw std::runtime_error("Unsupported lattice_type");
    }
    if (state != nullptr) {
      set_state(state);
    }
  }

  double J;
  int lattice_type;
  state_type const *state;

 private:
  mutable std::vector<int> m_original_value;

  bool m_use_nlist = true;

  /// \brief Neighbor list for formation energy
  std::vector<std::vector<Index>> m_nlist;

  /// \brief Neighbor list for delta formation energy
  std::vector<std::vector<Index>> m_flower_nlist;

 public:
  /// \brief Set the state the formation energy is calculated for
  void set_state(state_type const *_state) {
    this->state = _state;

    if (m_use_nlist == false) {
      return;
    }

    // build neighbor list:
    IsingConfiguration const &config = this->state->configuration;
    if (config.shape.size() != 2) {
      throw std::runtime_error("IsingConfiguration only supports 2d");
    }
    Index rows = config.shape[0];
    Index cols = config.shape[1];
    Eigen::VectorXi multi_index;
    Index i;
    Index j;
    Index i_neighbor;
    Index j_neighbor;

    if (this->lattice_type == 1) {
      m_nlist.clear();
      m_nlist.resize(config.n_sites);
      m_flower_nlist.clear();
      m_flower_nlist.resize(config.n_sites);
      for (Index l = 0; l < config.n_sites; ++l) {
        multi_index = config.from_linear_site_index(l);
        i = multi_index[0];
        j = multi_index[1];

        i_neighbor = config.within(i + 1, 0);
        m_nlist[l].push_back(config.to_linear_site_index(i_neighbor, j));
        m_flower_nlist[l].push_back(config.to_linear_site_index(i_neighbor, j));

        j_neighbor = config.within(j + 1, 1);
        m_nlist[l].push_back(config.to_linear_site_index(i, j_neighbor));
        m_flower_nlist[l].push_back(config.to_linear_site_index(i, j_neighbor));

        i_neighbor = config.within(i - 1, 0);
        m_flower_nlist[l].push_back(config.to_linear_site_index(i_neighbor, j));

        j_neighbor = config.within(j - 1, 1);
        m_flower_nlist[l].push_back(config.to_linear_site_index(i, j_neighbor));
      }
    } else {
      throw std::runtime_error("Invalid lattice_type");
    }
  }

  /// \brief Calculates Ising model formation energy (per supercell)
  double extensive_value() const {
    if (this->lattice_type == 1) {
      if (m_use_nlist) {
        IsingConfiguration const &config = this->state->configuration;
        Eigen::VectorXi const &occ = config.occupation();
        Index n_sites = config.n_sites;
        double e_formation = 0.0;
        for (Index l = 0; l < n_sites; ++l) {
          e_formation += occ(l) * (occ(m_nlist[l][0]) + occ(m_nlist[l][1]));
        }
        e_formation *= -this->J;
        return e_formation;
      } else {
        IsingConfiguration const &config = this->state->configuration;
        Index rows = config.shape[0];
        Index cols = config.shape[1];
        auto sites = config.occupation().reshaped(rows, cols);
        double e_formation = 0.0;
        for (Index i = 0; i < sites.rows(); ++i) {
          Index i_neighbor = config.within(i + 1, 0);
          e_formation += -this->J * sites.row(i).dot(sites.row(i_neighbor));
        }
        for (Index j = 0; j < sites.cols(); ++j) {
          Index j_neighbor = config.within(j + 1, 1);
          e_formation += -this->J * sites.col(j).dot(sites.col(j_neighbor));
        }
        return e_formation;
      }
    } else {
      throw std::runtime_error("Invalid lattice_type");
    }
  }

  /// \brief Calculates Ising model formation energy (per unit cell)
  double intensive_value() const {
    return this->extensive_value() / this->state->configuration.n_unitcells;
  }

  /// \brief Calculate the change in Ising model energy due to changing 1 site
  ///
  /// \param linear_site_index Linear site indices for one site that is flipped
  /// \param new_occ New occupant value
  ///
  /// \returns The change in the extensive formation energy (energy per
  /// supercell).
  ///
  double _single_occ_delta_extensive_value(Index linear_site_index,
                                           int new_occ) const {
    if (this->lattice_type == 1) {
      if (m_use_nlist) {
        IsingConfiguration const &config = this->state->configuration;
        Eigen::VectorXi const &occ = config.occupation();
        Index l = linear_site_index;
        return -this->J * (new_occ - occ(l)) *
               (occ(m_flower_nlist[l][0]) + occ(m_flower_nlist[l][1]) +
                occ(m_flower_nlist[l][2]) + occ(m_flower_nlist[l][3]));
      } else {
        auto const &config = this->state->configuration;
        Index rows = config.shape[0];
        Index cols = config.shape[1];
        auto sites = config.occupation().reshaped(rows, cols);

        Eigen::VectorXi multi_index =
            config.from_linear_site_index(linear_site_index);
        int i = multi_index[0];
        int j = multi_index[1];

        // change in site variable: +1 / -1
        // ds = s_final[i, j] - s_init[i, j]
        //   = -s_init[i, j] - s_init[i, j]
        //   = -2 * s_init[i, j]
        double ds = new_occ - sites(i, j);

        // change in formation energy:
        // -J * s_final[i, j] * (s[i + 1, j] + ... ) - -J * s_init[i, j] * (s[i
        // + 1, j] + ... ) = -J * (s_final[i, j] - s_init[i, j]) * (s[i + 1, j]
        // + ... ) = -J * ds * (s[i + 1, j] + ... )
        return -this->J * ds *
               (sites(i, config.within(j - 1, 1)) +
                sites(i, config.within(j + 1, 1)) +
                sites(config.within(i - 1, 0), j) +
                sites(config.within(i + 1, 0), j));
      }
    } else {
      throw std::runtime_error("Invalid lattice_type");
    }
  }

  /// \brief Calculate the change in Ising model energy due to changing 1 or
  /// more sites
  ///
  /// \param linear_site_index Linear site indices for sites that are flipped
  /// \param new_occ New occupant value on each site.
  /// \returns dE The change in the extensive formation energy (energy per
  /// supercell)
  double occ_delta_extensive_value(std::vector<Index> const &linear_site_index,
                                   std::vector<int> const &new_occ) const {
    auto &config = const_cast<IsingConfiguration &>(this->state->configuration);
    auto const &sites = config.occupation();

    if (linear_site_index.size() == 1) {
      return this->_single_occ_delta_extensive_value(linear_site_index[0],
                                                     new_occ[0]);
    } else {
      // calculate dE for each individual flip, applying changes as we go
      double dE = 0.0;
      m_original_value.clear();
      for (Index i = 0; i < linear_site_index.size(); ++i) {
        Index index = linear_site_index[i];
        int value = new_occ[i];
        dE += this->_single_occ_delta_extensive_value(index, value);
        m_original_value.push_back(config.occ(index));
        config.set_occ(index, value);
      }

      // unapply changes
      for (Index i = 0; i < m_original_value.size(); ++i) {
        config.set_occ(linear_site_index[i], m_original_value[i]);
      }
      return dE;
    }
  }
};

/// \brief Calculate parametric composition of IsingConfiguration
///
/// Notes:
/// - Implements PropertyCalculator protocol.
/// - This assumes state->configuration.occupation() has values +1/-1
/// - The parametric composition is x=1 if all sites are +1, 0 if all sites are
/// -1
template <typename ConditionsType>
class IsingComposition {
 public:
  typedef ConditionsType conditions_type;
  typedef IsingState<conditions_type> state_type;

  IsingComposition(state_type const *_state = nullptr) : state(_state) {
    if (state != nullptr) {
      set_state(state);
    }
  }

  state_type const *state;

  /// \brief Set state being calculated
  void set_state(state_type const *_state) {
    this->state = _state;
    if (this->state == nullptr) {
      throw std::runtime_error("Error: state is nullptr");
    }
  }

  /// \brief Return the number of independent compositions (size of composition
  /// vector)
  Index n_independent_compositions() const { return 1; }

  /// \brief Return parametric composition (extensive)
  Eigen::VectorXd extensive_value() const {
    Eigen::VectorXi const &occupation = this->state->configuration.occupation();
    Eigen::VectorXd result(1);
    result[0] = static_cast<double>(occupation.size() + occupation.sum()) / 2.0;
    return result;
  }

  /// \brief Return parametric composition (intensive)
  Eigen::VectorXd intensive_value() const {
    return this->extensive_value() / this->state->configuration.n_unitcells;
  }

  /// \brief Return change in parametric composition (extensive)
  Eigen::VectorXd occ_delta_extensive_value(
      std::vector<Index> const &linear_site_index,
      std::vector<int> const &new_occ) const {
    auto const &config = this->state->configuration;
    Eigen::VectorXd Ndx(1);
    Ndx[0] = 0.0;
    for (Index i = 0; i < linear_site_index.size(); ++i) {
      Ndx[0] += (new_occ[i] - config.occ(linear_site_index[i])) / 2.0;
    }
    return Ndx;
  }
};

/// \brief Holds methods and data for calculating Ising system properties
template <typename ConditionsType>
class IsingSystem {
 public:
  typedef ConditionsType conditions_type;
  typedef IsingState<conditions_type> state_type;
  typedef IsingFormationEnergy<conditions_type> formation_energy_f_type;
  typedef IsingComposition<conditions_type> composition_f_type;

  IsingSystem(formation_energy_f_type _formation_energy_calculator,
              composition_f_type _composition_calculator)
      : formation_energy_calculator(_formation_energy_calculator),
        composition_calculator(_composition_calculator) {}

  formation_energy_f_type formation_energy_calculator;
  composition_f_type composition_calculator;
};

}  // namespace basic_ising_eigen
}  // namespace models
}  // namespace monte
}  // namespace CASM