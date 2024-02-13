#ifndef CASM_monte_ising_cpp_model
#define CASM_monte_ising_cpp_model

#include "casm/casm_io/container/json_io.hh"
#include "casm/global/definitions.hh"
#include "casm/global/eigen.hh"
#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/ValueMap.hh"
#include "casm/monte/events/OccEvent.hh"

namespace CASM {
namespace monte {
namespace ising_cpp {

/// \brief Ising model configuration, using an Eigen::VectorXi
///
/// Simple configuration supports single site unit cells and 2d supercells
/// without off-diagonal transformation matrix components.
class IsingConfiguration {
 public:
  IsingConfiguration() : IsingConfiguration(Eigen::VectorXi::Zero(2), 1){};

  IsingConfiguration(Eigen::VectorXi _shape, int fill_value = 1)
      : shape(_shape) {
    if (this->shape.size() != 2) {
      throw std::runtime_error("IsingConfiguration only supports 2d");
    }
    m_occupation =
        Eigen::VectorXi::Constant(this->shape[0] * this->shape[1], fill_value);
    this->n_sites = m_occupation.size();
    this->n_variable_sites = m_occupation.size();
    this->n_unitcells = m_occupation.size();
  }

  /// \brief Dimensions of the supercell, i.e. [10, 10] for a 10x10 2D supercell
  Eigen::VectorXi shape;

  /// \brief Total number of sites in the supercell
  Index n_sites;

  /// \brief Number of variable sites in the supercell
  Index n_variable_sites;

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
  Index to_linear_site_index(
      Eigen::Ref<Eigen::VectorXi const> multi_index) const {
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
class IsingState {
 public:
  IsingState(IsingConfiguration _configuration, ValueMap _conditions,
             ValueMap _properties = ValueMap())
      : configuration(_configuration),
        conditions(_conditions),
        properties(_properties) {}

  /// \brief Current Monte Carlo configuration
  IsingConfiguration configuration;

  /// \brief Current thermodynamic conditions
  ValueMap conditions;

  /// \brief Current calculated properties, if applicable
  ValueMap properties;
};

/// \brief Calculates formation energy for the Ising model
///
/// Currently implements Ising model on square lattice. Could add other lattice
/// types or anisotropic bond energies.
///
class IsingFormationEnergy {
 public:
  typedef IsingState state_type;

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

  /// \brief Ising model interaction energy
  double J;

  /// \brief Enable future implementation of square vs triangular lattice, etc.
  int lattice_type;

  /// \brief State being calculated. May be nullptr until set.
  state_type const *state;

 private:
  /// \brief Used internally if calculating multi-flip delta energy
  mutable std::vector<int> m_original_value;

  /// \brief If true, calculate formation energy by creating a neighbor list
  /// when `set_state` is called and using it when the energy is calculated.
  /// If false, always use `IsingConfiguration::within` to find neighbors.
  bool m_use_nlist = true;

  /// \brief Neighbor list for formation energy (avoids double counting
  /// neighbors)
  std::vector<std::vector<Index>> m_nlist;

  /// \brief Neighbor list for delta formation energy (includes all neighbors)
  std::vector<std::vector<Index>> m_flower_nlist;

 public:
  /// \brief Set the state the formation energy is calculated for
  void set_state(state_type const *_state) {
    this->state = throw_if_null(
        _state, "Error in IsingFormationEnergy::set_state: _state==nullptr");

    if (m_use_nlist == false) {
      return;
    }

    // build neighbor list:
    IsingConfiguration const &config = this->state->configuration;
    if (config.shape.size() != 2) {
      throw std::runtime_error("IsingConfiguration only supports 2d");
    }
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
  double per_supercell() const {
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
  double per_unitcell() const {
    return this->per_supercell() / this->state->configuration.n_unitcells;
  }

  /// \brief Calculate the change in Ising model energy due to changing 1 site
  ///
  /// \param linear_site_index Linear site indices for one site that is flipped
  /// \param new_occ New occupant value
  ///
  /// \returns The change in the per_supercell formation energy (energy per
  /// supercell).
  ///
  double _single_occ_delta_per_supercell(Index linear_site_index,
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
  /// \returns dE The change in the per_supercell formation energy (energy per
  /// supercell)
  double occ_delta_per_supercell(std::vector<Index> const &linear_site_index,
                                 std::vector<int> const &new_occ) const {
    auto &config = const_cast<IsingConfiguration &>(this->state->configuration);

    if (linear_site_index.size() == 1) {
      return this->_single_occ_delta_per_supercell(linear_site_index[0],
                                                   new_occ[0]);
    } else {
      // calculate dE for each individual flip, applying changes as we go
      double dE = 0.0;
      m_original_value.clear();
      for (Index i = 0; i < linear_site_index.size(); ++i) {
        Index index = linear_site_index[i];
        int value = new_occ[i];
        dE += this->_single_occ_delta_per_supercell(index, value);
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
/// - This assumes state->configuration.occupation() has values +1/-1
/// - The parametric composition is x=1 if all sites are +1, 0 if all sites are
/// -1
class IsingParamComposition {
 public:
  typedef IsingState state_type;

  IsingParamComposition(state_type const *_state = nullptr) : state(nullptr) {
    if (state != nullptr) {
      set_state(state);
    }
  }

  /// \brief State being calculated. May be nullptr until set.
  state_type const *state;

  /// \brief Set state being calculated
  void set_state(state_type const *_state) {
    this->state = throw_if_null(
        _state, "Error in IsingParamComposition::set_state: _state==nullptr");
  }

  /// \brief Return the number of independent compositions (size of composition
  /// vector)
  Index n_independent_compositions() const { return 1; }

  /// \brief Return parametric composition (per_supercell)
  Eigen::VectorXd per_supercell() const {
    Eigen::VectorXi const &occupation = this->state->configuration.occupation();
    Eigen::VectorXd result(1);
    result[0] = static_cast<double>(occupation.size() + occupation.sum()) / 2.0;
    return result;
  }

  /// \brief Return parametric composition (per_unitcell)
  Eigen::VectorXd per_unitcell() const {
    return this->per_supercell() / this->state->configuration.n_unitcells;
  }

  /// \brief Return change in parametric composition (per_supercell)
  Eigen::VectorXd occ_delta_per_supercell(
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
class IsingSystem {
 public:
  typedef IsingState state_type;
  typedef IsingFormationEnergy formation_energy_f_type;
  typedef IsingParamComposition param_composition_f_type;

  IsingSystem(formation_energy_f_type _formation_energy_calculator,
              param_composition_f_type _param_composition_calculator)
      : formation_energy_calculator(_formation_energy_calculator),
        param_composition_calculator(_param_composition_calculator) {}

  formation_energy_f_type formation_energy_calculator;
  param_composition_f_type param_composition_calculator;
};

}  // namespace ising_cpp
}  // namespace monte
}  // namespace CASM

#endif