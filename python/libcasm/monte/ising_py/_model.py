"""Example Ising model implementation, written in Python"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import numpy.typing as npt

from libcasm.monte import ValueMap
from libcasm.monte.events import IntVector, LongVector


class IsingConfiguration:
    """Ising model configuration, using a np.ndarray

    Simple configuration supports single site unit cells and supercells without
    off-diagonal transformation matrix components.

    """

    def __init__(
        self,
        shape: tuple = (10, 10),
        fill_value: int = 1,
    ):
        # (l, m): dimensions of supercell
        self.shape: tuple = shape
        """tuple: Dimensions of the supercell, i.e. (10, 10) for a 10x10 2D supercell
        """

        # sites: np.array, dtype=int32, with integer site occupation, col-major order
        self._occupation = np.full(
            shape=self.shape, fill_value=fill_value, dtype=np.int32, order="F"
        )

        self.n_sites: int = self._occupation.size
        """ Total number of sites in the supercell """

        self.n_variable_sites: int = self._occupation.size
        """ Number of variable sites in the supercell """

        self.n_unitcells: int = self._occupation.size
        """ Number of unit cells in the supercell, which is equal to n_variable_sites. 
        """

    def occupation(self) -> npt.NDArray[np.int32]:
        """Get the current occupation (as a read-only view)"""
        readonly_view = self._occupation.view()
        readonly_view.flags.writeable = False
        return readonly_view

    def set_occupation(self, occupation: npt.NDArray[np.int32]) -> None:
        """Set the current occupation, without changing supercell shape/size"""
        if self._occupation.shape != occupation.shape:
            raise Exception("Error in set_occupation: shape mismatch")
        self._occupation[:] = occupation

    def occ(self, linear_site_index: int) -> np.int32:
        """Get the current occupation of one site"""
        return self._occupation[
            np.unravel_index(linear_site_index, self.shape, order="F")
        ]

    def set_occ(self, linear_site_index: int, new_occ: int) -> None:
        """Set the current occupation of one site"""
        self._occupation[np.unravel_index(linear_site_index, self.shape, order="F")] = (
            new_occ
        )

    @staticmethod
    def from_dict(data: dict) -> IsingConfiguration:
        """Construct from a configuration dict"""

        config = IsingConfiguration(
            shape=tuple(data["shape"]),
        )
        occ = np.array(data["occupation"], dtype=np.int32)
        config.set_occupation(occ.reshape(config.shape, order="F"))
        return config

    def to_dict(self) -> dict:
        """Construct a configuration dict"""
        return {
            "shape": list(self.occupation().shape),
            "occupation": list(self.occupation().flatten(order="F")),
        }

    def within(self, index: int, dim: int):
        """Get index for periodic equivalent within the array"""
        return index % self.shape[dim]

    def from_linear_site_index(self, linear_site_index: int):
        """Column-major unrolling index to tuple of np.ndarray indices"""
        return np.unravel_index(linear_site_index, self.shape, order="F")

    def to_linear_site_index(self, multi_index: tuple):
        """Tuple of np.ndarray indices to column-major unrolling index"""
        return np.ravel_multi_index(multi_index, self.shape, order="F")


class IsingState:
    """Ising model state, including configuration and conditions

    Attributes
    ----------
    configuration: IsingConfiguration
        Current Monte Carlo configuration
    conditions: :class:`~libcasm.monte.ValueMap`
        Current thermodynamic conditions
    properties: :class:`~libcasm.monte.ValueMap`
        Current calculated properties

    """

    def __init__(
        self,
        configuration: IsingConfiguration,
        conditions: ValueMap,
    ):
        self.configuration = configuration
        self.conditions = conditions
        self.properties = ValueMap()


class IsingFormationEnergy:
    """Calculates the formation energy of an IsingState

    .. rubric:: Notes

    - This methods implements the isotropic Ising model on square lattice.
    - This method could be extended to add other lattice types or anisotropic bond
      energies.

    .. rubric:: Constructor

    Parameters
    ----------
    J: float = 1.0
        Ising model interaction energy.

    lattice_type: int
        Lattice type. One of:

        - 1: 2-dimensional square lattice, using IsingConfiguration

    state: Optional[IsingState] = None,
        The Monte Carlo state to calculate the formation energy

    """

    def __init__(
        self,
        J: float = 1.0,
        lattice_type: int = 1,
        state: Optional[IsingState] = None,
    ):
        self.J: float = J
        """ The Ising model interaction energy """

        if lattice_type not in [1]:
            raise Exception("Unsupported lattice_type")
        self.lattice_type: int = lattice_type
        """ The Ising model lattice type 
        
        One of:

        - 1: 2-dimensional square lattice, using IsingConfiguration
        
        """

        self.state: Optional[IsingState] = None
        """ The Monte Carlo state to calculate the formation energy for """

        if state is not None:
            self.set_state(state)

        self._original_value = IntVector()

    def set_state(self, state: IsingState):
        """Set the state the formation energy is calculated for

        Parameters
        ----------
        state: IsingState
            The state for which the formation energy is calculated
        """
        if self.lattice_type == 1:
            if not isinstance(state.configuration, IsingConfiguration):
                raise Exception("IsingConfiguration is required for lattice_type == 1")
        self.state = state

    def per_supercell(self) -> float:
        """Calculates and returns the Ising model formation energy (per supercell)"""

        # formation energy, E_formation = -\sum_{NN} J s_i s_j
        config = self.state.configuration

        # read-only view of occupation array
        sites = config.occupation()

        if self.lattice_type == 1:
            e_formation = 0.0
            for i in range(sites.shape[0]):
                i_neighbor = config.within(i + 1, dim=0)
                e_formation += -self.J * np.dot(sites[i, :], sites[i_neighbor, :])

            for j in range(sites.shape[1]):
                j_neighbor = config.within(j + 1, dim=1)
                e_formation += -self.J * np.dot(sites[:, j], sites[:, j_neighbor])
            return e_formation
        else:
            raise Exception("Invalid lattice_type")

    def per_unitcell(self) -> float:
        """Calculates and returns the Ising model formation energy (per unitcell)"""
        # formation energy, e_formation = (-\sum_{NN} J s_i s_j) / n_unitcells
        return self.per_supercell() / self.state.configuration.n_unitcells

    def _single_occ_delta_per_supercell(
        self,
        linear_site_index: int,
        new_occ: int,
    ) -> float:
        """Calculate the change in Ising model energy due to changing 1 site

        Parameters
        ----------
        linear_site_index: int
            Linear site indices for sites that are flipped
        new_occ: int
            New value on site.

        Returns
        -------
        dE: float
            The change in the per_supercell formation energy (energy per supercell).
        """
        config = self.state.configuration

        # read-only view of occupation array
        sites = config.occupation()

        if self.lattice_type == 1:
            i, j = config.from_linear_site_index(linear_site_index)

            # change in site variable: +1 / -1
            # ds = s_final[i,j] - s_init[i,j]
            #   = -s_init[i,j] - s_init[i,j]
            #   = -2 * s_init[i,j]
            ds = new_occ - sites[i, j]

            # change in formation energy:
            # -J * s_final[i,j]*(s[i+1,j] + ... ) - -J * s_init[i,j]*(s[i+1,j] + ... )
            # = -J * (s_final[i,j] - s_init[i,j]) * (s[i+1,j] + ... )
            # = -J * ds * (s[i+1,j] + ... )
            de_formation = (
                -self.J
                * ds
                * (
                    sites[i, config.within(j - 1, dim=1)]
                    + sites[i, config.within(j + 1, dim=1)]
                    + sites[config.within(i - 1, dim=0), j]
                    + sites[config.within(i + 1, dim=0), j]
                )
            )

            return de_formation
        else:
            raise Exception("Invalid lattice_type")

    def occ_delta_per_supercell(
        self,
        linear_site_index: LongVector,
        new_occ: IntVector,
    ) -> float:
        """Calculate and returns the change in Ising model energy (per supercell) due \
        to changing 1 or more sites

        Parameters
        ----------
        linear_site_index: LongVector
            Linear site indices for sites that are flipped
        new_occ: IntVector
            New value on each site.

        Returns
        -------
        dE: float
            The change in the per_supercell formation energy (energy per supercell).
        """
        config = self.state.configuration

        if len(linear_site_index) == 1:
            return self._single_occ_delta_per_supercell(
                linear_site_index[0], new_occ[0]
            )
        else:
            # calculate dE for each individual flip, applying changes as we go
            dE = 0.0
            self._original_value.clear()
            for i in range(len(linear_site_index)):
                _index = linear_site_index[i]
                _value = new_occ[i]
                dE += self._single_occ_delta_per_supercell(_index, _value)
                self._original_value.push_back(config.occ(_index))
                config.set_occ(_index, _value)

            # unapply changes
            for i in range(len(linear_site_index)):
                config.set_occ(linear_site_index[i], self._original_value[i])

            return dE

    def __deepcopy__(self, memo):
        return IsingFormationEnergy(
            J=copy.deepcopy(self.J),
            lattice_type=copy.deepcopy(self.lattice_type),
        )


class IsingParamComposition:
    """Calculates the parametric composition of an IsingState

    .. rubric:: Notes

    - This assumes ``state.configuration.occupation()`` has values +1/-1
    - This method defines the parametric composition, :math:`x`, as:

      - :math:`x=1`, if all sites are +1,
      - :math:`x=0`,  if all sites are -1

    - For details on the definition of the parametric composition, see
      :cite:t:`puchala2023casm`.

    .. rubric:: Constructor

    Parameters
    ----------
    state: Optional[IsingState] = None
        The state for which the parametric composition is calculated. May also be set
        using :func:`~IsingParamComposition.set_state`.

    """

    def __init__(
        self,
        state: Optional[IsingState] = None,
    ):
        self.state = None
        if state is not None:
            self.set_state(state)

    def set_state(
        self,
        state: IsingState,
    ):
        """Set the state the parametric composition is calculated for.

        Parameters
        ----------
        state: IsingState]
            The state the parametric composition is calculated for. May also be set
            using :func:`~IsingParamComposition.set_state`.

        """
        self.state = state
        if not isinstance(state.configuration, IsingConfiguration):
            raise Exception("IsingConfiguration is required for IsingParamComposition")

    def n_independent_compositions(self):
        """Returns the number of independent composition variables"""
        return 1

    def per_supercell(self) -> npt.NDArray[np.double]:
        """Calculates and returns the parametric composition (per_supercell)"""
        config = self.state.configuration
        n_variable_sites = config.n_variable_sites
        return np.array(
            [(n_variable_sites + np.sum(config.occupation())) / 2.0], dtype=np.double
        )

    def per_unitcell(self) -> npt.NDArray[np.double]:
        """Calculates and returns the parametric composition (per_unitcell)"""
        return self.per_supercell() / self.state.configuration.n_unitcells

    def occ_delta_per_supercell(
        self,
        linear_site_index: LongVector,
        new_occ: IntVector,
    ) -> npt.NDArray[np.double]:
        """Calculate and returns the change in parametric composition (per supercell) \
        due to changing 1 or more sites"""
        config = self.state.configuration
        Ndx = np.array([0.0], dtype=np.double)
        for i in range(len(linear_site_index)):
            _index = linear_site_index[i]
            _value = new_occ[i]
            Ndx[0] += _value - config.occ(_index)

        return Ndx


class IsingSystem:
    """Holds methods and data for calculating Ising system properties"""

    def __init__(
        self,
        formation_energy_calculator: IsingFormationEnergy,
        param_composition_calculator: Optional[IsingParamComposition] = None,
    ):
        self.formation_energy_calculator = formation_energy_calculator
        self.param_composition_calculator = param_composition_calculator
