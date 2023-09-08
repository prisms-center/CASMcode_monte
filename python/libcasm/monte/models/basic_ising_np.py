"""Example Ising model implementation, using a numpy array"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import numpy.typing as npt

from libcasm.monte import RandomNumberGenerator, ValueMap
from libcasm.monte.basic_run_typing import ConditionsType
from libcasm.monte.events import IntVector, LongVector, OccEvent, OccLocation


class IsingConfiguration:
    """Ising model configuration, using a np.ndarray

    Simple configuration supports single site unit cells and supercells without
    off-diagonal transformation matrix components.

    Attributes
    ----------
    shape: tuple
        Dimensions of the supercell, i.e. (10, 10) for a 10x10 2D supercell
    n_sites:
        Number of sites in the supercell
    n_unitcells:
        Number of unit cells in the supercell, which is equal to n_sites.
    """

    def __init__(
        self,
        shape: tuple = (10, 10),
        fill_value: int = 1,
    ):
        # (l, m): dimensions of supercell
        self.shape = shape

        # sites: np.array, dtype=int32, with integer site occupation, col-major order
        self._occupation = np.full(
            shape=self.shape, fill_value=fill_value, dtype=np.int32, order="F"
        )

        self.n_sites = self._occupation.size
        self.n_unitcells = self._occupation.size

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
        self._occupation[
            np.unravel_index(linear_site_index, self.shape, order="F")
        ] = new_occ

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
    """Ising state, including configuration and conditions

    Attributes
    ----------
    configuration: IsingConfiguration
        Current Monte Carlo configuration
    conditions: ConditionsType
        Current thermodynamic conditions
    properties: :class:`~libcasm.monte.ValueMap`
        Current calculated properties

    """

    def __init__(
        self,
        configuration: IsingConfiguration,
        conditions: ConditionsType,
    ):
        self.configuration = configuration
        self.conditions = conditions
        self.properties = ValueMap()


class IsingSemiGrandCanonicalEventGenerator:
    """Propose and apply semi-grand canonical Ising model events

    Implements OccEventGenerator protocol.

    Attributes
    ----------
    state: IsingState
        The current state for which events are proposed and applied
    occ_location: Optional[:class:`~libcasm.monte.events.OccLocation`]
        Some event generators require the use of OccLocation for proposing and
        applying events. If provided, this should be used.
    occ_event: monte.events.OccEvent
        The current proposed event
    """

    def __init__(
        self,
        state: Optional[IsingState] = None,
        occ_location: Optional[OccLocation] = None,
    ):
        # construct occ_event
        e = OccEvent()
        e.linear_site_index.clear()
        e.linear_site_index.append(0)
        e.new_occ.clear()
        e.new_occ.append(1)
        self.occ_event = e

        if state is not None:
            self.set_state(state, occ_location)

    def set_state(
        self,
        state: IsingState,
        occ_location: Optional[OccLocation] = None,
    ):
        """Set the current Monte Carlo state and occupant locations

        Parameters
        ----------
        state: IsingState
            The current state for which events are proposed and applied
        occ_location: Optional[:class:`~libcasm.monte.events.OccLocation`] = None
            Some event generators require the use of OccLocation for proposing and
            applying events. If provided, this should be used.
        """
        self.state = state

        self._max_linear_site_index = self.state.configuration.n_sites - 1

    def propose(
        self,
        random_number_generator: RandomNumberGenerator,
    ):
        """Propose a Monte Carlo occupation event, by setting self.occ_event"""
        self.occ_event.linear_site_index[0] = random_number_generator.random_int(
            self._max_linear_site_index
        )
        self.occ_event.new_occ[0] = -self.state.configuration.occ(
            self.occ_event.linear_site_index[0]
        )

    def apply(
        self,
    ):
        """Update the occupation of the current state, using self.occ_event"""
        self.state.configuration.set_occ(
            self.occ_event.linear_site_index[0], self.occ_event.new_occ[0]
        )


class IsingFormationEnergy:
    """Calculates formation energy for the Ising model

    Implements PropertyCalculatorType protocol.

    Currently implements Ising model on square lattice. Could add other lattice types or
    anisotropic bond energies.

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
        self.J = J

        if lattice_type not in [1]:
            raise Exception("Unsupported lattice_type")
        self.lattice_type = lattice_type

        self.state = None
        if state is not None:
            self.set_state(state)

        self._original_value = IntVector()

    def set_state(self, state) -> None:
        """Set the state the formation energy is calculated for"""
        if self.lattice_type == 1:
            if not isinstance(state.configuration, IsingConfiguration):
                raise Exception("IsingConfiguration is required for lattice_type == 1")
        self.state = state

    def extensive_value(self) -> float:
        """Calculates Ising model formation energy (per supercell) for self.state"""

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

    def intensive_value(self) -> float:
        """Calculates Ising model formation energy (per unitcell) for self.state"""
        # formation energy, e_formation = (-\sum_{NN} J s_i s_j) / n_unitcells
        return self.extensive_value() / self.state.configuration.n_unitcells

    def _single_occ_delta_extensive_value(
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
            The change in the extensive formation energy (energy per supercell).
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

    def occ_delta_extensive_value(
        self,
        linear_site_index: LongVector,
        new_occ: IntVector,
    ) -> float:
        """Calculate the change in Ising model energy due to changing 1 or more sites

        Parameters
        ----------
        linear_site_index: LongVector
            Linear site indices for sites that are flipped
        new_occ: IntVector
            New value on each site.

        Returns
        -------
        dE: float
            The change in the extensive formation energy (energy per supercell).
        """
        config = self.state.configuration

        if len(linear_site_index) == 1:
            return self._single_occ_delta_extensive_value(
                linear_site_index[0], new_occ[0]
            )
        else:
            # calculate dE for each individual flip, applying changes as we go
            dE = 0.0
            self._original_value.clear()
            for i in range(len(linear_site_index)):
                _index = linear_site_index[i]
                _value = new_occ[i]
                dE += self._single_occ_delta_extensive_value(_index, _value)
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


class IsingCompositionCalculator:
    """Calculate parametric composition from np.ndarray of +1/-1 site occupation

    Notes:
    - Implements PropertyCalculator protocol.
    - This assumes state.configuration.occupation() has values +1/-1
    - The parametric composition is x=1 if all sites are +1, 0 if all sites are -1
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
        state: Optional[IsingState] = None,
    ):
        """Set self.state"""
        self.state = state
        if not isinstance(state.configuration, IsingConfiguration):
            raise Exception(
                "IsingConfiguration is required for IsingCompositionCalculator"
            )

    def n_independent_compositions(self):
        return 1

    def extensive_value(self) -> npt.NDArray[np.double]:
        """Return parametric composition (extensive)"""
        config = self.state.configuration
        n_sites = config.n_sites
        return np.array(
            [(n_sites + np.sum(config.occupation())) / 2.0], dtype=np.double
        )

    def intensive_value(self) -> npt.NDArray[np.double]:
        """Return parametric composition"""
        return self.extensive_value() / self.state.configuration.n_unitcells

    def occ_delta_extensive_value(
        self,
        linear_site_index: LongVector,
        new_occ: IntVector,
    ) -> npt.NDArray[np.double]:
        """Return change in parametric composition (extensive)"""
        config = self.state.configuration
        Ndx = np.array([0.0], dtype=np.double)
        for i in range(len(linear_site_index)):
            _index = linear_site_index[i]
            _value = new_occ[i]
            Ndx[0] += (_value-config.occ(_index))

        return Ndx


class IsingSystem:
    """Holds methods and data for calculating Ising system properties"""

    def __init__(
        self,
        formation_energy_calculator: IsingFormationEnergy,
        composition_calculator: IsingCompositionCalculator,
    ):
        self.formation_energy_calculator = formation_energy_calculator
        self.composition_calculator = composition_calculator
