"""Example square lattice Ising Model semi-grand canonical Monte Carlo implementation"""
import copy
import math
from typing import Optional

import numpy as np

import libcasm.casmglobal as casmglobal
import libcasm.monte as monte


def periodic(i, l):
    return i % l


def linear_index_to_row_col(linear_site_index, l):
    # for l x l sites array (also works if l x m)
    j = linear_site_index // l  # integer division
    i = linear_site_index - j * l
    return (i, j)


def row_col_to_linear_index(i, j, l):
    # for l x l sites array (also works if l x m)
    return j * l + i


class SquareConfiguration:
    """Monte Carlo configuration, as a square np.ndarray"""

    def __init__(
        self,
        supercell_l: int = 10,
    ):
        # supercell_l: int, dimensions of square supercell
        self.supercell_l = supercell_l

        self.n_unitcells = self.supercell_l * self.supercell_l

        # sites: np.array, dtype=int32, with site occupation
        self.sites = np.zeros((supercell_l, supercell_l), dtype="int32")


class Conditions:
    def __init__(
        self,
        temperature: float,
        exchange_potential: np.ndarray,
    ):
        self.temperature = temperature
        self.beta = 1.0 / (casmglobal.KB * self.temperature)
        self.exchange_potential = exchange_potential


class SquareState:
    """Monte Carlo state

    Attributes
    ----------
    configuration: SquareConfiguration
        Current Monte Carlo configuration
    conditions: Conditions
        Current thermodynamic conditions
    properties: monte.ValueMap
        Current calculated properties

    """

    def __init__(
        self,
        configuration: SquareConfiguration,
        conditions: Conditions,
    ):
        self.configuration = configuration
        self.conditions = conditions
        self.properties = monte.ValueMap()


class SquareIsingFormationEnergy:
    """Calculates formation energy for Ising model on square lattice

    extensive formation energy, E_formation = -\sum_{NN} J s_i s_j

    """

    def __init__(
        self,
        J: float = 1.0,
        state: Optional[SquareState] = None,
    ):
        self.J = J
        self.state = state

    def set_state(self, state):
        """Set self.state"""
        self.state = state

    def extensive_formation_energy(self):
        """Calculates formation energy (extensive) for self.state"""
        # formation energy, E_formation = -\sum_{NN} J s_i s_j
        sites = self.state.configuration.sites
        e_formation = 0.0

        rows = sites.shape[0]
        for i in range(rows):
            i_neighbor = periodic(i + 1, rows)
            e_formation += -self.J * np.sum(sites[i, :] * sites[i_neighbor, :])

        cols = sites.shape[1]
        for j in range(cols):
            j_neighbor = periodic(j + 1, cols)
            e_formation += -self.J * np.sum(sites[:, j] * sites[:, j_neighbor])
        return e_formation

    def intensive_formation_energy(self):
        """Calculates formation energy (intensive) for self.state"""
        # formation energy, e_formation = (-\sum_{NN} J s_i s_j) / n_unitcells
        n_unitcells = self.state.configuration.sites.size
        return self.extensive_formation_energy() / n_unitcells

    def occ_delta_extensive_value(
        self,
        linear_site_index,
    ):
        config = self.state.configuration
        l = config.supercell_l
        sites = config.sites
        J = self.system.J

        # '//' is integer division
        j = linear_site_index // l
        i = linear_site_index - j * l

        # change in site variable:
        # ds = s_final[i,j] - s_init[i,j]
        #   = -s_init[i,j] - s_init[i,j]
        #   = -2 * s_init[i,j]
        ds = -2 * sites[i, j]

        # change in formation energy:
        # -J * s_final[i,j]*(s[i+1,j] + ... ) - -J * s_init[i,j]*(s[i+1,j] + ... )
        # = -J * (s_final[i,j] - s_init[i,j]) * (s[i+1,j] + ... )
        # = -J * ds * (s[i+1,j] + ... )
        de_formation = (
            -J
            * ds
            * (
                sites[i, periodic(j - 1, l)]
                + sites[i, periodic(j + 1, l)]
                + sites[periodic(i - 1, l), j]
                + sites[periodic(i + 1, l), j]
            )
        )

        return de_formation


class SquareIsingCompositionCalculator:
    """Calculate parametric composition from square np.ndarray of site occupation

    This assumes sites has shape=(l,m), and values +1/-1
    The parametric composition is x=1 if all sites are +1, 0 if all sites are -1
    """

    def __init__(self):
        pass

    def n_independent_compositions(self):
        return 1

    def extensive_parametric_composition(self, sites):
        """Return parameteric composition (extensive)"""
        return np.array([(sites.size + np.sum(sites)) / 2.0])

    def parametric_composition(self, sites):
        """Return parameteric composition"""
        return self.extensive_parametric_composition(sites) / sites.size

    def delta_extensive_parametric_composition(self, sites, linear_site_index):
        """Return change in parameteric composition (extensive)"""
        l = sites.shape[0]

        # '//' is integer division
        j = linear_site_index // l
        i = linear_site_index - j * l

        # change in n_unitcells * x
        return -sites[i, j]


class SquareSystem:
    def __init__(
        self,
        formation_energy_calculator,
        composition_calculator,
    ):
        self.formation_energy_calculator = formation_energy_calculator
        self.composition_calculator = composition_calculator


class SquareSemiGrandPotential:
    """Calculate the potential and changes in potential for a state"""

    def __init__(
        self,
        system: SquareSystem,
        state: Optional[SquareState] = None,
    ):
        # E = -\sum_{NN} J s_i s_j
        # x = \sum_i s_i / 2 / \N_u
        # \Omega = E - \N_u \sum_i \tilde{\mu}_i x_i
        self.system = system
        self.formation_energy_calculator = copy.deepcopy(
            system.formation_energy_calculator
        )
        self.composition_calculator = copy.deepcopy(system.composition_calculator)
        self.set_state(state)

    def set_state(self, state):
        self.state = state
        self.formation_energy_calculator.set_state(state)

    def extensive_value(self):
        mu = self.state.conditions.exchange_potential
        sites = self.state.configuration.sites

        # formation energy, e_formation = -\sum_{NN} J s_i s_j
        e_formation_extensive = self.formation_energy_calculator.extensive_value()

        # independent composition, n_unitcells * x = \sum_i s_i / 2
        x_extensive = self.composition_calculator.extensive_parametric_composition(
            sites
        )

        return e_formation_extensive - mu @ x_extensive

    def intensive_value(self):
        n_unitcells = self.state.configuration.sites.size
        return self.extensive_value() / n_unitcells

    def occ_delta_extensive_value(
        self,
        linear_site_index,
    ):
        mu = self.state.conditions.exchange_potential
        sites = self.state.configuration.sites

        de_formation_extensive = (
            self.formation_energy_calculator.occ_delta_extensive_value(
                linear_site_index
            )
        )

        # independent composition, n_unitcells * x = \sum_i s_i / 2
        dx_extensive = (
            self.composition_calculator.delta_extensive_parametric_composition(
                sites, linear_site_index
            )
        )

        # de_potential = e_potential_final - e_potential_init
        #   = (e_formation_final - mu @ x_extensive_final) -
        #     (e_formation_init - mu @ x_extensive_init)
        #   = de_formation - mu * dx_extensive
        return de_formation_extensive - mu @ dx_extensive


class SquareSemiGrandCanonicalCalculator:
    def __init__(
        self,
        system: SquareSystem,
    ):
        # This contains the system parameters
        self.system = system

        # The current state
        self.state = None

        # The potential calculator
        self.potential = None

        # Number of passes
        self.n_pass = None

        # The number of acceptances
        self.n_accept = None

        # The number of rejections
        self.n_reject = None

    def run(
        self,
        state: SquareState,
        sampling_functions: monte.StateSamplingFunctionMap,
        completion_check_params: monte.CompletionCheckParams,
        sample_period: int = 1,
        method_log: Optional[monte.MonteLog] = None,
        random_engine: Optional[monte.RandomNumberEngine] = None,
    ):
        """
        Arguments
        ---------
        state: SquareState
            Initial Monte Carlo state, including configuration and conditions
        sampling_functions: monte.StateSamplingFunctionMap
            The sampling functions to use
        completion_check_params:
            Controls when the run finishes
        sample_period: int = 1
            Number of passes per sample. One pass is one Monte Carlo step per site.
        method_log: Optional[monte.MonteLog] = None,
            Method log, for writing status updates. If None, default
            writes to "log.txt" every 10 minutes.
        random_engine: Optional[monte.RandomNumberEngine] = None
            Random number engine. Default constructs a new engine.

        Returns
        -------
        (samplers, completion_check_results)

            samplers: monte.SamplerMap
            completion_check_results: monte.CompletionCheckResults
        """
        self.state = state
        sites = self.state.configuration.sites
        beta = self.state.beta
        self.potential = SquareSemiGrandPotential(
            system=self.system,
            state=state,
        )

        # construct CompletionCheck
        completion_check = monte.CompletionCheck(completion_check_params)

        # construct RandomNumberGenerator
        if random_engine is None:
            random_engine = monte.RandomNumberEngine()
        rng = monte.RandomNumberGenerator(random_engine)

        # make samplers - for all requested quantities
        samplers = monte.SamplerMap()
        for name, f in sampling_functions:
            samplers[f.name] = monte.Sampler(
                shape=f.shape,
                component_names=f.component_names,
            )

        # this is required, but can be left with 0 samples to indicate unweighted
        sample_weight = monte.Sampler(shape=[])

        # method log also tracks elapsed clocktime
        if method_log is None:
            method_log = monte.MethodLog(
                logfile_path=str("log.txt"),
                log_frequency=600,
            )
        method_log.restart_clock()

        l = sites.shape[0]
        n_step = 0
        self.n_pass = 0
        self.n_accept = 0
        self.n_reject = 0
        n_sites = self.state.configuration.sites.size
        max_linear_site_index = n_sites - 1
        n_pass_next_sample = sample_period
        while not completion_check.count_check(
            samplers=samplers,
            sample_weight=sample_weight,
            count=self.n_pass,
            method_log=method_log,
        ):
            # propose a flip
            linear_site_index = rng.random_int(max_linear_site_index)
            dE_potential = self.potential.occ_delta_extensive_value(linear_site_index)

            # Accept / reject event
            if dE_potential < 0.0:
                accept = True
            else:
                accept = rng.random_float(1.0) < math.exp(-dE_potential * beta)
            if accept:
                self.n_accept += 1

                # '//' is integer division
                j = linear_site_index // l
                i = linear_site_index - j * l
                sites[i, j] = -sites[i, j]
            else:
                self.n_reject += 1

            # increment n_step & n_pass
            n_step += 1
            if n_step == n_sites:
                n_step = 0
                self.n_pass += 1

            # sample if due
            if self.n_pass == n_pass_next_sample:
                n_pass_next_sample += sample_period
                for name, f in sampling_functions.items():
                    samplers[name].append(f())

        return (samplers, completion_check.results())


### ~~~ StateSamplingFunction factories ~~~ ###
def make_parametric_composition_f(mc_calculator):
    """Returns a parametric composition (intensive) sampling function"""

    def f():
        # captures a reference to mc_calculator
        sites = mc_calculator.state.configuration.sites
        return mc_calculator.system.composition_calculator(sites)

    return monte.StateSamplingFunction(
        name="parametric_composition",
        description="Parametric composition",
        shape=[
            mc_calculator.system.composition_calculator.n_independent_compositions()
        ],
        function=f,
    )


def make_formation_energy_f(mc_calculator):
    """Returns a formation energy (intensive) sampling function"""

    def f():
        # captures a reference to mc_calculator
        return mc_calculator.potential.formation_energy_calculator.intensive_value()

    return monte.StateSamplingFunction(
        name="formation_energy",
        description="Intensive formation energy",
        shape=[],
        function=f,
    )


def make_potential_energy_f(mc_calculator):
    """Returns a potential energy (intensive) sampling function"""

    def f():
        # captures a reference to mc_calculator
        return mc_calculator.potential.intensive_value()

    return monte.StateSamplingFunction(
        name="potential_energy",
        description="Intensive potential energy",
        shape=[],
        function=f,
    )


def test_SquareIsing():
    ...
