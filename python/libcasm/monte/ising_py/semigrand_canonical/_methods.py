from __future__ import annotations

import copy
import json
import math
import os
from typing import Any, Callable, Optional

import numpy as np

import libcasm.casmglobal as casmglobal
import libcasm.monte.events as mcevents
from libcasm.monte import (
    MethodLog,
    RandomNumberEngine,
    RandomNumberGenerator,
    ValueMap,
)
from libcasm.monte.ising_py import (
    IsingFormationEnergy,
    IsingParamComposition,
    IsingState,
    IsingSystem,
)
from libcasm.monte.sampling import (
    CompletionCheck,
    CompletionCheckParams,
    Sampler,
    SamplerMap,
    StateSamplingFunction,
    StateSamplingFunctionMap,
    get_n_samples,
    scalar_as_vector,
)


class SemiGrandCanonicalConditions:
    """Semi-grand canonical ensemble thermodynamic conditions"""

    def __init__(
        self,
        temperature: float,
        exchange_potential: np.ndarray,
    ):
        """
        Parameters
        ----------
        temperature: float
            The temperature, :math:`T`.
        exchange_potential: np.ndarray
            The semi-grand canonical exchange potential, conjugate to the parametric
            composition that will be calculated by the `param_composition_calculator`
            of the system under consideration.
        """
        self.temperature: float = temperature
        """ The temperature, :math:`T`. """

        self.exchange_potential: np.ndarray = exchange_potential
        """ 
        The semi-grand canonical exchange potential, conjugate to the parametric 
        composition that will be calculated by the `param_composition_calculator`
        of the system under consideration.
        """

    @staticmethod
    def from_values(values: ValueMap) -> SemiGrandCanonicalConditions:
        """Construct from a conditions ValueMap"""
        if "temperature" not in values.scalar_values:
            raise Exception('Missing required condition: "temperature"')
        if "exchange_potential" not in values.vector_values:
            raise Exception('Missing required condition: "exchange_potential"')
        return SemiGrandCanonicalConditions(
            temperature=values.scalar_values["temperature"],
            exchange_potential=values.vector_values["exchange_potential"],
        )

    def to_values(self) -> ValueMap:
        """Construct a conditions ValueMap"""
        values = ValueMap()
        values.scalar_values["temperature"] = self.temperature
        values.vector_values["exchange_potential"] = self.exchange_potential
        return values

    @staticmethod
    def from_dict(data: dict) -> SemiGrandCanonicalConditions:
        """Construct from a conditions dict"""
        return SemiGrandCanonicalConditions.from_values(ValueMap.from_dict(data))

    @staticmethod
    def to_dict(self) -> dict:
        """Construct a conditions dict"""
        return self.to_values().to_dict()


class SemiGrandCanonicalPotential:
    """Calculates the semi-grand canonical energy and changes in energy

    Implements the (per_supercell) semi-grand canonical energy:

    .. code-block::

        E_sgc = E_formation - n_unitcells * (exchange_potential @ param_composition)

    """

    def __init__(
        self,
        system: IsingSystem,
    ):
        """

        .. rubric:: Constructor

        Parameters
        ----------
        system: libcasm.monte.ising_py.IsingSystem
            Holds parameterized formation energy and parametric composition
            calculators, without specifying at a particular state.

        """
        self.system: IsingSystem = system
        """Holds parameterized calculators, without specifying at a particular state. \
        This is a shared object. """

        self.formation_energy_calculator: IsingFormationEnergy = copy.deepcopy(
            system.formation_energy_calculator
        )
        """The formation energy calculator, set to calculate using the current state \
        during a run."""

        self.param_composition_calculator: IsingParamComposition = copy.deepcopy(
            system.param_composition_calculator
        )
        """The parametric composition calculator, set to calculate using the current \
        state during a run.
        
        This is expected to calculate the compositions conjugate to the the exchange 
        potentials provided by ``state.conditions.vector_values["exchange_potential"]``.
        """

        self.state: Optional[IsingState] = None
        """The current state during a run. 
        
        This is set from the input parameter. The `state.configuration` attribute must 
        be usable by the potential, formation energy, and parametric composition 
        calculators. The `state.conditions` attribute must be convertible to a
        :class:`SemiGrandCanonicalConditions` instance.
        """

        self.conditions: Optional[SemiGrandCanonicalConditions] = None
        """ The current state's conditions, set during `run`. """

    def set_state(self, state: IsingState, conditions: SemiGrandCanonicalConditions):
        """Set the current Monte Carlo state"""
        self.state = state
        self.conditions = conditions

        self.formation_energy_calculator.set_state(state)
        self.param_composition_calculator.set_state(state)

    def per_supercell(self) -> float:
        """Set the current Monte Carlo state"""
        mu_exchange = self.conditions.exchange_potential

        # formation energy, e_formation = -\sum_{NN} J s_i s_j
        E_f = self.formation_energy_calculator.per_supercell()

        # independent composition, n_unitcells * x = \sum_i s_i / 2
        Nx = self.param_composition_calculator.per_supercell()

        return E_f - mu_exchange @ Nx

    def per_unitcell(self) -> float:
        return self.per_supercell() / self.state.configuration.n_unitcells

    def occ_delta_per_supercell(
        self,
        linear_site_index: mcevents.LongVector,
        new_occ: mcevents.IntVector,
    ):
        # de_potential = e_potential_final - e_potential_init
        #   = (e_formation_final - n_unitcells * mu @ x_final) -
        #     (e_formation_init - n_unitcells * mu @ x_init)
        #   = de_formation - n_unitcells * mu * dx

        dE_f = self.formation_energy_calculator.occ_delta_per_supercell(
            linear_site_index, new_occ
        )
        mu_exchange = self.conditions.exchange_potential
        Ndx = self.param_composition_calculator.occ_delta_per_supercell(
            linear_site_index, new_occ
        )

        return dE_f - mu_exchange @ Ndx


class SemiGrandCanonicalData:
    """Holds semi-grand canonical Metropolis Monte Carlo run data and results"""

    def __init__(
        self,
        sampling_functions: StateSamplingFunctionMap,
        n_steps_per_pass: int,
        completion_check_params: CompletionCheckParams,
    ):
        """
        .. rubric:: Constructor

        Parameters
        ----------
        sampling_functions: StateSamplingFunctionMap
            The sampling functions to use.

        n_steps_per_pass: int
            Number of steps per pass

        completion_check_params: CompletionCheckParams
            Controls when the run finishes
        """
        self.sampling_functions: StateSamplingFunctionMap = sampling_functions
        """ The sampling functions to use """

        self.samplers: SamplerMap = SamplerMap()
        """ Holds sampled data """

        for name, f in self.sampling_functions.items():
            self.samplers[f.name] = Sampler(
                shape=f.shape,
                component_names=f.component_names,
            )
        self.sample_weight: Sampler = Sampler(shape=[])
        """ Sample weights remain empty (unweighted). Included for compatibility with \
        statistics calculators. """

        self.n_pass: int = int(0)
        """ Total number of passes completed. One pass is equal to one Monte Carlo \
        step per variable site in the configuration. """

        self.n_steps_per_pass: int = n_steps_per_pass
        """ Number of steps per pass """

        self.n_accept: int = int(0)
        """ The number of acceptances during a run """

        self.n_reject: int = int(0)
        """ The number of rejections during a run """

        self.completion_check: CompletionCheck = CompletionCheck(
            completion_check_params
        )
        """ The completion checker used during a run """

    def acceptance_rate(self) -> float:
        _n_accept = float(self.n_accept)
        _n_reject = float(self.n_reject)
        _total = _n_accept + _n_reject
        return _n_accept / _total

    def rejection_rate(self) -> float:
        _n_accept = float(self.n_accept)
        _n_reject = float(self.n_reject)
        _total = _n_accept + _n_reject
        return _n_reject / _total

    def reset(self):
        """Reset attributes set during `run`"""
        for name, sampler in self.samplers.items():
            sampler.clear()
        self.sample_weight.clear()
        self.n_pass = int(0)
        self.n_accept = int(0)
        self.n_reject = int(0)
        self.completion_check.reset()

    def to_dict(self) -> dict:
        """Convert to dict, excluding samplers

        Returns
        -------
        data: dict
            Monte Carlo data as a dict, includes:

            - completion_check_results: CompletionCheckResults, as dict
            - n_pass: int
            - n_steps_per_pass: int
            - n_accept: int
            - n_reject: int
            - acceptance_rate: float
            - rejection_rate: float
        """
        return {
            "completion_check_results": self.completion_check.results().to_dict(),
            "n_pass": self.n_pass,
            "n_steps_per_pass": self.n_steps_per_pass,
            "n_accept": self.n_accept,
            "n_reject": self.n_reject,
            "acceptance_rate": self.acceptance_rate(),
            "rejection_rate": self.rejection_rate(),
        }


def default_write_status(
    mc_calculator: Any,
    method_log: MethodLog,
) -> None:
    """Write status to log file and screen

    Parameters
    ----------
    mc_calculator: Any
        The Monte Carlo calculator to write status for.
    method_log: MethodLog
        Method log, for writing status updates.
    """

    ### write status ###
    data = mc_calculator.data
    completion_check = data.completion_check
    n_pass = data.n_pass
    n_samples = get_n_samples(data.samplers)
    n_variable_sites = mc_calculator.state.configuration.n_variable_sites
    param_composition_calculator = mc_calculator.param_composition_calculator
    formation_energy_calculator = mc_calculator.formation_energy_calculator

    ## Formatting...
    param_composition_fmt = ".4f"
    formation_energy_fmt = ".4e"
    prec_fmt = ".4e"
    np_formatter = {"float_kind": lambda x: f"{x:{param_composition_fmt}}"}

    ## Print passes, simulated and clock time
    steps = n_pass * n_variable_sites
    time_s = method_log.time_s()

    timing_str = (
        f"Passes={n_pass}, "
        f"Samples={n_samples}, "
        f"ClockTime(s)={time_s:.2f}, "
        f"Steps/Second={steps/time_s:.2e}, "
        f"Seconds/Step={time_s/steps:.2e}"
    )
    print(timing_str)

    ## Print current property status
    param_composition_str = np.array2string(
        param_composition_calculator.per_unitcell(),
        formatter=np_formatter,
    )
    formation_energy = formation_energy_calculator.per_unitcell()
    print(
        f"  "
        f"ParametricComposition={param_composition_str}, "
        f"FormationEnergy={formation_energy:{formation_energy_fmt}}"
    )

    results = data.completion_check.results()

    def finish():
        """Things to do when finished"""
        method_log.reset()
        method_log.print(json.dumps(results.to_dict(), sort_keys=True, indent=2))
        method_log.begin_lap()

    ## Print AllEquilibrated=? status
    all_equilibrated = results.equilibration_check_results.all_equilibrated
    print(f"  " f"AllEquilibrated={all_equilibrated}")
    if not all_equilibrated:
        finish()
        return

    ## Print AllConverted=? status
    all_converged = results.convergence_check_results.all_converged
    print(f"  " f"AllConverged={all_converged}")
    if all_converged:
        finish()
        return

    ## Print individual requested convergence status
    converge_results = results.convergence_check_results.individual_results
    for key, req in completion_check.params().requested_precision.items():
        stats = converge_results[key].stats
        calc_abs_prec = stats.calculated_precision
        mean = stats.mean
        calc_rel_prec = math.fabs(calc_abs_prec / stats.mean)
        if req.abs_convergence_is_required:
            print(
                f"  - {key.sampler_name}({key.component_index}): "
                f"mean={mean:{prec_fmt}}, "
                f"abs_prec={calc_abs_prec:{prec_fmt}} "
                f"< "
                f"requested={req.abs_precision:{prec_fmt}} "
                f"== {calc_abs_prec < req.abs_precision}"
            )
        if req.rel_convergence_is_required:
            print(
                f"  - {key.sampler_name}({key.component_index}): "
                f"mean={mean:{prec_fmt}}, "
                f"rel_prec={calc_rel_prec:{prec_fmt}} "
                f"< "
                f"requested={req.rel_precision:{prec_fmt}} "
                f"== {calc_rel_prec < req.rel_precision}"
            )

    finish()
    return


class SemiGrandCanonicalEventGenerator:
    """Propose and apply semi-grand canonical Ising model events

    .. rubric:: Constructor

    Parameters
    ----------
    state: Optional[libcasm.monte.ising_py.IsingState] = None
        The current state for which events are proposed and applied
    """

    def __init__(
        self,
        state: Optional[IsingState] = None,
    ):
        # construct occ_event
        e = mcevents.OccEvent()
        e.linear_site_index.clear()
        e.linear_site_index.append(0)
        e.new_occ.clear()
        e.new_occ.append(1)
        self.occ_event: mcevents.OccEvent = e
        """ The current proposed event """

        self.state: Optional[IsingState] = None
        """ The current state for which events are proposed and applied """

        if state is not None:
            self.set_state(state)

    def set_state(
        self,
        state: IsingState,
    ):
        """Set the state for which events are proposed and applied

        Parameters
        ----------
        state: libcasm.monte.ising_py.IsingState
            The state for which events are proposed and applied
        """
        self.state = state

        self._max_linear_site_index = self.state.configuration.n_variable_sites - 1

    def propose(
        self,
        random_number_generator: RandomNumberGenerator,
    ) -> mcevents.OccEvent:
        """Propose a Monte Carlo occupation event

        Parameters
        ----------
        random_number_generator: RandomNumberGenerator
            The random number generator used to propose an event

        Returns
        -------
        occ_event: mcevents.OccEvent
            The current proposed event. This is also stored in ``self.occ_event``.
        """
        self.occ_event.linear_site_index[0] = random_number_generator.random_int(
            self._max_linear_site_index
        )
        self.occ_event.new_occ[0] = -self.state.configuration.occ(
            self.occ_event.linear_site_index[0]
        )
        return self.occ_event

    def apply(
        self,
        occ_event: mcevents.OccEvent,
    ):
        """Update the occupation of the current state

        Parameters
        ----------
        occ_event: mcevents.OccEvent
            The event which is applied to update the occupation of the current state

        """
        self.state.configuration.set_occ(
            occ_event.linear_site_index[0], occ_event.new_occ[0]
        )


class SemiGrandCanonicalCalculator:
    """A semi-grand canonical Monte Carlo calculator"""

    def __init__(
        self,
        system: IsingSystem,
    ):
        """

        .. rubric:: Constructor

        Parameters
        ----------
        system: libcasm.monte.ising_py.IsingSystem
            Holds parameterized formation energy and parametric composition
            calculators, without specifying at a particular state.

        """
        # This contains the system parameters, state-independent
        self.system: IsingSystem = system
        """Holds parameterized calculators, without specifying at a particular state. \
        This is a shared object."""

        # The potential calculator, set to current state during `run`
        self.potential: SemiGrandCanonicalPotential = None
        """ The semi-grand canonical energy calculator, set to calculate using the \
        `formation_energy_calculator` and `param_composition_calculator` during `run`.\
        """

        # The formation energy calculator, set to current state during `run`
        self.formation_energy_calculator: IsingFormationEnergy = None
        """The formation energy calculator, set to calculate using the current state \
        during `run`."""

        # The formation energy calculator, set to current state during `run`
        self.param_composition_calculator: IsingParamComposition = None
        """ The parametric composition calculator, set to calculate using the current \
        state during a run. This is expected to calculate the compositions conjugate \
        to the exchange potentials provided by \
        ``state.conditions.vector_values["exchange_potential"]``. """

        # Holds semi-grand canonical Metropolis Monte Carlo run data and results
        self.data: SemiGrandCanonicalData = None
        """ Holds semi-grand canonical Metropolis Monte Carlo run data and results """

        # Current state, set during `run`
        self.state: IsingState = None
        """ The current state during a run. 
        
        This is set from the input parameter. The `state.configuration` attribute must 
        be a type usable by the potential, formation energy, and parametric composition
        calculators.
        """

        # SemiGrandCanonicalConditions, set during `run`
        self.conditions: SemiGrandCanonicalConditions = None
        """The current state's conditions, set during a run."""

    def run(
        self,
        state: IsingState,
        sampling_functions: StateSamplingFunctionMap,
        completion_check_params: CompletionCheckParams,
        event_generator: SemiGrandCanonicalEventGenerator,
        sample_period: int = 1,
        method_log: Optional[MethodLog] = None,
        random_engine: Optional[RandomNumberEngine] = None,
        write_status_f: Optional[Callable] = None,
    ):
        """Run a semi-grand canonical Monte Carlo calculation

        Notes
        -----
        On completion, results can be obtained from the :attr:`data` attribute.

        Parameters
        ----------
        state: libcasm.monte.ising_py.IsingState
            Initial Monte Carlo state, including configuration and conditions. The
            configuration type must be supported by the calculators provided by
            the `system` constructor parameter. The `conditions` must be
            convertible to :class:`SemiGrandCanonicalConditions`. This is mutated
            during the calculation.
        sampling_functions: StateSamplingFunctionMap
            The sampling functions to use
        completion_check_params: CompletionCheckParams
            Controls when the run finishes
        event_generator: SemiGrandCanonicalEventGenerator
            An event generator, that can propose a new event and apply an accepted
            event.
        sample_period: int = 1
            Number of passes per sample. One pass is one Monte Carlo step per site.
        method_log: Optional[MethodLog] = None,
            Method log, for writing status updates. If None, default
            writes to "status.json" every 10 minutes.
        random_engine: Optional[RandomNumberEngine] = None
            Random number engine. Default constructs a new engine.
        write_status_f: Optional[Callable] = None
            Function with signature

            .. code-block:: Python

                def f(
                  mc_calculator: SemiGrandCanonicalCalculator,
                  method_log: MethodLog,
                ) -> None:
                    ...

            accepting `self` as the first argument, that writes status updates,
            after a new sample has been taken and due according to
            ``method_log.log_frequency()``. Default uses :func:`default_write_status`,
            which writes the current completion check results to
            ``method_log.logfile_path()`` and prints a summary of the current state and
            sampled data to stdout.

        """
        ### Setup ####

        # set state
        self.state = state
        self.conditions = SemiGrandCanonicalConditions.from_values(
            self.state.conditions
        )
        temperature = self.conditions.temperature
        n_steps_per_pass = self.state.configuration.n_variable_sites

        # set potential and other calculators
        self.potential = SemiGrandCanonicalPotential(
            system=self.system,
        )
        self.potential.set_state(self.state, self.conditions)
        self.formation_energy_calculator = self.potential.formation_energy_calculator
        self.param_composition_calculator = self.potential.param_composition_calculator

        def dpotential_f(e: mcevents.OccEvent) -> float:
            return self.potential.occ_delta_per_supercell(
                e.linear_site_index, e.new_occ
            )

        # set event generator
        event_generator = copy.deepcopy(event_generator)
        event_generator.set_state(
            state=self.state,
        )

        def propose_event_f(rng: RandomNumberGenerator) -> mcevents.OccEvent:
            return event_generator.propose(rng)

        def apply_event_f(e: mcevents.OccEvent):
            event_generator.apply(e)

        # construct Monte Carlo data structure
        self.data = SemiGrandCanonicalData(
            sampling_functions, n_steps_per_pass, completion_check_params
        )

        ### Setup next steps ###  (equal to basic_occupation_metropolis)
        data = self.data
        beta = 1.0 / (casmglobal.KB * temperature)

        # construct RandomNumberGenerator
        random_number_generator = RandomNumberGenerator(random_engine)

        # method log also tracks elapsed clocktime
        if method_log is None:
            method_log = MethodLog(
                logfile_path=os.path.join(os.getcwd(), "status.json"),
                log_frequency=600,
            )
        method_log.restart_clock()
        method_log.begin_lap()

        # used in main loop
        n_pass_next_sample = sample_period
        n_step = 0

        # write_status_f
        if write_status_f is None:
            write_status_f = default_write_status

        ### Main loop ####
        while not data.completion_check.count_check(
            samplers=data.samplers,
            sample_weight=data.sample_weight,
            count=data.n_pass,
            method_log=method_log,
        ):
            # Propose an event
            event = propose_event_f(random_number_generator)

            # Calculate change in potential energy (per_supercell) due to event
            delta_potential_energy = dpotential_f(event)

            # Accept / reject event
            if delta_potential_energy < 0.0:
                accept = True
            else:
                accept = random_number_generator.random_real(1.0) < math.exp(
                    -delta_potential_energy * beta
                )

            # if accept:
            if accept:
                data.n_accept += 1
                apply_event_f(event)
            else:
                data.n_reject += 1

            # increment n_step & n_pass
            n_step += 1
            if n_step == data.n_steps_per_pass:
                n_step = 0
                data.n_pass += 1

            # sample if due
            if data.n_pass == n_pass_next_sample:
                n_pass_next_sample += sample_period
                for name, f in data.sampling_functions.items():
                    data.samplers[name].append(f())

                # write status if due
                if (
                    method_log.log_frequency() is not None
                    and method_log.lap_time() >= method_log.log_frequency()
                ):
                    write_status_f(self, method_log)

        ### Finish ####
        write_status_f(self, method_log)

        return


def make_param_composition_f(mc_calculator):
    """Returns a parametric composition sampling function

    The sampling function "param_composition" gets the
    parametric composition from:

    .. code-block:: Python

        mc_calculator.param_composition_calculator.per_unitcell()

    The number of parametric composition components is obtained from:

    .. code-block:: Python

        mc_calculator.system.param_composition_calculator.n_independent_compositions()

    Parameters
    ----------
    mc_calculator: Any
        The Monte Carlo calculator to be sampled.

    Returns
    -------
    param_composition_f: StateSamplingFunction
        A parametric composition sampling function that samples `mc_calculator`.
    """

    def f():
        # captures a reference to mc_calculator
        return mc_calculator.param_composition_calculator.per_unitcell()

    return StateSamplingFunction(
        name="param_composition",
        description="Parametric composition",
        shape=[
            # n_independent_compositions is independent of the state
            mc_calculator.system.param_composition_calculator.n_independent_compositions()
        ],
        function=f,
    )


def make_formation_energy_f(mc_calculator):
    """Returns a formation energy (per unitcell) sampling function

    The sampling function "formation_energy" gets the formation energy
    (per unitcell) from:

    .. code-block:: Python

        mc_calculator.formation_energy_calculator.per_unitcell()

    Parameters
    ----------
    mc_calculator: Any
        The Monte Carlo calculator to be sampled.

    Returns
    -------
    formation_energy_f: StateSamplingFunction
        A formation energy (per unitcell) sampling function that samples
        `mc_calculator`.
    """

    def f():
        # captures a reference to mc_calculator
        return scalar_as_vector(
            mc_calculator.formation_energy_calculator.per_unitcell()
        )

    return StateSamplingFunction(
        name="formation_energy",
        description="Intensive formation energy",
        shape=[],  # scalar
        function=f,
    )


def make_potential_energy_f(mc_calculator):
    """Returns a potential energy (per unitcell) sampling function

    The sampling function "potential_energy" gets the potential
    energy (per unitcell) from:

    .. code-block:: Python

        mc_calculator.potential.per_unitcell()

    Parameters
    ----------
    mc_calculator: Any
        The Monte Carlo calculator to be sampled.

    Returns
    -------
    potential_energy_f: StateSamplingFunction
        A potential energy (per unitcell) sampling function that samples
        `mc_calculator`.
    """

    def f():
        # captures a reference to mc_calculator
        return scalar_as_vector(mc_calculator.potential.per_unitcell())

    return StateSamplingFunction(
        name="potential_energy",
        description="Intensive potential energy",
        shape=[],  # scalar
        function=f,
    )
