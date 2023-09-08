from __future__ import annotations

import copy
import json
import math
import os
from typing import Optional

import numpy as np

import libcasm.casmglobal as casmglobal
from libcasm.monte import (
    CompletionCheck,
    CompletionCheckParams,
    CompletionCheckResults,
    MethodLog,
    RandomNumberEngine,
    RandomNumberGenerator,
    Sampler,
    SamplerMap,
    StateSamplingFunctionMap,
    ValueMap,
    get_n_samples,
)
from libcasm.monte.basic_run_typing import (
    BasicMonteCarloOccEventCalculatorType,
    OccEventGeneratorType,
    StateType,
    SystemType,
)
from libcasm.monte.events import (
    IntVector,
    LongVector,
)


class SemiGrandCanonicalConditions:
    """Semi-grand canonical ensemble thermodynamic conditions

    Attributes
    ----------
    temperature: float
        The temperature, :math:`T`.
    beta: float
        The reciprocal temperature, :math:`\beta = 1/(k_B T)`.
    exchange_potential: np.ndarray
        The semi-grand canonical exchange potential, conjugate to the
        parametric composition that will be calculated by the `composition_calculator`
        of the system under consideration.

    Parameters
    ----------
    temperature: float
        The temperature, :math:`T`.
    exchange_potential: np.ndarray
        The semi-grand canonical exchange potential, conjugate to the
        parametric composition that will be calculated by the `composition_calculator`
        of the system under consideration.
    """

    def __init__(
        self,
        temperature: float,
        exchange_potential: np.ndarray,
    ):
        self.temperature = temperature
        self.beta = 1.0 / (casmglobal.KB * self.temperature)
        self.exchange_potential = exchange_potential

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

    Implements the (extensive) semi-grand canonical energy:

    .. code-block::

        E_sgc = E_formation - n_unitcells * (exchange_potential @ param_composition)

    Attributes
    ----------
    system: :class:`libcasm.monte.basic_run_typing.SystemType`
        Holds parameterized calculators, without specifying at a particular state.
        This is a shared object.
    state: :class:`~libcasm.monte.basic_run_typing.StateType`
        The current state during `run`. This is copied from the input parameter. The
        `state.configuration` attribute must be a
        :class:`~libcasm.monte.basic_run_typing.ConfigurationType` usable by the
        potential, formation energy, and parametric composition calculators. The
        `state.conditions` attribute must be a
        :class:`~libcasm.monte.calculators.basic_semigrand_canonical.SemiGrandCanonicalConditions`
        instance.
    formation_energy_calculator: PropertyCalculatorType
        The formation energy calculator, set to calculate using the current state
        during `run`.
    composition_calculator: \
    :class:`~libcasm.monte.basic_run_typing.PropertyCalculatorType`
        The parametric composition calculator, set to calculate using the current state
        during `run`. This is expected to calculate the compositions conjugate to the
        the exchange potentials provided by `state.conditions.exchange_potential`.

    Parameters
    ----------
    system: :class:`libcasm.monte.basic_run_typing.SystemType`
        Holds parameterized calculators, without specifying at a particular state. In
        particular, must provide:

        - formation_energy_calculator: \
        :class:`libcasm.monte.basic_run_typing.PropertyCalculatorType`
            - A formation energy calculator
        - composition_calculator: \
        class:`libcasm.monte.basic_run_typing.PropertyCalculatorType`
            - A parametric composition calculator

        The provided calculators must be able to calculate properties using the
        configuration type provided to the Monte Carlo calculator
        :func:`~libcasm.monte.calculators.basic_semigrand_canonical
        .SemiGrandCanonicalConditions.run` method as part of the `state` parameter.

    """

    def __init__(
        self,
        system: SystemType,
        state: Optional[StateType] = None,
    ):
        self.system = system
        self.formation_energy_calculator = copy.deepcopy(
            system.formation_energy_calculator
        )
        self.composition_calculator = copy.deepcopy(system.composition_calculator)
        if state is not None:
            self.set_state(state)

    def set_state(self, state: StateType):
        """Set the current Monte Carlo state"""
        self.state = state

        self.formation_energy_calculator.set_state(state)
        self.composition_calculator.set_state(state)

    def extensive_value(self) -> float:
        """Set the current Monte Carlo state"""
        mu_exchange = self.state.conditions.exchange_potential

        # formation energy, e_formation = -\sum_{NN} J s_i s_j
        E_f = self.formation_energy_calculator.extensive_value()

        # independent composition, n_unitcells * x = \sum_i s_i / 2
        Nx = self.composition_calculator.extensive_value()

        return E_f - mu_exchange @ Nx

    def intensive_value(self) -> float:
        return self.extensive_value() / self.state.configuration.n_unitcells

    def occ_delta_extensive_value(
        self,
        linear_site_index: LongVector,
        new_occ: IntVector,
    ):
        # de_potential = e_potential_final - e_potential_init
        #   = (e_formation_final - n_unitcells * mu @ x_final) -
        #     (e_formation_init - n_unitcells * mu @ x_init)
        #   = de_formation - n_unitcells * mu * dx

        dE_f = self.formation_energy_calculator.occ_delta_extensive_value(
            linear_site_index, new_occ
        )
        mu_exchange = self.state.conditions.exchange_potential
        Ndx = self.composition_calculator.occ_delta_extensive_value(
            linear_site_index, new_occ
        )

        return dE_f - mu_exchange @ Ndx


class SemiGrandCanonicalCalculatorResults:
    """Monte Carlo calculator results data structure

    Attributes
    ----------
    samplers: :class:`~libcasm.monte.SamplerMap`
        Holds sampled data
    completion_check_results: :class:`~libcasm.monte.CompletionCheckResults`
        Completion check results
    n_pass: int
        Total number of passes completed
    acceptance_rate: float
        Fraction of proposed Monte Carlo events accepted
    rejection_rate: float
        Fraction of proposed Monte Carlo events rejected
    """

    def __init__(
        self,
        samplers: SamplerMap,
        completion_check_results: CompletionCheckResults,
        n_pass: int,
        acceptance_rate: float,
        rejection_rate: float,
    ):
        self.samplers = samplers
        self.completion_check_results = completion_check_results
        self.n_pass = n_pass
        self.acceptance_rate = acceptance_rate
        self.rejection_rate =rejection_rate

    def to_dict(self) -> dict:
        """Convert to dict, excluding samplers

        Returns
        -------
        data: dict
            Results as a dict, includes:

            - completion_check_results: CompletionCheckResults, as dict
            - n_pass: int
            - acceptance_rate: float
            - rejection_rate: float
        """
        return {
            "completion_check_results": self.completion_check_results.to_dict(),
            "n_pass": self.n_pass,
            "acceptance_rate": self.acceptance_rate,
            "rejection_rate": self.rejection_rate,
        }


def default_write_status(
    mc_calculator: BasicMonteCarloOccEventCalculatorType,
) -> None:
    """Write status to log file and screen
    
    Parameters
    ----------
    mc_calculator: \
    :class:`~libcasm.monte.basic_run_typing.BasicMonteCarloOccEventCalculatorType`
        The Monte Carlo calculator to write status for.
        
    :return: 
    """
    method_log = mc_calculator.method_log

    ### write status ###
    completion_check = mc_calculator.completion_check
    n_pass = mc_calculator.n_pass
    n_samples = get_n_samples(mc_calculator.samplers)
    n_sites = mc_calculator.state.configuration.n_sites
    composition_calculator = mc_calculator.composition_calculator
    formation_energy_calculator = mc_calculator.formation_energy_calculator

    ## Formatting...
    param_composition_fmt = ".4f"
    formation_energy_fmt = ".4e"
    prec_fmt = ".4e"
    np_formatter = {"float_kind": lambda x: f"{x:{param_composition_fmt}}"}

    ## Print passes, simulated and clock time
    steps = n_pass * n_sites
    time_s = method_log.time_s()

    timing_str = f"Passes={n_pass}, "
    if mc_calculator.time is not None:
        timing_str += f"SimulatedTime={time_s:.2e}, "
    timing_str += (
        f"Samples={n_samples}, "
        f"ClockTime(s)={time_s:.2f}, "
        f"Steps/Second={steps/time_s:.2e}, "
        f"Seconds/Step={time_s/steps:.2e}"
    )
    print(timing_str)

    ## Print current property status
    param_composition_str = np.array2string(
        composition_calculator.intensive_value(),
        formatter=np_formatter,
    )
    formation_energy = formation_energy_calculator.intensive_value()
    print(
        f"  "
        f"ParametricComposition={param_composition_str}, "
        f"FormationEnergy={formation_energy:{formation_energy_fmt}}"
    )

    results = completion_check.results()
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


class SemiGrandCanonicalCalculator:
    """A semi-grand canonical Monte Carlo calculator

    Notes
    -----
    - Implements \
    :class:`libcasm.monte.basic_run_typing.BasicMonteCarloOccEventCalculatorType`

    Attributes
    ----------
    system: :class:`libcasm.monte.basic_run_typing.SystemType`
        Holds parameterized calculators, without specifying at a particular state.
        This is a shared object.
    state: :class:`~libcasm.monte.basic_run_typing.StateType`
        The current state during `run`. This is copied from the input parameter. The
        `state.configuration` attribute must be a
        :class:`~libcasm.monte.basic_run_typing.ConfigurationType` usable by the
        potential, formation energy, and parametric composition calculators. The
        `state.conditions` attribute must be a
        :class:`~libcasm.monte.calculators.basic_semigrand_canonical.SemiGrandCanonicalConditions`
        instance.
    potential: \
    :class:`~libcasm.monte.calculators.basic_semigrand_canonical.SemiGrandCanonicalPotential`
        The semi-grand canonical energy calculator, set to calculate using the
        `formation_energy_calculator` and `composition_calculator` during `run`.
    formation_energy_calculator: PropertyCalculatorType
        The formation energy calculator, set to calculate using the current state
        during `run`.
    composition_calculator: \
    :class:`~libcasm.monte.basic_run_typing.PropertyCalculatorType`
        The parametric composition calculator, set to calculate using the current state
        during `run`. This is expected to calculate the compositions conjugate to the
        the exchange potentials provided by `state.conditions.exchange_potential`.
    samplers: :class:`~libcasm.monte.SamplerMap`
        Holds sampled data during `run`
    sample_weight: :class:`~libcasm.monte.Sampler`
        Sample weights remain empty (unweighted)
    n_pass: int
        Number of passes during `run`. One pass is equal to one Monte Carlo step per
        variable site in the configuration.
    time: None
        Time is not simulated by this calculator
    n_accept: int
        The number of acceptances during `run`
    n_reject: int
        The number of rejections during `run`
    completion_check: :class:`~libcasm.monte.CompletionCheck`
        The completion checker used during `run`
    event_generator: :class:`~libcasm.monte.basic_run_typing.OccEventGeneratorType`
        The event generator used during `run`.
    sample_period: int
        The number of passes per sample during `run`
    method_log: :class:`~libcasm.monte.MethodLog`
        The logger used during `run`
    random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
        The random number generator used during `run`

    Parameters
    ----------
    system: :class:`libcasm.monte.basic_run_typing.SystemType`
        Holds parameterized calculators, without specifying at a particular state. In
        particular, must have attributes:

        - formation_energy_calculator: \
        :class:`libcasm.monte.basic_run_typing.PropertyCalculatorType`
            - A formation energy calculator
            - Must be copy-able
        - composition_calculator: \
        class:`libcasm.monte.basic_run_typing.PropertyCalculatorType`
            - A parametric composition calculator
            - Must be copy-able

        The provided calculators must be able to calculate properties using the
        configuration type provided to the Monte Carlo calculator
        :func:`~libcasm.monte.calculators.basic_semigrand_canonical.
        SemiGrandCanonicalConditions.run` method as part of the `state` parameter.

    """

    def __init__(
        self,
        system: SystemType,
    ):
        # This contains the system parameters, state-independent
        self.system = system

        self.reset()

    def reset(self) -> None:
        """Reset attributes set during `run`

        Listing calculator attributes makes it easier to understand what data is
        available and to implement methods, such as status printing or sampling function
        factories, as standalone functions that accept the calculator as an argument.

        """
        # The current state during `run`
        self.state = None

        # The potential calculator, set to current state during `run`
        self.potential = None

        # The formation energy calculator, set to current state during `run`
        self.formation_energy_calculator = None

        # The formation energy calculator, set to current state during `run`
        self.composition_calculator = None

        # Holds sampled data during a `run`
        self.samplers = None

        # Sample weights remain empty (unweighted)
        self.sample_weight = None

        # Number of passes during `run`
        self.n_pass = None

        # The number of acceptances during `run`
        self.n_accept = None

        # The number of rejections during `run`
        self.n_reject = None

        # The completion checker during `run`
        self.completion_check = None

        # The event generator during `run`
        self.event_generator = None

        # The number of passes per sample during `run`
        self.sample_period = None

        # The logger used during `run`
        self.method_log = None

        # The random number generator during `run`
        self.random_number_generator = None


    def run(
        self,
        state: StateType,
        sampling_functions: StateSamplingFunctionMap,
        completion_check_params: CompletionCheckParams,
        event_generator: OccEventGeneratorType,
        sample_period: int = 1,
        method_log: Optional[MethodLog] = None,
        random_engine: Optional[RandomNumberEngine] = None,
        write_status_f=default_write_status,
    ) -> SemiGrandCanonicalCalculatorResults:
        """Run a semi-grand canonical Monte Carlo calculation

        Parameters
        ----------
        state: :class:`~libcasm.monte.basic_run_typing.StateType`
            Initial Monte Carlo state, including configuration and conditions. The
            configuration type must be supported by the calculators provided by
            the `system` constructor parameter. The `conditions` type must be
            :class:`~libcasm.monte.calculators.basic_semigrand_canonical.
            SemiGrandCanonicalConditions`. This is mutated during the calculation.
        sampling_functions: :class:`~libcasm.monte.StateSamplingFunctionMap`
            The sampling functions to use
        completion_check_params: :class:`~libcasm.monte.CompletionCheckParams`
            Controls when the run finishes
        event_generator: :class:`~libcasm.monte.basic_run_typing.OccEventGeneratorType`
            An OccEventGeneratorType, that can propose a new event and apply an accepted
            event. For example, an
            :class:`~libcasm.monte.models.ising.IsingEventGenerator` instance.
        sample_period: int = 1
            Number of passes per sample. One pass is one Monte Carlo step per site.
        method_log: Optional[:class:`~libcasm.monte.MethodLog`] = None,
            Method log, for writing status updates. If None, default
            writes to "status.json" every 10 minutes.
        random_engine: Optional[:class:`~libcasm.monte.RandomNumberEngine`] = None
            Random number engine. Default constructs a new engine.

        Returns
        -------
        results: :class:`~libcasm.monte.calculators.basic_semigrand_canonical.
        SemiGrandCanonicalCalculatorResults`
            Holds sampled data and completion check results
        """

        ### Setup ####

        # set state
        self.state = state
        n_sites = self.state.configuration.n_sites
        beta = self.state.conditions.beta

        # set potential and other calculators
        self.potential = SemiGrandCanonicalPotential(
            system=self.system,
            state=self.state,
        )
        self.formation_energy_calculator = self.potential.formation_energy_calculator
        self.composition_calculator = self.potential.composition_calculator

        # set event generator
        self.event_generator = copy.deepcopy(event_generator)
        self.event_generator.set_state(
            state=self.state,
        )

        # construct CompletionCheck
        self.completion_check = CompletionCheck(completion_check_params)

        # construct RandomNumberGenerator
        if random_engine is None:
            random_engine = RandomNumberEngine()
        self.random_number_generator = RandomNumberGenerator(random_engine)

        # make samplers - for all requested quantities
        self.samplers = SamplerMap()
        for name, f in sampling_functions.items():
            self.samplers[f.name] = Sampler(
                shape=f.shape,
                component_names=f.component_names,
            )

        # this is required, but can be left with 0 samples to indicate unweighted
        self.sample_weight = Sampler(shape=[])

        # method log also tracks elapsed clocktime
        logfile_path = os.path.join(os.getcwd(), "status.json")
        if method_log is None:
            method_log = MethodLog(
                logfile_path=logfile_path,
                log_frequency=600,
            )
        self.method_log = method_log
        self.method_log.restart_clock()
        self.method_log.begin_lap()

        ### Main loop ####
        print("\n~~~Beginning~~~\n")

        n_step = 0
        self.n_pass = 0
        self.n_accept = 0
        self.n_reject = 0
        n_pass_next_sample = sample_period
        while not self.completion_check.count_check(
            samplers=self.samplers,
            sample_weight=self.sample_weight,
            count=self.n_pass,
            method_log=self.method_log,
        ):
            # propose a flip
            self.event_generator.propose(self.random_number_generator)
            dE_potential = self.potential.occ_delta_extensive_value(
                linear_site_index=self.event_generator.occ_event.linear_site_index,
                new_occ=self.event_generator.occ_event.new_occ,
            )

            # Accept / reject event
            if dE_potential < 0.0:
                accept = True
            else:
                accept = self.random_number_generator.random_real(1.0) < math.exp(
                    -dE_potential * beta
                )

            if accept:
                self.n_accept += 1
                self.event_generator.apply()
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
                    self.samplers[name].append(f())

                # write status if due
                if (method_log.log_frequency() is not None and
                        method_log.lap_time() >= method_log.log_frequency()):
                    write_status_f(self)

        ### Return results ####
        print("\n~~~Finished~~~\n")
        write_status_f(self)

        n_total = self.n_accept + self.n_reject

        return SemiGrandCanonicalCalculatorResults(
            samplers=self.samplers,
            completion_check_results=self.completion_check.results(),
            n_pass=self.n_pass,
            acceptance_rate=self.n_accept / n_total,
            rejection_rate=self.n_reject / n_total,
        )
