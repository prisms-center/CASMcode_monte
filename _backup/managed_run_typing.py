"""PEP 544 Protocols for typing Monte Carlo calculators that support managed runs

This is a first draft.

A Monte Carlo calculator that supports managed runs enables use of:

- state_generator: which specify a sequence of Monte Carlo states to calculate
- sampling fixtures: which specify a number of quantities to sample, how often, how to
  check for completion, and how to output results, etc.
- run_manager: specifies what run data to store (initial states, final states, etc.) to
  enable analysis and restarts, and how to use >1 sampling fixture (run to one is
  complete? all are complete?)

"""
from typing import Optional, Protocol

import numpy as np
import numpy.typing as npt

from libcasm.basic_run_typing import (
    ConfigurationType,
    OccEventGeneratorType,
    PropertyCalculatorType,
    StateType,
)
from libcasm.monte import (
    CompletionCheckParams,
    CompletionCheckResults,
    MethodLog,
    RandomNumberEngine,
    RunManagerParams,
    Sampler,
    SamplerMap,
    SamplingParams,
    StateSamplingFunctionMap,
    ValueMap,
)
from libcasm.monte.events import (
    OccLocation,
)


class ManagedRunDataType(Protocol):
    """Results common to all sampling fixtures: initial state, final state, etc.

    Attributes
    ----------
    initial_state: Optional[StateType]
        Initial state of the run
    final_state: Optional[StateType]
        Final state of the run
    conditions: ValueMap
        Conditions for the run
    transformation_matrix_to_super: np.ndarray
        Supercell shape
    n_unitcells: int
        Supercell volume
    """

    initial_state: Optional[StateType]
    final_state: Optional[StateType]
    conditions: ValueMap
    transformation_matrix_to_super: np.ndarray
    n_unitcells: int


class ManagedRunResultsType(Protocol):
    """Results from one sampling fixture

    Attributes
    ----------
    elapsed_clocktime: Optional[float]
        Elapsed clocktime
    samplers: SamplerMap
        Stores sampled data
    analysis_results: dict[str, np.ndarray]
        Results from analysis functions, as a dict of analysis_name : value.
    sample_count: Optional[list[int]]
        Vector of counts (could be pass or step) when a sample occurred. May be empty
        if not applicable.
    sample_time: Optional[list[float]]
        Optional vector of times when a sample occurred. May be empty if not applicable.
    sample_weight: Sampler
        Weights given to samples. May be empty if not applicable.
    sample_clocktime: list[float]
        Vector of clocktimes when a sample occurred. May be empty if not applicable.
    sample_trajectory: list[ConfigurationType]
        Vector of the configuration when a sample occurred. May be empty if not
        applicable.
    completion_check_results: CompletionCheckResults
        Completion check results
    n_accept: int
        Number of acceptances
    n_reject: int
        Number of rejections
    """

    elapsed_clocktime: Optional[float]
    samplers: SamplerMap
    analysis_results: dict[str, np.ndarray]
    sample_count: list[int]
    sample_time: list[float]
    sample_weight: Sampler
    sample_clocktime: list[float]
    sample_trajectory: list[ConfigurationType]
    completion_check_results: CompletionCheckResults
    n_accept: int
    n_reject: int


class ResultsIOType(Protocol):
    def write(
        self, results: ManagedRunResultsType, conditions: ValueMap, run_index: int
    ) -> None:
        ...


class ResultsAnalysisFunctionType(Protocol):
    """Implement a function to analyze run results"""

    def __call__(
        self,
        run_data: ManagedRunDataType,
        results: ManagedRunResultsType,
    ) -> npt.NDArray[np.double]:
        ...


class SamplingFixtureParamsType(Protocol):
    """Parameters controlling a sampling fixture

    Attributes
    ----------
    label: str
        Label, to distinguish multiple sampling fixtures
    sampling_functions: StateSamplingFunctionMap
        State sampling functions to use
    analysis_functions: dict[str, ResultsAnalysisFunctionType]
        Results analysis functions to use
    sampling_params: SamplingParams
        When and what to sample
    completion_check_params: CompletionCheckParams
        When to finish a Monte Carlo run
    results_io: ResultsIOType
        Implements results IO
    method_log: MethodLog
        Method logging

    """

    label: str
    sampling_functions: StateSamplingFunctionMap
    analysis_functions: dict[str, ResultsAnalysisFunctionType]
    sampling_params: SamplingParams
    completion_check_params: CompletionCheckParams
    results_io: ResultsIOType
    method_log: MethodLog


class SamplingFixtureType(Protocol):
    """Implements sampling and completion checking"""

    def label(self) -> str:
        ...

    def params(self) -> SamplingFixtureParamsType:
        ...

    def initialize(self, state: StateType, steps_per_pass: int) -> None:
        ...

    def is_complete(self) -> bool:
        ...

    def write_status(self, run_index: int) -> None:
        """Write current status to method_log"""
        ...

    def increment_n_accept(self) -> None:
        ...

    def increment_n_reject(self) -> None:
        ...

    def increment_step(self) -> None:
        ...

    def set_time(self, event_time: float) -> None:
        ...

    def push_back_sample_weight(self, weight: float) -> None:
        ...

    def sample_data(self, state: StateType) -> None:
        ...

    def finalize(
        self,
        state: StateType,
        run_index: int,
        run_data: ManagedRunDataType,
    ) -> None:
        ...


class SampleActionType(Protocol):
    """Sample actions customize what occurs pre- and post- sampling

    Example use:
    - For KMC calculations, a pre-sampling action gets the current atom positions,
      and a post-sampling action copies the current atom positions to use as the
      previous atom positions for calculating displacements at the next sample time.
    """

    def __call__(self, fixture: SamplingFixtureType, state: StateType) -> None:
        ...


class RunManagerType(Protocol):
    """Manages run data and one or more sampling fixtures

    Attributes
    ----------
    engine: RandomNumberEngine
        Random number engine
    sampling_fixtures: list[SamplingFixtureType]
        The sampling fixtures being managed
    current_run: ManagedRunDataType
        Data for the current run
    completed_runs: list[ManagedRunDataType]
        Data from completed runs
    """

    params: RunManagerParams
    engine: RandomNumberEngine
    sampling_fixtures: list[SamplingFixtureType]
    current_run: ManagedRunDataType
    completed_runs: list[ManagedRunDataType]

    def read_completed_runs(self) -> None:
        """Read completed runs, to enable restarts"""
        ...

    def initialize(self, state: StateType, steps_per_pass: int) -> None:
        """Initialize samplers and counters for a new run"""
        ...

    def next_sampling_fixture(self) -> SamplingFixtureType:
        """Get the next sampling fixture due to take a sample"""
        ...

    def next_sampling_time(self) -> float:
        """Get the time the next sampling fixture is due to take a sample"""
        ...

    def write_status_if_due(self) -> None:
        """Write to sampling fixture status logs, if due"""
        ...

    def increment_n_accept(self) -> None:
        """Increment number of accepts events for all sampling fixtures"""
        ...

    def increment_n_reject(self) -> None:
        """Increment number of rejected events for all sampling fixtures"""
        ...

    def increment_step(self) -> None:
        """Increment number of steps for all sampling fixtures"""
        ...

    def set_time(self, event_time: float) -> None:
        """Increment time for all sampling fixtures"""
        ...

    def sample_data_by_count_if_due(
        self,
        state: StateType,
        pre_sample_f: SampleActionType,
        post_sample_f: SampleActionType,
    ) -> None:
        """For all sampling fixtures, sample data if due by count"""
        ...

    def sample_data_by_time_if_due(
        self,
        event_time: float,
        state: StateType,
        pre_sample_f: SampleActionType,
        post_sample_f: SampleActionType,
    ) -> None:
        """For all sampling fixtures, sample data if due by time"""
        ...

    def update_next_sampling_fixture(self) -> None:
        """Check sampling fixtures for next due"""
        ...

    def is_complete(self) -> bool:
        """Check if the managed run is complete"""
        ...

    def finalize(self, final_state: StateType) -> None:
        """Write results for each sampling fixture and write completed runs"""
        ...


class ManagedMonteCarloOccEventCalculatorType(Protocol):
    """The system implementation

    Notes
    -----
    - Sampling functions can expect that attributes which are PropertyCalculator are
      set to calculate self.state by the `run` method.
    -
    """

    state: StateType
    occ_location: OccLocation
    formation_energy_calculator: Optional[PropertyCalculatorType]
    composition_calculator: Optional[PropertyCalculatorType]
    potential: PropertyCalculatorType

    def run(
        self,
        state: StateType,
        occ_location: OccLocation,
        run_manager: RunManagerType,
        event: OccEventGeneratorType,
    ) -> None:
        """Runs the Monte Carlo calculator to completion

        Parameters
        ----------
        state: State
            The initial configuration and thermodynamic conditions. Will
            be evolved during the run.
        occ_location: OccLocation
            Tracks occupant locations. Should already be initialized with `state`. Will
            be evolved during the run.
        run_manager: RunManagerType
            Manages sampling fixtures and run data
        event: OccEventGeneratorType
            Proposes and applies events

        """
        ...
