from typing import Callable, Optional

import libcasm.monte._monte as _monte

# from ._monte import (
#     _CompletionCheckParams,
#     CutoffCheckParams,
#     IndividualEquilibrationResult,
#     RequestedPrecision,
#     default_equilibration_check,
# )


class CompletionCheckParams(_monte.CompletionCheckParams):
    """Parameters that determine if a simulation is complete

    CompletionCheckParams allow:

    - setting the requested precision for convergence of sampled data
    - setting cutoff parameters, forcing the simulation to keep running to meet certain
      minimums (number of steps or passes, number of samples, amount of simulated time
      or elapsed clocktime), or stop when certain maximums are met
    - controlling when completion checks are performed
    - customizing the method used to calculate statistics
    - customizing the method used to check for equilibration.

    Parameters
    ----------
    requested_precision : Optional[:class:`~libcasm.monte.RequestedPrecisionMap`] = None
        Requested precision for convergence of sampler components. When all components
        reach the requested precision, and all `cutoff_params` minimums are met,
        then the completion check returns True, indicating the Monte Carlo simulation
        is complete.
    cutoff_params: Optional[:class:`~libcasm.monte.CutoffCheckParams`] = None,
        Cutoff check parameters allow setting limits on the Monte Carlo simulation to
        prevent calculations from stopping too soon or running too long. If None, no
        cutoffs are applied.
    calc_statistics_f: Optional[Callable] = None,
        A function for calculating :class:`~libcasm.monte.BasicStatistics` from
        sampled data, with signature:

        .. code-block:: Python

            def calc_statistics_f(
                observations: np.ndarray,
                sample_weight: np.ndarray,
            ) -> libcasm.monte.BasicStatistics:
                ...

        If None, the default is :class:`~libcasm.monte.BasicStatisticsCalculator`.
    equilibration_check_f: Optional[Callable] = None,
        A function for checking equilibration of sampled data, with signature:

        .. code-block:: Python

            def equilibration_check_f(
                observations: np.ndarray,
                sample_weight: np.ndarray,
                requested_precision: libcasm.monte.RequestedPrecision,
            ) -> libcasm.monte.IndividualEquilibrationResult:
                ...

        If None, the default is :class:`~libcasm.monte.default_equilibration_check`.
    log_spacing: bool = True
        If True, use logarithmic spacing for completion checking; else use linear
        spacing. For linear spacing, the n-th check will be taken when:

        .. code-block:: Python

            sample = round( check_begin + (check_period / checks_per_period) * n )

        For logarithmic spacing, the n-th check will be taken when:

        .. code-block:: Python

            sample = round( check_begin + check_period ^ ( (n + check_shift) /
                           checks_per_period ) )

        The default value is True, for logarithmic spacing.
    check_begin: float = 0.0,
        The sample to begin completion checking.
    check_period: float = 10.0,
        The linear sample checking period, or logarithmic spacing base.
    checks_per_period: float = 1.0,
        The number of checks per checking period.
    check_shift: float = 1.0,
        The shift for the logarithmic spacing exponent.
    """

    def __init__(
        self,
        requested_precision: Optional[_monte.RequestedPrecisionMap] = None,
        cutoff_params: Optional[_monte.CutoffCheckParams] = None,
        equilibration_check_f: Optional[Callable] = None,
        calc_statistics_f: Optional[Callable] = None,
        log_spacing: bool = True,
        check_begin: float = 0.0,
        check_period: float = 10.0,
        checks_per_period: float = 1.0,
        check_shift: float = 1.0,
    ):
        _monte.CompletionCheckParams.__init__(self)
        if cutoff_params is None:
            cutoff_params = _monte.CutoffCheckParams()
        if equilibration_check_f is None:
            equilibration_check_f = _monte.default_equilibration_check
        if calc_statistics_f is None:
            calc_statistics_f = _monte.BasicStatisticsCalculator()
        if requested_precision is None:
            requested_precision = _monte.RequestedPrecisionMap()

        self.cutoff_params = cutoff_params
        self.equilibration_check_f = equilibration_check_f
        self.calc_statistics_f = calc_statistics_f
        self.requested_precision = requested_precision
        self.log_spacing = log_spacing
        self.check_begin = check_begin
        self.check_period = check_period
        self.checks_per_period = checks_per_period
        self.check_shift = check_shift
