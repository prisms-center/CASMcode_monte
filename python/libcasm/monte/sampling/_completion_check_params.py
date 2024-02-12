from typing import Callable, Optional

import libcasm.monte.sampling._monte_sampling as _sampling


class CompletionCheckParams(_sampling.CompletionCheckParams):
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
    log_spacing: bool = False
        If True, use logarithmic spacing for completion checking; else use linear
        spacing. For linear spacing, the n-th check will be taken when:

        .. code-block:: Python

            sample = check_begin + check_period * n

        For logarithmic spacing, the n-th check will be taken when:

        .. code-block:: Python

            sample = check_begin + round( check_base ** (n + check_shift) )

        However, if sample(n) - sample(n-1) > `check_period_max`, then subsequent
        samples are taken every `check_period_max` samples.

        For linear spacing, the default is to check for completion after `100`,
          `200`, `300`, etc. samples are taken.

        For log spacing, the default is to check for completion after `100`,
          `1000`, `10000`, `20000`, `30000`, etc. samples are taken.

        The default value is False, for linear spacing.
    check_begin:  Optional[int] = None
        The earliest sample to begin completion checking. Default is 100 for linear
        spacing and 0 for log spacing.
    check_period:  Optional[int] = None
        The linear completion checking period. Default is 100.
    check_base: Optional[float] = None
        The logarithmic completion checking base. Default is 10.
    check_shift: Optional[float] = None
        The shift for the logarithmic spacing exponent. Default is 2.
    check_period_max: Optional[int] = None
        The maximum check spacing for logarithmic check spacing. Default is 10000.
    """

    def __init__(
        self,
        requested_precision: Optional[_sampling.RequestedPrecisionMap] = None,
        cutoff_params: Optional[_sampling.CutoffCheckParams] = None,
        equilibration_check_f: Optional[Callable] = None,
        calc_statistics_f: Optional[Callable] = None,
        log_spacing: bool = False,
        check_begin: Optional[int] = None,
        check_period: Optional[int] = None,
        check_base: Optional[float] = None,
        check_shift: Optional[float] = None,
        check_period_max: Optional[int] = None,
    ):
        _sampling.CompletionCheckParams.__init__(self)
        if cutoff_params is None:
            cutoff_params = _sampling.CutoffCheckParams()
        if equilibration_check_f is None:
            equilibration_check_f = _sampling.default_equilibration_check
        if calc_statistics_f is None:
            calc_statistics_f = _sampling.BasicStatisticsCalculator()
        if requested_precision is None:
            requested_precision = _sampling.RequestedPrecisionMap()

        if log_spacing is False:
            self.check_begin = 100
            self.check_period = 100
        else:
            self.check_begin = 0
            self.check_base = 10.0
            self.check_shift = 2.0
            self.check_period_max = 10000

        self.cutoff_params = cutoff_params
        self.equilibration_check_f = equilibration_check_f
        self.calc_statistics_f = calc_statistics_f
        self.requested_precision = requested_precision
        self.log_spacing = log_spacing

        if check_begin is not None:
            self.check_begin = check_begin
        if check_period is not None:
            self.check_period = check_period
        if check_base is not None:
            self.check_base = check_base
        if check_shift is not None:
            self.check_shift = check_shift
        if check_period_max is not None:
            self.check_period_max = check_period_max
