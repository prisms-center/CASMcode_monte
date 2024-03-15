Completion Checks
=================

The :class:`~libcasm.monte.CompletionCheck` class can be used to check when a Monte Carlo simulation is complete. It is constructed with a :class:`~libcasm.monte.CompletionCheckParams` instance which allows:

- setting the requested absolute or relative precision for convergence of sampled data
- setting cutoff parameters for number of steps or passes, number of samples, amount of simulated time
  or amount of elapsed clocktime, with:

  - minimums, which force the simulation to keep running until a minimium value is reached
  - maximums, which force the simulation to stop running when a maximum value is reached

- controlling when completion checks are performed
- customizing the method used to check for equilibration
- customizing the method used to calculate statistics

A :class:`~libcasm.monte.CompletionCheckParams` instance can be read from a Python dict for input file parsing.

Performing completion checks
----------------------------

Example usage:

.. code-block:: Python

    import libcasm.monte as monte

    # ~~~ Initialize a random number engine ~~~
    random_number_engine = monte.RandomNumberEngine()

    # ~~~ User input: Specify the system ~~~
    # For example, the prim, cluster expansions, composition axes, etc.
    prim = ...
    composition_axes = ...
    formation_energy_clexulator = ...
    formation_energy_coeff = ...

    # ~~~ Construct the system ~~~
    # Construct a data structure that makes system data accessible
    # to the Monte Carlo calculator and sampling functions
    system = SystemType(
        prim=prim,
        composition_axes=composition_axes,
        formation_energy_clexulator=formation_energy_clex_clexulator,
        formation_energy_coeff=formation_energy_clex_coeff,
        ...
    )

    # ~~~ Construct the Monte Carlo calculator ~~~
    # A Monte Carlo calculator implements the Monte Carlo method and holds
    # method specific data along with access to the system.
    # For example, this would have the current configuration and conditions
    mc_calculator = MonteCarloCalculatorType(
        system,
        random_number_engine,
        ...
    )

    # ~~~ Construct sampling functions ~~~
    # Sampling functions should be given access to the mc_calculator to
    # be able to take samples when called, and stored in a StateSamplingFunctionMap
    sampling_functions = monte.StateSamplingFunctionMap()
    _functions = [
        make_formation_energy_f(mc_calculator),
        make_potential_energy_f(mc_calculator),
        make_parametric_composition_f(mc_calculator),
    ]
    for _f in _functions:
        sampling_functions[_f.name] = _f

    # ~~~ User input: Set CompletionCheckParams ~~~
    params = monte.CompletionCheckParams()
    params.cutoff_params.max_count = 100

    completion_check = monte.CompletionCheck(params)

    # ~~~ User input: Specify quantities to be sampled ~~~
    # should be keys into sampling_functions
    quantities = [
        "formation_energy",
        "potential_energy",
        "parametric_composition",
    ]

    # ~~~ Construct data samplers - for all requested quantities ~~~
    samplers = monte.SamplerMap()
    for quantity_name in quantities:
        if quantity_name in sampling_functions:
            f = sampling_functions[quantity_name]
            samplers[f.name] = monte.Sampler(
                shape=f.shape,
                component_names=f.component_names,
            )

    # this is required, but can be left with 0 samples to indicate unweighted
    sample_weight = monte.Sampler(shape=[])

    # method log specifies where to periodically write status messages
    # internally, method log tracks the elapsed clock time
    method_log = monte.MethodLog(path_to_logfile)

    n_steps = 0
    i_sample_period = 0
    sample_period = 1000
    while not completion_check.count_check(
        samplers=samplers,
        sample_weight=sample_weight,
        count=n_steps,
        method_log=method_log,
    ):
        # ... do Monte Carlo step ...
        n_steps += 1

        # ... periodically sample data ...
        if i_sample_period == sample_period:
            for name, f in sampling_functions.items():
                samplers[name].append(f())

Examples
--------

Run for a user-specified number of steps:

.. code-block:: Python

    import libcasm.monte as monte
    params = monte.CompletionCheckParams()
    params.cutoff_params.max_count = 100

    completion_check = monte.CompletionCheck(params)

    n_steps = 0
    while not completion_check.count_check(
        samplers=samplers,
        sample_weight=sample_weight,
        count=n_steps,
        method_log=method_log,
    ):
        # ... do Monte Carlo step ...
        n_steps += 1

Set a limit for the maximum elapsed clock time (in seconds):

.. note::
    Because checking the clock time at every Monte Carlo step slows down calculations excessively, clock time limits are only checked after the number of samples changes, and the actual clock time at which a simulation is stopped may longer than the limit set.

.. code-block:: Python

    import libcasm.monte as monte
    params = monte.CompletionCheckParams()
    params.cutoff_params.max_count = 3.6e2 # stop after running at least 1 hr

    completion_check = monte.CompletionCheck(params)

    n_steps = 0
    while not completion_check.count_check(
        samplers=samplers,
        sample_weight=sample_weight,
        count=n_steps,
        method_log=method_log,
    ):
        # ... do Monte Carlo step ...
        n_steps += 1



The ensemble average value of a property can be estimated from Metropolis Monte Carlo simulations as

.. math::

    \langle X \rangle \approx \bar{X} = \frac{\sum_l^N X_l}{N},

where:

- :math:`X_l` is the :math:`l`-th of :math:`N` observations of property :math:`X`
- :math:`\langle X \rangle` is the ensemble average
- :math:`\bar{X}` is the mean of the observations.

The error, :math:`\bar{X} - \langle X \rangle`, is normally distributed and approaches zero in the limit of large :math:`N` according to the central limit theorem,

.. math::

    \bar{X}-\langle X \rangle \approx \mathcal{N}(0, \sigma^2/N).

For stationary distributions, the variance of the error, :math:`\sigma^2`, can be calculated from the lag :math:`k` autocovariance between observations :math:`X_j` and :math:`X_{j+k}`  as

.. math::

    \sigma^2 = \gamma_0 + 2 \sum^\infty_{k=1} \gamma_k,

    \gamma_k = \mathrm{Cov}\left( X_j, X_{j+k} \right).

After an initial equilibration period, samples drawn from Monte Carlo simulations have a stationary distribution. Therefore, the error can be estimated from the observations by estimating :math:`\sum^\infty_{k=1} \gamma_k`.

If the autocovariance decays like :math:`\gamma_k = \gamma_0 \rho^{-|k|}`, then the infinite sum can be evaluated to give

.. math::

    \sigma^2 = \gamma_0 \left(\frac{1+\rho}{1-\rho}\right).


Van de Walle and Asta introduced methods for determining the initial equilibration period from which observations should be discarded before calculating the mean, and for calculating :math:`\rho` from the remaining observations to estimate the error in the mean. These methods are implemented in CASM in the :func:`~libcasm.monte.default_equilibration_check` and :class:`~libcasm.monte.BasicStatisticsCalculator` methods, respectively, and used as the defaults for automatic convergence to user-specified precision.

Users may also implement and use alternative methods through the `equilibration_check_f` and `calc_statistics_f` parameters of the :class:`~libcasm.monte.CompletionCheckParams` class.


Equilibration check
-------------------

The :func:`~libcasm.monte.default_equilibration_check` method partitions an array of
observations into three ranges:

- the equilibriation stage, ``[0, start1)``,
- the first partition, ``[start1, start2)``,
- and the second partition, ``[start2, N)``,

where `N` is ``len(observations)``, and `start1` and `start2` are indices into
the observations array such that ``0 <= start1 < start2 <= N``, and the number
of elements in the first and second partition are the same (within 1).

The simulation is considered equilibrated at observation `start1` if the
mean of the elements in the first and second partition are approximately equal
to the desired precsion, ``(abs(mean1 - mean2) < prec)``.

Additionally, in CASM, the value `start1` is incremented as much as needed to ensure
that the equilibration stage has observations on either side of the overall mean.

The result of :func:`~libcasm.monte.default_equilibration_check` is of type :class:`~libcasm.monte.IndividualEquilibrationResult`, which has two attributes which are set as follows:

- If all observations are approximately equal, then:

  - ``is_equilibrated = True``
  - ``N_samples_for_equilibration = 0``

- If the equilibration conditions are met, the result contains:

  - ``is_equilibrated = true``
  - ``N_samples_for_equilibration = start1``

- If the equilibration conditions are not met, the result contains:

  - ``is_equilibrated = false``
  - ``N_samples_for_equilibration = <undefined>``

If samples are weighted, as in the n-fold way algorithm, then the same
partitioning method is used, but with weighted observations calculated using:

.. code-block:: Python

    weighted_observation[i] = sample_weight[i] * observation[i] * N / W

where:

.. code-block:: Python

    W = np.sum(sample_weight)

The same weight_factor ``N/W`` applies for all properties.


Calculated precision
--------------------

The value of :math:`\rho` depends on the details of the system and the Monte Carlo method. The method of estimating :math:`\rho` introduced by Van de Walle and Asta `` and implemented in :class:`~libcasm.monte.BasicStatisticsCalculator`, calculates :math:`\hat{\rho}`, an estimate for :math:`\rho`, by searching for the smallest value of :math:`k` for which :math:`\hat{\gamma}_k/\hat{\gamma}_0 \le 1/2`.

If samples are weighted, as in the n-fold way algorithm, then :class:`~libcasm.monte.BasicStatisticsCalculator` uses one of two optional approaches which re-sample the data to generate :math:`N'` equally weighted observations. The weighted observations can be considered a time series, where the time interval associated with each sample value is equal to the sample weight. Then the time series can be sampled at :math:`N'` regular intervals. Given the resampled data, the two approaches are:

1. Calculate :math:`\bar{X} = \sum_l X_l w_l / \sum_l w_l` and :math:`\hat{\gamma}_0 = \mathrm{Var}(X)` directly from the original observations and sample weights, and only calculate :math:`\hat{\rho}` from resampled observations
2. Calculate :math:`\bar{X}`, :math:`\hat{\gamma}_0`, and :math:`\hat{\rho}` from the resampled observations

Given :math:`\hat{\gamma}_0` and :math:`\hat{\rho}`, the error in the mean, :math:`\bar{X} \pm p`, is calculated to a user-specified confidence level according to

.. math::

    p = \sqrt{2} * \mathrm{erf}^{-1}(c) \sqrt{ \frac{\hat{\gamma}_0}{N} \left(\frac{1+\hat{\rho}}{1-\hat{\rho}}\right) }

where :math:`c` is the confidence level, and :math:`p` is the calculated precision.