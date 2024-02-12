import libcasm.monte as monte
import libcasm.monte.sampling as sampling


def insert_sampling_functions(
    function_map: sampling.StateSamplingFunctionMap,
    functions: list[sampling.StateSamplingFunction],
):
    for f in functions:
        function_map[f.name] = f


def make_random_sampling_function(
    name: str,
    random_engine: monte.RandomNumberEngine,
    mean: float = 0.0,
    amp: float = 1.0,
    description: str = "",
):
    rng = monte.RandomNumberGenerator(random_engine)

    def f():
        return [mean + rng.random_real(amp) - amp / 2.0]

    return sampling.StateSamplingFunction(
        name=name,
        description=description,
        shape=[],
        function=f,
        component_names=[""],
    )


def test_SamplingFunction_1(tmp_path):
    # ~~~ system and sampling functions ~~~

    # random_engine
    random_engine = monte.RandomNumberEngine()

    # make sampling functions:
    sampling_functions = sampling.StateSamplingFunctionMap()

    insert_sampling_functions(
        sampling_functions,
        [
            make_random_sampling_function(
                name="e", mean=1.0, amp=0.1, random_engine=random_engine
            ),
            make_random_sampling_function(
                name="v", mean=20.0, amp=1.0, random_engine=random_engine
            ),
        ],
    )

    # ~~~ method input ~~~

    # quantities to sample:
    quantities = ["e", "v"]

    # completion check params
    completion_check_params = sampling.CompletionCheckParams()
    completion_check_params.cutoff_params.min_sample = 100

    # requested precision of quantity "e"
    e_abs_precision = 0.001
    e_key = sampling.SamplerComponent(
        sampler_name="e",
        component_name="",
        component_index=0,
    )
    requested_precision = sampling.RequestedPrecision(abs=e_abs_precision)
    completion_check_params.requested_precision[e_key] = requested_precision

    # requested precision of quantity "v"
    v_abs_precision = 0.01
    v_key = sampling.SamplerComponent(
        sampler_name="v",
        component_name="",
        component_index=0,
    )
    requested_precision = sampling.RequestedPrecision(abs=v_abs_precision)
    completion_check_params.requested_precision[v_key] = requested_precision

    # ~~~ method ~~~

    # completion check
    completion_check = sampling.CompletionCheck(completion_check_params)

    # make samplers - for all requested quantities
    samplers = sampling.SamplerMap()
    for quantity_name in quantities:
        if quantity_name in sampling_functions:
            f = sampling_functions[quantity_name]
            samplers[f.name] = sampling.Sampler(
                shape=f.shape,
                component_names=f.component_names,
            )

    # this is required, but can be left with 0 samples to indicate unweighted
    sample_weight = sampling.Sampler(shape=[])

    # method log also tracks elapsed clocktime
    method_log = monte.MethodLog(str(tmp_path / "log.txt"))

    n_steps = 0
    while not completion_check.count_check(
        samplers=samplers,
        sample_weight=sample_weight,
        count=n_steps,
        method_log=method_log,
    ):
        n_steps += 1

        if n_steps % 10 == 0:
            for name, f in sampling_functions.items():
                samplers[name].append(f())

    results = completion_check.results()

    # ~~~ tests ~~~

    assert sampling.get_n_samples(samplers) >= 100
    assert results.is_complete

    # equilibration check results
    print(results.equilibration_check_results.to_dict())
    assert results.equilibration_check_results.all_equilibrated
    assert len(results.equilibration_check_results.individual_results) == 2

    # convergence check results
    print(results.convergence_check_results.to_dict())
    assert results.convergence_check_results.all_converged
    assert len(results.convergence_check_results.individual_results) == 2

    # no max cutoffs, so sampled data must be converged
    converge_results = results.convergence_check_results.individual_results
    assert converge_results[e_key].stats.calculated_precision < e_abs_precision
    assert converge_results[v_key].stats.calculated_precision < v_abs_precision
