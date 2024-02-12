import libcasm.monte as monte
import libcasm.monte.sampling as sampling


def test_CompletionCheck_1(tmp_path):
    # completion check params
    completion_check_params = sampling.CompletionCheckParams()
    completion_check_params.cutoff_params.max_count = 12

    # completion check
    completion_check = sampling.CompletionCheck(completion_check_params)

    # samplers
    samplers = sampling.SamplerMap()
    samplers["e"] = sampling.Sampler(shape=[])
    samplers["x"] = sampling.Sampler(shape=[3])

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
            samplers["e"].append([0])
            samplers["x"].append([0, 0, 0])

    assert n_steps == 12


def test_CompletionCheck_2(tmp_path):
    # completion check params
    completion_check_params = sampling.CompletionCheckParams()
    completion_check_params.cutoff_params.min_sample = 100

    # requested precision
    e_abs_precision = 0.001
    e_key = sampling.SamplerComponent(
        sampler_name="e",
        component_name="",
        component_index=0,
    )
    requested_precision = sampling.RequestedPrecision(abs=e_abs_precision)
    completion_check_params.requested_precision[e_key] = requested_precision

    v_abs_precision = 0.01
    v_key = sampling.SamplerComponent(
        sampler_name="v",
        component_name="",
        component_index=0,
    )
    requested_precision = sampling.RequestedPrecision(abs=v_abs_precision)
    completion_check_params.requested_precision[v_key] = requested_precision

    # completion check
    completion_check = sampling.CompletionCheck(completion_check_params)

    # samplers
    samplers = sampling.SamplerMap()
    samplers["e"] = sampling.Sampler(shape=[], component_names=[""])
    samplers["v"] = sampling.Sampler(shape=[], component_names=[""])

    # this is required, but can be left with 0 samples to indicate unweighted
    sample_weight = sampling.Sampler(shape=[])

    # method log also tracks elapsed clocktime
    method_log = monte.MethodLog(str(tmp_path / "log.txt"))

    # random number generator
    rng = monte.RandomNumberGenerator()

    # random variable e
    e_mean = 1.0
    e_amp = 0.1

    # random variable v
    rng = monte.RandomNumberGenerator()
    v_mean = 20.0
    v_amp = 1.0

    n_steps = 0
    while not completion_check.count_check(
        samplers=samplers,
        sample_weight=sample_weight,
        count=n_steps,
        method_log=method_log,
    ):
        n_steps += 1

        e = e_mean + rng.random_real(e_amp) - e_amp / 2.0
        v = v_mean + rng.random_real(v_amp) - v_amp / 2.0

        if n_steps % 10 == 0:
            samplers["e"].append([e])
            samplers["v"].append([v])

    results = completion_check.results()

    assert sampling.get_n_samples(samplers) >= 100
    assert results.is_complete

    # equilibration check results
    # print(results.equilibration_check_results.to_dict())
    assert results.equilibration_check_results.all_equilibrated
    assert len(results.equilibration_check_results.individual_results) == 2

    # convergence check results
    # print(results.convergence_check_results.to_dict())
    assert results.convergence_check_results.all_converged
    assert len(results.convergence_check_results.individual_results) == 2

    # no max cutoffs, so sampled data must be converged
    converge_results = results.convergence_check_results.individual_results
    assert converge_results[e_key].stats.calculated_precision < e_abs_precision
    assert converge_results[v_key].stats.calculated_precision < v_abs_precision
