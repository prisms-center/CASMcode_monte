"""Test Python implemented property ising_cpp with Python implemented Monte Carlo \
loop
"""

import json
import pathlib

import libcasm.monte as monte
import libcasm.monte.ising_py as ising
import libcasm.monte.ising_py.semigrand_canonical as sgc
import libcasm.monte.sampling as sampling


def test_ising_semigrand_canonical_py():
    # construct a SemiGrandCanonicalCalculator
    mc_calculator = sgc.SemiGrandCanonicalCalculator(
        system=ising.IsingSystem(
            formation_energy_calculator=ising.IsingFormationEnergy(
                J=0.1,
                lattice_type=1,
            ),
            param_composition_calculator=ising.IsingParamComposition(),
        )
    )

    # construct sampling functions
    sampling_functions = sampling.StateSamplingFunctionMap()
    for f in [
        sgc.make_param_composition_f(mc_calculator),
        sgc.make_formation_energy_f(mc_calculator),
        sgc.make_potential_energy_f(mc_calculator),
    ]:
        sampling_functions[f.name] = f

    # construct the initial state
    shape = (25, 25)
    initial_state = ising.IsingState(
        configuration=ising.IsingConfiguration(
            shape=shape,
        ),
        conditions=monte.ValueMap.from_dict(
            {
                "temperature": 2000.0,
                "exchange_potential": [0.0],
            }
        ),
    )

    # set the initial occupation explicitly here (default is all +1)
    for linear_site_index in range(initial_state.configuration.n_sites):
        initial_state.configuration.set_occ(linear_site_index, 1)

    # create an Ising model semi-grand canonical event proposer / applier
    event_generator = sgc.SemiGrandCanonicalEventGenerator()

    # completion check params
    completion_check_params = sampling.CompletionCheckParams()
    completion_check_params.cutoff_params.min_sample = 100
    completion_check_params.log_spacing = False
    completion_check_params.check_begin = 100
    completion_check_params.check_period = 10

    # Set requested precision
    sampling.converge(sampling_functions, completion_check_params).set_precision(
        "potential_energy", abs=0.001
    ).set_precision("param_composition", abs=0.001)

    # Create a logger
    method_log = monte.MethodLog(
        logfile_path=str(pathlib.Path(".").absolute() / "status.json"),
        log_frequency=0.2,
    )

    # Run
    mc_calculator.run(
        state=initial_state,
        sampling_functions=sampling_functions,
        completion_check_params=completion_check_params,
        event_generator=event_generator,
        sample_period=1,
        method_log=method_log,
        random_engine=None,
    )

    samplers = mc_calculator.data.samplers
    results = mc_calculator.data.completion_check.results()

    print(json.dumps(results.to_dict(), indent=2))

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
    for key, req in completion_check_params.requested_precision.items():
        assert converge_results[key].stats.calculated_precision < req.abs_precision
