import json
import pathlib

import numpy as np

import libcasm.monte as monte
import libcasm.monte.calculators.basic_semigrand_canonical as sgc
import libcasm.monte.models.basic_ising_np as ising
import libcasm.monte.models.sampling_functions as sf


def test_basic_Ising_np():
    # construct a SemiGrandCanonicalCalculator
    print("here 0")
    mc_calculator = sgc.SemiGrandCanonicalCalculator(
        system=ising.IsingSystem(
            formation_energy_calculator=ising.IsingFormationEnergy(
                J=0.1,
                lattice_type=1,
            ),
            composition_calculator=ising.IsingCompositionCalculator(),
        )
    )

    # construct sampling functions
    print("here 1")
    sampling_functions = monte.StateSamplingFunctionMap()
    for f in [
        sf.make_param_composition_f(mc_calculator),
        sf.make_formation_energy_f(mc_calculator),
        sf.make_potential_energy_f(mc_calculator),
    ]:
        sampling_functions[f.name] = f

    # construct the initial state
    print("here 2")
    shape = (25, 25)
    initial_state = ising.IsingState(
        configuration=ising.IsingConfiguration(
            shape=shape,
        ),
        conditions=sgc.SemiGrandCanonicalConditions(
            temperature=2000.0,
            exchange_potential=np.array([0.0]),
        ),
    )

    # set the initial occupation explicitly here (default is all +1)
    print("here 3")
    initial_state.configuration.set_occupation(
        np.full(shape=shape, fill_value=1, dtype=np.int32, order="F")
    )

    # create an Ising model semi-grand canonical event proposer / applier
    print("here 4")
    event_generator = ising.IsingSemiGrandCanonicalEventGenerator()

    # completion check params
    print("here 5")
    completion_check_params = monte.CompletionCheckParams()
    completion_check_params.cutoff_params.min_sample = 100
    completion_check_params.log_spacing = False
    completion_check_params.check_begin = 100
    completion_check_params.check_period = 10

    # Set requested precision
    print("here 6")
    monte.converge(sampling_functions, completion_check_params).set_precision(
        "potential_energy", abs=0.001
    ).set_precision("param_composition", abs=0.001)

    # Create a logger
    print("here 7")
    method_log = monte.MethodLog(
        logfile_path=str(pathlib.Path(".").absolute() / "status.json"),
        log_frequency=0.2,
    )

    # Run
    print("here 8")
    mc_results = mc_calculator.run(
        state=initial_state,
        sampling_functions=sampling_functions,
        completion_check_params=completion_check_params,
        event_generator=event_generator,
        sample_period=1,
        method_log=method_log,
        random_engine=None,
    )

    print("here 9")
    samplers = mc_results.samplers
    results = mc_results.completion_check_results

    print(json.dumps(results.to_dict(), indent=2))

    assert monte.get_n_samples(samplers) >= 100
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

