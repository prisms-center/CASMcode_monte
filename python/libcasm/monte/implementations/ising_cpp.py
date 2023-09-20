from ._monte_implementations_ising_cpp import (
  SemiGrandCanonicalConditions,
  SemiGrandCanonicalPotential,
  SemiGrandCanonicalData,
  SemiGrandCanonicalCalculator,
)

from typing import Any
import numpy as np
import libcasm.monte as monte


def custom_default_write_status(
    mc_calculator: Any,
    method_log: monte.MethodLog,
) -> None:
    """Write status to log file and screen

    Parameters
    ----------
    mc_calculator: Any
        The Monte Carlo calculator to write status for.

    :return:
    """
    ### write status ###
    data = mc_calculator.data
    completion_check = data.completion_check
    n_pass = data.n_pass
    n_samples = monte.get_n_samples(data.samplers)
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


def custom_make_param_composition_f(mc_calculator):
    """Returns a parametric composition sampling function

    The sampling function "param_composition" gets the
    parametric composition from:

    .. code-block:: Python

        mc_calculator.param_composition_calculator.per_unitcell()

    The number of parametric composition components is obtained from:

    .. code-block:: Python

        mc_calculator.system.param_composition_calculator.n_independent_compositions()


    """

    def f():
        # captures a reference to mc_calculator
        return mc_calculator.param_composition_calculator.per_unitcell()

    return monte.StateSamplingFunction(
        name="param_composition",
        description="Parametric composition",
        shape=[
            # n_independent_compositions is independent of the state
            mc_calculator.system.param_composition_calculator.n_independent_compositions()
        ],
        function=f,
    )


def custom_make_formation_energy_f(mc_calculator):
    """Returns a formation energy (per unitcell) sampling function

    The sampling function "formation_energy" gets the formation energy
    (per unitcell) from:

    .. code-block:: Python

        mc_calculator.formation_energy_calculator.per_unitcell()

    """
    def f():
        # captures a reference to mc_calculator
        return monte.scalar_as_vector(
            mc_calculator.formation_energy_calculator.per_unitcell()
        )

    return monte.StateSamplingFunction(
        name="formation_energy",
        description="Intensive formation energy",
        shape=[],  # scalar
        function=f,
    )


def custom_make_potential_energy_f(mc_calculator):
    """Returns a potential energy (per unitcell) sampling function

    The sampling function "potential_energy" gets the potential
    energy (per unitcell) from:

    .. code-block:: Python

        mc_calculator.potential.per_unitcell()

    """
    def f():
        # captures a reference to mc_calculator
        return monte.scalar_as_vector(mc_calculator.potential.per_unitcell())

    return monte.StateSamplingFunction(
        name="potential_energy",
        description="Intensive potential energy",
        shape=[],  # scalar
        function=f,
    )


def custom_make_configuration_json_f(mc_calculator):
    """Returns a configuration JSON sampling function

    The JSON sampling function "configuration" gets the current configuration
    using:

    .. code-block:: Python

        mc_calculator.state.configuration.to_dict()

    """
    def f():
        # captures a reference to mc_calculator
        return mc_calculator.state.configuration.to_dict()

    return monte.jsonStateSamplingFunction(
        name="configuration",
        description="Configuration as JSON",
        function=f,
    )
