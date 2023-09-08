"""State sampling functions

State sampling functions are constructed with a reference to a
Monte Carlo calculator (`mc_calculator`) so that when they are
called, they can sample a property of the current state of the
calculator (`mc_calculator.state`).

Example:

def my_sampling_function(mc_calculator):

    def f():
        sites = mc_calculator.state.configuration.sites
        return mc_calculator.system.composition_calculator(sites)

    return monte.StateSamplingFunction(
        name="param_composition",
        description="Parametric composition",
        shape=[
            mc_calculator.system.composition_calculator.n_independent_compositions()
        ],
        function=f,
    )
"""
from .._monte import StateSamplingFunction, scalar_as_vector


def make_param_composition_f(mc_calculator):
    """Returns a parametric composition sampling function

    The sampling function "param_composition" gets the
    parametric composition from:

    .. code-block:: Python

        mc_calculator.potential.composition_calculator.intensive_value()

    """

    def f():
        # captures a reference to mc_calculator
        return mc_calculator.composition_calculator.intensive_value()

    return StateSamplingFunction(
        name="param_composition",
        description="Parametric composition",
        shape=[
            mc_calculator.system.composition_calculator.n_independent_compositions()
        ],
        function=f,
        component_names=[""],
    )


def make_formation_energy_f(mc_calculator):
    """Returns a formation energy (per unitcell) sampling function

    The sampling function "formation_energy" gets the formation energy
    (per unitcell) from:

    .. code-block:: Python

        mc_calculator.potential.formation_energy_calculator.intensive_value()

    """

    def f():
        # captures a reference to mc_calculator
        return scalar_as_vector(
            mc_calculator.formation_energy_calculator.intensive_value()
        )

    return StateSamplingFunction(
        name="formation_energy",
        description="Intensive formation energy",
        shape=[],  # scalar
        function=f,
        component_names=[""],
    )


def make_potential_energy_f(mc_calculator):
    """Returns a potential energy (per unitcell) sampling function

    The sampling function "potential_energy" gets the potential
    energy (per unitcell) from:

    .. code-block:: Python

        mc_calculator.potential.intensive_value()

    """

    def f():
        # captures a reference to mc_calculator
        return scalar_as_vector(mc_calculator.potential.intensive_value())

    return StateSamplingFunction(
        name="potential_energy",
        description="Intensive potential energy",
        shape=[],  # scalar
        function=f,
        component_names=[""],
    )
