import math

from libcasm.monte import RandomNumberGenerator


def metropolis_acceptance(
    delta_potential_energy: float,
    beta: float,
    random_number_generator: RandomNumberGenerator,
):
    """Metropolis acceptance criteria

    Equivalent to:

    .. code-block:: Python

        if delta_potential_energy < 0.0:
            return True
        else:
            return random_number_generator.random_real(1.0) < math.exp(
                -delta_potential_energy * beta

    Parameters
    ----------
    delta_potential_energy: float
        Change in potential energy (per_supercell) due to a
        proposed Monte Carlo event
    beta: float
        The reciprocal temperature, :math:`\beta = 1/(k_B T)`.
    random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
        Random number generator

    Returns
    -------
    accept: bool
        If True, accept the event; if False, reject it.
    """
    if delta_potential_energy < 0.0:
        return True
    else:
        return random_number_generator.random_real(1.0) < math.exp(
            -delta_potential_energy * beta
        )
