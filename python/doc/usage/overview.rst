Overview
========

The libcasm-monte package provides generic data structures and methods that can form the
building blocks for Monte Carlo simulation implementations. The Python interface is
provided for controlling simulations, data analysis, and initial testing. For the most
efficient simulations, extensions written in C++ are used to implement calculators and
checks used most frequently.

.. note::

    The libcasm-clexmonte_ package provides the CASM cluster expansion Monte Carlo
    implementations built using this package.

Primarily, the libcasm-monte package includes a Python interface to data structures
and methods written in C++ in the CASM::monte namespace:

- :mod:`libcasm.monte`: Provides random number generation, logging, and the
  :class:`ValueMap` data structure used throughout :mod:`libcasm.monte`.
- :mod:`libcasm.monte.sampling`: Provides data structures and methods for sampling data
  and checking for convergence of Monte Carlo calculations.
- :mod:`libcasm.monte.events`: Provides data structures and methods to help specify,
  propose, and apply Monte Carlo events that update discrete occupation variables.
- :mod:`libcasm.monte.methods`: Provides data structures and methods for implementing
  Monte Carlo methods, such as the Metropolis algorithm.


Monte Carlo implementations can be roughly divided into two parts: a "model" which
specifies how microstates are implemented and properties are calculated, and a
"calculator" which implements a particular Monte Carlo method for sampling a particular
thermodynamic ensemble.

CASM does not expect or require a standard interface to allow any model to work with
any calculator. Generally, it is expected that models and calculators can be built
re-using generic methods the libcasm-monte package provides as building blocks.

For tutorial and testing purposes libcasm-monte also includes:

- :mod:`libcasm.monte.ising_cpp`: An Ising model implementation and a semi-grand
  canonical ensemble Monte Carlo calculator, written in C++ using CASM::monte with a
  Python interface using libcasm-monte.
- :mod:`libcasm.monte.ising_py`: An Ising model implementation and a semi-grand
  canonical ensemble Monte Carlo calculator, written in Python using libcasm-monte.


Monte Carlo models
------------------

Generally, a model implements:

- a *configuration* data structure, to represent microstates,
- a *state* data structure, to represent a configuration and the current thermodynamic
  conditions,
- as many *property calculator* methods as necessary to calculate properties of
  configurations,
- a *system* data structure, to store property calculators, and handle input of data that
  is used by property calculators, such as parametric composition axes,
  order parameter definitions, neighbor lists, and cluster expansion basis sets and
  coefficients.

For example, the CASM cluster expansion model in libcasm-clexmonte_ is implemented
using:

- the :class:`~libcasm.clexmonte.Configuration` and :class:`~libcasm.clexmonte.State`
  classes to represent microstates and thermodynamic conditions, and
- the :class:`~libcasm.clexmonte.System` class to manage the data needed by the
  calculators.

It also makes use of:

- the :class:`~libcasm.clexulator.ClusterExpansion` class and related methods for
  calculating energies,
- the :class:`~libcasm.composition.CompositionCalculator` and
  :class:`~libcasm.composition.CompositionConverter` classes and related methods for
  calculating compositions,
- the :class:`~libcasm.clexulator.OrderParameter` class for calculating order
  parameters.


Monte Carlo calculators
-----------------------

Generally, a calculator includes:

- the *Monte Carlo calculator* class with a run method which implements a particular
  method for sampling properties of microstates in a particular thermodynamic ensemble,
- an *event generator* method, for proposing Monte Carlo events,
- a *potential calculator* method, for calculating changes in the thermodynamic
  potential due to an event, under given thermodynamic conditions.

For example, the :class:`libcasm.clexmonte.semigrand_canonical` package implements
Monte Carlo simulations in the semi-grand canonical ensemble for the CASM cluster
expansion model.

The :class:`libcasm.clexmonte.semigrand_canonical` package provides:

- the :class:`~libcasm.clexmonte.semigrand_canonical.SemiGrandCanonicalConditions`
  class for representing thermodynamic conditions,
- the :class:`~libcasm.clexmonte.semigrand_canonical.SemiGrandCanonicalEventGenerator`
  class for proposing events in the semi-grand canonical ensemble,
- the :class:`~libcasm.clexmonte.semigrand_canonical.SemiGrandCanonicalPotential`
  class for calculating changes in the semi-grand canonical energy due to the
  proposed events, and
- the :class:`~libcasm.clexmonte.semigrand_canonical.SemiGrandCanonicalCalculator`
  class for sampling microstates in the semi-grand canonical ensemble.

It also makes use of:

- :class:`~libcasm.monte.sampling.SamplingFixture` and
  :class:`~libcasm.monte.sampling.RunManager`, to control sampling, convergence
  checking, and results output.


Other CASM cluster expansion model calculator packages include:

- :mod:`~libcasm.clexmonte.canonical`: The standard CASM canonical Monte Carlo
  implementation using the Metropolis algorithm
- :mod:`~libcasm.clexmonte.semigrand_canonical`: The standard CASM semigrand-canonical Monte
  Carlo implementation using the Metropolis algorithm
- :mod:`~libcasm.clexmonte.kinetic`: The standard CASM kinetic Monte Carlo implementation
- :mod:`~libcasm.clexmonte.nfold`: Implements semigrand-canonical Monte
  Carlo calculations using the N-fold way algorithm
- :mod:`~libcasm.clexmonte.flex`: A flexible CASM Monte Carlo implementation that allows
  including a additional terms to the potential to enable umbrella sampling, special
  quasi-random structure (SQS) generation, and other approaches.

.. _libcasm-clexmonte: https://prisms-center.github.io/CASMcode_pydocs/libcasm/clexmonte/2.0/
