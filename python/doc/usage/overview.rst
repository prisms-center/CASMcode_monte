libcasm-monte Usage
===================

The libcasm-monte package provides data structures and methods that can form the
building blocks for Monte Carlo simulation implementations. The Python interface is
provided for controlling simulations, data analysis, and initial testing. For the most
efficient simulations, C++ extensions are used to implement calculators and checks
used in the inner loop.

.. note::

    The libcasm-clexmonte_ package provides Python interfaces to efficient C++
    implementations of the CASM cluster expansion Monte Carlo methods.


Overview
--------

CASM Monte Carlo implementations can be roughly divided into two parts: a "model" and a
"calculator". A model includes:

- a "configuration" data structure used to represent microstates of a crystal system,
- the methods used to calculate configuration properties, and
- a "system" data structure to organize access to the property calculators and any data
  they use.

and a calculator includes:

- an "event generator" method to propose events that update the configuration and
  allow sampling, and
- a "potential calculator" method to calculate changes in thermodynamic potential for
  the proposed events under given thermodynamic conditions, and
- methods to sample properties, check for convergence, and output results.

CASM does not expect or require a standard interface to allow any model to work with
any calculator. Generally, it is expected that one or more calculators are built for a
particular model re-using generic Monte Carlo methods the libcasm-monte package
provides as building blocks.

The libcasm-monte package includes:

- :mod:`libcasm.monte`: Provides random number generation, logging, and the
  :class:`ValueMap` data structure used throughout :mod:`libcasm.monte`.
- :mod:`libcasm.monte.sampling`: Provides data structures and methods for sampling data
  and checking for convergence of Monte Carlo calculations.
- :mod:`libcasm.monte.events`: Provides data structures and methods to help specify,
  propose, and apply Monte Carlo events that update the discrete occupation variables.
- :mod:`libcasm.monte.methods`: Provides data structures and methods for implementing
  Monte Carlo methods, such as the Metropolis algorithm.
- :mod:`libcasm.monte.ising_cpp`: An example Ising model and a semi-grand canonical
  calculator, implemented in C++ using CASM::monte with a Python interface using
  libcasm-monte. Provided for tutorial and testing purposes.
- :mod:`libcasm.monte.ising_py`: Ising model and a semi-grand canonical calculator,
  fully implemented in Python using libcasm-monte. Provided for tutorial and testing
  purposes.


Monte Carlo models
------------------

Generally, a model implements:

- a configuration data structure to represent microstates,
- a state data structure, to represent a configuration and the current thermodynamic
  conditions,
- as many property calculation methods as necessary to calculate properties of
  configurations,
- a system data structure, to store property calculators, and handle input of data that
  is used by property calculators, such as parametric composition axes,
  order parameter definitions, neighbor lists, and cluster expansion basis sets and
  coefficients.

For example, the CASM cluster expansion model in libcasm-clexmonte_ is implemented
using:

- the :class:`~libcasm.clexmonte.Configuration`, :class:`~libcasm.clexmonte.State`, and
  :class:`~libcasm.clexmonte.Conditions`, classes to represent microstates and
  thermodynamic conditions,
- the :class:`~libcasm.clexulator.ClusterExpansion` class and related methods for
  calculating energies,
- the :class:`~libcasm.composition.CompositionCalculator` and
  :class:`~libcasm.composition.CompositionConverter` classes and related methods for
  calculating compositions,
- the :class:`~libcasm.clexulator.OrderParameter` class for calculating order
  parameters, and
- the :class:`~libcasm.clexmonte.System` class to manage the data needed by the
  calculators.

Existing models:

- libcasm.clexmonte: The standard CASM cluster expansion Hamiltonian model
  implementation, using CASM data structures for configurations and CASM clexulators
  for evaluating energies
- libcasm.monte.ising_cpp: An example Ising model, implemented in C++ using CASM::monte
  with a Python interface using libcasm-monte.
- libcasm.monte.ising_py: An example Ising model, fully implemented in Python using
  libcasm-monte.


Monte Carlo calculators
-----------------------

A Monte Carlo calculator samples properties of microstates in a particular statistical
ensemble.

For example, the :class:`libcasm.clexmonte.semigrand_canonical` package implements
Monte Carlo simulations in the semi-grand canonical ensemble for the CASM cluster
expansion model.

The :class:`libcasm.clexmonte.semigrand_canonical` package provides:

- the :class:`~libcasm.clexmonte.semigrand_canonical.SemiGrandCanonicalPotential`
  class for calculating the semi-grand canonical energy,
- the :class:`~libcasm.clexmonte.semigrand_canonical.SemiGrandCanonicalCalculator`
  class for sampling microstates in the semi-grand canonical ensemble,

and it uses:

- :class:`~libcasm.clexmonte.Conditions`, for representing thermodynamic conditions, and
- :class:`~libcasm.monte.sampling.SamplingFixture` and
  :class:`~libcasm.monte.sampling.RunManager`, to specify sampling functions,
  sampling, and convergence checking criteria, and to store sampled data and output
  results.


Existing calculator packages:

- libcasm.clexmonte.canonical: The standard CASM canonical Monte Carlo implementation
  using the Metropolis algorithm
- libcasm.clexmonte.semigrand_canonical: The standard CASM semigrand-canonical Monte
  Carlo implementation using the Metropolis algorithm
- libcasm.clexmonte.kinetic: The standard CASM kinetic Monte Carlo implementation
- libcasm.clexmonte.nfold: Implements semigrand-canonical Monte
  Carlo calculations using the N-fold way algorithm
- libcasm.clexmonte.flex: A flexible CASM Monte Carlo implementation that allows
  including a additional terms to the potential to enable umbrella sampling, special
  quasi-random structure (SQS) generation, and other approaches.
- libcasm.monte.ising_cpp.semigrand_canonical: An example semi-grand canonical Monte
  Carlo calculator for the Ising model, implemented in C++ using CASM::monte
  with a Python interface using libcasm-monte.
- libcasm.monte.ising_py.semigrand_canonical: An example semi-grand canonical Monte
  Carlo calculator for the Ising model, fully implemented in Python using libcasm-monte.

.. _libcasm-clexmonte: https://prisms-center.github.io/CASMcode_pydocs/libcasm/clexmonte/2.0/
