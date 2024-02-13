CASM Monte Carlo Overview
=========================


CASM Monte Carlo implementations consist of two parts: a "model" and a "calculator".

Here, the term "model" is used to refer to the choice of data structures used to
represent microstates of a crystal system and the choice of methods used to calculate
microstate properties. For example, the CASM cluster expansion model is implemented
using the :class:`~libcasm.clexmonte.Configuration` and
:class:`~libcasm.clexmonte.State` classes to represent microstates and thermodynamic
conditions, and the :class:`~libcasm.clexmonte.System` class to manage data needed for
calculating the cluster expansion predicted formation energy, the parametric
composition, order parameters, and other properties of a particular microstate or
allowed event.

The term "calculator" refers to methods used to implement a particular type of Monte
Carlo calculation and any data structures the implementation requires. For example,
the :class:`libcasm.clexmonte.semigrand_canonical` package implements semi-grand
canonical Monte Carlo calculations using the CASM cluster expansion model. CASM does not
expect or require a standard interface to allow any model to work with any calculator.

Generally, it is expected that one or more calculators are built for a particular model
re-using generic Monte Carlo methods the libcasm-monte package provides as building
blocks.

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


CASM Monte Carlo models
-----------------------

In a CASM model the following types are implemented:

- a ConfigurationType, to represent a configuration (microstate)
- a StateType, to represent a configuration and thermodynamic conditions
- as many as needed PropertyCalculatorType, to implement methods that calculate
  properties of a state
- a SystemType, to store property calculators and handle input of data that is used
  by property calculators, such as neighbor lists, order parameter definitions, and
  cluster expansion basis sets.

Existing models:

- libcasm.clexmonte: The standard CASM cluster expansion Hamiltonian model
  implementation, using CASM data structures for configurations and CASM clexulators
  for evaluating energies
- libcasm.monte.ising_cpp: An example Ising model, implemented in C++ using CASM::monte
  with a Python interface using libcasm-monte.
- libcasm.monte.ising_py: An example Ising model, fully implemented in Python using
  libcasm-monte.


CASM Monte Carlo calculators
----------------------------

In a CASM calculator the following types are implemented:

- a MonteCarloCalculatorType, which runs a particular type of Monte Carlo calculation
- a MonteCarloDataType data structure, which stores data needed by the Monte Carlo
  calculator
- a ConditionsType data structure, to represent thermodynamic conditions
- a PotentialType property calculator, which calculates a thermodynamic potential
- a EventGeneratorType, which proposes and applies events

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

