"""CASM Monte Carlo models

## CASM Monte Carlo implementation overview

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
re-using generic Monte Carlo methods as building blocks. The
:mod:`libcasm.monte.sampling` package provides data structures and methods for
sampling data and checking for convergence. The :mod:`libcasm.monte.events` package
provides data structures and methods to help specify, propose, and apply Monte Carlo
events.


## CASM Monte Carlo model components

In a CASM model the following types are implemented:

- a ConfigurationType, to represent a configuration (microstate)
- a StateType, to represent a configuration and thermodynamic conditions
- as many as needed PropertyCalculatorType, to implement methods that calculate
  properties of a state
- a SystemType, to store property calculators and handle input of data that is used
  by property calculators, such as neighbor lists, order parameter definitions, and
  cluster expansion basis sets.

Examples of existing models:
- libcasm.clexmonte: The standard CASM cluster expansion Hamiltonian model
  implementation, using CASM data structures for configurations and CASM clexulators
  for evaluating energies
- libcasm.monte.models.ising_cpp: An example C++ Ising model implementation for
  tutorial and testing purposes
- libcasm.monte.models.Ising_py: An example Python Ising model implementation for
  tutorial and testing purposes


## CASM Monte Carlo calculator components

In a CASM calculator the following types are implemented:

- a MonteCarloCalculatorType, which runs a particular type of Monte Carlo calculation
- a MonteCarloDataType data structure, which stores data needed by the Monte Carlo
  calculator
- a ConditionsType data structure, to represent thermodynamic conditions
- a PotentialType property calculator, which calculates a thermodynamic potential

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
- libcasm.monte.calculators: The standard CASM canonical Monte Carlo implementation
- libcasm.clexmonte.canonical: The standard CASM canonical Monte Carlo implementation

- libcasm.monte.models.ising_cpp: An example C++ Ising model implementation for
  tutorial and testing purposes
- libcasm.monte.models.Ising_py: An example Python Ising model implementation for
  tutorial and testing purposes

"""
