# Changelog

All notable changes to `libcasm-monte` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [2.0a2] - 2024-07-17

### Added

- Added to_json for CompletionCheckParams, SamplingFixtureParams, SamplingParams, jsonResultsIO
- Added "json_quantities" option to SamplingParams
- Added Conversions::species_list()

### Changed

- Use shared_ptr to hold sampling fixtures in RunManager
- Output scalar quantities under "value" key in JSON results output
- Allow MethodLog to output to stdout
- Allow constructing libcasm.monte.ValueMap from dict 


## [2.0a1] - 2024-03-15

The libcasm-monte package provides useful building blocks for Monte Carlo simulations. This includes:

- Sampling classes and functions
- Equilibration, convergence checking, and statistics calculation
- Generic results IO
- Supercell index conversions
- Generic event definitions, construction, and selection

This package includes the Python package libcasm.monte, which may be installed via pip install, using scikit-build, CMake, and pybind11. This release also includes documentation, built using Sphinx.
