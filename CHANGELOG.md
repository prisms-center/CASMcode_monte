# Changelog

All notable changes to `libcasm-monte` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-08-14

### Changed

- Set pybind11~=3.0


## [2.1.0] - 2025-08-07

### Changed

- Build Linux wheels using manylinux_2_28 (previously manylinux2014)
- Removed Cirrus CI testing


## [2.0.0] - 2025-05-03

### Changed

- Build for Python 3.13
- Restrict requires-python to ">=3.9,<3.14"
- Run CI tests using Python 3.13
- Build MacOS arm64 wheels using MacOS 15
- Build Linux wheels using Ubuntu 24.04


## [2.0a6] - 2024-02-11

### Changed

- Changed all random number engine type to default to the value set for the typedef `monte::default_engine_type`. The default engine is still `std::mt19937_64`.


## [2.0a5] - 2025-02-10

### Added

- Added `AtomInfo` and `AtomInfoMap` to `libcasm.monte.events`.
- Added more informative error messages for continuous 1d histogram functions that give a bad partition or infinite value

### Fixed

- Fixed documentation and binding errors in `libcasm.monte.events` and `libcasm.monte.sampling`.


## [2.0a4] - 2024-12-20

### Fixed

- Fixed error parsing SelectedEventFunctionParams from JSON.
- Fixed error merging log space Histogram1D which resulted in infinite loops.


## [2.0a3] - 2024-12-11

### Added

- Added `SamplingParams.json_sampler_names`, `SamplingParams.append_to_sampler_names`, `SamplingParams.remove_from_sampler_names`, `SamplingParams.extend_sampler_names`, `SamplingParams.append_to_json_sampler_names`, `SamplingParams.remove_from_json_sampler_names`, and `SamplingParams.extend_json_sampler_names`.
- Added `jsonSampler` and `jsonSamplerMap` to `libcasm.monte.sampling`.
- Added selected event data sampling methods to `libcasm.monte.sampling`.
- Added memory usage to results.

### Changed

- Changed `SamplingFixture::initialize` to copy sampling functions from the SamplingFixtureParams object so functions that sample changes do not need to check if a new run has begun.

### Fixed

- Fixed Conversions constructor to differentiate sublattices by symmetry and order of occupants. Documentation has been updated to reflect that the `asym` unit index used by Conversions, Mol, OccTransform, OccCandidate, OccCandidateList, etc. indicates sites that are equivalent by symmetry and order of occupants.


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
