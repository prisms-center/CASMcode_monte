cmake_minimum_required(VERSION 3.20)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version")

project(CASMcode_monte VERSION 2.2.0 LANGUAGES CXX)

# set CMAKE_INSTALL_X variables
include(GNUInstallDirs)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# try to use ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

##############################################
## Find dependencies

# Should find ZLIB::ZLIB
find_package(ZLIB)

# Find CASM
if(NOT DEFINED CASM_PREFIX)
  message(STATUS "CASM_PREFIX not defined")
  # try to find Python
  find_package (Python COMPONENTS Interpreter Development)
  if(DEFINED Python_EXECUTABLE)
    # if Python found, obtain CASM_PREFIX from the libcasm.casmglobal
    message(STATUS "found Python_EXECUTABLE: ${Python_EXECUTABLE}")
    message(STATUS "checking for libcasm-global")
    execute_process(
      COMMAND pip show libcasm-global
      RESULT_VARIABLE EXIT_CODE
      OUTPUT_QUIET
    )
    if (${EXIT_CODE} EQUAL 0)
      message(STATUS "found libcasm-global")
      execute_process(COMMAND ${Python_EXECUTABLE} -m libcasm.casmglobal --prefix
                      OUTPUT_VARIABLE CASM_PREFIX_RAW)
      string(STRIP ${CASM_PREFIX_RAW} CASM_PREFIX)
      message(STATUS "CASM_PREFIX: ${CASM_PREFIX}")
    else()
      message(STATUS "did not find libcasm-global")
    endif()
  endif()
endif()
if(DEFINED CASM_PREFIX)
  set(CASMcode_global_ROOT ${CASM_PREFIX}/share/CASMcode_global/cmake)
  set(CASMcode_crystallography_ROOT ${CASM_PREFIX}/share/CASMcode_crystallography/cmake)
endif()

find_package(CASMcode_global)
if(NOT CASMcode_global_FOUND)
  message(FATAL_ERROR "CMake failed to find CASMcode_global")
endif()
# if successful, we have CASM::casm_global

find_package(CASMcode_crystallography)
if(NOT CASMcode_crystallography_FOUND)
  message(FATAL_ERROR "CMake failed to find CASMcode_crystallography")
endif()
# if successful, we have CASM::casm_crystallography

# if no user CMAKE_INSTALL_PREFIX, use CASM_PREFIX if it exists
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if(DEFINED CASM_PREFIX)
    message(STATUS "CMAKE_INSTALL_PREFIX initialized to default, so updating CMAKE_INSTALL_PREFIX to CASM_PREFIX")
    set(CMAKE_INSTALL_PREFIX ${CASM_PREFIX} CACHE PATH "set CMAKE_INSTALL_PREFIX to CASM_PREFIX" FORCE)
    message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
  endif()
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

##############################################
## Build libcasm_monte

# create libcasm_monte
set(
  libcasm_monte_HEADERS
  ${PROJECT_SOURCE_DIR}/include/casm/monte/BasicStatistics.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/Conversions.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/MTRandEngine.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/MethodLog.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/RandomNumberGenerator.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/ValueMap.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/checks/CompletionCheck.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/checks/ConvergenceCheck.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/checks/CutoffCheck.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/checks/EquilibrationCheck.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/checks/io/json/CompletionCheck_json_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/checks/io/json/ConvergenceCheck_json_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/checks/io/json/CutoffCheck_json_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/checks/io/json/EquilibrationCheck_json_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/definitions.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/events/OccCandidate.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/events/OccEvent.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/events/OccEventProposal.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/events/OccLocation.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/events/io/OccCandidate_json_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/events/io/OccCandidate_stream_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/io/json/ValueMap_json_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/ising_cpp/basic_semigrand_canonical.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/ising_cpp/complete_semigrand_canonical.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/ising_cpp/model.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/methods/basic_occupation_metropolis.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/methods/kinetic_monte_carlo.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/methods/metropolis.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/methods/nfold.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/methods/occupation_metropolis.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/misc/BasicStructureTools.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/misc/LexicographicalCompare.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/misc/math.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/misc/memory_used.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/misc/polymorphic_method_json_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/run_management/Results.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/run_management/ResultsAnalysisFunction.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/run_management/RunManager.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/run_management/SamplingFixture.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/run_management/State.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/run_management/io/ResultsIO.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/run_management/io/json/ResultsIO_json_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/run_management/io/json/SamplingFixtureParams_json_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/run_management/io/json/jsonResultsIO.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/run_management/io/json/jsonResultsIO_impl.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/sampling/HistogramFunction.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/sampling/RequestedPrecisionConstructor.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/sampling/Sampler.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/sampling/SamplingParams.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/sampling/SelectedEventFunctions.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/sampling/StateSamplingFunction.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/sampling/io/json/Sampler_json_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/sampling/io/json/SamplingParams_json_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/monte/sampling/io/json/SelectedEventFunctions_json_io.hh
)
set(
  libcasm_monte_SOURCES
  ${PROJECT_SOURCE_DIR}/src/casm/monte/BasicStatistics.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/Conversions.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/checks/EquilibrationCheck.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/checks/io/json/CutoffCheck_json_io.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/checks/io/json/EquilibrationCheck_json_io.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/events/OccCandidate.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/events/OccLocation.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/events/io/OccCandidate_json_io.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/events/io/OccCandidate_stream_io.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/io/json/ValueMap_json_io.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/misc/BasicStructureTools.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/misc/memory_used.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/sampling/SelectedEventFunctions.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/sampling/io/json/Sampler_json_io.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/sampling/io/json/SamplingParams_json_io.cc
  ${PROJECT_SOURCE_DIR}/src/casm/monte/sampling/io/json/SelectedEventFunctions_json_io.cc
)
add_library(casm_monte SHARED ${libcasm_monte_SOURCES})
target_include_directories(casm_monte
  PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/casm/external>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/casm/external/gzstream>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/casm/external>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/casm/external/gzstream>
)
target_compile_options(casm_monte
  PUBLIC
    "-DCASM_MONTE_TXT_VERSION=\"${CMAKE_PROJECT_VERSION}\""
    -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=long
    -DGZSTREAM_NAMESPACE=gz
)
target_link_libraries(casm_monte
  ZLIB::ZLIB
  ${CMAKE_DL_LIBS}
  CASM::casm_global
  CASM::casm_crystallography
)
if(APPLE)
  set_target_properties(
    casm_monte PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(
    casm_monte PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()


##############################################
## Install libcasm_monte

# install header files in <prefix>/libcasm/include/,
# while preserving directory structure
foreach ( filevar ${libcasm_monte_HEADERS} )
  file(RELATIVE_PATH relfile ${PROJECT_SOURCE_DIR}/include/ ${filevar})
  get_filename_component( reldir ${relfile} DIRECTORY )
  install( FILES ${filevar} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${reldir} )
endforeach()

# install libcasm_monte in <prefix>/libcasm/lib/
install(
  TARGETS casm_monte
  EXPORT CASMcode_monteTargets
  DESTINATION lib)

##############################################
## Python extensions

# The CMake package config and target files are installed under the Python
# package root. This is necessary to ensure that all the relative paths in the
# helloTargets.cmake resolve correctly. It also provides encapsulation.
#
# The actual path used must be selected so that consuming projects can locate it
# via `find_package`. To support finding CMake packages in the Python package
# prefix, using `find_package`s default search path of
# `<prefix>/<name>/share/<name>*/cmake/` is reasonable. Adding the Python
# package installation prefix to CMAKE_PREFIX_PATH in combination with this path
# will allow `find_package` to find this package and any other package installed
# via a Python package if the CMake and Python packages are named the same.
set(CASM_CMAKE_PACKAGE_INSTALL_SUBDIR "share/CASMcode_monte/cmake")

install(
  EXPORT CASMcode_monteTargets
  NAMESPACE CASM::
  DESTINATION ${CASM_CMAKE_PACKAGE_INSTALL_SUBDIR})

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  CASMcode_monteConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/CASMcode_monteConfig.cmake.in" CASMcode_monteConfig.cmake
  INSTALL_DESTINATION ${CASM_CMAKE_PACKAGE_INSTALL_SUBDIR})

install(FILES "${PROJECT_BINARY_DIR}/CASMcode_monteConfig.cmake"
              "${PROJECT_BINARY_DIR}/CASMcode_monteConfigVersion.cmake"
        DESTINATION ${CASM_CMAKE_PACKAGE_INSTALL_SUBDIR})

# We are using the SKBUILD variable, which is defined when scikit-build is
# running the CMake build, to control building the Python wrapper. This allows
# the C++ project to be installed, standalone, when using the standard CMake
# build flow.
if(DEFINED SKBUILD)

  # call pybind11-config to obtain the root of the cmake package
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE pybind11_ROOT_RAW)
  string(STRIP ${pybind11_ROOT_RAW} pybind11_ROOT)
  find_package(pybind11)

  # The extension modules must load:
  # - the casm_global library
  # - the casm_crystallography library
  # - the casm_monte library
  # They can be found by setting a relative rpath

  ### libcasm.monte._monte ###
  pybind11_add_module(_monte MODULE
                      "${PROJECT_SOURCE_DIR}/python/src/monte.cpp")
  target_link_libraries(_monte PRIVATE
    CASM::casm_global
    CASM::casm_crystallography
    casm_monte
  )
  install(TARGETS _monte DESTINATION monte)
  if(APPLE)
    set_target_properties(
      _monte PROPERTIES INSTALL_RPATH "@loader_path/../lib")
  else()
    set_target_properties(
      _monte PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
  endif()

  ### libcasm.monte.events._monte_events ###
  pybind11_add_module(_monte_events MODULE
                      "${PROJECT_SOURCE_DIR}/python/src/monte_events.cpp")
  target_link_libraries(_monte_events PRIVATE
    CASM::casm_global
    CASM::casm_crystallography
    casm_monte
  )
  install(TARGETS _monte_events DESTINATION monte/events)
  if(APPLE)
    set_target_properties(
      _monte_events PROPERTIES INSTALL_RPATH "@loader_path/../../lib")
  else()
    set_target_properties(
      _monte_events PROPERTIES INSTALL_RPATH "$ORIGIN/../../lib")
  endif()

  ### libcasm.monte.methods._monte_methods ###
  pybind11_add_module(_monte_methods MODULE
                      "${PROJECT_SOURCE_DIR}/python/src/monte_methods.cpp")
  target_link_libraries(_monte_methods PRIVATE
    CASM::casm_global
    CASM::casm_crystallography
    casm_monte
  )
  install(TARGETS _monte_methods DESTINATION monte/methods)
  if(APPLE)
    set_target_properties(
      _monte_methods PROPERTIES INSTALL_RPATH "@loader_path/../../lib")
  else()
    set_target_properties(
      _monte_methods PROPERTIES INSTALL_RPATH "$ORIGIN/../../lib")
  endif()

  ### libcasm.monte.ising_cpp._monte_ising_cpp ###
  pybind11_add_module(_monte_ising_cpp MODULE
                      "${PROJECT_SOURCE_DIR}/python/src/monte_ising_cpp.cpp")
  target_link_libraries(_monte_ising_cpp PRIVATE
    CASM::casm_global
    CASM::casm_crystallography
    casm_monte
  )
  install(TARGETS _monte_ising_cpp DESTINATION monte/ising_cpp)
  if(APPLE)
    set_target_properties(
      _monte_ising_cpp PROPERTIES INSTALL_RPATH "@loader_path/../../lib")
  else()
    set_target_properties(
      _monte_ising_cpp PROPERTIES INSTALL_RPATH "$ORIGIN/../../lib")
  endif()

  ### libcasm.monte.ising_cpp.semigrand_canonical._monte_ising_cpp_semigrand_canonical ###
  pybind11_add_module(_monte_ising_cpp_semigrand_canonical MODULE
                      "${PROJECT_SOURCE_DIR}/python/src/monte_ising_cpp_semigrand_canonical.cpp")
  target_link_libraries(_monte_ising_cpp_semigrand_canonical PRIVATE
    CASM::casm_global
    CASM::casm_crystallography
    casm_monte
  )
  install(TARGETS _monte_ising_cpp_semigrand_canonical DESTINATION monte/ising_cpp/semigrand_canonical)
  if(APPLE)
    set_target_properties(
      _monte_ising_cpp_semigrand_canonical PROPERTIES INSTALL_RPATH "@loader_path/../../lib")
  else()
    set_target_properties(
      _monte_ising_cpp_semigrand_canonical PROPERTIES INSTALL_RPATH "$ORIGIN/../../lib")
  endif()

  ### libcasm.monte.sampling._monte_sampling ###
  pybind11_add_module(_monte_sampling MODULE
                      "${PROJECT_SOURCE_DIR}/python/src/monte_sampling.cpp")
  target_link_libraries(_monte_sampling PRIVATE
    CASM::casm_global
    casm_monte
  )
  install(TARGETS _monte_sampling DESTINATION monte/sampling)
  if(APPLE)
    set_target_properties(
      _monte_sampling PROPERTIES INSTALL_RPATH "@loader_path/../../lib")
  else()
    set_target_properties(
      _monte_sampling PROPERTIES INSTALL_RPATH "$ORIGIN/../../lib")
  endif()


endif()
