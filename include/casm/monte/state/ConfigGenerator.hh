#ifndef CASM_monte_ConfigGenerator
#define CASM_monte_ConfigGenerator

#include <vector>

#include "casm/monte/definitions.hh"

namespace CASM {
namespace monte {

/// \brief A ConfigGenerator generates a configuration given a set of
///     conditions and results from previous runs
///
/// Notes:
/// - The template parameter _RunInfoType is specified by a particular Monte
///   Carlo method implementation.
/// - _RunInfoType allows customization of what
///   information is provided to a particular configuration generation method.
///   In the basic case, it will be the final state for each run. Templating
///   allows support for more complex cases where the next state could be
///   generated based on the sampled data collected during previous runs.
template <typename _ConfigType, typename _RunInfoType>
class ConfigGenerator {
 public:
  typedef _ConfigType ConfigType;
  typedef _RunInfoType RunInfoType;

  virtual ~ConfigGenerator() {}

  /// \brief Generate a configuration, using information from a set of
  /// conditions and info from previous runs
  virtual ConfigType operator()(ValueMap const &conditions,
                                std::vector<RunInfoType> const &run_info) = 0;
};

}  // namespace monte
}  // namespace CASM

#endif
