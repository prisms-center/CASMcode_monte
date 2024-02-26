#ifndef CASM_monte_run_management_ResultsIO_json_io
#define CASM_monte_run_management_ResultsIO_json_io

#include "casm/monte/definitions.hh"
#include "casm/monte/misc/polymorphic_method_json_io.hh"
#include "casm/monte/run_management/io/json/jsonResultsIO_impl.hh"

namespace CASM {

template <typename T>
class InputParser;

namespace monte {

/// \brief Construct ResultsIO from JSON
template <typename ResultsType>
void parse(InputParser<ResultsIO<ResultsType>> &parser,
           MethodParserMap<ResultsIO<ResultsType>> const &results_io_methods);

/// \brief Construct jsonResultsIO from JSON
template <typename ConfigType, typename StatisticsType>
void parse(InputParser<monte::jsonResultsIO<
               ResultsIO<Results<ConfigType, StatisticsType>>>> &parser);

// ~~~ Definitions ~~~

/// \brief Construct ResultsIO from JSON
///
/// A ResultsIO method implements reading and writing Monte Carlo output.
///
/// Expected:
///   method: string (required)
///     The name of the chosen state generation method. Currently, the only
///     option is:
///     - "json": monte::jsonResultsIO
///
///   kwargs: dict (optional, default={})
///     Method-specific options. See documentation for particular methods:
///     - "json": `parse(monte::jsonResultsIO<config_type> &)`
///
template <typename ResultsType>
void parse(InputParser<ResultsIO<ResultsType>> &parser,
           MethodParserMap<ResultsIO<ResultsType>> const &results_io_methods) {
  parse_polymorphic_method(parser, results_io_methods);
}

/// \brief Construct jsonResultsIO from JSON
///
/// The "json" results IO method writes results to JSON files:
/// - summary.json: Summarizes results from each inidividual run in arrays
/// - run.<index>/trajectory.json: Optional output file (one per each
///   individual run) is a JSON array of the configuration at the time a sample
///   was taken.
/// - run.<index>/observations.json: Optional output file (one per each
///   individual run) contains all sampled data.
///
/// Expected:
///   output_dir: string (required)
///     Specifies the directory where results should be written.
///
///   write_observations: bool (default=false)
///     If true, write an `"observations.json"` file for each individual run.
///
///   write_trajectory: bool (default=false)
///     If true, write an `"trajectory.json"` file for each individual run.
///
template <typename ConfigType, typename StatisticsType>
void parse(
    InputParser<monte::jsonResultsIO<Results<ConfigType, StatisticsType>>>
        &parser) {
  std::string output_dir = "output";
  parser.optional(output_dir, "output_dir");

  bool write_observations = false;
  parser.optional(write_observations, "write_observations");

  bool write_trajectory = false;
  parser.optional(write_trajectory, "write_trajectory");

  if (parser.valid()) {
    parser.value = std::make_unique<
        monte::jsonResultsIO<Results<ConfigType, StatisticsType>>>(
        output_dir, write_trajectory, write_observations);
  }
}

}  // namespace monte
}  // namespace CASM

#endif
