#ifndef CASM_monte_MethodLog
#define CASM_monte_MethodLog

#include <fstream>

#include "casm/casm_io/Log.hh"
#include "casm/global/definitions.hh"
#include "casm/global/filesystem.hh"

namespace CASM {
namespace monte {

struct MethodLog {
  /// Where to write log messages - writes to file if not empty
  fs::path logfile_path;

  /// Where to write log messages - set from logfile_path on reset() call
  std::shared_ptr<std::ofstream> fout;

  /// Where to write log messages
  Log log;

  /// How often to log method status, in seconds, if this has value.
  std::optional<double> log_frequency;

  void reset() {
    if (!logfile_path.empty()) {
      fs::create_directories(logfile_path.parent_path());
      fout = std::make_shared<std::ofstream>(logfile_path);
      log.reset(*fout);
    }
  }

  void reset_to_stdout() {
    fout.reset();
    log.reset();
  }
};

}  // namespace monte
}  // namespace CASM

#endif
