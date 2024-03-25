#ifndef CASM_monte_results_io_ResultsIO
#define CASM_monte_results_io_ResultsIO

#include <vector>

#include "casm/casm_io/json/jsonParser.hh"
#include "casm/global/definitions.hh"
#include "casm/misc/cloneable_ptr.hh"

namespace CASM {
namespace monte {

struct ValueMap;

template <typename _ResultsType>
class ResultsIO : public notstd::Cloneable {
  ABSTRACT_CLONEABLE(ResultsIO)
 public:
  typedef _ResultsType results_type;

  virtual void write(results_type const &results, ValueMap const &conditions,
                     Index run_index) = 0;

  virtual jsonParser to_json() = 0;
};

}  // namespace monte
}  // namespace CASM

#endif
