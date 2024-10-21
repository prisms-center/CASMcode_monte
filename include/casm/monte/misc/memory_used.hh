#ifndef CASM_monte_misc_memory_used
#define CASM_monte_misc_memory_used

#include <cstddef>
#include <string>

namespace CASM {

/// \brief The amount of memory currently being used by this process, in bytes.
std::size_t memory_used(bool resident);

/// \brief Memory usage in MiB
double memory_used_MiB(bool resident);

/// \brief Memory usage as a human-readable string.
std::string convert_size(std::size_t size_bytes);

}  // namespace CASM

#endif  // CASM_monte_misc_memory_used
