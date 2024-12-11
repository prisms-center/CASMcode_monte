#include "casm/monte/misc/memory_used.hh"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

// -- Memory usage --
// https://stackoverflow.com/a/372525
// Changed getpagesize() to sysconf(_SC_PAGESIZE)

#ifdef __linux__
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <mach/mach_init.h>
#include <mach/task.h>
#endif

#ifdef _WINDOWS
#include <windows.h>
#else
#include <sys/resource.h>
#endif

namespace CASM {

/// \brief The amount of memory currently being used by this process, in bytes.
///
/// By default, returns the full virtual arena, but if resident=true,
/// it will report just the resident set in RAM (if supported on that OS).
std::size_t memory_used(bool resident) {
#if defined(__linux__)
  // Ugh, getrusage doesn't work well on Linux.  Try grabbing info
  // directly from the /proc pseudo-filesystem.  Reading from
  // /proc/self/statm gives info on your own process, as one line of
  // numbers that are: virtual mem program size, resident set size,
  // shared pages, text/code, data/stack, library, dirty pages.  The
  // mem sizes should all be multiplied by the page size.
  std::size_t size = 0;
  FILE *file = fopen("/proc/self/statm", "r");
  if (file) {
    unsigned long vm = 0;
    fscanf(file, "%lu", &vm);  // Just need the first num: vm size
    fclose(file);
    long pagesize = sysconf(_SC_PAGESIZE);
    size = (std::size_t)vm * pagesize;
  }
  return size;

#elif defined(__APPLE__)
  // Inspired by:
  // http://miknight.blogspot.com/2005/11/resident-set-size-in-mac-os-x.html
  struct task_basic_info t_info;
  mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;
  task_info(current_task(), TASK_BASIC_INFO, (task_info_t)&t_info,
            &t_info_count);
  std::size_t size = (resident ? t_info.resident_size : t_info.virtual_size);
  return size;

#elif defined(_WINDOWS)
  // According to MSDN...
  PROCESS_MEMORY_COUNTERS counters;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
    return counters.PagefileUsage;
  else
    return 0;

#else
  // No idea what platform this is
  return 0;  // Punt
#endif
}

/// \brief Memory usage in MiB
double memory_used_MiB(bool resident) {
  return memory_used(resident) / (1024.0 * 1024.0);
}

/// \brief Memory usage as a human-readable string.
std::string convert_size(std::size_t size_bytes) {
  // Based on # https://stackoverflow.com/a/14822210
  if (size_bytes == 0) {
    return "0B";
  }
  std::vector<std::string> size_name = {"B",   "KiB", "MiB", "GiB", "TiB",
                                        "PiB", "EiB", "ZiB", "YiB"};
  double i = std::floor(std::log2(size_bytes) / std::log2(1024));
  double s = size_bytes / std::pow(1024, i);
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << s
     << size_name[static_cast<int>(i)];
  return ss.str();
}

}  // namespace CASM

// -- End Memory usage --
