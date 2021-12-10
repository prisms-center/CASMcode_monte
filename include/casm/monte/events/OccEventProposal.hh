#ifndef CASM_monte_OccEventProposal
#define CASM_monte_OccEventProposal

#include <vector>
class MTRand;

namespace CASM {
namespace monte {

struct OccEvent;
class OccLocation;
class OccSwap;

/// \brief Typedef of function pointer
///
/// Could be `propose_canonical_event`, `propose_grand_canonical_event`, or a
/// similar custom function.
typedef OccEvent &(*ProposeOccEventFuntionType)(OccEvent &e,
                                                OccLocation const &,
                                                std::vector<OccSwap> const &,
                                                MTRand &);

/// \brief Choose a swap type from a list of allowed canonical swap types
OccSwap const &choose_canonical_swap(OccLocation const &occ_location,
                                     std::vector<OccSwap> const &canonical_swap,
                                     MTRand &mtrand);

/// \brief Propose canonical OccEvent of particular swap type
OccEvent &propose_canonical_event_from_swap(OccEvent &e,
                                            OccLocation const &occ_location,
                                            OccSwap const &swap,
                                            MTRand &mtrand);

/// \brief Propose canonical OccEvent from list of swap types
OccEvent &propose_canonical_event(OccEvent &e, OccLocation const &occ_location,
                                  std::vector<OccSwap> const &canonical_swap,
                                  MTRand &mtrand);

/// \brief Choose a swap type from a list of allowed grand canonical swap types
OccSwap const &choose_grand_canonical_swap(
    OccLocation const &occ_location,
    std::vector<OccSwap> const &grand_canonical_swap, MTRand &mtrand);

/// \brief Propose grand canonical OccEvent of particular swap type
OccEvent &propose_grand_canonical_event_from_swap(
    OccEvent &e, OccLocation const &occ_location, OccSwap const &swap,
    MTRand &mtrand);

/// \brief Propose grand canonical OccEvent from list of swap types
OccEvent &propose_grand_canonical_event(
    OccEvent &e, OccLocation const &occ_location,
    std::vector<OccSwap> const &grand_canonical_swap, MTRand &mtrand);

}  // namespace monte
}  // namespace CASM

#endif
