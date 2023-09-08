#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// nlohmann::json binding
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "pybind11_json/pybind11_json.hpp"

// std
#include <random>

// CASM
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/monte/Conversions.hh"
#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/events/OccCandidate.hh"
#include "casm/monte/events/OccEvent.hh"
#include "casm/monte/events/OccEventProposal.hh"
#include "casm/monte/events/OccLocation.hh"
#include "casm/monte/events/io/OccCandidate_json_io.hh"
#include "casm/monte/events/io/OccCandidate_stream_io.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

// used for libcasm.monte:
typedef std::mt19937_64 engine_type;
typedef monte::RandomNumberGenerator<engine_type> generator_type;

monte::OccCandidateList make_OccCandidateList(
    monte::Conversions const &convert,
    std::optional<std::vector<monte::OccCandidate>> candidates) {
  if (candidates.has_value()) {
    return monte::OccCandidateList(*candidates, convert);
  }
  return monte::OccCandidateList(convert);
}

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<CASM::Index>);
PYBIND11_MAKE_OPAQUE(std::vector<CASM::monte::Atom>);
PYBIND11_MAKE_OPAQUE(std::vector<CASM::monte::AtomTraj>);
PYBIND11_MAKE_OPAQUE(std::vector<CASM::monte::Mol>);
PYBIND11_MAKE_OPAQUE(std::vector<CASM::monte::OccTransform>);

PYBIND11_MODULE(_monte_events, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        Monte Carlo simulation events

        libcasm.monte.events
        --------------------

        Data structures for representing (kinetic) Monte Carlo events, and methods
        for proposing and applying events.
    )pbdoc";
  py::module::import("libcasm.xtal");

  py::bind_vector<std::vector<int>>(m, "IntVector");
  py::bind_vector<std::vector<Index>>(m, "LongVector");
  py::bind_vector<std::vector<monte::Atom>>(m, "AtomVector");
  py::bind_vector<std::vector<monte::AtomTraj>>(m, "AtomTrajVector");
  py::bind_vector<std::vector<monte::Mol>>(m, "MolVector");
  py::bind_vector<std::vector<monte::OccTransform>>(m, "OccTransformVector");

  py::class_<monte::Atom>(m, "Atom", R"pbdoc(
      Track the position of individual atoms, as if no periodic boundaries

      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          Default constructor only.
          )pbdoc")
      .def_readwrite("translation", &monte::Atom::translation,
                     R"pbdoc(
          np.ndarray(shape=[3,1], dtype=np.int_): Current translation, \
          in fractional coordinates, as if no periodic boundary
          )pbdoc")
      .def_readwrite("n_jumps", &monte::Atom::n_jumps,
                     R"pbdoc(
          int: Current number of jumps
          )pbdoc");

  py::class_<monte::Mol>(m, "Mol", R"pbdoc(
      Represents the occupant on a site

      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          Default constructor only.
          )pbdoc")
      .def_readwrite("id", &monte::Mol::id,
                     R"pbdoc(
          int: Location of molecule in OccLocation mol list.
          )pbdoc")
      .def_readwrite("component_id", &monte::Mol::component,
                     R"pbdoc(
          LongVector: Location of component atoms in OccLocation atom list.
          )pbdoc")
      .def_readwrite("linear_site_index", &monte::Mol::l,
                     R"pbdoc(
          int: Location in configuration occupation vector.
          )pbdoc")
      .def_readwrite("asymmetric_unit_index", &monte::Mol::asym,
                     R"pbdoc(
          int: Current site asymmetric unit index. Must be consistent with `linear_site_index`.
          )pbdoc")
      .def_readwrite("mol_location_index", &monte::Mol::loc,
                     R"pbdoc(
          int: Location in OccLocation mol location list
          )pbdoc");

  py::class_<monte::OccTransform>(m, "OccTransform", R"pbdoc(
      Information used to update :class:`~libcasm.events.OccLocation`

      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          Default constructor only.
          )pbdoc")
      .def_readwrite("linear_site_index", &monte::OccTransform::l,
                     R"pbdoc(
          int: Location in configuration occupation list being transformed.
          )pbdoc")
      .def_readwrite("mol_id", &monte::OccTransform::mol_id,
                     R"pbdoc(
          int: Location in OccLocation mol list being transformed.
          )pbdoc")
      .def_readwrite("asym", &monte::OccTransform::asym,
                     R"pbdoc(
          int: Asymmetric unit index of site being transformed.
          )pbdoc")
      .def_readwrite("from_species", &monte::OccTransform::from_species,
                     R"pbdoc(
          int: Species index, as defined by \
          :class:`~libcasm.monte.Conversions`, before transformation.
          )pbdoc")
      .def_readwrite("to_species", &monte::OccTransform::to_species,
                     R"pbdoc(
          int: Species index, as defined by \
          :class:`~libcasm.monte.Conversions`, after transformation.
          )pbdoc");

  py::class_<monte::AtomLocation>(m, "AtomLocation", R"pbdoc(
    Specify a specific atom location, on a site, or in a molecule

    )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
        Default constructor only.
        )pbdoc")
      .def_readwrite("linear_site_index", &monte::AtomLocation::l,
                     R"pbdoc(
        int: Location in configuration occupation list.
        )pbdoc")
      .def_readwrite("mol_id", &monte::AtomLocation::mol_id,
                     R"pbdoc(
        int: Location in OccLocation mol list.
        )pbdoc")
      .def_readwrite("mol_comp", &monte::AtomLocation::mol_comp,
                     R"pbdoc(
        int: Location in Mol components list.
        )pbdoc");

  py::class_<monte::AtomTraj>(m, "AtomTraj", R"pbdoc(
    Specifies a trajectory from one AtomLocation to another

    )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
        Default constructor only.
        )pbdoc")
      .def_readwrite("from", &monte::AtomTraj::from,
                     R"pbdoc(
        AtomLocation: Initial AtomLocation.
        )pbdoc")
      .def_readwrite("to", &monte::AtomTraj::to,
                     R"pbdoc(
        AtomLocation: Final AtomLocation.
        )pbdoc")
      .def_readwrite("delta_ijk", &monte::AtomTraj::delta_ijk,
                     R"pbdoc(
        np.ndarray(shape=[3,1], dtype=np.int_): Amount to increment Atom \
        translation, in fractional coordinates
        )pbdoc");

  py::class_<monte::OccEvent>(m, "OccEvent", R"pbdoc(
      Describes a Monte Carlo event that modifies occupation

      )pbdoc")
      .def(py::init<>(),
           R"pbdoc(
          Default constructor only.
          )pbdoc")
      .def_readwrite("linear_site_index", &monte::OccEvent::linear_site_index,
                     R"pbdoc(
          LongVector: Linear site indices, indicating on which sites the occupation \
          will be modified.
          )pbdoc")
      .def_readwrite("new_occ", &monte::OccEvent::new_occ,
                     R"pbdoc(
          IntVector: Occupant indices, indicating the new occupation index on the \
          sites being modified.
          )pbdoc")
      .def_readwrite("occ_transform", &monte::OccEvent::occ_transform,
                     R"pbdoc(
          OccTransformVector: Information used to update occupant tracking \
          information stored in :class:`~libcasm.monte.event.OccLocation`.
          )pbdoc")
      .def_readwrite("atom_traj", &monte::OccEvent::atom_traj,
                     R"pbdoc(
          OccTransformVector: Information used to update occupant location \
          information stored in :class:`~libcasm.monte.event.OccLocation` - \
          use if tracking species trajectories for kinetic Monte Carlo.
          )pbdoc")
      .def("__copy__",
           [](monte::OccEvent const &self) { return monte::OccEvent(self); })
      .def("__deepcopy__", [](monte::OccEvent const &self, py::dict) {
        return monte::OccEvent(self);
      });

  py::class_<monte::OccCandidate>(m, "OccCandidate", R"pbdoc(
    A pair of asymmetric unit index and species index, indicating a type of
    occupant that may be chosen for Monte Carlo events

    )pbdoc")
      .def(py::init<Index, Index>(),
           R"pbdoc(
          Parameters
          ----------
          asymmetric_unit_index: int
              Asymmetric unit index
          species_index: int
              Species index, distinguishing each allowed site occupant, including
              distinct molecular orientations if applicable.
          )pbdoc",
           py::arg("asymmetric_unit_index"), py::arg("species_index"))
      .def_readwrite("asymmetric_unit_index", &monte::OccCandidate::asym,
                     R"pbdoc(
          int: Asymmetric unit index
          )pbdoc")
      .def_readwrite("species_index", &monte::OccCandidate::asym,
                     R"pbdoc(
          int: Species index, distinguishing each allowed site occupant, including\
          distinct molecular orientations if applicable.
          )pbdoc")
      .def(
          "is_valid",
          [](monte::OccCandidate const &self,
             monte::Conversions const &convert) {
            return is_valid(convert, self);
          },
          R"pbdoc(
          Checks if indices are valid.

          Parameters
          ----------
          convert: :class:`~libcasm.monte.Conversions`
              The `convert` instance provides the number of asymmetric unit sites and
              species.

          Returns
          -------
          result: bool
              True if `asymmetric_unit_index` and `species_index` are valid.
          )pbdoc",
          py::arg("convert"))
      .def(py::self < py::self,
           "Compares as if (asymmetric_unit_index, species_index)")
      .def(py::self <= py::self,
           "Compares as if (asymmetric_unit_index, species_index)")
      .def(py::self > py::self,
           "Compares as if (asymmetric_unit_index, species_index)")
      .def(py::self >= py::self,
           "Compares as if (asymmetric_unit_index, species_index)")
      .def(py::self == py::self,
           "Compares as if (asymmetric_unit_index, species_index)")
      .def(py::self != py::self,
           "Compares as if (asymmetric_unit_index, species_index)")
      .def("__copy__",
           [](monte::OccCandidate const &self) {
             return monte::OccCandidate(self);
           })
      .def("__deepcopy__", [](monte::OccCandidate const &self,
                              py::dict) { return monte::OccCandidate(self); })
      .def(
          "to_dict",
          [](monte::OccCandidate const &self,
             monte::Conversions const &convert) {
            jsonParser json;
            to_json(self, convert, json);
            return json;
          },
          R"pbdoc(
         Represent the OccCandidate as a Python dict

         Parameters
         ----------
         convert : :class:`~libcasm.monte.Conversions`
             Provides index conversions

         Returns
         -------
         data : dict
              The OccCandidate as a Python dict
         )pbdoc")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             monte::Conversions const &convert) -> monte::OccCandidate {
            jsonParser json{data};
            return jsonConstructor<monte::OccCandidate>::from_json(json,
                                                                   convert);
          },
          R"pbdoc(
         Construct an OccCandidate from a Python dict

         Parameters
         ----------
         data : dict
             The OccCandidate representation

         convert : :class:`~libcasm.monte.Conversions`
             Provides index conversions

         Returns
         -------
         candidate : :class:`~libcasm.monte.events.OccCandidate`
              The OccCandidate
         )pbdoc");

  py::class_<monte::OccSwap>(m, "OccSwap", R"pbdoc(
    Represents a Monte Carlo event that swaps occupants

    )pbdoc")
      .def(py::init<const monte::OccCandidate &, const monte::OccCandidate &>(),
           R"pbdoc(
          Parameters
          ----------
          first: :class:`~libcasm.monte.events.OccCandidate`
              The first candidate occupant
          second: :class:`~libcasm.monte.events.OccCandidate`
              The second candidate occupant
          )pbdoc",
           py::arg("first"), py::arg("second"))
      .def_readwrite("first", &monte::OccSwap::cand_a,
                     R"pbdoc(
          :class:`~libcasm.monte.events.OccCandidate`: The first candidate occupant
          )pbdoc")
      .def_readwrite("second", &monte::OccSwap::cand_b,
                     R"pbdoc(
          :class:`~libcasm.monte.events.OccCandidate`: The second candidate occupant
          )pbdoc")
      .def("reverse", &monte::OccSwap::sort,
           R"pbdoc(
          Transforms self so that `first` and `second` are reversed.
          )pbdoc")
      .def("sort", &monte::OccSwap::sort,
           R"pbdoc(
          Mutates self so that (first, second) <= (second, first).
          )pbdoc")
      .def("sorted", &monte::OccSwap::sorted,
           R"pbdoc(
          bool: Returns True if already sorted
          )pbdoc")
      .def(
          "is_valid",
          [](monte::OccSwap const &self, monte::Conversions const &convert) {
            return is_valid(convert, self);
          },
          R"pbdoc(
          Checks if `first` and `second` are valid.

          Parameters
          ----------
          convert: :class:`~libcasm.monte.Conversions`
              The `convert` instance provides the number of asymmetric unit sites and
              species.

          Returns
          -------
          result: bool
              True if `first` and `second` are valid.
          )pbdoc",
          py::arg("convert"))
      .def(py::self < py::self, "Compares as if (first, second)")
      .def(py::self <= py::self, "Compares as if (first, second)")
      .def(py::self > py::self, "Compares as if (first, second)")
      .def(py::self >= py::self, "Compares as if (first, second)")
      .def(py::self == py::self, "Compares as if (first, second)")
      .def(py::self != py::self, "Compares as if (first, second)")
      .def("__copy__",
           [](monte::OccSwap const &self) { return monte::OccSwap(self); })
      .def("__deepcopy__", [](monte::OccSwap const &self,
                              py::dict) { return monte::OccSwap(self); })
      .def(
          "to_dict",
          [](monte::OccSwap const &self, monte::Conversions const &convert) {
            jsonParser json;
            to_json(self, convert, json);
            return json;
          },
          R"pbdoc(
         Represent the OccSwap as a Python dict

         Parameters
         ----------
         convert : :class:`~libcasm.monte.Conversions`
             Provides index conversions

         Returns
         -------
         data : dict
              The OccSwap as a Python dict
         )pbdoc")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             monte::Conversions const &convert) -> monte::OccSwap {
            jsonParser json{data};
            return jsonConstructor<monte::OccSwap>::from_json(json, convert);
          },
          R"pbdoc(
         Construct an OccSwap from a Python dict

         Parameters
         ----------
         data : dict
             The OccSwap representation

         convert : :class:`~libcasm.monte.Conversions`
             Provides index conversions

         Returns
         -------
         swap : :class:`~libcasm.monte.events.OccSwap`
              The OccSwap
         )pbdoc");

  py::class_<monte::OccCandidateList>(m, "OccCandidateList", R"pbdoc(
    Stores a list of allowed OccCandidate

    )pbdoc")
      .def(py::init<>(&make_OccCandidateList),
           R"pbdoc(
          Parameters
          ----------
          convert: :class:`~libcasm.monte.Conversions`
              The `convert` instance provides the number of asymmetric unit sites and
              species.
          candidates: Optional[list[:class:`~libcasm.monte.events.OccCandidate`]] = None
              A custom list of candidate occupant types for Monte Carlo events. If None,
              then all possible candidates are constructed.
          )pbdoc",
           py::arg("convert"), py::arg("candidates") = std::nullopt)
      .def(
          "index",
          [](monte::OccCandidateList const &self,
             monte::OccCandidate const &cand) { return self.index(cand); },
          R"pbdoc(
          int: Return index of `candidate` in the list, or len(self) if not allowed
          )pbdoc",
          py::arg("candidate"))
      .def(
          "matching_index",
          [](monte::OccCandidateList const &self, Index asym,
             Index species_index) { return self.index(asym, species_index); },
          R"pbdoc(
          int: Return index of `candidate` with matching (asymmetric_unit_index, species_index) in the list, or len(self) if not allowed
          )pbdoc",
          py::arg("asymmetric_unit_index"), py::arg("species_index"))
      .def(
          "__getitem__",
          [](monte::OccCandidateList const &self, Index candidate_index) {
            return self[candidate_index];
          },
          R"pbdoc(
         int: Return index of `candidate` in the list, or len(self) if not allowed
         )pbdoc")
      .def("__len__",
           [](monte::OccCandidateList const &self) { return self.size(); })
      .def(
          "__iter__",  // for x in occ_candidate_list
          [](monte::OccCandidateList const &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<
              0, 1>() /* Essential: keep object alive while iterator exists */)
      .def(
          "to_dict",
          [](monte::OccCandidateList const &self,
             monte::Conversions const &convert) {
            jsonParser json;
            to_json(self, convert, json);
            return json;
          },
          R"pbdoc(
          Represent the OccCandidateList as a Python dict

          Includes the possible canonical and semi-grand canonical events
          that can be generated from the candidates.

          Parameters
          ----------
          convert : :class:`~libcasm.monte.Conversions`
              Provides index conversions

          Returns
          -------
          data : dict
              The OccCandidateList as a Python dict
          )pbdoc")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data,
             monte::Conversions const &convert) -> monte::OccCandidateList {
            jsonParser json{data};
            return jsonConstructor<monte::OccCandidateList>::from_json(json,
                                                                       convert);
          },
          R"pbdoc(
          Construct an OccCandidateList from a Python dict

          Parameters
          ----------
          data : dict
              The OccCandidateList representation

          convert : :class:`~libcasm.monte.Conversions`
              Provides index conversions

          Returns
          -------
          candidate_list : :class:`~libcasm.monte.events.OccCandidateList`
              The OccCandidateList
          )pbdoc");

  m.def("is_allowed_canonical_swap", &monte::allowed_canonical_swap,
        R"pbdoc(
        Check that candidates form an allowed canonical Monte Carlo swap

        Checks that:
        - `first` and `second` are valid
        - the `species_index` are different and allowed on both asymmetric unit sites


        Parameters
        ----------
        convert: :class:`~libcasm.monte.Conversions`
            Provides index conversions
        first: :class:`~libcasm.monte.events.OccCandidate`
            The first candidate occupant
        second: :class:`~libcasm.monte.events.OccCandidate`
            The second candidate occupant

        Returns
        -------
        is_allowed : bool
            True if candidates form an allowed canonical Monte Carlo swap
        )pbdoc",
        py::arg("convert"), py::arg("first"), py::arg("second"));

  m.def("make_canonical_swaps", &monte::make_canonical_swaps,
        R"pbdoc(
        Make all allowed OccSwap for canonical Monte Carlo events

        Parameters
        ----------
        convert: :class:`~libcasm.monte.Conversions`
            Provides index conversions
        occ_candidate_list: :class:`~libcasm.monte.events.OccCandidateList`
            The allowed candidate occupants

        Returns
        -------
        canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for canonical Monte Carlo events. This
            does not allow both forward and reverse swaps to be included.
        )pbdoc",
        py::arg("convert"), py::arg("occ_candidate_list"));

  m.def("is_allowed_semigrand_canonical_swap",
        &monte::allowed_grand_canonical_swap,
        R"pbdoc(
        Check that candidates form an allowed semi-grand canonical Monte Carlo swap

        Checks that:
        - `first` and `second` are valid
        - the `asymmetric_unit_index` are the same
        - the `species_index` are different and both allowed on the asymmetric unit site

        Parameters
        ----------
        convert: :class:`~libcasm.monte.Conversions`
            Provides index conversions
        first: :class:`~libcasm.monte.events.OccCandidate`
            The first candidate occupant
        second: :class:`~libcasm.monte.events.OccCandidate`
            The second candidate occupant

        Returns
        -------
        is_allowed : bool
            True if candidates form an allowed semi-grand canonical Monte Carlo swap
        )pbdoc",
        py::arg("convert"), py::arg("first"), py::arg("second"));

  m.def("make_semigrad_canonical_swaps", &monte::make_grand_canonical_swaps,
        R"pbdoc(
        Make all allowed OccSwap for semi-grand canonical Monte Carlo events

        Parameters
        ----------
        convert: :class:`~libcasm.monte.Conversions`
            Provides index conversions
        occ_candidate_list: :class:`~libcasm.monte.events.OccCandidateList`
            The allowed candidate occupants

        Returns
        -------
        semigrand_canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for semi-grand canonical Monte Carlo events.
            This does include both forward and reverse swaps.
        )pbdoc",
        py::arg("convert"), py::arg("occ_candidate_list"));

  m.def("swaps_allowed_per_unitcell", &monte::get_n_allowed_per_unitcell,
        R"pbdoc(
        For semi-grand canonical swaps, get the number of possible events per unit cell

        Parameters
        ----------
        convert: :class:`~libcasm.monte.Conversions`
            Provides index conversions
        semigrand_canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for semi-grand canonical Monte Carlo events.
            This does include both forward and reverse swaps.


        Returns
        -------
        result: int
            Total number of possible swaps per unit cell, using the multiplicity
            for each site asymmetric unit.
        )pbdoc",
        py::arg("convert"), py::arg("occ_candidate_list"));

  py::class_<monte::OccLocation>(m, "OccLocation", R"pbdoc(
    Specify a specific atom location, on a site, or in a molecule

    )pbdoc")
      .def(py::init<const monte::Conversions &, const monte::OccCandidateList &,
                    bool>(),
           R"pbdoc(
          Parameters
          ----------
          convert: :class:`~libcasm.monte.Conversions`
              The `convert` instance provides the number of asymmetric unit sites and
              species.
          occ_candidate_list: :class:`~libcasm.monte.events.OccCandidateList`
              A list of candidate occupant types for Monte Carlo events.
          update_species: bool

          )pbdoc",
           py::arg("convert"), py::arg("occ_candidate_list"),
           py::arg("update_species") = false)
      .def("initialize", &monte::OccLocation::initialize,
           R"pbdoc(
          Fill tables with current occupation info
          )pbdoc",
           py::arg("occupation"))
      .def("apply", &monte::OccLocation::apply,
           R"pbdoc(
          Update occupation vector and this to reflect that `event` occurred.
          )pbdoc",
           py::arg("event"), py::arg("occupation"))
      .def(
          "choose_mol_by_candidate_index",
          [](monte::OccLocation const &self, Index cand_index,
             generator_type &random_number_generator) {
            return self.choose_mol(cand_index, random_number_generator);
          },
          R"pbdoc(
          Stochastically choose an occupant of a particular OccCandidate type.
          )pbdoc",
          py::arg("candidate_index"), py::arg("random_number_generator"))
      .def(
          "choose_mol",
          [](monte::OccLocation const &self, monte::OccCandidate const &cand,
             generator_type &random_number_generator) {
            return self.choose_mol(cand, random_number_generator);
          },
          R"pbdoc(
          Stochastically choose an occupant of a particular OccCandidate type.
          )pbdoc",
          py::arg("candidate"), py::arg("random_number_generator"))
      .def("mol_size", &monte::OccLocation::mol_size,
           R"pbdoc(
          Total number of mutating sites.
          )pbdoc")
      .def(
          "mol",
          [](monte::OccLocation &self, Index mol_id) {
            return self.mol(mol_id);
          },
          R"pbdoc(
          Access Mol by id (location of molecule in mol list).
          )pbdoc")
      .def("atom_size", &monte::OccLocation::mol_size,
           R"pbdoc(
          Total number of atoms.
          )pbdoc")
      .def(
          "atom",
          [](monte::OccLocation &self, Index mol_id) {
            return self.mol(mol_id);
          },
          R"pbdoc(
          Access Atom by id (location of atom in atom list).
          )pbdoc")
      .def("atom_positions_cart", &monte::OccLocation::atom_positions_cart,
           R"pbdoc(
          Return current atom positions in cartesian coordinates, shape=(3, atom_size).
          )pbdoc")
      .def("atom_positions_cart_within",
           &monte::OccLocation::atom_positions_cart_within,
           R"pbdoc(
          Return current atom positions in cartesian coordinates, shape=(3, atom_size).
          )pbdoc")
      .def("initial_atom_species_index",
           &monte::OccLocation::initial_atom_species_index,
           R"pbdoc(
          Holds initial species index for each atom in atom position matrices.
          )pbdoc")
      .def("initial_atom_position_index",
           &monte::OccLocation::initial_atom_position_index,
           R"pbdoc(
          Holds initial atom position index for each atom in atom position matrices.
          )pbdoc")
      .def("current_atom_names", &monte::OccLocation::current_atom_names,
           R"pbdoc(
          Return current name for each atom in atom position matrices.
          )pbdoc")
      .def("current_atom_species_index",
           &monte::OccLocation::current_atom_species_index,
           R"pbdoc(
          Return current species index for atoms in atom position matrices.
          )pbdoc")
      .def("current_atom_position_index",
           &monte::OccLocation::current_atom_position_index,
           R"pbdoc(
          Return current atom position index for atoms in atom position matrices.
          )pbdoc")
      .def("current_atom_n_jumps", &monte::OccLocation::current_atom_n_jumps,
           R"pbdoc(
          Return number of jumps made by each atom.
          )pbdoc")
      .def("candidate_list", &monte::OccLocation::candidate_list,
           R"pbdoc(
          Access the OccCandidateList.
          )pbdoc")
      .def(
          "cand_size_by_candidate_index",
          [](monte::OccLocation const &self, Index cand_index) {
            return self.cand_size(cand_index);
          },
          R"pbdoc(
          Total number of mutating sites, of OccCandidate type, specified by index.
          )pbdoc",
          py::arg("candidate_index"))
      .def(
          "cand_size",
          [](monte::OccLocation const &self, monte::OccCandidate const &cand) {
            return self.cand_size(cand);
          },
          R"pbdoc(
          Total number of mutating sites, of OccCandidate type.
          )pbdoc",
          py::arg("candidate"))
      .def(
          "mol_id_by_candidate_index",
          [](monte::OccLocation const &self, Index cand_index, Index loc) {
            return self.mol_id(cand_index, loc);
          },
          R"pbdoc(
          Mol.id of a particular OccCandidate type, specified by index.
          )pbdoc",
          py::arg("candidate_index"), py::arg("location_index"))
      .def(
          "mol_id",
          [](monte::OccLocation const &self, monte::OccCandidate const &cand,
             Index loc) { return self.mol_id(cand, loc); },
          R"pbdoc(
          Mol.id of a particular OccCandidate type.
          )pbdoc",
          py::arg("candidate"), py::arg("location_index"))
      .def(
          "linear_site_index_to_mol_id",
          [](monte::OccLocation const &self, Index linear_site_index) {
            return self.l_to_mol_id(linear_site_index);
          },
          R"pbdoc(
          Convert from linear site index in configuration to variable site index (mol_id).
          )pbdoc",
          py::arg("linear_site_index"))
      .def("convert", &monte::OccLocation::convert,
           R"pbdoc(
           Get Conversions objects.
           )pbdoc");

  m.def(
      "choose_canonical_swap",
      [](monte::OccLocation const &occ_location,
         std::vector<monte::OccSwap> const &canonical_swaps,
         generator_type &random_number_generator) {
        return monte::choose_canonical_swap(occ_location, canonical_swaps,
                                            random_number_generator);
      },
      R"pbdoc(
        Choose a swap type from a list of allowed canonical swap types

        Parameters
        ----------
        occ_location: :class:`~libcasm.monte.OccLocation`
            Current occupant location list
        canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for canonical Monte Carlo events.
            This should not include both forward and reverse swaps.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        swap: :class:`~libcasm.monte.events.OccSwap`
            Chosen swap type.
        )pbdoc",
      py::arg("occ_location"), py::arg("canonical_swaps"),
      py::arg("random_number_generator"));

  m.def(
      "propose_canonical_event_from_swap",
      [](monte::OccEvent &e, monte::OccLocation const &occ_location,
         monte::OccSwap const &swap, generator_type &random_number_generator) {
        return monte::propose_canonical_event_from_swap(
            e, occ_location, swap, random_number_generator);
      },
      R"pbdoc(
        Propose canonical OccEvent of particular swap type

        Parameters
        ----------
        event: :class:`~libcasm.monte.events.OccEvent`
            Event to update based on the chosen OccSwap.
        occ_location: :class:`~libcasm.monte.events.OccLocation`
            Current occupant location list
        swap : :class:`~libcasm.monte.events.OccSwap`
            Chosen swap type.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        event: :class:`~libcasm.monte.events.OccEvent`
            Updated event based on the chosen swap type and particular event.

        )pbdoc",
      py::arg("event"), py::arg("occ_location"), py::arg("swap"),
      py::arg("random_number_generator"));

  m.def(
      "propose_canonical_event",
      [](monte::OccEvent &e, monte::OccLocation const &occ_location,
         std::vector<monte::OccSwap> const &canonical_swap,
         generator_type &random_number_generator) {
        return monte::propose_canonical_event(e, occ_location, canonical_swap,
                                              random_number_generator);
      },
      R"pbdoc(
        Propose canonical OccEvent from list of swap types

        Parameters
        ----------
        event: :class:`~libcasm.monte.events.OccEvent`
            Event to update based on the chosen OccSwap.
        occ_location: :class:`~libcasm.monte.events.OccLocation`
            Current occupant location list
        canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for canonical Monte Carlo events.
            This should not include both forward and reverse swaps.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        event: :class:`~libcasm.monte.events.OccEvent`
            Updated event based on the chosen swap type and particular event.

        )pbdoc",
      py::arg("event"), py::arg("occ_location"), py::arg("canonical_swaps"),
      py::arg("random_number_generator"));

  m.def(
      "choose_semigrand_canonical_swap",
      [](monte::OccLocation const &occ_location,
         std::vector<monte::OccSwap> const &semigrand_canonical_swaps,
         generator_type &random_number_generator) {
        return monte::choose_grand_canonical_swap(
            occ_location, semigrand_canonical_swaps, random_number_generator);
      },
      R"pbdoc(
        Choose a swap type from a list of allowed semi-grand canonical swap types

        Parameters
        ----------
        occ_location: :class:`~libcasm.monte.OccLocation`
            Current occupant location list
        semigrand_canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for semi-grand canonical Monte Carlo events.
            This should include both forward and reverse swaps.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        swap: :class:`~libcasm.monte.events.OccSwap`
            Chosen swap type.
        )pbdoc",
      py::arg("occ_location"), py::arg("semigrand_canonical_swaps"),
      py::arg("random_number_generator"));

  m.def(
      "propose_semigrand_canonical_event_from_swap",
      [](monte::OccEvent &e, monte::OccLocation const &occ_location,
         monte::OccSwap const &swap, generator_type &random_number_generator) {
        return monte::propose_grand_canonical_event_from_swap(
            e, occ_location, swap, random_number_generator);
      },
      R"pbdoc(
        Propose semi-grand canonical OccEvent of particular swap type

        Parameters
        ----------
        event: :class:`~libcasm.monte.events.OccEvent`
            Event to update based on the chosen OccSwap.
        occ_location: :class:`~libcasm.monte.events.OccLocation`
            Current occupant location list
        swap : :class:`~libcasm.monte.events.OccSwap`
            Chosen swap type.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        event: :class:`~libcasm.monte.events.OccEvent`
            Updated event based on the chosen swap type and particular event.

        )pbdoc",
      py::arg("event"), py::arg("occ_location"), py::arg("swap"),
      py::arg("random_number_generator"));

  m.def(
      "propose_semigrand_canonical_event",
      [](monte::OccEvent &e, monte::OccLocation const &occ_location,
         std::vector<monte::OccSwap> const &semigrand_canonical_swaps,
         generator_type &random_number_generator) {
        return monte::propose_grand_canonical_event(e, occ_location,
                                                    semigrand_canonical_swaps,
                                                    random_number_generator);
      },
      R"pbdoc(
        Propose semi-grand canonical OccEvent from list of swap types

        Parameters
        ----------
        event: :class:`~libcasm.monte.events.OccEvent`
            Event to update.
        occ_location: :class:`~libcasm.monte.events.OccLocation`
            Current occupant location list
        semigrand_canonical_swaps : List[:class:`~libcasm.monte.events.OccSwap`]
            A list of allowed OccSwap for semi-grand canonical Monte Carlo events.
            This should include both forward and reverse swaps.
        random_number_generator: :class:`~libcasm.monte.RandomNumberGenerator`
            Random number generator.

        Returns
        -------
        event: :class:`~libcasm.monte.events.OccEvent`
            Updated event based on the chosen swap type and particular event.

        )pbdoc",
      py::arg("event"), py::arg("occ_location"),
      py::arg("semigrand_canonical_swaps"), py::arg("random_number_generator"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
