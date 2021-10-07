#include "casm/monte/events/io/OccCandidate_json_io.hh"

#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/UnitCellCoord.hh"
#include "casm/monte/Conversions.hh"
#include "casm/monte/events/OccCandidate.hh"

namespace CASM {

monte::OccCandidate jsonConstructor<monte::OccCandidate>::from_json(
    const jsonParser &json, const monte::Conversions &convert) {
  return monte::OccCandidate(
      json["asym"].get<Index>(),
      convert.species_index(json["spec"].get<std::string>()));
}

jsonParser &to_json(monte::OccCandidate const &cand,
                    monte::Conversions const &convert, jsonParser &json) {
  json.put_obj();
  json["asym"] = cand.asym;
  json["spec"] = convert.species_name(cand.species_index);
  return json;
}

monte::OccSwap jsonConstructor<monte::OccSwap>::from_json(
    const jsonParser &json, const monte::Conversions &convert) {
  return monte::OccSwap(
      jsonConstructor<monte::OccCandidate>::from_json(json[0], convert),
      jsonConstructor<monte::OccCandidate>::from_json(json[1], convert));
}

jsonParser &to_json(monte::OccSwap const &swap,
                    monte::Conversions const &convert, jsonParser &json) {
  jsonParser tmp;
  json.put_array();
  json.push_back(to_json(swap.cand_a, convert, tmp));
  json.push_back(to_json(swap.cand_b, convert, tmp));
  return json;
}

jsonParser &to_json(monte::OccCandidateList const &list,
                    monte::Conversions const &convert, jsonParser &json) {
  jsonParser tmp;

  json.put_obj();

  json["candidate"].put_array();
  for (auto it = list.begin(); it != list.end(); ++it) {
    json["candidate"].push_back(to_json(*it, convert, tmp));
  }

  json["canonical_swap"].put_array();
  for (auto it = list.canonical_swap().begin();
       it != list.canonical_swap().end(); ++it) {
    json["candidate_swap"].push_back(to_json(*it, convert, tmp));
  }

  json["grand_canonical_swap"].put_array();
  for (auto it = list.grand_canonical_swap().begin();
       it != list.grand_canonical_swap().end(); ++it) {
    json["grand_candidate_swap"].push_back(to_json(*it, convert, tmp));
  }

  return json;
}

}  // namespace CASM
