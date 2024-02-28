#include "casm/monte/events/io/OccCandidate_json_io.hh"

#include "casm/casm_io/json/InputParser_impl.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/UnitCellCoord.hh"
#include "casm/monte/Conversions.hh"
#include "casm/monte/events/OccCandidate.hh"

namespace CASM {

monte::OccCandidate jsonConstructor<monte::OccCandidate>::from_json(
    const jsonParser &json, const monte::Conversions &convert) {
  InputParser<monte::OccCandidate> parser{json, convert};
  std::stringstream ss;
  ss << "Error: Invalid monte::OccCandidate object";
  report_and_throw_if_invalid(parser, err_log(), std::runtime_error{ss.str()});
  return std::move(*parser.value);
}

void parse(InputParser<monte::OccCandidate> &parser,
           monte::Conversions const &convert) {
  Index asym;
  parser.require(asym, "asym");

  std::string species_name;
  parser.require(species_name, "spec");

  Index species_index = convert.species_index(species_name);
  if (species_index == convert.species_size()) {
    std::stringstream ss;
    ss << "species name '" << species_name << "' is not a valid option";
    parser.insert_error("spec", ss.str());
  }

  if (parser.valid()) {
    parser.value = std::make_unique<monte::OccCandidate>(asym, species_index);
  }
}

jsonParser &to_json(monte::OccCandidate const &cand, jsonParser &json,
                    monte::Conversions const &convert) {
  json.put_obj();
  json["asym"] = cand.asym;
  json["spec"] = convert.species_name(cand.species_index);
  return json;
}

monte::OccSwap jsonConstructor<monte::OccSwap>::from_json(
    const jsonParser &json, const monte::Conversions &convert) {
  InputParser<monte::OccSwap> parser{json, convert};
  std::stringstream ss;
  ss << "Error: Invalid monte::OccSwap object";
  report_and_throw_if_invalid(parser, err_log(), std::runtime_error{ss.str()});
  return std::move(*parser.value);
}

void parse(InputParser<monte::OccSwap> &parser,
           monte::Conversions const &convert) {
  fs::path option;

  option = "0";
  auto cand_a_subparser = parser.subparse<monte::OccCandidate>(option, convert);

  option = "1";
  auto cand_b_subparser = parser.subparse<monte::OccCandidate>(option, convert);

  if (parser.valid()) {
    parser.value = std::make_unique<monte::OccSwap>(*cand_a_subparser->value,
                                                    *cand_b_subparser->value);
  }
}

jsonParser &to_json(monte::OccSwap const &swap, jsonParser &json,
                    monte::Conversions const &convert) {
  jsonParser tmp;
  json.put_array();
  json.push_back(to_json(swap.cand_a, tmp, convert));
  json.push_back(to_json(swap.cand_b, tmp, convert));
  return json;
}

monte::MultiOccSwap jsonConstructor<monte::MultiOccSwap>::from_json(
    const jsonParser &json, const monte::Conversions &convert) {
  InputParser<monte::MultiOccSwap> parser{json, convert};
  std::stringstream ss;
  ss << "Error: Invalid monte::MultiOccSwap object";
  report_and_throw_if_invalid(parser, err_log(), std::runtime_error{ss.str()});
  return std::move(*parser.value);
}

void parse(InputParser<monte::MultiOccSwap> &parser,
           monte::Conversions const &convert) {
  std::map<monte::OccSwap, int> swaps;

  jsonParser const &json = parser.self;
  if (json.is_array()) {
    int i = 0;
    for (auto it = json.begin(); it != json.end(); ++it) {
      fs::path element = std::to_string(i);
      auto swap_subparser =
          parser.subparse<monte::OccSwap>(element / "swap", convert);

      int count = 0;
      parser.require(count, element / "count");

      if (swap_subparser->valid()) {
        swaps.emplace(*swap_subparser->value, count);
      }
      ++i;
    }
  } else {
    std::stringstream ss;
    ss << "Could not construct monte::MultiOccSwap: not an array";
    parser.error.insert(ss.str());
  }

  if (parser.valid()) {
    parser.value = std::make_unique<monte::MultiOccSwap>(swaps);
  }
}

jsonParser &to_json(monte::MultiOccSwap const &multiswap, jsonParser &json,
                    monte::Conversions const &convert) {
  json.put_array();
  for (auto const &pair : multiswap.swaps) {
    jsonParser tmp;
    to_json(pair.first, tmp["swap"], convert);
    tmp["count"] = pair.second;
    json.push_back(tmp);
  }
  return json;
}

monte::OccCandidateList jsonConstructor<monte::OccCandidateList>::from_json(
    const jsonParser &json, const monte::Conversions &convert) {
  InputParser<monte::OccCandidateList> parser{json, convert};
  std::stringstream ss;
  ss << "Error: Invalid monte::OccCandidateList object";
  report_and_throw_if_invalid(parser, err_log(), std::runtime_error{ss.str()});
  return std::move(*parser.value);
}

void parse(InputParser<monte::OccCandidateList> &parser,
           monte::Conversions const &convert) {
  std::vector<monte::OccCandidate> candidates;
  jsonParser const &json = parser.self;
  if (json.is_array()) {
    std::vector<monte::OccCandidate> candidates;
    int i = 0;
    for (auto const &x : json) {
      auto subparser =
          parser.subparse<monte::OccCandidate>(std::to_string(i), convert);
      if (subparser->valid()) {
        candidates.emplace_back(std::move(*subparser->value));
      }
      ++i;
    }
    if (parser.valid()) {
      parser.value =
          std::make_unique<monte::OccCandidateList>(candidates, convert);
    }
    return;
  } else if (json.is_obj() && json.contains("candidate") &&
             json["candidate"].is_array()) {
    auto subparser =
        parser.subparse<monte::OccCandidateList>("candidate", convert);
    if (subparser->valid()) {
      parser.value = std::move(subparser->value);
    }
    return;
  } else {
    std::stringstream ss;
    ss << "Could not construct monte::OccCandidateList: not an array or an "
          "object with a \"candidate\" array.";
    parser.error.insert(ss.str());
    return;
  }
}

/// \brief Write OccCandidateList to json, including all possible canonical
/// and
///     grand canonical swaps
jsonParser &to_json(monte::OccCandidateList const &list, jsonParser &json,
                    monte::Conversions const &convert) {
  jsonParser tmp;

  json.put_obj();

  json["candidate"].put_array();
  for (auto it = list.begin(); it != list.end(); ++it) {
    json["candidate"].push_back(to_json(*it, tmp, convert));
  }

  json["canonical_swaps"].put_array();
  auto canonical_swaps = make_canonical_swaps(convert, list);
  for (auto it = canonical_swaps.begin(); it != canonical_swaps.end(); ++it) {
    json["canonical_swaps"].push_back(to_json(*it, tmp, convert));
  }

  json["grand_canonical_swaps"].put_array();
  auto semigrand_canonical_swaps =
      make_semigrand_canonical_swaps(convert, list);
  for (auto it = semigrand_canonical_swaps.begin();
       it != semigrand_canonical_swaps.end(); ++it) {
    json["grand_canonical_swaps"].push_back(to_json(*it, tmp, convert));
  }

  return json;
}

}  // namespace CASM
