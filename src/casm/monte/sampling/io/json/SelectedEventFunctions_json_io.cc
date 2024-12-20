#include "casm/monte/sampling/io/json/SelectedEventFunctions_json_io.hh"

#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/InputParser_impl.hh"
#include "casm/casm_io/json/optional.hh"
#include "casm/monte/sampling/SelectedEventFunctions.hh"

namespace CASM {
namespace monte {

/// \brief Construct CorrelationsDataParams from JSON
///
/// Expected:
///
///   jumps_per_position_sample: int (optional, default=1)
///     Every `jumps_per_position_sample` jumps of an individual atom, save
///     its position in Cartesian coordinates.
///
///   max_n_position_samples: int (optional, default=100)
///     The maximum number of positions to store for each atom.
///
///   output_incomplete_samples: bool (optional, default=false)
///     If true, output incomplete samples.
///
void parse(InputParser<CorrelationsDataParams> &parser) {
  CorrelationsDataParams params;
  parser.optional_else(params.jumps_per_position_sample,
                       "jumps_per_position_sample", Index(1));
  parser.optional_else(params.max_n_position_samples, "max_n_position_samples",
                       Index(100));
  parser.optional_else(params.output_incomplete_samples,
                       "output_incomplete_samples", false);
  parser.optional_else(params.stop_run_when_complete, "stop_run_when_complete",
                       false);
  if (parser.valid()) {
    parser.value = std::make_unique<CorrelationsDataParams>(params);
  }
}

/// \brief Convert CorrelationsDataParams to JSON
jsonParser &to_json(CorrelationsDataParams const &correlations_data_params,
                    jsonParser &json) {
  auto &params = correlations_data_params;
  json.put_obj();
  json["jumps_per_position_sample"] = params.jumps_per_position_sample;
  json["max_n_position_samples"] = params.max_n_position_samples;
  json["output_incomplete_samples"] = params.output_incomplete_samples;
  json["stop_run_when_complete"] = params.stop_run_when_complete;
  return json;
}

/// \brief Convert CorrelationsData to JSON
jsonParser &to_json(CorrelationsData const &correlations_data,
                    jsonParser &json) {
  auto &data = correlations_data;
  json.put_obj();

  // Insert all members of CorrelationsData to json:
  json["jumps_per_position_sample"] = data.jumps_per_position_sample;
  json["max_n_position_samples"] = data.max_n_position_samples;
  json["output_incomplete_samples"] = data.output_incomplete_samples;
  json["n_position_samples"] = data.n_position_samples;
  json["n_complete_samples"] = data.n_complete_samples;

  if (data.output_incomplete_samples) {
    json["step"] = data.step;
    json["pass"] = data.pass;
    json["time"] = data.time;
    json["atom_positions_cart"] = data.atom_positions_cart;
  } else {
    json["step"] = data.step.topRows(data.n_complete_samples);
    json["pass"] = data.pass.topRows(data.n_complete_samples);
    json["time"] = data.time.topRows(data.n_complete_samples);
    json["atom_positions_cart"] = jsonParser::array();
    for (Index i = 0; i < data.n_complete_samples; ++i) {
      json["atom_positions_cart"].push_back(data.atom_positions_cart[i]);
    }
  }
  return json;
}

namespace {

template <typename DiscreteHistogramType>
jsonParser &_discrete_histogram_to_json(DiscreteHistogramType const &histogram,
                                        jsonParser &json) {
  DiscreteHistogramType const &data = histogram;
  json.put_obj();
  json["shape"] = data.shape();
  json["max_size"] = data.max_size();
  json["max_size_exceeded"] = data.max_size_exceeded();
  json["size"] = data.size();
  json["sum"] = data.sum();
  if (!data.shape().empty()) {
    json["component_names"] = data.component_names();
  }

  // labels
  if (data.value_labels().has_value()) {
    json["labels"] = jsonParser::array();
    auto const &value_to_label = data.value_labels().value();
    for (auto v : data.values()) {
      if (value_to_label.count(v)) {
        json["labels"].push_back(value_to_label.at(v));
      } else {
        if (data.shape().size() == 0) {
          json["labels"].push_back(v(0));
        } else {
          json["labels"].push_back(v, jsonParser::as_array());
        }
      }
    }
  }

  // values
  if (data.shape().size() == 0) {
    // scalar
    json["values"] = jsonParser::array();
    for (auto const &v : data.values()) {
      json["values"].push_back(v(0));
    }
  } else {
    to_json(data.values(), json["values"], jsonParser::as_array());
  }

  json["count"] = data.count();
  json["fraction"] = data.fraction();
  json["out_of_range_count"] = data.out_of_range_count();
  return json;
}

}  // namespace

/// \brief Convert DiscreteVectorIntHistogram to JSON
jsonParser &to_json(DiscreteVectorIntHistogram const &histogram,
                    jsonParser &json) {
  return _discrete_histogram_to_json(histogram, json);
}

/// \brief Convert DiscreteVectorFloatHistogram to JSON
jsonParser &to_json(DiscreteVectorFloatHistogram const &histogram,
                    jsonParser &json) {
  return _discrete_histogram_to_json(histogram, json);
}

/// \brief Convert Histogram1D to JSON
jsonParser &to_json(Histogram1D const &histogram, jsonParser &json) {
  auto &data = histogram;
  json.put_obj();
  json["max_size"] = data.max_size();
  json["max_size_exceeded"] = data.max_size_exceeded();
  json["is_log"] = data.is_log();
  json["begin"] = data.begin();
  json["bin_width"] = data.bin_width();
  json["count"] = data.count();
  json["bin_coords"] = data.bin_coords();
  json["sum"] = data.sum();
  json["density"] = data.density();
  json["out_of_range_count"] = data.out_of_range_count();
  return json;
}

/// \brief Convert PartitionedHistogram1D to JSON
jsonParser &to_json(PartitionedHistogram1D const &histogram, jsonParser &json) {
  auto &data = histogram;
  json.put_obj();
  if (data.partition_names().size() == 1) {
    // If only 1 partition, just output like a single histogram
    to_json(data.histograms()[0], json);
  } else {
    // If >1 partition, output combined histogram and partitioned histograms
    to_json(data.combined_histogram(), json["combined"]);
    json["partition_names"] = data.partition_names();
    json["partitions"] = jsonParser::array();

    Index i_partition = 0;
    for (auto const &hist : data.histograms()) {
      double partition_frac = hist.sum() / data.combined_histogram().sum();
      jsonParser tjson;
      to_json(hist, tjson);
      tjson["partition_name"] = data.partition_names()[i_partition];
      tjson["partition_fraction"] =
          hist.sum() / data.combined_histogram().sum();
      tjson["partial_density"] = jsonParser::array();
      for (double x : hist.density()) {
        tjson["partial_density"].push_back(x * partition_frac);
      }
      json["partitions"].push_back(tjson);
      ++i_partition;
    }
  }

  return json;
}

/// \brief Construct SelectedEventFunctionParams from JSON
///
/// Expected:
///
///   "correlation_data_params": dict (default=None),
///       Options controlling the collection of hop correlation data (not
///       basis functions correlations). Options are:
///
///       "jumps_per_position_sample": int (optional, default=1)
///           Every `jumps_per_position_sample` jumps of an individual
///           atom, its position in Cartesian coordinates (as if hopping
///           without periodic boundaries) is saved.
///       "max_n_position_samples: int (optional, default=100)
///           The maximum number of positions to store for each atom.
///       "output_incomplete_samples: bool (optional, default=false)
///           If true, output all position data collected for every atom.
///           If false, only output positions if all atoms jumped the
///           required number of times.
///       "stop_run_when_complete": bool (optional, default=false)
///           If true, add an additional completion check to stop the run
///           when the maximum number of position samples is reached. If
///           false, continue running until the standard completion check
///           is met.
///
///   "quantities": list[str] (default=[])
///       Names of quantities to collect for each selected event and
///       store in histograms.
///   "tol": dict[str,float] (default={})
///       Tolerance values (float) for comparing discrete floating point
///        values, by quantity name. May be provided to override the
///       default value for a particular quantity.
///   "bin_width": dict[str,float] (default={})
///       Histogram bin width values (float), by quantity name. May be
///       provided to override the default value for a particular
///       quantity.
///   "initial_begin": dict[str,float] (default={})
///       Initial bin coordinate value (float), by quantity name. The bin
///       number for a particular value is calculated as
///       `(value - begin) / bin_width`, so the range for bin `i` is
///       [begin, begin + i*bin_width). Coordinates are adjusted to fit
///       the data encountered by starting `begin` at `initial_begin` and
///       adjusting it as necessary by multiples of `bin_width`. May be
///       provided to override the default value for a particular
///       quantity.
///   "spacing": dict[str,str] (default={})
///       Bin coordinate spacing (one of "log" or "linear"), by quantity
///       name. May be provided to override the default value for a
///       particular quantity.
///   "max_size": dict[str,int] (default={})
///       Maximum number of bins / discrete values to keep. If adding an
///       additional data point would cause the number of bins / discrete
///       values to exceed `max_size`, the count / weight is instead added
///       to the `out_of_range_count`. May be provided to override the
///       default value for a particular quantity.
///
void parse(InputParser<SelectedEventFunctionParams> &parser) {
  parser.value = std::make_unique<SelectedEventFunctionParams>();

  SelectedEventFunctionParams &params = *parser.value;

  auto subparser =
      parser.subparse_if<CorrelationsDataParams>("correlations_data_params");
  if (subparser->valid() && subparser->value != nullptr) {
    params.correlations_data_params = std::move(*subparser->value);
  }

  parser.optional(params.function_names, "quantities");

  for (const std::string &name : params.function_names) {
    // Check for user-specified "tol"
    std::unique_ptr<double> tol =
        parser.optional<double>(fs::path("tol") / name);
    if (tol) {
      params.tol[name] = *tol;
    }

    // Check for user-specified "bin_width"
    std::unique_ptr<double> bin_width =
        parser.optional<double>(fs::path("bin_width") / name);
    if (bin_width) {
      params.bin_width[name] = *bin_width;
    }

    // Check for user-specified "initial_begin"
    std::unique_ptr<double> initial_begin =
        parser.optional<double>(fs::path("initial_begin") / name);
    if (initial_begin) {
      params.initial_begin[name] = *initial_begin;
    }

    // Check for user-specified "spacing"
    std::unique_ptr<std::string> spacing =
        parser.optional<std::string>(fs::path("spacing") / name);
    if (spacing) {
      if (*spacing == "log") {
        params.is_log[name] = true;
      } else if (*spacing == "linear") {
        params.is_log[name] = false;
      } else {
        std::stringstream msg;
        msg << "Error: \"" << *spacing
            << "\" is not a valid spacing option for \"" << name << "\".";
        parser.insert_error(fs::path("spacing") / name, msg.str());
      }
    }

    // Check for user-specified "max_size"
    std::unique_ptr<Index> max_size =
        parser.optional<Index>(fs::path("max_size") / name);
    if (max_size) {
      params.max_size[name] = *max_size;
    }
  }

  if (!parser.valid()) {
    parser.value.reset();
  }
}

/// \brief Convert SelectedEventFunctionParams to JSON
jsonParser &to_json(
    SelectedEventFunctionParams const &selected_event_data_params,
    jsonParser &json) {
  auto &params = selected_event_data_params;
  json.put_obj();
  if (params.correlations_data_params.has_value()) {
    json["correlations_data_params"] = *params.correlations_data_params;
  }
  json["quantities"] = params.function_names;
  if (!params.tol.empty()) {
    json["tol"] = params.tol;
  }
  if (!params.bin_width.empty()) {
    json["bin_width"] = params.bin_width;
  }
  if (!params.initial_begin.empty()) {
    json["initial_begin"] = params.initial_begin;
  }
  if (!params.is_log.empty()) {
    json["spacing"] = jsonParser::object();
    for (auto const &pair : params.is_log) {
      json["spacing"][pair.first] = pair.second ? "log" : "linear";
    }
  }
  if (!params.max_size.empty()) {
    json["max_size"] = params.max_size;
  }

  return json;
}

/// \brief Convert SelectedEventData to JSON
jsonParser &to_json(SelectedEventData const &selected_event_data,
                    jsonParser &json) {
  auto &data = selected_event_data;
  json.put_obj();

  // Insert all members of SelectedEventData to json:
  if (data.correlations_data.has_value()) {
    json["correlations_data"] = *data.correlations_data;
  }

  int n_histograms = data.discrete_vector_int_histograms.size() +
                     data.discrete_vector_float_histograms.size() +
                     data.continuous_1d_histograms.size();

  if (n_histograms > 0) {
    json["histograms"] = jsonParser::object();

    for (auto const &pair : data.discrete_vector_int_histograms) {
      to_json(pair.second, json["histograms"][pair.first]);
    }
    for (auto const &pair : data.discrete_vector_float_histograms) {
      to_json(pair.second, json["histograms"][pair.first]);
    }
    for (auto const &pair : data.continuous_1d_histograms) {
      to_json(pair.second, json["histograms"][pair.first]);
    }
  }

  return json;
}

}  // namespace monte
}  // namespace CASM
