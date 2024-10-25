#ifndef CASM_monte_sampling_SelectedEventData_json_io
#define CASM_monte_sampling_SelectedEventData_json_io

namespace CASM {

template <typename T>
class InputParser;
class jsonParser;

namespace monte {
struct CorrelationsDataParams;
struct CorrelationsData;
class DiscreteVectorIntHistogram;
class DiscreteVectorFloatHistogram;
class Histogram1D;
class PartitionedHistogram1D;
struct SelectedEventFunctionParams;
struct SelectedEventData;

/// \brief Construct CorrelationsDataParams from JSON
void parse(InputParser<CorrelationsDataParams> &parser);

/// \brief Convert CorrelationsDataParams to JSON
jsonParser &to_json(CorrelationsDataParams const &correlations_data_params,
                    jsonParser &json);

/// \brief Convert CorrelationsData to JSON
jsonParser &to_json(CorrelationsData const &correlations_data,
                    jsonParser &json);

/// \brief Convert DiscreteVectorIntHistogram to JSON
jsonParser &to_json(DiscreteVectorIntHistogram const &histogram,
                    jsonParser &json);

/// \brief Convert DiscreteVectorFloatHistogram to JSON
jsonParser &to_json(DiscreteVectorFloatHistogram const &histogram,
                    jsonParser &json);

/// \brief Convert Histogram1D to JSON
jsonParser &to_json(Histogram1D const &histogram, jsonParser &json);

/// \brief Convert PartitionedHistogram1D to JSON
jsonParser &to_json(PartitionedHistogram1D const &histogram, jsonParser &json);

/// \brief Construct SelectedEventFunctionParams from JSON
void parse(InputParser<SelectedEventFunctionParams> &parser);

/// \brief Convert SelectedEventFunctionParams to JSON
jsonParser &to_json(
    SelectedEventFunctionParams const &selected_event_data_params,
    jsonParser &json);

/// \brief Convert SelectedEventData to JSON
jsonParser &to_json(SelectedEventData const &selected_event_data,
                    jsonParser &json);

}  // namespace monte
}  // namespace CASM

#endif
