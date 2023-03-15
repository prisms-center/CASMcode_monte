// #include "casm/monte/checks/io/json/ConvergenceCheck_json_io.hh"
//
// #include "casm/casm_io/json/jsonParser.hh"
// #include "casm/monte/checks/ConvergenceCheck.hh"
//
// namespace CASM {
// namespace monte {
//
// /// \brief IndividualConvergenceCheckResult to JSON
// template<typename StatisticsType>
// jsonParser &to_json(IndividualConvergenceCheckResult<StatisticsType> const
// &value,
//                     jsonParser &json) {
//   json.put_obj();
//   json["is_converged"] = value.is_converged;
//   json["requested_precision"] = value.requested_precision;
//   json["stats"] = value.stats
//   return json;
// }
//
// /// \brief ConvergenceCheckResults to JSON
// template<typename StatisticsType>
// jsonParser &to_json(ConvergenceCheckResults<StatisticsType> const &value,
// jsonParser &json) {
//   json.put_obj();
//   json["all_converged"] = value.all_converged;
//   json["N_samples_for_statistics"] = value.N_samples_for_statistics;
//   json["individual_results"];
//   for (auto const &pair : value.individual_results) {
//     std::string name =
//         pair.first.sampler_name + "(" + pair.first.component_name + ")";
//     json["individual_results"][name] = pair.second;
//   }
//   return json;
// }
//
// }  // namespace monte
// }  // namespace CASM
