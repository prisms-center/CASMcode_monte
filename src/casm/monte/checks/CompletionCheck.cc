// #include "casm/monte/checks/CompletionCheck.hh"
//
// namespace CASM {
// namespace monte {
//
// template<typename StatisticsType>
// CompletionCheck<StatisticsType>::CompletionCheck(CompletionCheckParams
// params)
//     : m_params(params) {
//   m_results.params = m_params;
//   m_results.confidence = m_params.confidence;
//   m_results.is_complete = false;
// }
//
// /// \brief Check for equilibration and convergence, then set m_results
// template<typename StatisticsType>
// void CompletionCheck<StatisticsType>::_check(
//     std::map<std::string, std::shared_ptr<Sampler>> const &samplers,
//     std::vector<double> const &sample_weight, std::optional<CountType> count,
//     std::optional<TimeType> time, CountType n_samples) {
//   // if auto convergence mode:
//   if (m_params.requested_precision.size()) {
//     m_results.convergence_check_performed = true;
//     std::cout << "~~~ equilibration and convergence check ~~~" << std::endl;
//
//     // check for equilibration
//     bool check_all = false;
//     m_results.equilibration_check_results = equilibration_check(
//         m_params.equilibration_check_f, m_params.requested_precision,
//         samplers, sample_weight, check_all);
//
//     // if all requested to converge are equilibrated, then check convergence
//     if (m_results.equilibration_check_results.all_equilibrated) {
//       m_results.convergence_check_results =
//           convergence_check(m_params.calc_statistics_f,
//           m_params.requested_precision,
//                             m_params.confidence,
//                             m_results.equilibration_check_results
//                                 .N_samples_for_all_to_equilibrate,
//                             samplers, sample_weight);
//     }
//
//     // if all requested to converge are converged, then complete
//     if (m_results.convergence_check_results.all_converged) {
//       m_results.is_complete = true;
//       return;
//     }
//   }
// }
//
// }  // namespace monte
// }  // namespace CASM
