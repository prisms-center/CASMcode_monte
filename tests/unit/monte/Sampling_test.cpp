#include "casm/casm_io/Log.hh"
#include "casm/composition/CompositionCalculator.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/external/MersenneTwister/MersenneTwister.h"
#include "casm/monte/Conversions.hh"
#include "casm/monte/events/OccCandidate.hh"
#include "casm/monte/events/OccEventProposal.hh"
#include "casm/monte/events/OccLocation.hh"
#include "casm/monte/state/State.hh"
#include "casm/monte/state/StateSampler.hh"
#include "gtest/gtest.h"
#include "testConfiguration.hh"
#include "teststructures.hh"

using namespace CASM;

void random_config(test::Configuration &config, monte::Conversions &convert,
                   MTRand &mtrand);

std::shared_ptr<monte::Sampler> run_case(
    std::shared_ptr<xtal::BasicStructure const> shared_prim, Eigen::Matrix3l T,
    MTRand &mtrand,
    monte::StateSamplingFunction<test::Configuration> &function) {
  ScopedNullLogging logging;

  monte::Conversions convert(*shared_prim, T);

  // config with default occupation
  test::Configuration config(shared_prim->basis().size(), T);
  monte::State<test::Configuration> state{config};
  random_config(state.configuration, convert, mtrand);
  Eigen::VectorXi &occupation = state.configuration.occupation;

  // construct OccCandidateList
  monte::OccCandidateList cand_list(convert);
  auto canonical_swaps = make_canonical_swaps(convert, cand_list);

  // construct OccLocation
  monte::OccLocation occ_loc(convert, cand_list);
  occ_loc.initialize(occupation);

  // construct Sampler
  auto shared_sampler =
      std::make_shared<monte::Sampler>(function.component_names);

  Index count = 0;
  monte::OccEvent e;
  while (count < 1000000) {
    if (count % 1000 == 0) {
      shared_sampler->push_back(function(state));
    }
    propose_canonical_event(e, occ_loc, canonical_swaps, mtrand);
    occ_loc.apply(e, occupation);
    ++count;
  }
  return shared_sampler;
}

TEST(SamplingTest, CompNSamplingTest) {
  MTRand mtrand;
  auto shared_prim =
      std::make_shared<xtal::BasicStructure const>(test::ZrO_prim());

  Eigen::Matrix3l T = Eigen::Matrix3l::Identity() * 9;

  std::vector<std::string> components = {"Zr", "Va", "O"};
  composition::CompositionCalculator composition_calculator(
      {"Zr", "Va", "O"}, xtal::allowed_molecule_names(*shared_prim));
  monte::StateSamplingFunction<test::Configuration> comp_n_sampling_f(
      "comp_n", "Composition per unit cell", components,
      [&](monte::State<test::Configuration> const &state) {
        return composition_calculator.mean_num_each_component(
            state.configuration.occupation);
      });

  std::shared_ptr<monte::Sampler> shared_sampler =
      run_case(shared_prim, T, mtrand, comp_n_sampling_f);

  EXPECT_EQ(shared_sampler->n_samples(), 1000);
  EXPECT_EQ(shared_sampler->n_components(), 3);
}
