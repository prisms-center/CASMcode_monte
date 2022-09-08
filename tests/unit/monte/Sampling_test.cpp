#include "casm/casm_io/Log.hh"
#include "casm/composition/CompositionCalculator.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/monte/Conversions.hh"
#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/events/OccCandidate.hh"
#include "casm/monte/events/OccEventProposal.hh"
#include "casm/monte/events/OccLocation.hh"
#include "casm/monte/state/State.hh"
#include "casm/monte/state/StateSampler.hh"
#include "gtest/gtest.h"
#include "testConfiguration.hh"
#include "teststructures.hh"

using namespace CASM;

template <typename GeneratorType>
std::shared_ptr<monte::Sampler> run_case(
    std::shared_ptr<xtal::BasicStructure const> shared_prim, Eigen::Matrix3l T,
    GeneratorType &random_number_generator,
    monte::StateSamplingFunction<test::Configuration> &function) {
  ScopedNullLogging logging;

  monte::Conversions convert(*shared_prim, T);

  // config with default occupation
  test::Configuration config(shared_prim->basis().size(), T);
  monte::State<test::Configuration> state{config};
  test::random_config(state.configuration, convert, random_number_generator);
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
    propose_canonical_event(e, occ_loc, canonical_swaps,
                            random_number_generator);
    occ_loc.apply(e, occupation);
    ++count;
  }
  return shared_sampler;
}

class SamplingTest : public testing::Test {
 protected:
  typedef std::mt19937_64 engine_type;
  typedef monte::RandomNumberGenerator<engine_type> generator_type;
  generator_type random_number_generator;
};

TEST_F(SamplingTest, CompNSamplingTest) {
  auto shared_prim =
      std::make_shared<xtal::BasicStructure const>(test::ZrO_prim());

  Eigen::Matrix3l T = Eigen::Matrix3l::Identity() * 9;

  std::vector<std::string> components = {"Zr", "Va", "O"};
  composition::CompositionCalculator composition_calculator(
      {"Zr", "Va", "O"}, xtal::allowed_molecule_names(*shared_prim));
  std::vector<Index> shape;
  shape.push_back(components.size());
  monte::StateSamplingFunction<test::Configuration> comp_n_sampling_f(
      "comp_n", "Composition per unit cell", components, shape,
      [&](monte::State<test::Configuration> const &state) {
        return composition_calculator.mean_num_each_component(
            state.configuration.occupation);
      });

  std::shared_ptr<monte::Sampler> shared_sampler =
      run_case(shared_prim, T, random_number_generator, comp_n_sampling_f);

  EXPECT_EQ(shared_sampler->n_samples(), 1000);
  EXPECT_EQ(shared_sampler->n_components(), 3);
}
