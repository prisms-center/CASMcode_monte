#include "casm/casm_io/Log.hh"
#include "casm/composition/CompositionCalculator.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/monte/Conversions.hh"
#include "casm/monte/RandomNumberGenerator.hh"
#include "casm/monte/events/OccCandidate.hh"
#include "casm/monte/events/OccEventProposal.hh"
#include "casm/monte/events/OccLocation.hh"
#include "casm/monte/run_management/State.hh"
#include "casm/monte/sampling/StateSamplingFunction.hh"
#include "gtest/gtest.h"
#include "testConfiguration.hh"
#include "teststructures.hh"

using namespace CASM;

struct CalculationTest {
  /// Prim
  std::shared_ptr<xtal::BasicStructure const> shared_prim;

  /// Random number generator
  monte::RandomNumberGenerator<monte::default_engine_type>
      random_number_generator;

  /// Current state
  monte::State<test::Configuration> const *state;

  /// Current supercell
  Eigen::Matrix3l transformation_matrix_to_super;

  CalculationTest(
      std::shared_ptr<xtal::BasicStructure const> _shared_prim,
      std::shared_ptr<monte::default_engine_type> _random_number_engine =
          std::shared_ptr<monte::default_engine_type>())
      : shared_prim(_shared_prim),
        random_number_generator(_random_number_engine),
        state(nullptr) {}

  std::shared_ptr<monte::Sampler> run_case(
      Eigen::Matrix3l T, monte::StateSamplingFunction &function) {
    ScopedNullLogging logging;

    monte::Conversions convert(*shared_prim, T);

    // config with default occupation
    test::Configuration config(shared_prim->basis().size(), T);
    monte::State<test::Configuration> state{config};
    test::random_config(state.configuration, convert, random_number_generator);
    Eigen::VectorXi &occupation = state.configuration.occupation;

    // set pointers
    this->transformation_matrix_to_super = T;
    this->state = &state;

    // construct OccCandidateList
    monte::OccCandidateList cand_list(convert);
    auto canonical_swaps = make_canonical_swaps(convert, cand_list);

    // construct OccLocation
    monte::OccLocation occ_loc(convert, cand_list);
    occ_loc.initialize(occupation);

    // construct Sampler
    auto shared_sampler = std::make_shared<monte::Sampler>(
        function.shape, function.component_names);

    Index count = 0;
    monte::OccEvent e;
    while (count < 1000000) {
      if (count % 1000 == 0) {
        shared_sampler->push_back(function());
      }
      propose_canonical_event(e, occ_loc, canonical_swaps,
                              random_number_generator);
      occ_loc.apply(e, occupation);
      ++count;
    }
    return shared_sampler;
  }
};

class SamplingTest : public testing::Test {};

TEST_F(SamplingTest, CompNSamplingTest) {
  // construct calculation
  auto calculation = std::make_shared<CalculationTest>(
      std::make_shared<xtal::BasicStructure const>(test::ZrO_prim()));

  // construct sampling function with copy of calculation shared pointer
  std::vector<std::string> components = {"Zr", "Va", "O"};
  std::vector<Index> shape;
  shape.push_back(components.size());
  monte::StateSamplingFunction comp_n_sampling_f(
      "comp_n", "Composition per unit cell", components, shape,
      [calculation, components]() {
        composition::CompositionCalculator composition_calculator(
            components,
            xtal::allowed_molecule_names(*calculation->shared_prim));
        return composition_calculator.mean_num_each_component(
            calculation->state->configuration.occupation);
      });

  // run a "calculation" - this will set calculation->state pointer so the
  // sampling function samples the current calculation state
  Eigen::Matrix3l T = Eigen::Matrix3l::Identity() * 9;
  std::shared_ptr<monte::Sampler> shared_sampler =
      calculation->run_case(T, comp_n_sampling_f);

  EXPECT_EQ(shared_sampler->n_samples(), 1000);
  EXPECT_EQ(shared_sampler->n_components(), 3);
}
