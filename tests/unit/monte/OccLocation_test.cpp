#include "casm/monte/events/OccLocation.hh"

#include "casm/casm_io/Log.hh"
#include "casm/external/MersenneTwister/MersenneTwister.h"
#include "casm/monte/Conversions.hh"
#include "casm/monte/events/OccCandidate.hh"
#include "casm/monte/events/OccEventProposal.hh"
#include "casm/monte/events/io/OccCandidate_stream_io.hh"
#include "gtest/gtest.h"
#include "testConfiguration.hh"
#include "teststructures.hh"

using namespace CASM;

/// Set config to random occupation
void random_config(test::Configuration &config, monte::Conversions &convert,
                   MTRand &mtrand) {
  config.occupation.setZero();
  for (Index l = 0; l < config.occupation.size(); ++l) {
    int Nocc = convert.occ_size(convert.l_to_asym(l));
    config.occupation[l] = mtrand.randInt(Nocc - 1);
  }
}

/// Set a single non-default occupant
void dilute_config(test::Configuration &config, monte::Conversions &convert,
                   MTRand &mtrand) {
  config.occupation.setZero();
  for (Index i = 0; i < convert.species_size(); ++i) {
    for (Index l = 0; l < config.occupation.size(); ++l) {
      Index asym = convert.l_to_asym(l);
      if (config.occupation[l] == 0 && convert.species_allowed(asym, i)) {
        config.occupation[l] = convert.occ_index(asym, i);
        break;
      }
    }
  }
}

void check_occ_init(test::Configuration &config, monte::OccLocation &occ_loc,
                    monte::Conversions &convert,
                    monte::OccCandidateList &cand_list) {
  // check OccLocation initialization
  for (Index mol_id = 0; mol_id < occ_loc.mol_size(); ++mol_id) {
    ASSERT_EQ(mol_id, occ_loc.mol(mol_id).id);
  }

  for (Index l = 0; l < config.occupation.size(); ++l) {
    Index mol_id = occ_loc.l_to_mol_id(l);
    if (mol_id == occ_loc.mol_size()) {  // non-variable site
      continue;
    }
    auto &mol = occ_loc.mol(mol_id);

    ASSERT_EQ(mol.l, l);
    ASSERT_EQ(mol.id, mol_id);

    Index asym = convert.l_to_asym(l);
    ASSERT_EQ(asym, mol.asym);

    ASSERT_EQ(config.occupation[l], convert.occ_index(asym, mol.species_index));
    ASSERT_EQ(convert.species_index(asym, config.occupation[l]),
              mol.species_index);

    Index cand_index = cand_list.index(mol.asym, mol.species_index);
    ASSERT_EQ(mol.id, occ_loc.mol_id(cand_index, mol.loc));
  }
}

void check_occ(test::Configuration &config, monte::OccEvent &e,
               monte::OccLocation &occ_loc, monte::Conversions &convert,
               monte::OccCandidateList &cand_list) {
  // check that occ_loc / config / mol are consistent for initial state of
  // config
  for (const auto &occ : e.occ_transform) {
    Index l = occ.l;
    Index mol_id = occ_loc.l_to_mol_id(l);
    auto &mol = occ_loc.mol(mol_id);

    ASSERT_EQ(mol.l, l);
    ASSERT_EQ(mol.id, mol_id);
    ASSERT_EQ(occ.mol_id, mol_id);

    Index asym = convert.l_to_asym(l);
    ASSERT_EQ(asym, mol.asym);

    ASSERT_EQ(config.occupation[l], convert.occ_index(asym, mol.species_index));
    ASSERT_EQ(convert.species_index(asym, config.occupation[l]),
              mol.species_index);

    Index cand_index = cand_list.index(mol.asym, mol.species_index);
    ASSERT_EQ(mol.id, occ_loc.mol_id(cand_index, mol.loc));
  }
}

template <typename ConfigInit>
void run_case(std::shared_ptr<xtal::BasicStructure const> shared_prim,
              Eigen::Matrix3l T, ConfigInit f, MTRand &mtrand) {
  ScopedNullLogging logging;

  monte::Conversions convert(*shared_prim, T);

  test::Configuration config(shared_prim->basis().size(), T);
  f(config, convert, mtrand);

  // construct OccCandidateList
  monte::OccCandidateList cand_list(convert);
  auto canonical_swaps = make_canonical_swaps(convert, cand_list);

  // construct OccLocation
  monte::OccLocation occ_loc(convert, cand_list);
  occ_loc.initialize(config.occupation);

  check_occ_init(config, occ_loc, convert, cand_list);

  Index count = 0;
  monte::OccEvent e;
  while (count < 1000000) {
    // if(count % 100000 == 0) { std::cout << "count: " << count << std::endl; }
    propose_canonical_event(e, occ_loc, canonical_swaps, mtrand);
    check_occ(config, e, occ_loc, convert, cand_list);
    occ_loc.apply(e, config.occupation);
    check_occ(config, e, occ_loc, convert, cand_list);
    ++count;
  }
}

TEST(OccLocationTest, ZrO_RandomConfig) {
  MTRand mtrand;
  auto shared_prim =
      std::make_shared<xtal::BasicStructure const>(test::ZrO_prim());
  Eigen::Matrix3l T;
  T << 9, 0, 0, 0, 9, 0, 0, 0, 9;
  run_case(shared_prim, T, random_config, mtrand);
}

TEST(OccLocationTest, ZrO_DiluteConfig) {
  MTRand mtrand;
  auto shared_prim =
      std::make_shared<xtal::BasicStructure const>(test::ZrO_prim());
  Eigen::Matrix3l T;
  T << 9, 0, 0, 0, 9, 0, 0, 0, 9;
  run_case(shared_prim, T, dilute_config, mtrand);
}
TEST(OccLocationTest, FCCTernary_RandomConfig) {
  MTRand mtrand;
  auto shared_prim =
      std::make_shared<xtal::BasicStructure const>(test::FCC_ternary_prim());
  Eigen::Matrix3l T;
  T << 9, 0, 0, 0, 9, 0, 0, 0, 9;
  run_case(shared_prim, T, random_config, mtrand);
}

TEST(OccLocationTest, FCCTernary_DiluteConfig) {
  MTRand mtrand;
  auto shared_prim =
      std::make_shared<xtal::BasicStructure const>(test::FCC_ternary_prim());
  Eigen::Matrix3l T;
  T << 9, 0, 0, 0, 9, 0, 0, 0, 9;
  run_case(shared_prim, T, dilute_config, mtrand);
}
