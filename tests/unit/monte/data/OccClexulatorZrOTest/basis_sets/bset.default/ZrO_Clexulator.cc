#include <cstddef>

#include "casm/clexulator/BaseClexulator.hh"
#include "casm/clexulator/BasicClexParamPack.hh"
#include "casm/global/eigen.hh"

/****** PROJECT SPECIFICATIONS ******

         ****** prim.json ******

{
  "basis" : [
    {
      "coordinate" : [ -0.000000008333, -0.000000016667, 0.000000000000 ],
      "occupants" : [ "Zr" ]
    },
    {
      "coordinate" : [ 0.666666658333, 0.333333316667, 0.500000000000 ],
      "occupants" : [ "Zr" ]
    },
    {
      "coordinate" : [ 0.333333325000, 0.666666650000, 0.250000000000 ],
      "occupants" : [ "Va", "O" ]
    },
    {
      "coordinate" : [ 0.333333325000, 0.666666650000, 0.750000000000 ],
      "occupants" : [ "Va", "O" ]
    }
  ],
  "coordinate_mode" : "Fractional",
  "lattice_vectors" : [
    [ 3.233986856383, 0.000000000000, 0.000000000000 ],
    [ -1.616993428191, 2.800714773133, 0.000000000000 ],
    [ 0.000000000000, 0.000000000000, 5.168678340000 ]
  ],
  "title" : "ZrO"
}

        ****** bspecs.json ******

{
  "basis_function_specs" : {
    "dof_specs" : {
      "occ" : {
        "site_basis_functions" : "OCCUPATION"
      }
    },
    "dofs" : [ "occ" ],
    "global_max_poly_order" : -1,
    "param_pack_type" : "DEFAULT"
  },
  "cluster_specs" : {
    "method" : "periodic_max_length",
    "params" : {
      "generating_group" : [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
15, 16, 17, 18, 19, 20, 21, 22, 23 ], "orbit_branch_specs" : { "0" : {
          "max_length" : 0.000000000000
        },
        "1" : {
          "max_length" : 0.000000000000
        },
        "2" : {
          "max_length" : 9.000000000000
        },
        "3" : {
          "max_length" : 7.000000000000
        },
        "4" : {
          "max_length" : 6.000000000000
        }
      }
    }
  }
}

**/

/// \brief Returns a clexulator::BaseClexulator* owning a ZrO_Clexulator
extern "C" CASM::clexulator::BaseClexulator *make_ZrO_Clexulator();

namespace CASM {
namespace clexulator {

/****** GENERATED CLEXPARAMPACK DEFINITION ******/

typedef BasicClexParamPack ParamPack;

/****** GENERATED CLEXULATOR DEFINITION ******/

class ZrO_Clexulator : public clexulator::BaseClexulator {
 public:
  ZrO_Clexulator();

  ~ZrO_Clexulator();

  ClexParamPack const &param_pack() const override { return m_params; }

  ClexParamPack &param_pack() override { return m_params; }

  template <typename Scalar>
  Scalar eval_bfunc_0_0() const;

  template <typename Scalar>
  Scalar eval_bfunc_1_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_1_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_1_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_1_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_1_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_2_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_2_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_2_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_2_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_2_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_3_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_3_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_3_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_3_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_3_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_4_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_4_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_4_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_4_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_4_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_5_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_5_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_5_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_5_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_5_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_6_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_6_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_6_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_6_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_6_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_7_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_7_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_7_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_7_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_7_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_8_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_8_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_8_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_8_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_8_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_9_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_9_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_9_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_9_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_9_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_10_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_10_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_10_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_10_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_10_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_11_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_11_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_11_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_11_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_11_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_12_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_12_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_12_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_12_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_12_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_13_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_13_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_13_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_13_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_13_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_14_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_14_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_14_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_14_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_14_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_15_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_15_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_15_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_15_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_15_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_16_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_16_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_16_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_16_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_16_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_17_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_17_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_17_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_17_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_17_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_18_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_18_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_18_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_18_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_18_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_19_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_19_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_19_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_19_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_19_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_20_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_20_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_20_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_20_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_20_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_21_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_21_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_21_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_21_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_21_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_22_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_22_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_22_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_22_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_22_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_23_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_23_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_23_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_23_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_23_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_24_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_24_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_24_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_24_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_24_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_25_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_25_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_25_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_25_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_25_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_26_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_26_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_26_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_26_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_26_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_27_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_27_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_27_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_27_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_27_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_28_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_28_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_28_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_28_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_28_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_29_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_29_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_29_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_29_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_29_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_30_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_30_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_30_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_30_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_30_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_31_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_31_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_31_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_31_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_31_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_32_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_32_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_32_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_32_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_32_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_33_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_33_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_33_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_33_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_33_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_34_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_34_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_34_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_34_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_34_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_35_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_35_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_35_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_35_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_35_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_36_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_36_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_36_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_36_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_36_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_37_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_37_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_37_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_37_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_37_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_38_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_38_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_38_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_38_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_38_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_39_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_39_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_39_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_39_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_39_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_40_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_40_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_40_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_40_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_40_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_41_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_41_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_41_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_41_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_41_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_42_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_42_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_42_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_42_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_42_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_43_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_43_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_43_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_43_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_43_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_44_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_44_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_44_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_44_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_44_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_45_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_45_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_45_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_45_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_45_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_46_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_46_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_46_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_46_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_46_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_47_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_47_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_47_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_47_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_47_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_48_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_48_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_48_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_48_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_48_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_49_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_49_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_49_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_49_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_49_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_50_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_50_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_50_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_50_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_50_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_51_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_51_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_51_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_51_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_51_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_52_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_52_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_52_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_52_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_52_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_53_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_53_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_53_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_53_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_53_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_54_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_54_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_54_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_54_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_54_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_55_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_55_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_55_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_55_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_55_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_56_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_56_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_56_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_56_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_56_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_57_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_57_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_57_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_57_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_57_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_58_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_58_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_58_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_58_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_58_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_59_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_59_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_59_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_59_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_59_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_60_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_60_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_60_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_60_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_60_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_61_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_61_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_61_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_61_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_61_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_62_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_62_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_62_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_62_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_62_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_63_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_63_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_63_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_63_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_63_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_64_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_64_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_64_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_64_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_64_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_65_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_65_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_65_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_65_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_65_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_66_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_66_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_66_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_66_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_66_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_67_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_67_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_67_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_67_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_67_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_68_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_68_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_68_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_68_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_68_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_69_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_69_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_69_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_69_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_69_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_70_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_70_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_70_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_70_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_70_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_71_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_71_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_71_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_71_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_71_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_72_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_72_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_72_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_72_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_72_0_at_1(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar eval_bfunc_73_0() const;

  template <typename Scalar>
  Scalar site_eval_bfunc_73_0_at_0() const;
  template <typename Scalar>
  Scalar site_eval_bfunc_73_0_at_1() const;

  template <typename Scalar>
  Scalar site_deval_bfunc_73_0_at_0(int occ_i, int occ_f) const;

  template <typename Scalar>
  Scalar site_deval_bfunc_73_0_at_1(int occ_i, int occ_f) const;

 private:
  // ParamPack object, which stores temporary data for calculations
  mutable ParamPack m_params;

  // typedef for method pointers of scalar type double
  typedef double (ZrO_Clexulator::*BasisFuncPtr_0)() const;

  // typedef for method pointers
  typedef double (ZrO_Clexulator::*DeltaBasisFuncPtr_0)(int, int) const;

  // array of pointers to member functions for calculating basis functions of
  // scalar type double
  BasisFuncPtr_0 m_orbit_func_table_0[74];

  // array of pointers to member functions for calculating flower functions of
  // scalar type double
  BasisFuncPtr_0 m_flower_func_table_0[2][74];

  // array of pointers to member functions for calculating DELTA flower
  // functions of scalar type double
  DeltaBasisFuncPtr_0 m_delta_func_table_0[2][74];

  // Occupation Function tables for basis sites in asymmetric unit 1:
  //   - basis site 2:
  double m_occ_func_2_0[2];

  //   - basis site 3:
  double m_occ_func_3_0[2];

  // ClexParamPack allocation for evaluated correlations
  ParamPack::Key m_corr_param_key;
  // ClexParamPack allocation for DoF occ
  ParamPack::Key m_occ_site_func_param_key;

  /// \brief Clone the ZrO_Clexulator
  BaseClexulator *_clone() const override { return new ZrO_Clexulator(*this); }

  /// \brief Calculate contribution to global correlations from one unit cell
  /// Result is recorded in ClexParamPack
  void _calc_global_corr_contribution() const override;

  /// \brief Calculate contribution to global correlations from one unit cell
  /// /// Result is recorded in double array starting at corr_begin
  void _calc_global_corr_contribution(double *corr_begin) const override;

  /// \brief Calculate contribution to select global correlations from one unit
  /// cell into ClexParamPack Result is recorded in ClexParamPack
  void _calc_restricted_global_corr_contribution(
      size_type const *ind_list_begin,
      size_type const *ind_list_end) const override;

  /// \brief Calculate contribution to select global correlations from one unit
  /// cell Result is recorded in double array starting at corr_begin
  void _calc_restricted_global_corr_contribution(
      double *corr_begin, size_type const *ind_list_begin,
      size_type const *ind_list_end) const override;

  /// \brief Calculate point correlations about neighbor site 'nlist_ind'
  /// For global clexulators, 'nlist_ind' only ranges over sites in the cell
  /// For local clexulators, 'nlist_ind' ranges over all sites in the
  /// neighborhood Result is recorded in ClexParamPack
  void _calc_point_corr(int nlist_ind) const override;

  /// \brief Calculate point correlations about neighbor site 'nlist_ind'
  /// For global clexulators, 'nlist_ind' only ranges over sites in the cell
  /// For local clexulators, 'nlist_ind' ranges over all sites in the
  /// neighborhood Result is recorded in double array starting at corr_begin
  void _calc_point_corr(int nlist_ind, double *corr_begin) const override;

  /// \brief Calculate select point correlations about neighbor site 'nlist_ind'
  /// For global clexulators, 'nlist_ind' only ranges over sites in the cell
  /// For local clexulators, 'nlist_ind' ranges over all sites in the
  /// neighborhood Result is recorded in ClexParamPack
  void _calc_restricted_point_corr(
      int nlist_ind, size_type const *ind_list_begin,
      size_type const *ind_list_end) const override;

  /// \brief Calculate select point correlations about neighbor site 'nlist_ind'
  /// For global clexulators, 'nlist_ind' only ranges over sites in the cell
  /// For local clexulators, 'nlist_ind' ranges over all sites in the
  /// neighborhood Result is recorded in double array starting at corr_begin
  void _calc_restricted_point_corr(
      int nlist_ind, double *corr_begin, size_type const *ind_list_begin,
      size_type const *ind_list_end) const override;

  /// \brief Calculate the change in point correlations due to changing an
  /// occupant at neighbor site 'nlist_ind' For global clexulators, 'nlist_ind'
  /// only ranges over sites in the cell For local clexulators, 'nlist_ind'
  /// ranges over all sites in the neighborhood Result is recorded in
  /// ClexParamPack
  void _calc_delta_point_corr(int nlist_ind, int occ_i,
                              int occ_f) const override;

  /// \brief Calculate the change in point correlations due to changing an
  /// occupant at neighbor site 'nlist_ind' For global clexulators, 'nlist_ind'
  /// only ranges over sites in the cell For local clexulators, 'nlist_ind'
  /// ranges over all sites in the neighborhood Result is recorded in double
  /// array starting at corr_begin
  void _calc_delta_point_corr(int nlist_ind, int occ_i, int occ_f,
                              double *corr_begin) const override;

  /// \brief Calculate the change in select point correlations due to changing
  /// an occupant at neighbor site 'nlist_ind' For global clexulators,
  /// 'nlist_ind' only ranges over sites in the cell For local clexulators,
  /// 'nlist_ind' ranges over all sites in the neighborhood Result is recorded
  /// in ClexParamPack
  void _calc_restricted_delta_point_corr(
      int nlist_ind, int occ_i, int occ_f, size_type const *ind_list_begin,
      size_type const *ind_list_end) const override;

  /// \brief Calculate the change in select point correlations due to changing
  /// an occupant at neighbor site 'nlist_ind' For global clexulators,
  /// 'nlist_ind' only ranges over sites in the cell For local clexulators,
  /// 'nlist_ind' ranges over all sites in the neighborhood Result is recorded
  /// in double array starting at corr_begin
  void _calc_restricted_delta_point_corr(
      int nlist_ind, int occ_i, int occ_f, double *corr_begin,
      size_type const *ind_list_begin,
      size_type const *ind_list_end) const override;

  template <typename Scalar>
  void _global_prepare() const;

  template <typename Scalar>
  void _point_prepare(int nlist_ind) const;

  // Occupation Function evaluators and accessors for basis site 2:
  double const &eval_occ_func_2_0(const int &nlist_ind) const {
    return m_occ_func_2_0[_occ(nlist_ind)];
  }

  double const &occ_func_2_0(const int &nlist_ind) const {
    return m_params.read(m_occ_site_func_param_key, 0, nlist_ind);
  }

  // Occupation Function evaluators and accessors for basis site 3:
  double const &eval_occ_func_3_0(const int &nlist_ind) const {
    return m_occ_func_3_0[_occ(nlist_ind)];
  }

  double const &occ_func_3_0(const int &nlist_ind) const {
    return m_params.read(m_occ_site_func_param_key, 0, nlist_ind);
  }

  // default functions for basis function evaluation
  template <typename Scalar>
  Scalar zero_func() const {
    return Scalar(0.0);
  }

  template <typename Scalar>
  Scalar zero_func(int, int) const {
    return Scalar(0.0);
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ZrO_Clexulator::ZrO_Clexulator() : BaseClexulator(225, 74, 2) {
  m_occ_func_2_0[0] = -0.0000000000, m_occ_func_2_0[1] = 1.0000000000;

  m_occ_func_3_0[0] = -0.0000000000, m_occ_func_3_0[1] = 1.0000000000;

  m_occ_site_func_param_key = m_params.allocate("occ_site_func", 1, 225, true);

  m_corr_param_key = m_params.allocate("corr", corr_size(), 1, false);

  m_orbit_func_table_0[0] = &ZrO_Clexulator::eval_bfunc_0_0<double>;
  m_orbit_func_table_0[1] = &ZrO_Clexulator::eval_bfunc_1_0<double>;
  m_orbit_func_table_0[2] = &ZrO_Clexulator::eval_bfunc_2_0<double>;
  m_orbit_func_table_0[3] = &ZrO_Clexulator::eval_bfunc_3_0<double>;
  m_orbit_func_table_0[4] = &ZrO_Clexulator::eval_bfunc_4_0<double>;
  m_orbit_func_table_0[5] = &ZrO_Clexulator::eval_bfunc_5_0<double>;
  m_orbit_func_table_0[6] = &ZrO_Clexulator::eval_bfunc_6_0<double>;
  m_orbit_func_table_0[7] = &ZrO_Clexulator::eval_bfunc_7_0<double>;
  m_orbit_func_table_0[8] = &ZrO_Clexulator::eval_bfunc_8_0<double>;
  m_orbit_func_table_0[9] = &ZrO_Clexulator::eval_bfunc_9_0<double>;
  m_orbit_func_table_0[10] = &ZrO_Clexulator::eval_bfunc_10_0<double>;
  m_orbit_func_table_0[11] = &ZrO_Clexulator::eval_bfunc_11_0<double>;
  m_orbit_func_table_0[12] = &ZrO_Clexulator::eval_bfunc_12_0<double>;
  m_orbit_func_table_0[13] = &ZrO_Clexulator::eval_bfunc_13_0<double>;
  m_orbit_func_table_0[14] = &ZrO_Clexulator::eval_bfunc_14_0<double>;
  m_orbit_func_table_0[15] = &ZrO_Clexulator::eval_bfunc_15_0<double>;
  m_orbit_func_table_0[16] = &ZrO_Clexulator::eval_bfunc_16_0<double>;
  m_orbit_func_table_0[17] = &ZrO_Clexulator::eval_bfunc_17_0<double>;
  m_orbit_func_table_0[18] = &ZrO_Clexulator::eval_bfunc_18_0<double>;
  m_orbit_func_table_0[19] = &ZrO_Clexulator::eval_bfunc_19_0<double>;
  m_orbit_func_table_0[20] = &ZrO_Clexulator::eval_bfunc_20_0<double>;
  m_orbit_func_table_0[21] = &ZrO_Clexulator::eval_bfunc_21_0<double>;
  m_orbit_func_table_0[22] = &ZrO_Clexulator::eval_bfunc_22_0<double>;
  m_orbit_func_table_0[23] = &ZrO_Clexulator::eval_bfunc_23_0<double>;
  m_orbit_func_table_0[24] = &ZrO_Clexulator::eval_bfunc_24_0<double>;
  m_orbit_func_table_0[25] = &ZrO_Clexulator::eval_bfunc_25_0<double>;
  m_orbit_func_table_0[26] = &ZrO_Clexulator::eval_bfunc_26_0<double>;
  m_orbit_func_table_0[27] = &ZrO_Clexulator::eval_bfunc_27_0<double>;
  m_orbit_func_table_0[28] = &ZrO_Clexulator::eval_bfunc_28_0<double>;
  m_orbit_func_table_0[29] = &ZrO_Clexulator::eval_bfunc_29_0<double>;
  m_orbit_func_table_0[30] = &ZrO_Clexulator::eval_bfunc_30_0<double>;
  m_orbit_func_table_0[31] = &ZrO_Clexulator::eval_bfunc_31_0<double>;
  m_orbit_func_table_0[32] = &ZrO_Clexulator::eval_bfunc_32_0<double>;
  m_orbit_func_table_0[33] = &ZrO_Clexulator::eval_bfunc_33_0<double>;
  m_orbit_func_table_0[34] = &ZrO_Clexulator::eval_bfunc_34_0<double>;
  m_orbit_func_table_0[35] = &ZrO_Clexulator::eval_bfunc_35_0<double>;
  m_orbit_func_table_0[36] = &ZrO_Clexulator::eval_bfunc_36_0<double>;
  m_orbit_func_table_0[37] = &ZrO_Clexulator::eval_bfunc_37_0<double>;
  m_orbit_func_table_0[38] = &ZrO_Clexulator::eval_bfunc_38_0<double>;
  m_orbit_func_table_0[39] = &ZrO_Clexulator::eval_bfunc_39_0<double>;
  m_orbit_func_table_0[40] = &ZrO_Clexulator::eval_bfunc_40_0<double>;
  m_orbit_func_table_0[41] = &ZrO_Clexulator::eval_bfunc_41_0<double>;
  m_orbit_func_table_0[42] = &ZrO_Clexulator::eval_bfunc_42_0<double>;
  m_orbit_func_table_0[43] = &ZrO_Clexulator::eval_bfunc_43_0<double>;
  m_orbit_func_table_0[44] = &ZrO_Clexulator::eval_bfunc_44_0<double>;
  m_orbit_func_table_0[45] = &ZrO_Clexulator::eval_bfunc_45_0<double>;
  m_orbit_func_table_0[46] = &ZrO_Clexulator::eval_bfunc_46_0<double>;
  m_orbit_func_table_0[47] = &ZrO_Clexulator::eval_bfunc_47_0<double>;
  m_orbit_func_table_0[48] = &ZrO_Clexulator::eval_bfunc_48_0<double>;
  m_orbit_func_table_0[49] = &ZrO_Clexulator::eval_bfunc_49_0<double>;
  m_orbit_func_table_0[50] = &ZrO_Clexulator::eval_bfunc_50_0<double>;
  m_orbit_func_table_0[51] = &ZrO_Clexulator::eval_bfunc_51_0<double>;
  m_orbit_func_table_0[52] = &ZrO_Clexulator::eval_bfunc_52_0<double>;
  m_orbit_func_table_0[53] = &ZrO_Clexulator::eval_bfunc_53_0<double>;
  m_orbit_func_table_0[54] = &ZrO_Clexulator::eval_bfunc_54_0<double>;
  m_orbit_func_table_0[55] = &ZrO_Clexulator::eval_bfunc_55_0<double>;
  m_orbit_func_table_0[56] = &ZrO_Clexulator::eval_bfunc_56_0<double>;
  m_orbit_func_table_0[57] = &ZrO_Clexulator::eval_bfunc_57_0<double>;
  m_orbit_func_table_0[58] = &ZrO_Clexulator::eval_bfunc_58_0<double>;
  m_orbit_func_table_0[59] = &ZrO_Clexulator::eval_bfunc_59_0<double>;
  m_orbit_func_table_0[60] = &ZrO_Clexulator::eval_bfunc_60_0<double>;
  m_orbit_func_table_0[61] = &ZrO_Clexulator::eval_bfunc_61_0<double>;
  m_orbit_func_table_0[62] = &ZrO_Clexulator::eval_bfunc_62_0<double>;
  m_orbit_func_table_0[63] = &ZrO_Clexulator::eval_bfunc_63_0<double>;
  m_orbit_func_table_0[64] = &ZrO_Clexulator::eval_bfunc_64_0<double>;
  m_orbit_func_table_0[65] = &ZrO_Clexulator::eval_bfunc_65_0<double>;
  m_orbit_func_table_0[66] = &ZrO_Clexulator::eval_bfunc_66_0<double>;
  m_orbit_func_table_0[67] = &ZrO_Clexulator::eval_bfunc_67_0<double>;
  m_orbit_func_table_0[68] = &ZrO_Clexulator::eval_bfunc_68_0<double>;
  m_orbit_func_table_0[69] = &ZrO_Clexulator::eval_bfunc_69_0<double>;
  m_orbit_func_table_0[70] = &ZrO_Clexulator::eval_bfunc_70_0<double>;
  m_orbit_func_table_0[71] = &ZrO_Clexulator::eval_bfunc_71_0<double>;
  m_orbit_func_table_0[72] = &ZrO_Clexulator::eval_bfunc_72_0<double>;
  m_orbit_func_table_0[73] = &ZrO_Clexulator::eval_bfunc_73_0<double>;

  m_flower_func_table_0[0][0] = &ZrO_Clexulator::zero_func<double>;
  m_flower_func_table_0[0][1] =
      &ZrO_Clexulator::site_eval_bfunc_1_0_at_0<double>;
  m_flower_func_table_0[0][2] =
      &ZrO_Clexulator::site_eval_bfunc_2_0_at_0<double>;
  m_flower_func_table_0[0][3] =
      &ZrO_Clexulator::site_eval_bfunc_3_0_at_0<double>;
  m_flower_func_table_0[0][4] =
      &ZrO_Clexulator::site_eval_bfunc_4_0_at_0<double>;
  m_flower_func_table_0[0][5] =
      &ZrO_Clexulator::site_eval_bfunc_5_0_at_0<double>;
  m_flower_func_table_0[0][6] =
      &ZrO_Clexulator::site_eval_bfunc_6_0_at_0<double>;
  m_flower_func_table_0[0][7] =
      &ZrO_Clexulator::site_eval_bfunc_7_0_at_0<double>;
  m_flower_func_table_0[0][8] =
      &ZrO_Clexulator::site_eval_bfunc_8_0_at_0<double>;
  m_flower_func_table_0[0][9] =
      &ZrO_Clexulator::site_eval_bfunc_9_0_at_0<double>;
  m_flower_func_table_0[0][10] =
      &ZrO_Clexulator::site_eval_bfunc_10_0_at_0<double>;
  m_flower_func_table_0[0][11] =
      &ZrO_Clexulator::site_eval_bfunc_11_0_at_0<double>;
  m_flower_func_table_0[0][12] =
      &ZrO_Clexulator::site_eval_bfunc_12_0_at_0<double>;
  m_flower_func_table_0[0][13] =
      &ZrO_Clexulator::site_eval_bfunc_13_0_at_0<double>;
  m_flower_func_table_0[0][14] =
      &ZrO_Clexulator::site_eval_bfunc_14_0_at_0<double>;
  m_flower_func_table_0[0][15] =
      &ZrO_Clexulator::site_eval_bfunc_15_0_at_0<double>;
  m_flower_func_table_0[0][16] =
      &ZrO_Clexulator::site_eval_bfunc_16_0_at_0<double>;
  m_flower_func_table_0[0][17] =
      &ZrO_Clexulator::site_eval_bfunc_17_0_at_0<double>;
  m_flower_func_table_0[0][18] =
      &ZrO_Clexulator::site_eval_bfunc_18_0_at_0<double>;
  m_flower_func_table_0[0][19] =
      &ZrO_Clexulator::site_eval_bfunc_19_0_at_0<double>;
  m_flower_func_table_0[0][20] =
      &ZrO_Clexulator::site_eval_bfunc_20_0_at_0<double>;
  m_flower_func_table_0[0][21] =
      &ZrO_Clexulator::site_eval_bfunc_21_0_at_0<double>;
  m_flower_func_table_0[0][22] =
      &ZrO_Clexulator::site_eval_bfunc_22_0_at_0<double>;
  m_flower_func_table_0[0][23] =
      &ZrO_Clexulator::site_eval_bfunc_23_0_at_0<double>;
  m_flower_func_table_0[0][24] =
      &ZrO_Clexulator::site_eval_bfunc_24_0_at_0<double>;
  m_flower_func_table_0[0][25] =
      &ZrO_Clexulator::site_eval_bfunc_25_0_at_0<double>;
  m_flower_func_table_0[0][26] =
      &ZrO_Clexulator::site_eval_bfunc_26_0_at_0<double>;
  m_flower_func_table_0[0][27] =
      &ZrO_Clexulator::site_eval_bfunc_27_0_at_0<double>;
  m_flower_func_table_0[0][28] =
      &ZrO_Clexulator::site_eval_bfunc_28_0_at_0<double>;
  m_flower_func_table_0[0][29] =
      &ZrO_Clexulator::site_eval_bfunc_29_0_at_0<double>;
  m_flower_func_table_0[0][30] =
      &ZrO_Clexulator::site_eval_bfunc_30_0_at_0<double>;
  m_flower_func_table_0[0][31] =
      &ZrO_Clexulator::site_eval_bfunc_31_0_at_0<double>;
  m_flower_func_table_0[0][32] =
      &ZrO_Clexulator::site_eval_bfunc_32_0_at_0<double>;
  m_flower_func_table_0[0][33] =
      &ZrO_Clexulator::site_eval_bfunc_33_0_at_0<double>;
  m_flower_func_table_0[0][34] =
      &ZrO_Clexulator::site_eval_bfunc_34_0_at_0<double>;
  m_flower_func_table_0[0][35] =
      &ZrO_Clexulator::site_eval_bfunc_35_0_at_0<double>;
  m_flower_func_table_0[0][36] =
      &ZrO_Clexulator::site_eval_bfunc_36_0_at_0<double>;
  m_flower_func_table_0[0][37] =
      &ZrO_Clexulator::site_eval_bfunc_37_0_at_0<double>;
  m_flower_func_table_0[0][38] =
      &ZrO_Clexulator::site_eval_bfunc_38_0_at_0<double>;
  m_flower_func_table_0[0][39] =
      &ZrO_Clexulator::site_eval_bfunc_39_0_at_0<double>;
  m_flower_func_table_0[0][40] =
      &ZrO_Clexulator::site_eval_bfunc_40_0_at_0<double>;
  m_flower_func_table_0[0][41] =
      &ZrO_Clexulator::site_eval_bfunc_41_0_at_0<double>;
  m_flower_func_table_0[0][42] =
      &ZrO_Clexulator::site_eval_bfunc_42_0_at_0<double>;
  m_flower_func_table_0[0][43] =
      &ZrO_Clexulator::site_eval_bfunc_43_0_at_0<double>;
  m_flower_func_table_0[0][44] =
      &ZrO_Clexulator::site_eval_bfunc_44_0_at_0<double>;
  m_flower_func_table_0[0][45] =
      &ZrO_Clexulator::site_eval_bfunc_45_0_at_0<double>;
  m_flower_func_table_0[0][46] =
      &ZrO_Clexulator::site_eval_bfunc_46_0_at_0<double>;
  m_flower_func_table_0[0][47] =
      &ZrO_Clexulator::site_eval_bfunc_47_0_at_0<double>;
  m_flower_func_table_0[0][48] =
      &ZrO_Clexulator::site_eval_bfunc_48_0_at_0<double>;
  m_flower_func_table_0[0][49] =
      &ZrO_Clexulator::site_eval_bfunc_49_0_at_0<double>;
  m_flower_func_table_0[0][50] =
      &ZrO_Clexulator::site_eval_bfunc_50_0_at_0<double>;
  m_flower_func_table_0[0][51] =
      &ZrO_Clexulator::site_eval_bfunc_51_0_at_0<double>;
  m_flower_func_table_0[0][52] =
      &ZrO_Clexulator::site_eval_bfunc_52_0_at_0<double>;
  m_flower_func_table_0[0][53] =
      &ZrO_Clexulator::site_eval_bfunc_53_0_at_0<double>;
  m_flower_func_table_0[0][54] =
      &ZrO_Clexulator::site_eval_bfunc_54_0_at_0<double>;
  m_flower_func_table_0[0][55] =
      &ZrO_Clexulator::site_eval_bfunc_55_0_at_0<double>;
  m_flower_func_table_0[0][56] =
      &ZrO_Clexulator::site_eval_bfunc_56_0_at_0<double>;
  m_flower_func_table_0[0][57] =
      &ZrO_Clexulator::site_eval_bfunc_57_0_at_0<double>;
  m_flower_func_table_0[0][58] =
      &ZrO_Clexulator::site_eval_bfunc_58_0_at_0<double>;
  m_flower_func_table_0[0][59] =
      &ZrO_Clexulator::site_eval_bfunc_59_0_at_0<double>;
  m_flower_func_table_0[0][60] =
      &ZrO_Clexulator::site_eval_bfunc_60_0_at_0<double>;
  m_flower_func_table_0[0][61] =
      &ZrO_Clexulator::site_eval_bfunc_61_0_at_0<double>;
  m_flower_func_table_0[0][62] =
      &ZrO_Clexulator::site_eval_bfunc_62_0_at_0<double>;
  m_flower_func_table_0[0][63] =
      &ZrO_Clexulator::site_eval_bfunc_63_0_at_0<double>;
  m_flower_func_table_0[0][64] =
      &ZrO_Clexulator::site_eval_bfunc_64_0_at_0<double>;
  m_flower_func_table_0[0][65] =
      &ZrO_Clexulator::site_eval_bfunc_65_0_at_0<double>;
  m_flower_func_table_0[0][66] =
      &ZrO_Clexulator::site_eval_bfunc_66_0_at_0<double>;
  m_flower_func_table_0[0][67] =
      &ZrO_Clexulator::site_eval_bfunc_67_0_at_0<double>;
  m_flower_func_table_0[0][68] =
      &ZrO_Clexulator::site_eval_bfunc_68_0_at_0<double>;
  m_flower_func_table_0[0][69] =
      &ZrO_Clexulator::site_eval_bfunc_69_0_at_0<double>;
  m_flower_func_table_0[0][70] =
      &ZrO_Clexulator::site_eval_bfunc_70_0_at_0<double>;
  m_flower_func_table_0[0][71] =
      &ZrO_Clexulator::site_eval_bfunc_71_0_at_0<double>;
  m_flower_func_table_0[0][72] =
      &ZrO_Clexulator::site_eval_bfunc_72_0_at_0<double>;
  m_flower_func_table_0[0][73] =
      &ZrO_Clexulator::site_eval_bfunc_73_0_at_0<double>;

  m_flower_func_table_0[1][0] = &ZrO_Clexulator::zero_func<double>;
  m_flower_func_table_0[1][1] =
      &ZrO_Clexulator::site_eval_bfunc_1_0_at_1<double>;
  m_flower_func_table_0[1][2] =
      &ZrO_Clexulator::site_eval_bfunc_2_0_at_1<double>;
  m_flower_func_table_0[1][3] =
      &ZrO_Clexulator::site_eval_bfunc_3_0_at_1<double>;
  m_flower_func_table_0[1][4] =
      &ZrO_Clexulator::site_eval_bfunc_4_0_at_1<double>;
  m_flower_func_table_0[1][5] =
      &ZrO_Clexulator::site_eval_bfunc_5_0_at_1<double>;
  m_flower_func_table_0[1][6] =
      &ZrO_Clexulator::site_eval_bfunc_6_0_at_1<double>;
  m_flower_func_table_0[1][7] =
      &ZrO_Clexulator::site_eval_bfunc_7_0_at_1<double>;
  m_flower_func_table_0[1][8] =
      &ZrO_Clexulator::site_eval_bfunc_8_0_at_1<double>;
  m_flower_func_table_0[1][9] =
      &ZrO_Clexulator::site_eval_bfunc_9_0_at_1<double>;
  m_flower_func_table_0[1][10] =
      &ZrO_Clexulator::site_eval_bfunc_10_0_at_1<double>;
  m_flower_func_table_0[1][11] =
      &ZrO_Clexulator::site_eval_bfunc_11_0_at_1<double>;
  m_flower_func_table_0[1][12] =
      &ZrO_Clexulator::site_eval_bfunc_12_0_at_1<double>;
  m_flower_func_table_0[1][13] =
      &ZrO_Clexulator::site_eval_bfunc_13_0_at_1<double>;
  m_flower_func_table_0[1][14] =
      &ZrO_Clexulator::site_eval_bfunc_14_0_at_1<double>;
  m_flower_func_table_0[1][15] =
      &ZrO_Clexulator::site_eval_bfunc_15_0_at_1<double>;
  m_flower_func_table_0[1][16] =
      &ZrO_Clexulator::site_eval_bfunc_16_0_at_1<double>;
  m_flower_func_table_0[1][17] =
      &ZrO_Clexulator::site_eval_bfunc_17_0_at_1<double>;
  m_flower_func_table_0[1][18] =
      &ZrO_Clexulator::site_eval_bfunc_18_0_at_1<double>;
  m_flower_func_table_0[1][19] =
      &ZrO_Clexulator::site_eval_bfunc_19_0_at_1<double>;
  m_flower_func_table_0[1][20] =
      &ZrO_Clexulator::site_eval_bfunc_20_0_at_1<double>;
  m_flower_func_table_0[1][21] =
      &ZrO_Clexulator::site_eval_bfunc_21_0_at_1<double>;
  m_flower_func_table_0[1][22] =
      &ZrO_Clexulator::site_eval_bfunc_22_0_at_1<double>;
  m_flower_func_table_0[1][23] =
      &ZrO_Clexulator::site_eval_bfunc_23_0_at_1<double>;
  m_flower_func_table_0[1][24] =
      &ZrO_Clexulator::site_eval_bfunc_24_0_at_1<double>;
  m_flower_func_table_0[1][25] =
      &ZrO_Clexulator::site_eval_bfunc_25_0_at_1<double>;
  m_flower_func_table_0[1][26] =
      &ZrO_Clexulator::site_eval_bfunc_26_0_at_1<double>;
  m_flower_func_table_0[1][27] =
      &ZrO_Clexulator::site_eval_bfunc_27_0_at_1<double>;
  m_flower_func_table_0[1][28] =
      &ZrO_Clexulator::site_eval_bfunc_28_0_at_1<double>;
  m_flower_func_table_0[1][29] =
      &ZrO_Clexulator::site_eval_bfunc_29_0_at_1<double>;
  m_flower_func_table_0[1][30] =
      &ZrO_Clexulator::site_eval_bfunc_30_0_at_1<double>;
  m_flower_func_table_0[1][31] =
      &ZrO_Clexulator::site_eval_bfunc_31_0_at_1<double>;
  m_flower_func_table_0[1][32] =
      &ZrO_Clexulator::site_eval_bfunc_32_0_at_1<double>;
  m_flower_func_table_0[1][33] =
      &ZrO_Clexulator::site_eval_bfunc_33_0_at_1<double>;
  m_flower_func_table_0[1][34] =
      &ZrO_Clexulator::site_eval_bfunc_34_0_at_1<double>;
  m_flower_func_table_0[1][35] =
      &ZrO_Clexulator::site_eval_bfunc_35_0_at_1<double>;
  m_flower_func_table_0[1][36] =
      &ZrO_Clexulator::site_eval_bfunc_36_0_at_1<double>;
  m_flower_func_table_0[1][37] =
      &ZrO_Clexulator::site_eval_bfunc_37_0_at_1<double>;
  m_flower_func_table_0[1][38] =
      &ZrO_Clexulator::site_eval_bfunc_38_0_at_1<double>;
  m_flower_func_table_0[1][39] =
      &ZrO_Clexulator::site_eval_bfunc_39_0_at_1<double>;
  m_flower_func_table_0[1][40] =
      &ZrO_Clexulator::site_eval_bfunc_40_0_at_1<double>;
  m_flower_func_table_0[1][41] =
      &ZrO_Clexulator::site_eval_bfunc_41_0_at_1<double>;
  m_flower_func_table_0[1][42] =
      &ZrO_Clexulator::site_eval_bfunc_42_0_at_1<double>;
  m_flower_func_table_0[1][43] =
      &ZrO_Clexulator::site_eval_bfunc_43_0_at_1<double>;
  m_flower_func_table_0[1][44] =
      &ZrO_Clexulator::site_eval_bfunc_44_0_at_1<double>;
  m_flower_func_table_0[1][45] =
      &ZrO_Clexulator::site_eval_bfunc_45_0_at_1<double>;
  m_flower_func_table_0[1][46] =
      &ZrO_Clexulator::site_eval_bfunc_46_0_at_1<double>;
  m_flower_func_table_0[1][47] =
      &ZrO_Clexulator::site_eval_bfunc_47_0_at_1<double>;
  m_flower_func_table_0[1][48] =
      &ZrO_Clexulator::site_eval_bfunc_48_0_at_1<double>;
  m_flower_func_table_0[1][49] =
      &ZrO_Clexulator::site_eval_bfunc_49_0_at_1<double>;
  m_flower_func_table_0[1][50] =
      &ZrO_Clexulator::site_eval_bfunc_50_0_at_1<double>;
  m_flower_func_table_0[1][51] =
      &ZrO_Clexulator::site_eval_bfunc_51_0_at_1<double>;
  m_flower_func_table_0[1][52] =
      &ZrO_Clexulator::site_eval_bfunc_52_0_at_1<double>;
  m_flower_func_table_0[1][53] =
      &ZrO_Clexulator::site_eval_bfunc_53_0_at_1<double>;
  m_flower_func_table_0[1][54] =
      &ZrO_Clexulator::site_eval_bfunc_54_0_at_1<double>;
  m_flower_func_table_0[1][55] =
      &ZrO_Clexulator::site_eval_bfunc_55_0_at_1<double>;
  m_flower_func_table_0[1][56] =
      &ZrO_Clexulator::site_eval_bfunc_56_0_at_1<double>;
  m_flower_func_table_0[1][57] =
      &ZrO_Clexulator::site_eval_bfunc_57_0_at_1<double>;
  m_flower_func_table_0[1][58] =
      &ZrO_Clexulator::site_eval_bfunc_58_0_at_1<double>;
  m_flower_func_table_0[1][59] =
      &ZrO_Clexulator::site_eval_bfunc_59_0_at_1<double>;
  m_flower_func_table_0[1][60] =
      &ZrO_Clexulator::site_eval_bfunc_60_0_at_1<double>;
  m_flower_func_table_0[1][61] =
      &ZrO_Clexulator::site_eval_bfunc_61_0_at_1<double>;
  m_flower_func_table_0[1][62] =
      &ZrO_Clexulator::site_eval_bfunc_62_0_at_1<double>;
  m_flower_func_table_0[1][63] =
      &ZrO_Clexulator::site_eval_bfunc_63_0_at_1<double>;
  m_flower_func_table_0[1][64] =
      &ZrO_Clexulator::site_eval_bfunc_64_0_at_1<double>;
  m_flower_func_table_0[1][65] =
      &ZrO_Clexulator::site_eval_bfunc_65_0_at_1<double>;
  m_flower_func_table_0[1][66] =
      &ZrO_Clexulator::site_eval_bfunc_66_0_at_1<double>;
  m_flower_func_table_0[1][67] =
      &ZrO_Clexulator::site_eval_bfunc_67_0_at_1<double>;
  m_flower_func_table_0[1][68] =
      &ZrO_Clexulator::site_eval_bfunc_68_0_at_1<double>;
  m_flower_func_table_0[1][69] =
      &ZrO_Clexulator::site_eval_bfunc_69_0_at_1<double>;
  m_flower_func_table_0[1][70] =
      &ZrO_Clexulator::site_eval_bfunc_70_0_at_1<double>;
  m_flower_func_table_0[1][71] =
      &ZrO_Clexulator::site_eval_bfunc_71_0_at_1<double>;
  m_flower_func_table_0[1][72] =
      &ZrO_Clexulator::site_eval_bfunc_72_0_at_1<double>;
  m_flower_func_table_0[1][73] =
      &ZrO_Clexulator::site_eval_bfunc_73_0_at_1<double>;

  m_delta_func_table_0[0][0] = &ZrO_Clexulator::zero_func<double>;
  m_delta_func_table_0[0][1] =
      &ZrO_Clexulator::site_deval_bfunc_1_0_at_0<double>;
  m_delta_func_table_0[0][2] =
      &ZrO_Clexulator::site_deval_bfunc_2_0_at_0<double>;
  m_delta_func_table_0[0][3] =
      &ZrO_Clexulator::site_deval_bfunc_3_0_at_0<double>;
  m_delta_func_table_0[0][4] =
      &ZrO_Clexulator::site_deval_bfunc_4_0_at_0<double>;
  m_delta_func_table_0[0][5] =
      &ZrO_Clexulator::site_deval_bfunc_5_0_at_0<double>;
  m_delta_func_table_0[0][6] =
      &ZrO_Clexulator::site_deval_bfunc_6_0_at_0<double>;
  m_delta_func_table_0[0][7] =
      &ZrO_Clexulator::site_deval_bfunc_7_0_at_0<double>;
  m_delta_func_table_0[0][8] =
      &ZrO_Clexulator::site_deval_bfunc_8_0_at_0<double>;
  m_delta_func_table_0[0][9] =
      &ZrO_Clexulator::site_deval_bfunc_9_0_at_0<double>;
  m_delta_func_table_0[0][10] =
      &ZrO_Clexulator::site_deval_bfunc_10_0_at_0<double>;
  m_delta_func_table_0[0][11] =
      &ZrO_Clexulator::site_deval_bfunc_11_0_at_0<double>;
  m_delta_func_table_0[0][12] =
      &ZrO_Clexulator::site_deval_bfunc_12_0_at_0<double>;
  m_delta_func_table_0[0][13] =
      &ZrO_Clexulator::site_deval_bfunc_13_0_at_0<double>;
  m_delta_func_table_0[0][14] =
      &ZrO_Clexulator::site_deval_bfunc_14_0_at_0<double>;
  m_delta_func_table_0[0][15] =
      &ZrO_Clexulator::site_deval_bfunc_15_0_at_0<double>;
  m_delta_func_table_0[0][16] =
      &ZrO_Clexulator::site_deval_bfunc_16_0_at_0<double>;
  m_delta_func_table_0[0][17] =
      &ZrO_Clexulator::site_deval_bfunc_17_0_at_0<double>;
  m_delta_func_table_0[0][18] =
      &ZrO_Clexulator::site_deval_bfunc_18_0_at_0<double>;
  m_delta_func_table_0[0][19] =
      &ZrO_Clexulator::site_deval_bfunc_19_0_at_0<double>;
  m_delta_func_table_0[0][20] =
      &ZrO_Clexulator::site_deval_bfunc_20_0_at_0<double>;
  m_delta_func_table_0[0][21] =
      &ZrO_Clexulator::site_deval_bfunc_21_0_at_0<double>;
  m_delta_func_table_0[0][22] =
      &ZrO_Clexulator::site_deval_bfunc_22_0_at_0<double>;
  m_delta_func_table_0[0][23] =
      &ZrO_Clexulator::site_deval_bfunc_23_0_at_0<double>;
  m_delta_func_table_0[0][24] =
      &ZrO_Clexulator::site_deval_bfunc_24_0_at_0<double>;
  m_delta_func_table_0[0][25] =
      &ZrO_Clexulator::site_deval_bfunc_25_0_at_0<double>;
  m_delta_func_table_0[0][26] =
      &ZrO_Clexulator::site_deval_bfunc_26_0_at_0<double>;
  m_delta_func_table_0[0][27] =
      &ZrO_Clexulator::site_deval_bfunc_27_0_at_0<double>;
  m_delta_func_table_0[0][28] =
      &ZrO_Clexulator::site_deval_bfunc_28_0_at_0<double>;
  m_delta_func_table_0[0][29] =
      &ZrO_Clexulator::site_deval_bfunc_29_0_at_0<double>;
  m_delta_func_table_0[0][30] =
      &ZrO_Clexulator::site_deval_bfunc_30_0_at_0<double>;
  m_delta_func_table_0[0][31] =
      &ZrO_Clexulator::site_deval_bfunc_31_0_at_0<double>;
  m_delta_func_table_0[0][32] =
      &ZrO_Clexulator::site_deval_bfunc_32_0_at_0<double>;
  m_delta_func_table_0[0][33] =
      &ZrO_Clexulator::site_deval_bfunc_33_0_at_0<double>;
  m_delta_func_table_0[0][34] =
      &ZrO_Clexulator::site_deval_bfunc_34_0_at_0<double>;
  m_delta_func_table_0[0][35] =
      &ZrO_Clexulator::site_deval_bfunc_35_0_at_0<double>;
  m_delta_func_table_0[0][36] =
      &ZrO_Clexulator::site_deval_bfunc_36_0_at_0<double>;
  m_delta_func_table_0[0][37] =
      &ZrO_Clexulator::site_deval_bfunc_37_0_at_0<double>;
  m_delta_func_table_0[0][38] =
      &ZrO_Clexulator::site_deval_bfunc_38_0_at_0<double>;
  m_delta_func_table_0[0][39] =
      &ZrO_Clexulator::site_deval_bfunc_39_0_at_0<double>;
  m_delta_func_table_0[0][40] =
      &ZrO_Clexulator::site_deval_bfunc_40_0_at_0<double>;
  m_delta_func_table_0[0][41] =
      &ZrO_Clexulator::site_deval_bfunc_41_0_at_0<double>;
  m_delta_func_table_0[0][42] =
      &ZrO_Clexulator::site_deval_bfunc_42_0_at_0<double>;
  m_delta_func_table_0[0][43] =
      &ZrO_Clexulator::site_deval_bfunc_43_0_at_0<double>;
  m_delta_func_table_0[0][44] =
      &ZrO_Clexulator::site_deval_bfunc_44_0_at_0<double>;
  m_delta_func_table_0[0][45] =
      &ZrO_Clexulator::site_deval_bfunc_45_0_at_0<double>;
  m_delta_func_table_0[0][46] =
      &ZrO_Clexulator::site_deval_bfunc_46_0_at_0<double>;
  m_delta_func_table_0[0][47] =
      &ZrO_Clexulator::site_deval_bfunc_47_0_at_0<double>;
  m_delta_func_table_0[0][48] =
      &ZrO_Clexulator::site_deval_bfunc_48_0_at_0<double>;
  m_delta_func_table_0[0][49] =
      &ZrO_Clexulator::site_deval_bfunc_49_0_at_0<double>;
  m_delta_func_table_0[0][50] =
      &ZrO_Clexulator::site_deval_bfunc_50_0_at_0<double>;
  m_delta_func_table_0[0][51] =
      &ZrO_Clexulator::site_deval_bfunc_51_0_at_0<double>;
  m_delta_func_table_0[0][52] =
      &ZrO_Clexulator::site_deval_bfunc_52_0_at_0<double>;
  m_delta_func_table_0[0][53] =
      &ZrO_Clexulator::site_deval_bfunc_53_0_at_0<double>;
  m_delta_func_table_0[0][54] =
      &ZrO_Clexulator::site_deval_bfunc_54_0_at_0<double>;
  m_delta_func_table_0[0][55] =
      &ZrO_Clexulator::site_deval_bfunc_55_0_at_0<double>;
  m_delta_func_table_0[0][56] =
      &ZrO_Clexulator::site_deval_bfunc_56_0_at_0<double>;
  m_delta_func_table_0[0][57] =
      &ZrO_Clexulator::site_deval_bfunc_57_0_at_0<double>;
  m_delta_func_table_0[0][58] =
      &ZrO_Clexulator::site_deval_bfunc_58_0_at_0<double>;
  m_delta_func_table_0[0][59] =
      &ZrO_Clexulator::site_deval_bfunc_59_0_at_0<double>;
  m_delta_func_table_0[0][60] =
      &ZrO_Clexulator::site_deval_bfunc_60_0_at_0<double>;
  m_delta_func_table_0[0][61] =
      &ZrO_Clexulator::site_deval_bfunc_61_0_at_0<double>;
  m_delta_func_table_0[0][62] =
      &ZrO_Clexulator::site_deval_bfunc_62_0_at_0<double>;
  m_delta_func_table_0[0][63] =
      &ZrO_Clexulator::site_deval_bfunc_63_0_at_0<double>;
  m_delta_func_table_0[0][64] =
      &ZrO_Clexulator::site_deval_bfunc_64_0_at_0<double>;
  m_delta_func_table_0[0][65] =
      &ZrO_Clexulator::site_deval_bfunc_65_0_at_0<double>;
  m_delta_func_table_0[0][66] =
      &ZrO_Clexulator::site_deval_bfunc_66_0_at_0<double>;
  m_delta_func_table_0[0][67] =
      &ZrO_Clexulator::site_deval_bfunc_67_0_at_0<double>;
  m_delta_func_table_0[0][68] =
      &ZrO_Clexulator::site_deval_bfunc_68_0_at_0<double>;
  m_delta_func_table_0[0][69] =
      &ZrO_Clexulator::site_deval_bfunc_69_0_at_0<double>;
  m_delta_func_table_0[0][70] =
      &ZrO_Clexulator::site_deval_bfunc_70_0_at_0<double>;
  m_delta_func_table_0[0][71] =
      &ZrO_Clexulator::site_deval_bfunc_71_0_at_0<double>;
  m_delta_func_table_0[0][72] =
      &ZrO_Clexulator::site_deval_bfunc_72_0_at_0<double>;
  m_delta_func_table_0[0][73] =
      &ZrO_Clexulator::site_deval_bfunc_73_0_at_0<double>;

  m_delta_func_table_0[1][0] = &ZrO_Clexulator::zero_func<double>;
  m_delta_func_table_0[1][1] =
      &ZrO_Clexulator::site_deval_bfunc_1_0_at_1<double>;
  m_delta_func_table_0[1][2] =
      &ZrO_Clexulator::site_deval_bfunc_2_0_at_1<double>;
  m_delta_func_table_0[1][3] =
      &ZrO_Clexulator::site_deval_bfunc_3_0_at_1<double>;
  m_delta_func_table_0[1][4] =
      &ZrO_Clexulator::site_deval_bfunc_4_0_at_1<double>;
  m_delta_func_table_0[1][5] =
      &ZrO_Clexulator::site_deval_bfunc_5_0_at_1<double>;
  m_delta_func_table_0[1][6] =
      &ZrO_Clexulator::site_deval_bfunc_6_0_at_1<double>;
  m_delta_func_table_0[1][7] =
      &ZrO_Clexulator::site_deval_bfunc_7_0_at_1<double>;
  m_delta_func_table_0[1][8] =
      &ZrO_Clexulator::site_deval_bfunc_8_0_at_1<double>;
  m_delta_func_table_0[1][9] =
      &ZrO_Clexulator::site_deval_bfunc_9_0_at_1<double>;
  m_delta_func_table_0[1][10] =
      &ZrO_Clexulator::site_deval_bfunc_10_0_at_1<double>;
  m_delta_func_table_0[1][11] =
      &ZrO_Clexulator::site_deval_bfunc_11_0_at_1<double>;
  m_delta_func_table_0[1][12] =
      &ZrO_Clexulator::site_deval_bfunc_12_0_at_1<double>;
  m_delta_func_table_0[1][13] =
      &ZrO_Clexulator::site_deval_bfunc_13_0_at_1<double>;
  m_delta_func_table_0[1][14] =
      &ZrO_Clexulator::site_deval_bfunc_14_0_at_1<double>;
  m_delta_func_table_0[1][15] =
      &ZrO_Clexulator::site_deval_bfunc_15_0_at_1<double>;
  m_delta_func_table_0[1][16] =
      &ZrO_Clexulator::site_deval_bfunc_16_0_at_1<double>;
  m_delta_func_table_0[1][17] =
      &ZrO_Clexulator::site_deval_bfunc_17_0_at_1<double>;
  m_delta_func_table_0[1][18] =
      &ZrO_Clexulator::site_deval_bfunc_18_0_at_1<double>;
  m_delta_func_table_0[1][19] =
      &ZrO_Clexulator::site_deval_bfunc_19_0_at_1<double>;
  m_delta_func_table_0[1][20] =
      &ZrO_Clexulator::site_deval_bfunc_20_0_at_1<double>;
  m_delta_func_table_0[1][21] =
      &ZrO_Clexulator::site_deval_bfunc_21_0_at_1<double>;
  m_delta_func_table_0[1][22] =
      &ZrO_Clexulator::site_deval_bfunc_22_0_at_1<double>;
  m_delta_func_table_0[1][23] =
      &ZrO_Clexulator::site_deval_bfunc_23_0_at_1<double>;
  m_delta_func_table_0[1][24] =
      &ZrO_Clexulator::site_deval_bfunc_24_0_at_1<double>;
  m_delta_func_table_0[1][25] =
      &ZrO_Clexulator::site_deval_bfunc_25_0_at_1<double>;
  m_delta_func_table_0[1][26] =
      &ZrO_Clexulator::site_deval_bfunc_26_0_at_1<double>;
  m_delta_func_table_0[1][27] =
      &ZrO_Clexulator::site_deval_bfunc_27_0_at_1<double>;
  m_delta_func_table_0[1][28] =
      &ZrO_Clexulator::site_deval_bfunc_28_0_at_1<double>;
  m_delta_func_table_0[1][29] =
      &ZrO_Clexulator::site_deval_bfunc_29_0_at_1<double>;
  m_delta_func_table_0[1][30] =
      &ZrO_Clexulator::site_deval_bfunc_30_0_at_1<double>;
  m_delta_func_table_0[1][31] =
      &ZrO_Clexulator::site_deval_bfunc_31_0_at_1<double>;
  m_delta_func_table_0[1][32] =
      &ZrO_Clexulator::site_deval_bfunc_32_0_at_1<double>;
  m_delta_func_table_0[1][33] =
      &ZrO_Clexulator::site_deval_bfunc_33_0_at_1<double>;
  m_delta_func_table_0[1][34] =
      &ZrO_Clexulator::site_deval_bfunc_34_0_at_1<double>;
  m_delta_func_table_0[1][35] =
      &ZrO_Clexulator::site_deval_bfunc_35_0_at_1<double>;
  m_delta_func_table_0[1][36] =
      &ZrO_Clexulator::site_deval_bfunc_36_0_at_1<double>;
  m_delta_func_table_0[1][37] =
      &ZrO_Clexulator::site_deval_bfunc_37_0_at_1<double>;
  m_delta_func_table_0[1][38] =
      &ZrO_Clexulator::site_deval_bfunc_38_0_at_1<double>;
  m_delta_func_table_0[1][39] =
      &ZrO_Clexulator::site_deval_bfunc_39_0_at_1<double>;
  m_delta_func_table_0[1][40] =
      &ZrO_Clexulator::site_deval_bfunc_40_0_at_1<double>;
  m_delta_func_table_0[1][41] =
      &ZrO_Clexulator::site_deval_bfunc_41_0_at_1<double>;
  m_delta_func_table_0[1][42] =
      &ZrO_Clexulator::site_deval_bfunc_42_0_at_1<double>;
  m_delta_func_table_0[1][43] =
      &ZrO_Clexulator::site_deval_bfunc_43_0_at_1<double>;
  m_delta_func_table_0[1][44] =
      &ZrO_Clexulator::site_deval_bfunc_44_0_at_1<double>;
  m_delta_func_table_0[1][45] =
      &ZrO_Clexulator::site_deval_bfunc_45_0_at_1<double>;
  m_delta_func_table_0[1][46] =
      &ZrO_Clexulator::site_deval_bfunc_46_0_at_1<double>;
  m_delta_func_table_0[1][47] =
      &ZrO_Clexulator::site_deval_bfunc_47_0_at_1<double>;
  m_delta_func_table_0[1][48] =
      &ZrO_Clexulator::site_deval_bfunc_48_0_at_1<double>;
  m_delta_func_table_0[1][49] =
      &ZrO_Clexulator::site_deval_bfunc_49_0_at_1<double>;
  m_delta_func_table_0[1][50] =
      &ZrO_Clexulator::site_deval_bfunc_50_0_at_1<double>;
  m_delta_func_table_0[1][51] =
      &ZrO_Clexulator::site_deval_bfunc_51_0_at_1<double>;
  m_delta_func_table_0[1][52] =
      &ZrO_Clexulator::site_deval_bfunc_52_0_at_1<double>;
  m_delta_func_table_0[1][53] =
      &ZrO_Clexulator::site_deval_bfunc_53_0_at_1<double>;
  m_delta_func_table_0[1][54] =
      &ZrO_Clexulator::site_deval_bfunc_54_0_at_1<double>;
  m_delta_func_table_0[1][55] =
      &ZrO_Clexulator::site_deval_bfunc_55_0_at_1<double>;
  m_delta_func_table_0[1][56] =
      &ZrO_Clexulator::site_deval_bfunc_56_0_at_1<double>;
  m_delta_func_table_0[1][57] =
      &ZrO_Clexulator::site_deval_bfunc_57_0_at_1<double>;
  m_delta_func_table_0[1][58] =
      &ZrO_Clexulator::site_deval_bfunc_58_0_at_1<double>;
  m_delta_func_table_0[1][59] =
      &ZrO_Clexulator::site_deval_bfunc_59_0_at_1<double>;
  m_delta_func_table_0[1][60] =
      &ZrO_Clexulator::site_deval_bfunc_60_0_at_1<double>;
  m_delta_func_table_0[1][61] =
      &ZrO_Clexulator::site_deval_bfunc_61_0_at_1<double>;
  m_delta_func_table_0[1][62] =
      &ZrO_Clexulator::site_deval_bfunc_62_0_at_1<double>;
  m_delta_func_table_0[1][63] =
      &ZrO_Clexulator::site_deval_bfunc_63_0_at_1<double>;
  m_delta_func_table_0[1][64] =
      &ZrO_Clexulator::site_deval_bfunc_64_0_at_1<double>;
  m_delta_func_table_0[1][65] =
      &ZrO_Clexulator::site_deval_bfunc_65_0_at_1<double>;
  m_delta_func_table_0[1][66] =
      &ZrO_Clexulator::site_deval_bfunc_66_0_at_1<double>;
  m_delta_func_table_0[1][67] =
      &ZrO_Clexulator::site_deval_bfunc_67_0_at_1<double>;
  m_delta_func_table_0[1][68] =
      &ZrO_Clexulator::site_deval_bfunc_68_0_at_1<double>;
  m_delta_func_table_0[1][69] =
      &ZrO_Clexulator::site_deval_bfunc_69_0_at_1<double>;
  m_delta_func_table_0[1][70] =
      &ZrO_Clexulator::site_deval_bfunc_70_0_at_1<double>;
  m_delta_func_table_0[1][71] =
      &ZrO_Clexulator::site_deval_bfunc_71_0_at_1<double>;
  m_delta_func_table_0[1][72] =
      &ZrO_Clexulator::site_deval_bfunc_72_0_at_1<double>;
  m_delta_func_table_0[1][73] =
      &ZrO_Clexulator::site_deval_bfunc_73_0_at_1<double>;

  m_weight_matrix.row(0) << 2, -1, 0;
  m_weight_matrix.row(1) << -1, 2, 0;
  m_weight_matrix.row(2) << 0, 0, 5;

  m_sublat_indices = std::set<int>{2, 3};

  m_n_sublattices = 4;

  m_neighborhood = std::set<xtal::UnitCell>{
      xtal::UnitCell(-3, -2, -1), xtal::UnitCell(-3, -2, 0),
      xtal::UnitCell(-3, -2, 1),  xtal::UnitCell(-3, -1, -1),
      xtal::UnitCell(-3, -1, 0),  xtal::UnitCell(-3, -1, 1),
      xtal::UnitCell(-2, -3, -1), xtal::UnitCell(-2, -3, 0),
      xtal::UnitCell(-2, -3, 1),  xtal::UnitCell(-2, -2, -1),
      xtal::UnitCell(-2, -2, 0),  xtal::UnitCell(-2, -2, 1),
      xtal::UnitCell(-2, -1, -1), xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, -1, 1),  xtal::UnitCell(-2, 0, -1),
      xtal::UnitCell(-2, 0, 0),   xtal::UnitCell(-2, 0, 1),
      xtal::UnitCell(-2, 1, -1),  xtal::UnitCell(-2, 1, 0),
      xtal::UnitCell(-2, 1, 1),   xtal::UnitCell(-1, -3, -1),
      xtal::UnitCell(-1, -3, 0),  xtal::UnitCell(-1, -3, 1),
      xtal::UnitCell(-1, -2, -1), xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -2, 1),  xtal::UnitCell(-1, -1, -2),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, -1, 2),
      xtal::UnitCell(-1, 0, -2),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 0, 2),   xtal::UnitCell(-1, 1, -1),
      xtal::UnitCell(-1, 1, 0),   xtal::UnitCell(-1, 1, 1),
      xtal::UnitCell(-1, 2, -1),  xtal::UnitCell(-1, 2, 0),
      xtal::UnitCell(-1, 2, 1),   xtal::UnitCell(0, -2, -1),
      xtal::UnitCell(0, -2, 0),   xtal::UnitCell(0, -2, 1),
      xtal::UnitCell(0, -1, -2),  xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, -1, 2),   xtal::UnitCell(0, 0, -2),
      xtal::UnitCell(0, 0, -1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 0, 1),    xtal::UnitCell(0, 0, 2),
      xtal::UnitCell(0, 1, -2),   xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(0, 1, 2),    xtal::UnitCell(0, 2, -1),
      xtal::UnitCell(0, 2, 0),    xtal::UnitCell(0, 2, 1),
      xtal::UnitCell(1, -2, -1),  xtal::UnitCell(1, -2, 0),
      xtal::UnitCell(1, -2, 1),   xtal::UnitCell(1, -1, -1),
      xtal::UnitCell(1, -1, 0),   xtal::UnitCell(1, -1, 1),
      xtal::UnitCell(1, 0, -2),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 0, 2),    xtal::UnitCell(1, 1, -2),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 1, 2),
      xtal::UnitCell(1, 2, -1),   xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(1, 2, 1),    xtal::UnitCell(1, 3, -1),
      xtal::UnitCell(1, 3, 0),    xtal::UnitCell(1, 3, 1),
      xtal::UnitCell(2, -1, -1),  xtal::UnitCell(2, -1, 0),
      xtal::UnitCell(2, -1, 1),   xtal::UnitCell(2, 0, -1),
      xtal::UnitCell(2, 0, 0),    xtal::UnitCell(2, 0, 1),
      xtal::UnitCell(2, 1, -1),   xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 1, 1),    xtal::UnitCell(2, 2, -1),
      xtal::UnitCell(2, 2, 0),    xtal::UnitCell(2, 2, 1),
      xtal::UnitCell(2, 3, -1),   xtal::UnitCell(2, 3, 0),
      xtal::UnitCell(2, 3, 1),    xtal::UnitCell(3, 1, -1),
      xtal::UnitCell(3, 1, 0),    xtal::UnitCell(3, 1, 1),
      xtal::UnitCell(3, 2, -1),   xtal::UnitCell(3, 2, 0),
      xtal::UnitCell(3, 2, 1)};

  m_orbit_neighborhood.resize(corr_size());
  m_orbit_site_neighborhood.resize(corr_size());
  m_orbit_neighborhood[1] = std::set<xtal::UnitCell>{xtal::UnitCell(0, 0, 0)};

  m_orbit_site_neighborhood[1] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, 0, 0, 0), xtal::UnitCellCoord(3, 0, 0, 0)};

  m_orbit_neighborhood[2] = std::set<xtal::UnitCell>{xtal::UnitCell(0, 0, -1),
                                                     xtal::UnitCell(0, 0, 0),
                                                     xtal::UnitCell(0, 0, 1)};

  m_orbit_site_neighborhood[2] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, 0, 0, -1), xtal::UnitCellCoord(2, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 0), xtal::UnitCellCoord(2, 0, 0, 1)};

  m_orbit_neighborhood[3] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, 0), xtal::UnitCell(-1, 0, 0),
      xtal::UnitCell(0, -1, 0),  xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, 0),   xtal::UnitCell(1, 0, 0),
      xtal::UnitCell(1, 1, 0)};

  m_orbit_site_neighborhood[3] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -1, -1, 0), xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, 0, 0),  xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, 0, -1, 0),  xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),   xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, 0),   xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 1, 0, 0),   xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 1, 0),   xtal::UnitCellCoord(3, 1, 1, 0)};

  m_orbit_neighborhood[4] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[4] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1)};

  m_orbit_neighborhood[5] = std::set<xtal::UnitCell>{xtal::UnitCell(0, 0, -1),
                                                     xtal::UnitCell(0, 0, 0),
                                                     xtal::UnitCell(0, 0, 1)};

  m_orbit_site_neighborhood[5] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, 0, 0, -1), xtal::UnitCellCoord(3, 0, 0, -1),
      xtal::UnitCellCoord(2, 0, 0, 0),  xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 0, 1),  xtal::UnitCellCoord(3, 0, 0, 1)};

  m_orbit_neighborhood[6] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, 0), xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, 1, 0),  xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(1, -1, 0),  xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 1, 0)};

  m_orbit_site_neighborhood[6] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, 0), xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -1, -2, 0), xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(2, -1, 1, 0),  xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),   xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 1, -1, 0),  xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(2, 1, 2, 0),   xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),   xtal::UnitCellCoord(3, 2, 1, 0)};

  m_orbit_neighborhood[7] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 1),
      xtal::UnitCell(-1, 0, -1),  xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 1),    xtal::UnitCell(1, 1, -1),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[7] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1)};

  m_orbit_neighborhood[8] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, -1), xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, -1, 1),  xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 1, -1),   xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 1, 1)};

  m_orbit_site_neighborhood[8] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1)};

  m_orbit_neighborhood[9] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, 0), xtal::UnitCell(-2, 0, 0),
      xtal::UnitCell(0, -2, 0),  xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 2, 0),   xtal::UnitCell(2, 0, 0),
      xtal::UnitCell(2, 2, 0)};

  m_orbit_site_neighborhood[9] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -2, 0), xtal::UnitCellCoord(3, -2, -2, 0),
      xtal::UnitCellCoord(2, -2, 0, 0),  xtal::UnitCellCoord(3, -2, 0, 0),
      xtal::UnitCellCoord(2, 0, -2, 0),  xtal::UnitCellCoord(3, 0, -2, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),   xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 2, 0),   xtal::UnitCellCoord(3, 0, 2, 0),
      xtal::UnitCellCoord(2, 2, 0, 0),   xtal::UnitCellCoord(3, 2, 0, 0),
      xtal::UnitCellCoord(2, 2, 2, 0),   xtal::UnitCellCoord(3, 2, 2, 0)};

  m_orbit_neighborhood[10] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, 0, -1),
      xtal::UnitCell(-2, 0, 0),   xtal::UnitCell(-2, 0, 1),
      xtal::UnitCell(0, -2, -1),  xtal::UnitCell(0, -2, 0),
      xtal::UnitCell(0, -2, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 2, -1),   xtal::UnitCell(0, 2, 0),
      xtal::UnitCell(0, 2, 1),    xtal::UnitCell(2, 0, -1),
      xtal::UnitCell(2, 0, 0),    xtal::UnitCell(2, 0, 1),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[10] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[11] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, -1), xtal::UnitCell(-2, -1, 1),
      xtal::UnitCell(-1, -2, -1), xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(1, -1, -1),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 1),    xtal::UnitCell(2, 1, -1),
      xtal::UnitCell(2, 1, 1)};

  m_orbit_site_neighborhood[11] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, -1), xtal::UnitCellCoord(3, -2, -1, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 1, -1, -1),  xtal::UnitCellCoord(3, 1, -1, 1),
      xtal::UnitCellCoord(2, 1, 2, -1),   xtal::UnitCellCoord(3, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 1)};

  m_orbit_neighborhood[12] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, -1), xtal::UnitCell(-2, -1, 1),
      xtal::UnitCell(-1, -2, -1), xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(1, -1, -1),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 1),    xtal::UnitCell(2, 1, -1),
      xtal::UnitCell(2, 1, 1)};

  m_orbit_site_neighborhood[12] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(2, -1, -2, -1), xtal::UnitCellCoord(3, -1, -2, 1),
      xtal::UnitCellCoord(2, -1, 1, -1),  xtal::UnitCellCoord(3, -1, 1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(2, 2, 1, -1),   xtal::UnitCellCoord(3, 2, 1, 1)};

  m_orbit_neighborhood[13] = std::set<xtal::UnitCell>{
      xtal::UnitCell(0, 0, -2), xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0), xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 0, 2)};

  m_orbit_site_neighborhood[13] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, 0, 0, -2), xtal::UnitCellCoord(2, 0, 0, -1),
      xtal::UnitCellCoord(2, 0, 0, 0),  xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 1),  xtal::UnitCellCoord(2, 0, 0, 2)};

  m_orbit_neighborhood[14] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 1),
      xtal::UnitCell(-2, 0, -1),  xtal::UnitCell(-2, 0, 1),
      xtal::UnitCell(0, -2, -1),  xtal::UnitCell(0, -2, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 2, -1),
      xtal::UnitCell(0, 2, 1),    xtal::UnitCell(2, 0, -1),
      xtal::UnitCell(2, 0, 1),    xtal::UnitCell(2, 2, -1),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[14] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -2, -1), xtal::UnitCellCoord(3, -2, -2, -1),
      xtal::UnitCellCoord(2, -2, -2, 1),  xtal::UnitCellCoord(3, -2, -2, 1),
      xtal::UnitCellCoord(2, -2, 0, -1),  xtal::UnitCellCoord(3, -2, 0, -1),
      xtal::UnitCellCoord(2, -2, 0, 1),   xtal::UnitCellCoord(3, -2, 0, 1),
      xtal::UnitCellCoord(2, 0, -2, -1),  xtal::UnitCellCoord(3, 0, -2, -1),
      xtal::UnitCellCoord(2, 0, -2, 1),   xtal::UnitCellCoord(3, 0, -2, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 2, -1),   xtal::UnitCellCoord(3, 0, 2, -1),
      xtal::UnitCellCoord(2, 0, 2, 1),    xtal::UnitCellCoord(3, 0, 2, 1),
      xtal::UnitCellCoord(2, 2, 0, -1),   xtal::UnitCellCoord(3, 2, 0, -1),
      xtal::UnitCellCoord(2, 2, 0, 1),    xtal::UnitCellCoord(3, 2, 0, 1),
      xtal::UnitCellCoord(2, 2, 2, -1),   xtal::UnitCellCoord(3, 2, 2, -1),
      xtal::UnitCellCoord(2, 2, 2, 1),    xtal::UnitCellCoord(3, 2, 2, 1)};

  m_orbit_neighborhood[15] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -2), xtal::UnitCell(-1, -1, -1),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, -1, 2),
      xtal::UnitCell(-1, 0, -2),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 1),   xtal::UnitCell(-1, 0, 2),
      xtal::UnitCell(0, -1, -2),  xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, -1, 2),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -2),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(0, 1, 2),    xtal::UnitCell(1, 0, -2),
      xtal::UnitCell(1, 0, -1),   xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 0, 2),    xtal::UnitCell(1, 1, -2),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 1),
      xtal::UnitCell(1, 1, 2)};

  m_orbit_site_neighborhood[15] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -2), xtal::UnitCellCoord(2, -1, -1, -1),
      xtal::UnitCellCoord(3, -1, -1, 1),  xtal::UnitCellCoord(2, -1, -1, 2),
      xtal::UnitCellCoord(3, -1, 0, -2),  xtal::UnitCellCoord(2, -1, 0, -1),
      xtal::UnitCellCoord(3, -1, 0, 1),   xtal::UnitCellCoord(2, -1, 0, 2),
      xtal::UnitCellCoord(3, 0, -1, -2),  xtal::UnitCellCoord(2, 0, -1, -1),
      xtal::UnitCellCoord(3, 0, -1, 1),   xtal::UnitCellCoord(2, 0, -1, 2),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -2),   xtal::UnitCellCoord(2, 0, 1, -1),
      xtal::UnitCellCoord(3, 0, 1, 1),    xtal::UnitCellCoord(2, 0, 1, 2),
      xtal::UnitCellCoord(3, 1, 0, -2),   xtal::UnitCellCoord(2, 1, 0, -1),
      xtal::UnitCellCoord(3, 1, 0, 1),    xtal::UnitCellCoord(2, 1, 0, 2),
      xtal::UnitCellCoord(3, 1, 1, -2),   xtal::UnitCellCoord(2, 1, 1, -1),
      xtal::UnitCellCoord(3, 1, 1, 1),    xtal::UnitCellCoord(2, 1, 1, 2)};

  m_orbit_neighborhood[16] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-3, -2, 0), xtal::UnitCell(-3, -1, 0),
      xtal::UnitCell(-2, -3, 0), xtal::UnitCell(-2, 1, 0),
      xtal::UnitCell(-1, -3, 0), xtal::UnitCell(-1, 2, 0),
      xtal::UnitCell(0, 0, 0),   xtal::UnitCell(1, -2, 0),
      xtal::UnitCell(1, 3, 0),   xtal::UnitCell(2, -1, 0),
      xtal::UnitCell(2, 3, 0),   xtal::UnitCell(3, 1, 0),
      xtal::UnitCell(3, 2, 0)};

  m_orbit_site_neighborhood[16] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -3, -2, 0), xtal::UnitCellCoord(3, -3, -2, 0),
      xtal::UnitCellCoord(2, -3, -1, 0), xtal::UnitCellCoord(3, -3, -1, 0),
      xtal::UnitCellCoord(2, -2, -3, 0), xtal::UnitCellCoord(3, -2, -3, 0),
      xtal::UnitCellCoord(2, -2, 1, 0),  xtal::UnitCellCoord(3, -2, 1, 0),
      xtal::UnitCellCoord(2, -1, -3, 0), xtal::UnitCellCoord(3, -1, -3, 0),
      xtal::UnitCellCoord(2, -1, 2, 0),  xtal::UnitCellCoord(3, -1, 2, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),   xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 1, -2, 0),  xtal::UnitCellCoord(3, 1, -2, 0),
      xtal::UnitCellCoord(2, 1, 3, 0),   xtal::UnitCellCoord(3, 1, 3, 0),
      xtal::UnitCellCoord(2, 2, -1, 0),  xtal::UnitCellCoord(3, 2, -1, 0),
      xtal::UnitCellCoord(2, 2, 3, 0),   xtal::UnitCellCoord(3, 2, 3, 0),
      xtal::UnitCellCoord(2, 3, 1, 0),   xtal::UnitCellCoord(3, 3, 1, 0),
      xtal::UnitCellCoord(2, 3, 2, 0),   xtal::UnitCellCoord(3, 3, 2, 0)};

  m_orbit_neighborhood[17] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-3, -2, -1), xtal::UnitCell(-3, -2, 0),
      xtal::UnitCell(-3, -2, 1),  xtal::UnitCell(-3, -1, -1),
      xtal::UnitCell(-3, -1, 0),  xtal::UnitCell(-3, -1, 1),
      xtal::UnitCell(-2, -3, -1), xtal::UnitCell(-2, -3, 0),
      xtal::UnitCell(-2, -3, 1),  xtal::UnitCell(-2, 1, -1),
      xtal::UnitCell(-2, 1, 0),   xtal::UnitCell(-2, 1, 1),
      xtal::UnitCell(-1, -3, -1), xtal::UnitCell(-1, -3, 0),
      xtal::UnitCell(-1, -3, 1),  xtal::UnitCell(-1, 2, -1),
      xtal::UnitCell(-1, 2, 0),   xtal::UnitCell(-1, 2, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(1, -2, -1),
      xtal::UnitCell(1, -2, 0),   xtal::UnitCell(1, -2, 1),
      xtal::UnitCell(1, 3, -1),   xtal::UnitCell(1, 3, 0),
      xtal::UnitCell(1, 3, 1),    xtal::UnitCell(2, -1, -1),
      xtal::UnitCell(2, -1, 0),   xtal::UnitCell(2, -1, 1),
      xtal::UnitCell(2, 3, -1),   xtal::UnitCell(2, 3, 0),
      xtal::UnitCell(2, 3, 1),    xtal::UnitCell(3, 1, -1),
      xtal::UnitCell(3, 1, 0),    xtal::UnitCell(3, 1, 1),
      xtal::UnitCell(3, 2, -1),   xtal::UnitCell(3, 2, 0),
      xtal::UnitCell(3, 2, 1)};

  m_orbit_site_neighborhood[17] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -3, -2, -1), xtal::UnitCellCoord(2, -3, -2, 0),
      xtal::UnitCellCoord(3, -3, -2, 0),  xtal::UnitCellCoord(2, -3, -2, 1),
      xtal::UnitCellCoord(3, -3, -1, -1), xtal::UnitCellCoord(2, -3, -1, 0),
      xtal::UnitCellCoord(3, -3, -1, 0),  xtal::UnitCellCoord(2, -3, -1, 1),
      xtal::UnitCellCoord(3, -2, -3, -1), xtal::UnitCellCoord(2, -2, -3, 0),
      xtal::UnitCellCoord(3, -2, -3, 0),  xtal::UnitCellCoord(2, -2, -3, 1),
      xtal::UnitCellCoord(3, -2, 1, -1),  xtal::UnitCellCoord(2, -2, 1, 0),
      xtal::UnitCellCoord(3, -2, 1, 0),   xtal::UnitCellCoord(2, -2, 1, 1),
      xtal::UnitCellCoord(3, -1, -3, -1), xtal::UnitCellCoord(2, -1, -3, 0),
      xtal::UnitCellCoord(3, -1, -3, 0),  xtal::UnitCellCoord(2, -1, -3, 1),
      xtal::UnitCellCoord(3, -1, 2, -1),  xtal::UnitCellCoord(2, -1, 2, 0),
      xtal::UnitCellCoord(3, -1, 2, 0),   xtal::UnitCellCoord(2, -1, 2, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 1, -2, -1),  xtal::UnitCellCoord(2, 1, -2, 0),
      xtal::UnitCellCoord(3, 1, -2, 0),   xtal::UnitCellCoord(2, 1, -2, 1),
      xtal::UnitCellCoord(3, 1, 3, -1),   xtal::UnitCellCoord(2, 1, 3, 0),
      xtal::UnitCellCoord(3, 1, 3, 0),    xtal::UnitCellCoord(2, 1, 3, 1),
      xtal::UnitCellCoord(3, 2, -1, -1),  xtal::UnitCellCoord(2, 2, -1, 0),
      xtal::UnitCellCoord(3, 2, -1, 0),   xtal::UnitCellCoord(2, 2, -1, 1),
      xtal::UnitCellCoord(3, 2, 3, -1),   xtal::UnitCellCoord(2, 2, 3, 0),
      xtal::UnitCellCoord(3, 2, 3, 0),    xtal::UnitCellCoord(2, 2, 3, 1),
      xtal::UnitCellCoord(3, 3, 1, -1),   xtal::UnitCellCoord(2, 3, 1, 0),
      xtal::UnitCellCoord(3, 3, 1, 0),    xtal::UnitCellCoord(2, 3, 1, 1),
      xtal::UnitCellCoord(3, 3, 2, -1),   xtal::UnitCellCoord(2, 3, 2, 0),
      xtal::UnitCellCoord(3, 3, 2, 0),    xtal::UnitCellCoord(2, 3, 2, 1)};

  m_orbit_neighborhood[18] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, 0), xtal::UnitCell(-1, 0, 0),
      xtal::UnitCell(0, -1, 0),  xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, 0),   xtal::UnitCell(1, 0, 0),
      xtal::UnitCell(1, 1, 0)};

  m_orbit_site_neighborhood[18] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -1, -1, 0), xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, 0, 0),  xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, 0, -1, 0),  xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),   xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, 0),   xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 1, 0, 0),   xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 1, 0),   xtal::UnitCellCoord(3, 1, 1, 0)};

  m_orbit_neighborhood[19] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[19] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(3, 0, 0, -1),   xtal::UnitCellCoord(2, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 0),    xtal::UnitCellCoord(2, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1)};

  m_orbit_neighborhood[20] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[20] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1)};

  m_orbit_neighborhood[21] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[21] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1)};

  m_orbit_neighborhood[22] = std::set<xtal::UnitCell>{xtal::UnitCell(0, 0, -1),
                                                      xtal::UnitCell(0, 0, 0),
                                                      xtal::UnitCell(0, 0, 1)};

  m_orbit_site_neighborhood[22] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, 0, 0, -1), xtal::UnitCellCoord(3, 0, 0, -1),
      xtal::UnitCellCoord(2, 0, 0, 0),  xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 0, 1),  xtal::UnitCellCoord(3, 0, 0, 1)};

  m_orbit_neighborhood[23] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[23] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, -1),   xtal::UnitCellCoord(3, 0, 0, -1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 0, 1),    xtal::UnitCellCoord(3, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1)};

  m_orbit_neighborhood[24] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, 0), xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, 0), xtal::UnitCell(-1, 0, 0),
      xtal::UnitCell(-1, 1, 0),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, 0, 0),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(1, -1, 0),  xtal::UnitCell(1, 0, 0),
      xtal::UnitCell(1, 1, 0),   xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 1, 0)};

  m_orbit_site_neighborhood[24] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, 0), xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -1, -2, 0), xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(2, -1, -1, 0), xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, 0, 0),  xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, -1, 1, 0),  xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(2, 0, -1, 0),  xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),   xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, 0),   xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 1, -1, 0),  xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(2, 1, 0, 0),   xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 1, 0),   xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 1, 2, 0),   xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),   xtal::UnitCellCoord(3, 2, 1, 0)};

  m_orbit_neighborhood[25] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, 0),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(1, -1, 0),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 1, 0)};

  m_orbit_site_neighborhood[25] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, 0),  xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -1, -2, 0),  xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(2, -1, 1, 0),   xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, -1, 0),   xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(2, 1, 2, 0),    xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),    xtal::UnitCellCoord(3, 2, 1, 0)};

  m_orbit_neighborhood[26] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, 0), xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, 1, 0),  xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(1, -1, 0),  xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 1, 0)};

  m_orbit_site_neighborhood[26] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, 0), xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -1, -2, 0), xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(2, -1, 1, 0),  xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),   xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 1, -1, 0),  xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(2, 1, 2, 0),   xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),   xtal::UnitCellCoord(3, 2, 1, 0)};

  m_orbit_neighborhood[27] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[27] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 0),  xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 0),   xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 0),   xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(3, 0, 0, -1),   xtal::UnitCellCoord(2, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 0),    xtal::UnitCellCoord(2, 0, 0, 1),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 0),    xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 0),    xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 0),    xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1)};

  m_orbit_neighborhood[28] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[28] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 0),  xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 0),   xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 0),   xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 0),    xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 0),    xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 0),    xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1)};

  m_orbit_neighborhood[29] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[29] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 0),  xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 0),   xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 0),   xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, -1),   xtal::UnitCellCoord(3, 0, 0, -1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 0, 1),    xtal::UnitCellCoord(3, 0, 0, 1),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 0),    xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 0),    xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 0),    xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1)};

  m_orbit_neighborhood[30] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[30] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 0),  xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 0),   xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 0),   xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 0),    xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 0),    xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 0),    xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1)};

  m_orbit_neighborhood[31] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[31] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 0),  xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 0),   xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 0),   xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 0),    xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 0),    xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 0),    xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1)};

  m_orbit_neighborhood[32] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 1),
      xtal::UnitCell(-1, 0, -1),  xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, 0),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(1, -1, 0),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 1),    xtal::UnitCell(1, 1, -1),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 1, 0)};

  m_orbit_site_neighborhood[32] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, 0),  xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -1, -2, 0),  xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(2, -1, 1, 0),   xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, -1, 0),   xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1),
      xtal::UnitCellCoord(2, 1, 2, 0),    xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),    xtal::UnitCellCoord(3, 2, 1, 0)};

  m_orbit_neighborhood[33] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, -1), xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, -1, 1),  xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 1, -1),   xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 1, 1)};

  m_orbit_site_neighborhood[33] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1)};

  m_orbit_neighborhood[34] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, -1), xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, -1, 1),  xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 1, -1),   xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 1, 1)};

  m_orbit_site_neighborhood[34] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1)};

  m_orbit_neighborhood[35] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, -1), xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, -1, 1),  xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 1, -1),   xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 1, 1)};

  m_orbit_site_neighborhood[35] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(3, 0, 0, -1),   xtal::UnitCellCoord(2, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 0),    xtal::UnitCellCoord(2, 0, 0, 1),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1)};

  m_orbit_neighborhood[36] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, -1), xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, -1, 1),  xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 1, -1),   xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 1, 1)};

  m_orbit_site_neighborhood[36] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(3, 0, 0, -1),   xtal::UnitCellCoord(2, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 0),    xtal::UnitCellCoord(2, 0, 0, 1),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1)};

  m_orbit_neighborhood[37] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, -1), xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, -1, 1),  xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 1, -1),   xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 1, 1)};

  m_orbit_site_neighborhood[37] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 0),  xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 0),   xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 0),   xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 0),    xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 0),    xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 0),    xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1)};

  m_orbit_neighborhood[38] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, -1), xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, -1, 1),  xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 1, -1),   xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 1, 1)};

  m_orbit_site_neighborhood[38] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 0),  xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 0),   xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 0),   xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 0),    xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 0),    xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 0),    xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1)};

  m_orbit_neighborhood[39] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, -1), xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, -1, 1),  xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 1, -1),   xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 1, 1)};

  m_orbit_site_neighborhood[39] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(2, 0, 0, -1),   xtal::UnitCellCoord(3, 0, 0, -1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 0, 1),    xtal::UnitCellCoord(3, 0, 0, 1),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1)};

  m_orbit_neighborhood[40] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, -1), xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, -1, 1),  xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 1, -1),   xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 1, 1)};

  m_orbit_site_neighborhood[40] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1)};

  m_orbit_neighborhood[41] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, 0), xtal::UnitCell(-2, 0, 0),
      xtal::UnitCell(-1, -1, 0), xtal::UnitCell(-1, 0, 0),
      xtal::UnitCell(0, -2, 0),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, 0, 0),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 2, 0),   xtal::UnitCell(1, 0, 0),
      xtal::UnitCell(1, 1, 0),   xtal::UnitCell(2, 0, 0),
      xtal::UnitCell(2, 2, 0)};

  m_orbit_site_neighborhood[41] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -2, 0), xtal::UnitCellCoord(3, -2, -2, 0),
      xtal::UnitCellCoord(2, -2, 0, 0),  xtal::UnitCellCoord(3, -2, 0, 0),
      xtal::UnitCellCoord(2, -1, -1, 0), xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, 0, 0),  xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, 0, -2, 0),  xtal::UnitCellCoord(3, 0, -2, 0),
      xtal::UnitCellCoord(2, 0, -1, 0),  xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),   xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, 0),   xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 0, 2, 0),   xtal::UnitCellCoord(3, 0, 2, 0),
      xtal::UnitCellCoord(2, 1, 0, 0),   xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 1, 0),   xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 2, 0, 0),   xtal::UnitCellCoord(3, 2, 0, 0),
      xtal::UnitCellCoord(2, 2, 2, 0),   xtal::UnitCellCoord(3, 2, 2, 0)};

  m_orbit_neighborhood[42] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, 0),  xtal::UnitCell(-2, 0, 0),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -2, 0),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(0, 2, 0),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(2, 0, 0),
      xtal::UnitCell(2, 2, 0)};

  m_orbit_site_neighborhood[42] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -2, 0),  xtal::UnitCellCoord(3, -2, -2, 0),
      xtal::UnitCellCoord(2, -2, 0, 0),   xtal::UnitCellCoord(3, -2, 0, 0),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(2, 0, -2, 0),   xtal::UnitCellCoord(3, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(2, 0, 2, 0),    xtal::UnitCellCoord(3, 0, 2, 0),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(2, 2, 0, 0),    xtal::UnitCellCoord(3, 2, 0, 0),
      xtal::UnitCellCoord(2, 2, 2, 0),    xtal::UnitCellCoord(3, 2, 2, 0)};

  m_orbit_neighborhood[43] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, 0), xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, 0, 0),  xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, 0), xtal::UnitCell(-1, 0, 0),
      xtal::UnitCell(-1, 1, 0),  xtal::UnitCell(0, -2, 0),
      xtal::UnitCell(0, -1, 0),  xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, 0),   xtal::UnitCell(0, 2, 0),
      xtal::UnitCell(1, -1, 0),  xtal::UnitCell(1, 0, 0),
      xtal::UnitCell(1, 1, 0),   xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 0, 0),   xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 2, 0)};

  m_orbit_site_neighborhood[43] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -2, 0), xtal::UnitCellCoord(3, -2, -2, 0),
      xtal::UnitCellCoord(2, -2, -1, 0), xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -2, 0, 0),  xtal::UnitCellCoord(3, -2, 0, 0),
      xtal::UnitCellCoord(2, -1, -2, 0), xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(2, -1, -1, 0), xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, 0, 0),  xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, -1, 1, 0),  xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(2, 0, -2, 0),  xtal::UnitCellCoord(3, 0, -2, 0),
      xtal::UnitCellCoord(2, 0, -1, 0),  xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),   xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, 0),   xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 0, 2, 0),   xtal::UnitCellCoord(3, 0, 2, 0),
      xtal::UnitCellCoord(2, 1, -1, 0),  xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(2, 1, 0, 0),   xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 1, 0),   xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 1, 2, 0),   xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 0, 0),   xtal::UnitCellCoord(3, 2, 0, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),   xtal::UnitCellCoord(3, 2, 1, 0),
      xtal::UnitCellCoord(2, 2, 2, 0),   xtal::UnitCellCoord(3, 2, 2, 0)};

  m_orbit_neighborhood[44] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, 0),  xtal::UnitCell(-2, 0, 0),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 1),
      xtal::UnitCell(-1, 0, -1),  xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -2, 0),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(0, 2, 0),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 1),    xtal::UnitCell(1, 1, -1),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(2, 0, 0),
      xtal::UnitCell(2, 2, 0)};

  m_orbit_site_neighborhood[44] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -2, 0),  xtal::UnitCellCoord(3, -2, -2, 0),
      xtal::UnitCellCoord(2, -2, 0, 0),   xtal::UnitCellCoord(3, -2, 0, 0),
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(2, 0, -2, 0),   xtal::UnitCellCoord(3, 0, -2, 0),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(2, 0, 2, 0),    xtal::UnitCellCoord(3, 0, 2, 0),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1),
      xtal::UnitCellCoord(2, 2, 0, 0),    xtal::UnitCellCoord(3, 2, 0, 0),
      xtal::UnitCellCoord(2, 2, 2, 0),    xtal::UnitCellCoord(3, 2, 2, 0)};

  m_orbit_neighborhood[45] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, 0),  xtal::UnitCell(-2, -1, -1),
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-2, -1, 1),
      xtal::UnitCell(-2, 0, 0),   xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, -2, 0),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(0, 2, 0),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 0, 0),    xtal::UnitCell(2, 1, -1),
      xtal::UnitCell(2, 1, 0),    xtal::UnitCell(2, 1, 1),
      xtal::UnitCell(2, 2, 0)};

  m_orbit_site_neighborhood[45] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -2, 0),  xtal::UnitCellCoord(3, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(2, -2, 0, 0),   xtal::UnitCellCoord(3, -2, 0, 0),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(2, 0, -2, 0),   xtal::UnitCellCoord(3, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(2, 0, 2, 0),    xtal::UnitCellCoord(3, 0, 2, 0),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(2, 2, 0, 0),    xtal::UnitCellCoord(3, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1),
      xtal::UnitCellCoord(2, 2, 2, 0),    xtal::UnitCellCoord(3, 2, 2, 0)};

  m_orbit_neighborhood[46] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, 0),  xtal::UnitCell(-2, -1, -1),
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-2, -1, 1),
      xtal::UnitCell(-2, 0, 0),   xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, -2, 0),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(0, 2, 0),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 0, 0),    xtal::UnitCell(2, 1, -1),
      xtal::UnitCell(2, 1, 0),    xtal::UnitCell(2, 1, 1),
      xtal::UnitCell(2, 2, 0)};

  m_orbit_site_neighborhood[46] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -2, 0),  xtal::UnitCellCoord(3, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(2, -2, 0, 0),   xtal::UnitCellCoord(3, -2, 0, 0),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(2, 0, -2, 0),   xtal::UnitCellCoord(3, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(2, 0, 2, 0),    xtal::UnitCellCoord(3, 0, 2, 0),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(2, 2, 0, 0),    xtal::UnitCellCoord(3, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1),
      xtal::UnitCellCoord(2, 2, 2, 0),    xtal::UnitCellCoord(3, 2, 2, 0)};

  m_orbit_neighborhood[47] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, 0), xtal::UnitCell(-2, 0, 0),
      xtal::UnitCell(0, -2, 0),  xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 2, 0),   xtal::UnitCell(2, 0, 0),
      xtal::UnitCell(2, 2, 0)};

  m_orbit_site_neighborhood[47] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -2, 0), xtal::UnitCellCoord(3, -2, -2, 0),
      xtal::UnitCellCoord(2, -2, 0, 0),  xtal::UnitCellCoord(3, -2, 0, 0),
      xtal::UnitCellCoord(2, 0, -2, 0),  xtal::UnitCellCoord(3, 0, -2, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),   xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 2, 0),   xtal::UnitCellCoord(3, 0, 2, 0),
      xtal::UnitCellCoord(2, 2, 0, 0),   xtal::UnitCellCoord(3, 2, 0, 0),
      xtal::UnitCellCoord(2, 2, 2, 0),   xtal::UnitCellCoord(3, 2, 2, 0)};

  m_orbit_neighborhood[48] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, 0, -1),
      xtal::UnitCell(-2, 0, 0),   xtal::UnitCell(-2, 0, 1),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -2, -1),  xtal::UnitCell(0, -2, 0),
      xtal::UnitCell(0, -2, 1),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(0, 2, -1),   xtal::UnitCell(0, 2, 0),
      xtal::UnitCell(0, 2, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(2, 0, -1),
      xtal::UnitCell(2, 0, 0),    xtal::UnitCell(2, 0, 1),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[48] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[49] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, 0, -1),  xtal::UnitCell(-2, 0, 0),
      xtal::UnitCell(-2, 0, 1),   xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, 0),   xtal::UnitCell(0, -2, -1),
      xtal::UnitCell(0, -2, 0),   xtal::UnitCell(0, -2, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(0, 2, -1),
      xtal::UnitCell(0, 2, 0),    xtal::UnitCell(0, 2, 1),
      xtal::UnitCell(1, -1, 0),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 0, -1),   xtal::UnitCell(2, 0, 0),
      xtal::UnitCell(2, 0, 1),    xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[49] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(2, -2, -1, 0),  xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(2, -1, -2, 0),  xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(2, -1, 1, 0),   xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(2, 1, -1, 0),   xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(2, 1, 2, 0),    xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(2, 2, 1, 0),    xtal::UnitCellCoord(3, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[50] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, -1, 0),
      xtal::UnitCell(-2, 0, -1),  xtal::UnitCell(-2, 0, 0),
      xtal::UnitCell(-2, 0, 1),   xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, 0),   xtal::UnitCell(0, -2, -1),
      xtal::UnitCell(0, -2, 0),   xtal::UnitCell(0, -2, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(0, 2, -1),
      xtal::UnitCell(0, 2, 0),    xtal::UnitCell(0, 2, 1),
      xtal::UnitCell(1, -1, 0),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 0, -1),   xtal::UnitCell(2, 0, 0),
      xtal::UnitCell(2, 0, 1),    xtal::UnitCell(2, 1, 0),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[50] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(2, -2, -1, 0),  xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(2, -1, -2, 0),  xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(2, -1, 1, 0),   xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(2, 1, -1, 0),   xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(2, 1, 2, 0),    xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(2, 2, 1, 0),    xtal::UnitCellCoord(3, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[51] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, 0, -1),
      xtal::UnitCell(-2, 0, 0),   xtal::UnitCell(-2, 0, 1),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -2, -1),  xtal::UnitCell(0, -2, 0),
      xtal::UnitCell(0, -2, 1),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(0, 2, -1),   xtal::UnitCell(0, 2, 0),
      xtal::UnitCell(0, 2, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(2, 0, -1),
      xtal::UnitCell(2, 0, 0),    xtal::UnitCell(2, 0, 1),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[51] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 0),  xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 0),   xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 0),   xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 0),    xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 0),    xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 0),    xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[52] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, -1, -1),
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-2, -1, 1),
      xtal::UnitCell(-2, 0, -1),  xtal::UnitCell(-2, 0, 0),
      xtal::UnitCell(-2, 0, 1),   xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, -1, 0),  xtal::UnitCell(-1, 0, 0),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, -2, -1),
      xtal::UnitCell(0, -2, 0),   xtal::UnitCell(0, -2, 1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 2, -1),
      xtal::UnitCell(0, 2, 0),    xtal::UnitCell(0, 2, 1),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 0, 0),
      xtal::UnitCell(1, 1, 0),    xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 0, -1),   xtal::UnitCell(2, 0, 0),
      xtal::UnitCell(2, 0, 1),    xtal::UnitCell(2, 1, -1),
      xtal::UnitCell(2, 1, 0),    xtal::UnitCell(2, 1, 1),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[52] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(2, -1, -1, 0),  xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, 0, 0),   xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(2, 0, -1, 0),   xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, 0),    xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(2, 1, 0, 0),    xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 1, 0),    xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[53] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, -1, -1),
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-2, -1, 1),
      xtal::UnitCell(-2, 0, -1),  xtal::UnitCell(-2, 0, 0),
      xtal::UnitCell(-2, 0, 1),   xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, -1, 0),  xtal::UnitCell(-1, 0, 0),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, -2, -1),
      xtal::UnitCell(0, -2, 0),   xtal::UnitCell(0, -2, 1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 2, -1),
      xtal::UnitCell(0, 2, 0),    xtal::UnitCell(0, 2, 1),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 0, 0),
      xtal::UnitCell(1, 1, 0),    xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 0, -1),   xtal::UnitCell(2, 0, 0),
      xtal::UnitCell(2, 0, 1),    xtal::UnitCell(2, 1, -1),
      xtal::UnitCell(2, 1, 0),    xtal::UnitCell(2, 1, 1),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[53] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(2, -1, -1, 0),  xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, 0, 0),   xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(2, 0, -1, 0),   xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, 0),    xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(2, 1, 0, 0),    xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 1, 0),    xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[54] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, -1, -1),
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-2, -1, 1),
      xtal::UnitCell(-2, 0, -1),  xtal::UnitCell(-2, 0, 0),
      xtal::UnitCell(-2, 0, 1),   xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 1),
      xtal::UnitCell(-1, 0, -1),  xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, -2, -1),
      xtal::UnitCell(0, -2, 0),   xtal::UnitCell(0, -2, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(0, 2, -1),
      xtal::UnitCell(0, 2, 0),    xtal::UnitCell(0, 2, 1),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 1),    xtal::UnitCell(1, 1, -1),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 0, -1),   xtal::UnitCell(2, 0, 0),
      xtal::UnitCell(2, 0, 1),    xtal::UnitCell(2, 1, -1),
      xtal::UnitCell(2, 1, 0),    xtal::UnitCell(2, 1, 1),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[54] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[55] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, -1, -1),
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-2, -1, 1),
      xtal::UnitCell(-2, 0, -1),  xtal::UnitCell(-2, 0, 0),
      xtal::UnitCell(-2, 0, 1),   xtal::UnitCell(-1, -2, -1),
      xtal::UnitCell(-1, -2, 0),  xtal::UnitCell(-1, -2, 1),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 1),
      xtal::UnitCell(-1, 0, -1),  xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, -1),  xtal::UnitCell(-1, 1, 0),
      xtal::UnitCell(-1, 1, 1),   xtal::UnitCell(0, -2, -1),
      xtal::UnitCell(0, -2, 0),   xtal::UnitCell(0, -2, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(0, 2, -1),
      xtal::UnitCell(0, 2, 0),    xtal::UnitCell(0, 2, 1),
      xtal::UnitCell(1, -1, -1),  xtal::UnitCell(1, -1, 0),
      xtal::UnitCell(1, -1, 1),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 1),    xtal::UnitCell(1, 1, -1),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, -1),
      xtal::UnitCell(1, 2, 0),    xtal::UnitCell(1, 2, 1),
      xtal::UnitCell(2, 0, -1),   xtal::UnitCell(2, 0, 0),
      xtal::UnitCell(2, 0, 1),    xtal::UnitCell(2, 1, -1),
      xtal::UnitCell(2, 1, 0),    xtal::UnitCell(2, 1, 1),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[55] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(3, -2, -1, -1), xtal::UnitCellCoord(2, -2, -1, 0),
      xtal::UnitCellCoord(3, -2, -1, 0),  xtal::UnitCellCoord(2, -2, -1, 1),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(3, -1, -2, -1), xtal::UnitCellCoord(2, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -2, 0),  xtal::UnitCellCoord(2, -1, -2, 1),
      xtal::UnitCellCoord(2, -1, -1, -1), xtal::UnitCellCoord(3, -1, -1, -1),
      xtal::UnitCellCoord(2, -1, -1, 1),  xtal::UnitCellCoord(3, -1, -1, 1),
      xtal::UnitCellCoord(2, -1, 0, -1),  xtal::UnitCellCoord(3, -1, 0, -1),
      xtal::UnitCellCoord(2, -1, 0, 1),   xtal::UnitCellCoord(3, -1, 0, 1),
      xtal::UnitCellCoord(3, -1, 1, -1),  xtal::UnitCellCoord(2, -1, 1, 0),
      xtal::UnitCellCoord(3, -1, 1, 0),   xtal::UnitCellCoord(2, -1, 1, 1),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(2, 0, -1, -1),  xtal::UnitCellCoord(3, 0, -1, -1),
      xtal::UnitCellCoord(2, 0, -1, 1),   xtal::UnitCellCoord(3, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, -1),   xtal::UnitCellCoord(3, 0, 1, -1),
      xtal::UnitCellCoord(2, 0, 1, 1),    xtal::UnitCellCoord(3, 0, 1, 1),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(3, 1, -1, -1),  xtal::UnitCellCoord(2, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, -1, 0),   xtal::UnitCellCoord(2, 1, -1, 1),
      xtal::UnitCellCoord(2, 1, 0, -1),   xtal::UnitCellCoord(3, 1, 0, -1),
      xtal::UnitCellCoord(2, 1, 0, 1),    xtal::UnitCellCoord(3, 1, 0, 1),
      xtal::UnitCellCoord(2, 1, 1, -1),   xtal::UnitCellCoord(3, 1, 1, -1),
      xtal::UnitCellCoord(2, 1, 1, 1),    xtal::UnitCellCoord(3, 1, 1, 1),
      xtal::UnitCellCoord(3, 1, 2, -1),   xtal::UnitCellCoord(2, 1, 2, 0),
      xtal::UnitCellCoord(3, 1, 2, 0),    xtal::UnitCellCoord(2, 1, 2, 1),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(3, 2, 1, -1),   xtal::UnitCellCoord(2, 2, 1, 0),
      xtal::UnitCellCoord(3, 2, 1, 0),    xtal::UnitCellCoord(2, 2, 1, 1),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[56] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, 0, -1),
      xtal::UnitCell(-2, 0, 0),   xtal::UnitCell(-2, 0, 1),
      xtal::UnitCell(0, -2, -1),  xtal::UnitCell(0, -2, 0),
      xtal::UnitCell(0, -2, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 2, -1),   xtal::UnitCell(0, 2, 0),
      xtal::UnitCell(0, 2, 1),    xtal::UnitCell(2, 0, -1),
      xtal::UnitCell(2, 0, 0),    xtal::UnitCell(2, 0, 1),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[56] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(3, 0, 0, -1),   xtal::UnitCellCoord(2, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 0),    xtal::UnitCellCoord(2, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[57] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, 0, -1),
      xtal::UnitCell(-2, 0, 0),   xtal::UnitCell(-2, 0, 1),
      xtal::UnitCell(0, -2, -1),  xtal::UnitCell(0, -2, 0),
      xtal::UnitCell(0, -2, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 2, -1),   xtal::UnitCell(0, 2, 0),
      xtal::UnitCell(0, 2, 1),    xtal::UnitCell(2, 0, -1),
      xtal::UnitCell(2, 0, 0),    xtal::UnitCell(2, 0, 1),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[57] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(2, 0, 0, -1),   xtal::UnitCellCoord(3, 0, 0, -1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 0, 1),    xtal::UnitCellCoord(3, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[58] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, 0, -1),
      xtal::UnitCell(-2, 0, 0),   xtal::UnitCell(-2, 0, 1),
      xtal::UnitCell(0, -2, -1),  xtal::UnitCell(0, -2, 0),
      xtal::UnitCell(0, -2, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 2, -1),   xtal::UnitCell(0, 2, 0),
      xtal::UnitCell(0, 2, 1),    xtal::UnitCell(2, 0, -1),
      xtal::UnitCell(2, 0, 0),    xtal::UnitCell(2, 0, 1),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[58] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[59] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -2, -1), xtal::UnitCell(-2, -2, 0),
      xtal::UnitCell(-2, -2, 1),  xtal::UnitCell(-2, 0, -1),
      xtal::UnitCell(-2, 0, 0),   xtal::UnitCell(-2, 0, 1),
      xtal::UnitCell(0, -2, -1),  xtal::UnitCell(0, -2, 0),
      xtal::UnitCell(0, -2, 1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 2, -1),   xtal::UnitCell(0, 2, 0),
      xtal::UnitCell(0, 2, 1),    xtal::UnitCell(2, 0, -1),
      xtal::UnitCell(2, 0, 0),    xtal::UnitCell(2, 0, 1),
      xtal::UnitCell(2, 2, -1),   xtal::UnitCell(2, 2, 0),
      xtal::UnitCell(2, 2, 1)};

  m_orbit_site_neighborhood[59] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -2, -2, -1), xtal::UnitCellCoord(2, -2, -2, 0),
      xtal::UnitCellCoord(3, -2, -2, 0),  xtal::UnitCellCoord(2, -2, -2, 1),
      xtal::UnitCellCoord(3, -2, 0, -1),  xtal::UnitCellCoord(2, -2, 0, 0),
      xtal::UnitCellCoord(3, -2, 0, 0),   xtal::UnitCellCoord(2, -2, 0, 1),
      xtal::UnitCellCoord(3, 0, -2, -1),  xtal::UnitCellCoord(2, 0, -2, 0),
      xtal::UnitCellCoord(3, 0, -2, 0),   xtal::UnitCellCoord(2, 0, -2, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 2, -1),   xtal::UnitCellCoord(2, 0, 2, 0),
      xtal::UnitCellCoord(3, 0, 2, 0),    xtal::UnitCellCoord(2, 0, 2, 1),
      xtal::UnitCellCoord(3, 2, 0, -1),   xtal::UnitCellCoord(2, 2, 0, 0),
      xtal::UnitCellCoord(3, 2, 0, 0),    xtal::UnitCellCoord(2, 2, 0, 1),
      xtal::UnitCellCoord(3, 2, 2, -1),   xtal::UnitCellCoord(2, 2, 2, 0),
      xtal::UnitCellCoord(3, 2, 2, 0),    xtal::UnitCellCoord(2, 2, 2, 1)};

  m_orbit_neighborhood[60] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[60] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(3, 0, 0, -1),   xtal::UnitCellCoord(2, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 0),    xtal::UnitCellCoord(2, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1)};

  m_orbit_neighborhood[61] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[61] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(3, 0, 0, -1),   xtal::UnitCellCoord(2, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 0),    xtal::UnitCellCoord(2, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1)};

  m_orbit_neighborhood[62] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[62] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(3, 0, 0, -1),   xtal::UnitCellCoord(2, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 0),    xtal::UnitCellCoord(2, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1)};

  m_orbit_neighborhood[63] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[63] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(3, 0, 0, -1),   xtal::UnitCellCoord(2, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 0),    xtal::UnitCellCoord(2, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1)};

  m_orbit_neighborhood[64] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[64] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(3, 0, 0, -1),   xtal::UnitCellCoord(2, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 0),    xtal::UnitCellCoord(2, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1)};

  m_orbit_neighborhood[65] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[65] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, -1),   xtal::UnitCellCoord(3, 0, 0, -1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 0, 1),    xtal::UnitCellCoord(3, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1)};

  m_orbit_neighborhood[66] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(0, -1, -1),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, -1, 1),   xtal::UnitCell(0, 0, -1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 0, 1),
      xtal::UnitCell(0, 1, -1),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(0, 1, 1),    xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1)};

  m_orbit_site_neighborhood[66] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, -1),   xtal::UnitCellCoord(3, 0, 0, -1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 0, 1),    xtal::UnitCellCoord(3, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1)};

  m_orbit_neighborhood[67] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, 0), xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, 0), xtal::UnitCell(-1, 0, 0),
      xtal::UnitCell(-1, 1, 0),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, 0, 0),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(1, -1, 0),  xtal::UnitCell(1, 0, 0),
      xtal::UnitCell(1, 1, 0),   xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 1, 0)};

  m_orbit_site_neighborhood[67] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, 0), xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -1, -2, 0), xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(2, -1, -1, 0), xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, 0, 0),  xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, -1, 1, 0),  xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(2, 0, -1, 0),  xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),   xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, 0),   xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 1, -1, 0),  xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(2, 1, 0, 0),   xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 1, 0),   xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 1, 2, 0),   xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),   xtal::UnitCellCoord(3, 2, 1, 0)};

  m_orbit_neighborhood[68] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, 0),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, -1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 0, 1),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(1, -1, 0),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 1, 0)};

  m_orbit_site_neighborhood[68] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, 0),  xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -1, -2, 0),  xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(2, -1, 1, 0),   xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(3, 0, 0, -1),   xtal::UnitCellCoord(2, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 0, 0),    xtal::UnitCellCoord(2, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, -1, 0),   xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(2, 1, 2, 0),    xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),    xtal::UnitCellCoord(3, 2, 1, 0)};

  m_orbit_neighborhood[69] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, 0),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(1, -1, 0),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 1, 0)};

  m_orbit_site_neighborhood[69] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, 0),  xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -1, -2, 0),  xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(2, -1, 1, 0),   xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, -1, 0),   xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(2, 1, 2, 0),    xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),    xtal::UnitCellCoord(3, 2, 1, 0)};

  m_orbit_neighborhood[70] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, 0),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(1, -1, 0),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 1, 0)};

  m_orbit_site_neighborhood[70] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, 0),  xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -1, -2, 0),  xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(2, -1, 1, 0),   xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, -1, 0),   xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(2, 1, 2, 0),    xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),    xtal::UnitCellCoord(3, 2, 1, 0)};

  m_orbit_neighborhood[71] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, 0),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, -1),   xtal::UnitCell(0, 0, 0),
      xtal::UnitCell(0, 0, 1),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(1, -1, 0),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 1, 0)};

  m_orbit_site_neighborhood[71] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, 0),  xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -1, -2, 0),  xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(2, -1, 1, 0),   xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, -1),   xtal::UnitCellCoord(3, 0, 0, -1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 0, 1),    xtal::UnitCellCoord(3, 0, 0, 1),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, -1, 0),   xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(2, 1, 2, 0),    xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),    xtal::UnitCellCoord(3, 2, 1, 0)};

  m_orbit_neighborhood[72] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, 0), xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, 0), xtal::UnitCell(-1, 0, 0),
      xtal::UnitCell(-1, 1, 0),  xtal::UnitCell(0, -1, 0),
      xtal::UnitCell(0, 0, 0),   xtal::UnitCell(0, 1, 0),
      xtal::UnitCell(1, -1, 0),  xtal::UnitCell(1, 0, 0),
      xtal::UnitCell(1, 1, 0),   xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 1, 0)};

  m_orbit_site_neighborhood[72] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, 0), xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -1, -2, 0), xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(2, -1, -1, 0), xtal::UnitCellCoord(3, -1, -1, 0),
      xtal::UnitCellCoord(2, -1, 0, 0),  xtal::UnitCellCoord(3, -1, 0, 0),
      xtal::UnitCellCoord(2, -1, 1, 0),  xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(2, 0, -1, 0),  xtal::UnitCellCoord(3, 0, -1, 0),
      xtal::UnitCellCoord(2, 0, 0, 0),   xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(2, 0, 1, 0),   xtal::UnitCellCoord(3, 0, 1, 0),
      xtal::UnitCellCoord(2, 1, -1, 0),  xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(2, 1, 0, 0),   xtal::UnitCellCoord(3, 1, 0, 0),
      xtal::UnitCellCoord(2, 1, 1, 0),   xtal::UnitCellCoord(3, 1, 1, 0),
      xtal::UnitCellCoord(2, 1, 2, 0),   xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),   xtal::UnitCellCoord(3, 2, 1, 0)};

  m_orbit_neighborhood[73] = std::set<xtal::UnitCell>{
      xtal::UnitCell(-2, -1, 0),  xtal::UnitCell(-1, -2, 0),
      xtal::UnitCell(-1, -1, -1), xtal::UnitCell(-1, -1, 0),
      xtal::UnitCell(-1, -1, 1),  xtal::UnitCell(-1, 0, -1),
      xtal::UnitCell(-1, 0, 0),   xtal::UnitCell(-1, 0, 1),
      xtal::UnitCell(-1, 1, 0),   xtal::UnitCell(0, -1, -1),
      xtal::UnitCell(0, -1, 0),   xtal::UnitCell(0, -1, 1),
      xtal::UnitCell(0, 0, 0),    xtal::UnitCell(0, 1, -1),
      xtal::UnitCell(0, 1, 0),    xtal::UnitCell(0, 1, 1),
      xtal::UnitCell(1, -1, 0),   xtal::UnitCell(1, 0, -1),
      xtal::UnitCell(1, 0, 0),    xtal::UnitCell(1, 0, 1),
      xtal::UnitCell(1, 1, -1),   xtal::UnitCell(1, 1, 0),
      xtal::UnitCell(1, 1, 1),    xtal::UnitCell(1, 2, 0),
      xtal::UnitCell(2, 1, 0)};

  m_orbit_site_neighborhood[73] = std::set<xtal::UnitCellCoord>{
      xtal::UnitCellCoord(2, -2, -1, 0),  xtal::UnitCellCoord(3, -2, -1, 0),
      xtal::UnitCellCoord(2, -1, -2, 0),  xtal::UnitCellCoord(3, -1, -2, 0),
      xtal::UnitCellCoord(3, -1, -1, -1), xtal::UnitCellCoord(2, -1, -1, 0),
      xtal::UnitCellCoord(3, -1, -1, 0),  xtal::UnitCellCoord(2, -1, -1, 1),
      xtal::UnitCellCoord(3, -1, 0, -1),  xtal::UnitCellCoord(2, -1, 0, 0),
      xtal::UnitCellCoord(3, -1, 0, 0),   xtal::UnitCellCoord(2, -1, 0, 1),
      xtal::UnitCellCoord(2, -1, 1, 0),   xtal::UnitCellCoord(3, -1, 1, 0),
      xtal::UnitCellCoord(3, 0, -1, -1),  xtal::UnitCellCoord(2, 0, -1, 0),
      xtal::UnitCellCoord(3, 0, -1, 0),   xtal::UnitCellCoord(2, 0, -1, 1),
      xtal::UnitCellCoord(2, 0, 0, 0),    xtal::UnitCellCoord(3, 0, 0, 0),
      xtal::UnitCellCoord(3, 0, 1, -1),   xtal::UnitCellCoord(2, 0, 1, 0),
      xtal::UnitCellCoord(3, 0, 1, 0),    xtal::UnitCellCoord(2, 0, 1, 1),
      xtal::UnitCellCoord(2, 1, -1, 0),   xtal::UnitCellCoord(3, 1, -1, 0),
      xtal::UnitCellCoord(3, 1, 0, -1),   xtal::UnitCellCoord(2, 1, 0, 0),
      xtal::UnitCellCoord(3, 1, 0, 0),    xtal::UnitCellCoord(2, 1, 0, 1),
      xtal::UnitCellCoord(3, 1, 1, -1),   xtal::UnitCellCoord(2, 1, 1, 0),
      xtal::UnitCellCoord(3, 1, 1, 0),    xtal::UnitCellCoord(2, 1, 1, 1),
      xtal::UnitCellCoord(2, 1, 2, 0),    xtal::UnitCellCoord(3, 1, 2, 0),
      xtal::UnitCellCoord(2, 2, 1, 0),    xtal::UnitCellCoord(3, 2, 1, 0)};
}

ZrO_Clexulator::~ZrO_Clexulator() {
  // nothing here for now
}

/// \brief Calculate contribution to global correlations from one unit cell
void ZrO_Clexulator::_calc_global_corr_contribution(double *corr_begin) const {
  _calc_global_corr_contribution();
  for (size_type i = 0; i < corr_size(); i++) {
    *(corr_begin + i) =
        ParamPack::Val<double>::get(m_params, m_corr_param_key, i);
  }
}

/// \brief Calculate contribution to global correlations from one unit cell
void ZrO_Clexulator::_calc_global_corr_contribution() const {
  m_params.pre_eval();
  {
    _global_prepare<double>();
    for (size_type i = 0; i < corr_size(); i++) {
      ParamPack::Val<double>::set(m_params, m_corr_param_key, i,
                                  (this->*m_orbit_func_table_0[i])());
    }
  }
  m_params.post_eval();
}

/// \brief Calculate contribution to select global correlations from one unit
/// cell
void ZrO_Clexulator::_calc_restricted_global_corr_contribution(
    double *corr_begin, size_type const *ind_list_begin,
    size_type const *ind_list_end) const {
  _calc_restricted_global_corr_contribution(ind_list_begin, ind_list_end);
  for (; ind_list_begin < ind_list_end; ind_list_begin++) {
    *(corr_begin + *ind_list_begin) = ParamPack::Val<double>::get(
        m_params, m_corr_param_key, *ind_list_begin);
  }
}

/// \brief Calculate contribution to select global correlations from one unit
/// cell
void ZrO_Clexulator::_calc_restricted_global_corr_contribution(
    size_type const *ind_list_begin, size_type const *ind_list_end) const {
  m_params.pre_eval();
  {
    _global_prepare<double>();
    for (; ind_list_begin < ind_list_end; ind_list_begin++) {
      ParamPack::Val<double>::set(
          m_params, m_corr_param_key, *ind_list_begin,
          (this->*m_orbit_func_table_0[*ind_list_begin])());
    }
  }
  m_params.post_eval();
}

/// \brief Calculate point correlations about basis site 'nlist_ind'
void ZrO_Clexulator::_calc_point_corr(int nlist_ind, double *corr_begin) const {
  _calc_point_corr(nlist_ind);
  for (size_type i = 0; i < corr_size(); i++) {
    *(corr_begin + i) =
        ParamPack::Val<double>::get(m_params, m_corr_param_key, i);
  }
}

/// \brief Calculate point correlations about basis site 'nlist_ind'
void ZrO_Clexulator::_calc_point_corr(int nlist_ind) const {
  m_params.pre_eval();
  {
    _point_prepare<double>(nlist_ind);
    for (size_type i = 0; i < corr_size(); i++) {
      ParamPack::Val<double>::set(
          m_params, m_corr_param_key, i,
          (this->*m_flower_func_table_0[nlist_ind][i])());
    }
  }
  m_params.post_eval();
}

/// \brief Calculate select point correlations about basis site 'nlist_ind'
void ZrO_Clexulator::_calc_restricted_point_corr(
    int nlist_ind, double *corr_begin, size_type const *ind_list_begin,
    size_type const *ind_list_end) const {
  _calc_restricted_point_corr(nlist_ind, ind_list_begin, ind_list_end);
  for (; ind_list_begin < ind_list_end; ind_list_begin++) {
    *(corr_begin + *ind_list_begin) = ParamPack::Val<double>::get(
        m_params, m_corr_param_key, *ind_list_begin);
  }
}

/// \brief Calculate select point correlations about basis site 'nlist_ind'
void ZrO_Clexulator::_calc_restricted_point_corr(
    int nlist_ind, size_type const *ind_list_begin,
    size_type const *ind_list_end) const {
  m_params.pre_eval();
  {
    _point_prepare<double>(nlist_ind);
    for (; ind_list_begin < ind_list_end; ind_list_begin++) {
      ParamPack::Val<double>::set(
          m_params, m_corr_param_key, *ind_list_begin,
          (this->*m_flower_func_table_0[nlist_ind][*ind_list_begin])());
    }
  }
  m_params.post_eval();
}

/// \brief Calculate the change in point correlations due to changing an
/// occupant
void ZrO_Clexulator::_calc_delta_point_corr(int nlist_ind, int occ_i, int occ_f,
                                            double *corr_begin) const {
  _calc_delta_point_corr(nlist_ind, occ_i, occ_f);
  for (size_type i = 0; i < corr_size(); i++) {
    *(corr_begin + i) =
        ParamPack::Val<double>::get(m_params, m_corr_param_key, i);
  }
}

/// \brief Calculate the change in point correlations due to changing an
/// occupant
void ZrO_Clexulator::_calc_delta_point_corr(int nlist_ind, int occ_i,
                                            int occ_f) const {
  m_params.pre_eval();
  {
    _point_prepare<double>(nlist_ind);
    for (size_type i = 0; i < corr_size(); i++) {
      ParamPack::Val<double>::set(
          m_params, m_corr_param_key, i,
          (this->*m_delta_func_table_0[nlist_ind][i])(occ_i, occ_f));
    }
  }
  m_params.post_eval();
}

/// \brief Calculate the change in select point correlations due to changing an
/// occupant
void ZrO_Clexulator::_calc_restricted_delta_point_corr(
    int nlist_ind, int occ_i, int occ_f, double *corr_begin,
    size_type const *ind_list_begin, size_type const *ind_list_end) const {
  _calc_restricted_delta_point_corr(nlist_ind, occ_i, occ_f, ind_list_begin,
                                    ind_list_end);
  for (; ind_list_begin < ind_list_end; ind_list_begin++) {
    *(corr_begin + *ind_list_begin) = ParamPack::Val<double>::get(
        m_params, m_corr_param_key, *ind_list_begin);
  }
}

/// \brief Calculate the change in select point correlations due to changing an
/// occupant
void ZrO_Clexulator::_calc_restricted_delta_point_corr(
    int nlist_ind, int occ_i, int occ_f, size_type const *ind_list_begin,
    size_type const *ind_list_end) const {
  m_params.pre_eval();
  {
    _point_prepare<double>(nlist_ind);
    for (; ind_list_begin < ind_list_end; ind_list_begin++) {
      ParamPack::Val<double>::set(
          m_params, m_corr_param_key, *ind_list_begin,
          (this->*m_delta_func_table_0[nlist_ind][*ind_list_begin])(occ_i,
                                                                    occ_f));
    }
  }
  m_params.post_eval();
}

template <typename Scalar>
void ZrO_Clexulator::_point_prepare(int nlist_ind) const {
  switch (nlist_ind) {
    case 0:
      if (m_params.eval_mode(m_occ_site_func_param_key) != ParamPack::READ) {
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 0,
                                    eval_occ_func_2_0(0));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 2,
                                    eval_occ_func_2_0(2));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 4,
                                    eval_occ_func_2_0(4));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 6,
                                    eval_occ_func_2_0(6));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 8,
                                    eval_occ_func_2_0(8));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 10,
                                    eval_occ_func_2_0(10));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 12,
                                    eval_occ_func_2_0(12));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 14,
                                    eval_occ_func_2_0(14));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 16,
                                    eval_occ_func_2_0(16));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 18,
                                    eval_occ_func_2_0(18));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 20,
                                    eval_occ_func_2_0(20));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 22,
                                    eval_occ_func_2_0(22));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 24,
                                    eval_occ_func_2_0(24));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 26,
                                    eval_occ_func_2_0(26));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 28,
                                    eval_occ_func_2_0(28));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 30,
                                    eval_occ_func_2_0(30));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 32,
                                    eval_occ_func_2_0(32));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 34,
                                    eval_occ_func_2_0(34));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 36,
                                    eval_occ_func_2_0(36));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 38,
                                    eval_occ_func_2_0(38));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 40,
                                    eval_occ_func_2_0(40));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 42,
                                    eval_occ_func_2_0(42));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 44,
                                    eval_occ_func_2_0(44));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 46,
                                    eval_occ_func_2_0(46));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 48,
                                    eval_occ_func_2_0(48));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 50,
                                    eval_occ_func_2_0(50));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 52,
                                    eval_occ_func_2_0(52));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 54,
                                    eval_occ_func_2_0(54));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 56,
                                    eval_occ_func_2_0(56));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 58,
                                    eval_occ_func_2_0(58));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 60,
                                    eval_occ_func_2_0(60));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 62,
                                    eval_occ_func_2_0(62));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 64,
                                    eval_occ_func_2_0(64));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 66,
                                    eval_occ_func_2_0(66));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 68,
                                    eval_occ_func_2_0(68));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 70,
                                    eval_occ_func_2_0(70));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 72,
                                    eval_occ_func_2_0(72));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 74,
                                    eval_occ_func_2_0(74));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 76,
                                    eval_occ_func_2_0(76));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 78,
                                    eval_occ_func_2_0(78));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 80,
                                    eval_occ_func_2_0(80));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 82,
                                    eval_occ_func_2_0(82));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 84,
                                    eval_occ_func_2_0(84));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 86,
                                    eval_occ_func_2_0(86));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 88,
                                    eval_occ_func_2_0(88));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 90,
                                    eval_occ_func_2_0(90));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 92,
                                    eval_occ_func_2_0(92));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 94,
                                    eval_occ_func_2_0(94));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 96,
                                    eval_occ_func_2_0(96));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 98,
                                    eval_occ_func_2_0(98));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 100,
                                    eval_occ_func_2_0(100));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 102,
                                    eval_occ_func_2_0(102));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 104,
                                    eval_occ_func_2_0(104));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 106,
                                    eval_occ_func_2_0(106));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 108,
                                    eval_occ_func_2_0(108));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 110,
                                    eval_occ_func_2_0(110));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 112,
                                    eval_occ_func_2_0(112));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 114,
                                    eval_occ_func_2_0(114));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 116,
                                    eval_occ_func_2_0(116));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 118,
                                    eval_occ_func_2_0(118));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 120,
                                    eval_occ_func_2_0(120));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 122,
                                    eval_occ_func_2_0(122));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 124,
                                    eval_occ_func_2_0(124));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 126,
                                    eval_occ_func_2_0(126));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 128,
                                    eval_occ_func_2_0(128));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 130,
                                    eval_occ_func_2_0(130));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 132,
                                    eval_occ_func_2_0(132));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 134,
                                    eval_occ_func_2_0(134));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 136,
                                    eval_occ_func_2_0(136));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 1,
                                    eval_occ_func_3_0(1));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 3,
                                    eval_occ_func_3_0(3));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 5,
                                    eval_occ_func_3_0(5));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 7,
                                    eval_occ_func_3_0(7));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 9,
                                    eval_occ_func_3_0(9));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 11,
                                    eval_occ_func_3_0(11));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 13,
                                    eval_occ_func_3_0(13));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 15,
                                    eval_occ_func_3_0(15));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 17,
                                    eval_occ_func_3_0(17));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 19,
                                    eval_occ_func_3_0(19));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 21,
                                    eval_occ_func_3_0(21));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 23,
                                    eval_occ_func_3_0(23));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 25,
                                    eval_occ_func_3_0(25));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 27,
                                    eval_occ_func_3_0(27));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 29,
                                    eval_occ_func_3_0(29));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 31,
                                    eval_occ_func_3_0(31));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 33,
                                    eval_occ_func_3_0(33));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 35,
                                    eval_occ_func_3_0(35));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 37,
                                    eval_occ_func_3_0(37));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 39,
                                    eval_occ_func_3_0(39));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 41,
                                    eval_occ_func_3_0(41));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 43,
                                    eval_occ_func_3_0(43));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 45,
                                    eval_occ_func_3_0(45));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 47,
                                    eval_occ_func_3_0(47));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 49,
                                    eval_occ_func_3_0(49));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 51,
                                    eval_occ_func_3_0(51));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 53,
                                    eval_occ_func_3_0(53));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 55,
                                    eval_occ_func_3_0(55));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 57,
                                    eval_occ_func_3_0(57));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 59,
                                    eval_occ_func_3_0(59));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 61,
                                    eval_occ_func_3_0(61));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 63,
                                    eval_occ_func_3_0(63));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 65,
                                    eval_occ_func_3_0(65));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 67,
                                    eval_occ_func_3_0(67));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 71,
                                    eval_occ_func_3_0(71));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 75,
                                    eval_occ_func_3_0(75));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 79,
                                    eval_occ_func_3_0(79));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 83,
                                    eval_occ_func_3_0(83));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 87,
                                    eval_occ_func_3_0(87));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 91,
                                    eval_occ_func_3_0(91));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 95,
                                    eval_occ_func_3_0(95));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 99,
                                    eval_occ_func_3_0(99));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 103,
                                    eval_occ_func_3_0(103));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 107,
                                    eval_occ_func_3_0(107));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 111,
                                    eval_occ_func_3_0(111));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 115,
                                    eval_occ_func_3_0(115));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 117,
                                    eval_occ_func_3_0(117));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 119,
                                    eval_occ_func_3_0(119));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 121,
                                    eval_occ_func_3_0(121));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 123,
                                    eval_occ_func_3_0(123));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 125,
                                    eval_occ_func_3_0(125));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 127,
                                    eval_occ_func_3_0(127));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 129,
                                    eval_occ_func_3_0(129));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 131,
                                    eval_occ_func_3_0(131));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 133,
                                    eval_occ_func_3_0(133));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 135,
                                    eval_occ_func_3_0(135));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 137,
                                    eval_occ_func_3_0(137));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 151,
                                    eval_occ_func_3_0(151));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 155,
                                    eval_occ_func_3_0(155));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 159,
                                    eval_occ_func_3_0(159));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 163,
                                    eval_occ_func_3_0(163));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 167,
                                    eval_occ_func_3_0(167));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 171,
                                    eval_occ_func_3_0(171));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 175,
                                    eval_occ_func_3_0(175));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 179,
                                    eval_occ_func_3_0(179));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 183,
                                    eval_occ_func_3_0(183));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 187,
                                    eval_occ_func_3_0(187));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 191,
                                    eval_occ_func_3_0(191));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 195,
                                    eval_occ_func_3_0(195));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 199,
                                    eval_occ_func_3_0(199));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 203,
                                    eval_occ_func_3_0(203));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 207,
                                    eval_occ_func_3_0(207));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 211,
                                    eval_occ_func_3_0(211));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 215,
                                    eval_occ_func_3_0(215));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 219,
                                    eval_occ_func_3_0(219));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 223,
                                    eval_occ_func_3_0(223));
      }
      break;
    case 1:
      if (m_params.eval_mode(m_occ_site_func_param_key) != ParamPack::READ) {
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 0,
                                    eval_occ_func_2_0(0));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 2,
                                    eval_occ_func_2_0(2));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 4,
                                    eval_occ_func_2_0(4));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 6,
                                    eval_occ_func_2_0(6));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 8,
                                    eval_occ_func_2_0(8));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 10,
                                    eval_occ_func_2_0(10));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 12,
                                    eval_occ_func_2_0(12));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 14,
                                    eval_occ_func_2_0(14));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 16,
                                    eval_occ_func_2_0(16));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 18,
                                    eval_occ_func_2_0(18));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 20,
                                    eval_occ_func_2_0(20));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 22,
                                    eval_occ_func_2_0(22));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 24,
                                    eval_occ_func_2_0(24));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 26,
                                    eval_occ_func_2_0(26));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 28,
                                    eval_occ_func_2_0(28));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 30,
                                    eval_occ_func_2_0(30));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 32,
                                    eval_occ_func_2_0(32));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 34,
                                    eval_occ_func_2_0(34));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 36,
                                    eval_occ_func_2_0(36));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 38,
                                    eval_occ_func_2_0(38));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 40,
                                    eval_occ_func_2_0(40));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 42,
                                    eval_occ_func_2_0(42));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 44,
                                    eval_occ_func_2_0(44));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 46,
                                    eval_occ_func_2_0(46));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 48,
                                    eval_occ_func_2_0(48));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 50,
                                    eval_occ_func_2_0(50));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 52,
                                    eval_occ_func_2_0(52));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 54,
                                    eval_occ_func_2_0(54));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 56,
                                    eval_occ_func_2_0(56));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 58,
                                    eval_occ_func_2_0(58));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 60,
                                    eval_occ_func_2_0(60));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 62,
                                    eval_occ_func_2_0(62));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 64,
                                    eval_occ_func_2_0(64));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 68,
                                    eval_occ_func_2_0(68));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 72,
                                    eval_occ_func_2_0(72));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 76,
                                    eval_occ_func_2_0(76));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 80,
                                    eval_occ_func_2_0(80));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 84,
                                    eval_occ_func_2_0(84));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 88,
                                    eval_occ_func_2_0(88));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 92,
                                    eval_occ_func_2_0(92));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 96,
                                    eval_occ_func_2_0(96));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 100,
                                    eval_occ_func_2_0(100));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 104,
                                    eval_occ_func_2_0(104));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 108,
                                    eval_occ_func_2_0(108));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 112,
                                    eval_occ_func_2_0(112));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 114,
                                    eval_occ_func_2_0(114));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 116,
                                    eval_occ_func_2_0(116));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 118,
                                    eval_occ_func_2_0(118));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 120,
                                    eval_occ_func_2_0(120));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 122,
                                    eval_occ_func_2_0(122));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 124,
                                    eval_occ_func_2_0(124));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 126,
                                    eval_occ_func_2_0(126));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 128,
                                    eval_occ_func_2_0(128));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 130,
                                    eval_occ_func_2_0(130));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 132,
                                    eval_occ_func_2_0(132));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 134,
                                    eval_occ_func_2_0(134));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 136,
                                    eval_occ_func_2_0(136));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 152,
                                    eval_occ_func_2_0(152));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 156,
                                    eval_occ_func_2_0(156));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 160,
                                    eval_occ_func_2_0(160));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 164,
                                    eval_occ_func_2_0(164));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 168,
                                    eval_occ_func_2_0(168));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 172,
                                    eval_occ_func_2_0(172));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 176,
                                    eval_occ_func_2_0(176));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 180,
                                    eval_occ_func_2_0(180));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 184,
                                    eval_occ_func_2_0(184));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 188,
                                    eval_occ_func_2_0(188));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 192,
                                    eval_occ_func_2_0(192));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 196,
                                    eval_occ_func_2_0(196));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 200,
                                    eval_occ_func_2_0(200));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 204,
                                    eval_occ_func_2_0(204));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 208,
                                    eval_occ_func_2_0(208));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 212,
                                    eval_occ_func_2_0(212));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 216,
                                    eval_occ_func_2_0(216));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 220,
                                    eval_occ_func_2_0(220));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 224,
                                    eval_occ_func_2_0(224));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 1,
                                    eval_occ_func_3_0(1));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 3,
                                    eval_occ_func_3_0(3));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 5,
                                    eval_occ_func_3_0(5));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 7,
                                    eval_occ_func_3_0(7));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 9,
                                    eval_occ_func_3_0(9));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 11,
                                    eval_occ_func_3_0(11));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 13,
                                    eval_occ_func_3_0(13));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 15,
                                    eval_occ_func_3_0(15));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 17,
                                    eval_occ_func_3_0(17));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 19,
                                    eval_occ_func_3_0(19));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 21,
                                    eval_occ_func_3_0(21));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 23,
                                    eval_occ_func_3_0(23));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 25,
                                    eval_occ_func_3_0(25));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 27,
                                    eval_occ_func_3_0(27));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 29,
                                    eval_occ_func_3_0(29));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 31,
                                    eval_occ_func_3_0(31));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 33,
                                    eval_occ_func_3_0(33));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 35,
                                    eval_occ_func_3_0(35));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 37,
                                    eval_occ_func_3_0(37));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 39,
                                    eval_occ_func_3_0(39));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 41,
                                    eval_occ_func_3_0(41));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 43,
                                    eval_occ_func_3_0(43));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 45,
                                    eval_occ_func_3_0(45));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 47,
                                    eval_occ_func_3_0(47));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 49,
                                    eval_occ_func_3_0(49));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 51,
                                    eval_occ_func_3_0(51));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 53,
                                    eval_occ_func_3_0(53));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 55,
                                    eval_occ_func_3_0(55));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 57,
                                    eval_occ_func_3_0(57));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 59,
                                    eval_occ_func_3_0(59));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 61,
                                    eval_occ_func_3_0(61));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 63,
                                    eval_occ_func_3_0(63));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 65,
                                    eval_occ_func_3_0(65));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 67,
                                    eval_occ_func_3_0(67));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 69,
                                    eval_occ_func_3_0(69));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 71,
                                    eval_occ_func_3_0(71));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 73,
                                    eval_occ_func_3_0(73));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 75,
                                    eval_occ_func_3_0(75));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 77,
                                    eval_occ_func_3_0(77));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 79,
                                    eval_occ_func_3_0(79));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 81,
                                    eval_occ_func_3_0(81));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 83,
                                    eval_occ_func_3_0(83));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 85,
                                    eval_occ_func_3_0(85));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 87,
                                    eval_occ_func_3_0(87));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 89,
                                    eval_occ_func_3_0(89));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 91,
                                    eval_occ_func_3_0(91));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 93,
                                    eval_occ_func_3_0(93));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 95,
                                    eval_occ_func_3_0(95));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 97,
                                    eval_occ_func_3_0(97));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 99,
                                    eval_occ_func_3_0(99));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 101,
                                    eval_occ_func_3_0(101));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 103,
                                    eval_occ_func_3_0(103));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 105,
                                    eval_occ_func_3_0(105));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 107,
                                    eval_occ_func_3_0(107));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 109,
                                    eval_occ_func_3_0(109));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 111,
                                    eval_occ_func_3_0(111));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 113,
                                    eval_occ_func_3_0(113));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 115,
                                    eval_occ_func_3_0(115));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 117,
                                    eval_occ_func_3_0(117));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 119,
                                    eval_occ_func_3_0(119));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 121,
                                    eval_occ_func_3_0(121));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 123,
                                    eval_occ_func_3_0(123));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 125,
                                    eval_occ_func_3_0(125));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 127,
                                    eval_occ_func_3_0(127));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 129,
                                    eval_occ_func_3_0(129));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 131,
                                    eval_occ_func_3_0(131));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 133,
                                    eval_occ_func_3_0(133));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 135,
                                    eval_occ_func_3_0(135));
        ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 137,
                                    eval_occ_func_3_0(137));
      }
      break;
  }
}
template <typename Scalar>
void ZrO_Clexulator::_global_prepare() const {
  if (m_params.eval_mode(m_occ_site_func_param_key) != ParamPack::READ) {
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 0,
                                eval_occ_func_2_0(0));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 2,
                                eval_occ_func_2_0(2));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 4,
                                eval_occ_func_2_0(4));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 6,
                                eval_occ_func_2_0(6));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 8,
                                eval_occ_func_2_0(8));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 10,
                                eval_occ_func_2_0(10));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 12,
                                eval_occ_func_2_0(12));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 14,
                                eval_occ_func_2_0(14));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 16,
                                eval_occ_func_2_0(16));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 18,
                                eval_occ_func_2_0(18));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 20,
                                eval_occ_func_2_0(20));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 22,
                                eval_occ_func_2_0(22));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 24,
                                eval_occ_func_2_0(24));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 26,
                                eval_occ_func_2_0(26));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 28,
                                eval_occ_func_2_0(28));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 30,
                                eval_occ_func_2_0(30));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 32,
                                eval_occ_func_2_0(32));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 34,
                                eval_occ_func_2_0(34));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 36,
                                eval_occ_func_2_0(36));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 38,
                                eval_occ_func_2_0(38));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 40,
                                eval_occ_func_2_0(40));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 42,
                                eval_occ_func_2_0(42));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 44,
                                eval_occ_func_2_0(44));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 46,
                                eval_occ_func_2_0(46));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 48,
                                eval_occ_func_2_0(48));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 50,
                                eval_occ_func_2_0(50));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 52,
                                eval_occ_func_2_0(52));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 54,
                                eval_occ_func_2_0(54));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 56,
                                eval_occ_func_2_0(56));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 58,
                                eval_occ_func_2_0(58));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 60,
                                eval_occ_func_2_0(60));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 62,
                                eval_occ_func_2_0(62));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 64,
                                eval_occ_func_2_0(64));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 66,
                                eval_occ_func_2_0(66));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 68,
                                eval_occ_func_2_0(68));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 70,
                                eval_occ_func_2_0(70));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 72,
                                eval_occ_func_2_0(72));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 74,
                                eval_occ_func_2_0(74));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 76,
                                eval_occ_func_2_0(76));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 78,
                                eval_occ_func_2_0(78));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 80,
                                eval_occ_func_2_0(80));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 82,
                                eval_occ_func_2_0(82));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 84,
                                eval_occ_func_2_0(84));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 86,
                                eval_occ_func_2_0(86));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 88,
                                eval_occ_func_2_0(88));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 90,
                                eval_occ_func_2_0(90));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 92,
                                eval_occ_func_2_0(92));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 94,
                                eval_occ_func_2_0(94));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 96,
                                eval_occ_func_2_0(96));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 98,
                                eval_occ_func_2_0(98));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 100,
                                eval_occ_func_2_0(100));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 102,
                                eval_occ_func_2_0(102));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 104,
                                eval_occ_func_2_0(104));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 106,
                                eval_occ_func_2_0(106));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 108,
                                eval_occ_func_2_0(108));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 110,
                                eval_occ_func_2_0(110));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 112,
                                eval_occ_func_2_0(112));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 114,
                                eval_occ_func_2_0(114));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 116,
                                eval_occ_func_2_0(116));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 118,
                                eval_occ_func_2_0(118));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 120,
                                eval_occ_func_2_0(120));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 122,
                                eval_occ_func_2_0(122));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 124,
                                eval_occ_func_2_0(124));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 126,
                                eval_occ_func_2_0(126));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 128,
                                eval_occ_func_2_0(128));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 130,
                                eval_occ_func_2_0(130));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 132,
                                eval_occ_func_2_0(132));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 134,
                                eval_occ_func_2_0(134));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 136,
                                eval_occ_func_2_0(136));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 152,
                                eval_occ_func_2_0(152));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 156,
                                eval_occ_func_2_0(156));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 160,
                                eval_occ_func_2_0(160));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 164,
                                eval_occ_func_2_0(164));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 168,
                                eval_occ_func_2_0(168));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 172,
                                eval_occ_func_2_0(172));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 176,
                                eval_occ_func_2_0(176));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 180,
                                eval_occ_func_2_0(180));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 184,
                                eval_occ_func_2_0(184));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 188,
                                eval_occ_func_2_0(188));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 192,
                                eval_occ_func_2_0(192));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 196,
                                eval_occ_func_2_0(196));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 200,
                                eval_occ_func_2_0(200));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 204,
                                eval_occ_func_2_0(204));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 208,
                                eval_occ_func_2_0(208));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 212,
                                eval_occ_func_2_0(212));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 216,
                                eval_occ_func_2_0(216));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 220,
                                eval_occ_func_2_0(220));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 224,
                                eval_occ_func_2_0(224));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 1,
                                eval_occ_func_3_0(1));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 3,
                                eval_occ_func_3_0(3));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 5,
                                eval_occ_func_3_0(5));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 7,
                                eval_occ_func_3_0(7));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 9,
                                eval_occ_func_3_0(9));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 11,
                                eval_occ_func_3_0(11));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 13,
                                eval_occ_func_3_0(13));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 15,
                                eval_occ_func_3_0(15));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 17,
                                eval_occ_func_3_0(17));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 19,
                                eval_occ_func_3_0(19));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 21,
                                eval_occ_func_3_0(21));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 23,
                                eval_occ_func_3_0(23));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 25,
                                eval_occ_func_3_0(25));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 27,
                                eval_occ_func_3_0(27));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 29,
                                eval_occ_func_3_0(29));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 31,
                                eval_occ_func_3_0(31));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 33,
                                eval_occ_func_3_0(33));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 35,
                                eval_occ_func_3_0(35));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 37,
                                eval_occ_func_3_0(37));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 39,
                                eval_occ_func_3_0(39));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 41,
                                eval_occ_func_3_0(41));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 43,
                                eval_occ_func_3_0(43));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 45,
                                eval_occ_func_3_0(45));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 47,
                                eval_occ_func_3_0(47));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 49,
                                eval_occ_func_3_0(49));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 51,
                                eval_occ_func_3_0(51));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 53,
                                eval_occ_func_3_0(53));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 55,
                                eval_occ_func_3_0(55));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 57,
                                eval_occ_func_3_0(57));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 59,
                                eval_occ_func_3_0(59));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 61,
                                eval_occ_func_3_0(61));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 63,
                                eval_occ_func_3_0(63));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 65,
                                eval_occ_func_3_0(65));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 67,
                                eval_occ_func_3_0(67));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 69,
                                eval_occ_func_3_0(69));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 71,
                                eval_occ_func_3_0(71));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 73,
                                eval_occ_func_3_0(73));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 75,
                                eval_occ_func_3_0(75));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 77,
                                eval_occ_func_3_0(77));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 79,
                                eval_occ_func_3_0(79));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 81,
                                eval_occ_func_3_0(81));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 83,
                                eval_occ_func_3_0(83));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 85,
                                eval_occ_func_3_0(85));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 87,
                                eval_occ_func_3_0(87));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 89,
                                eval_occ_func_3_0(89));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 91,
                                eval_occ_func_3_0(91));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 93,
                                eval_occ_func_3_0(93));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 95,
                                eval_occ_func_3_0(95));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 97,
                                eval_occ_func_3_0(97));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 99,
                                eval_occ_func_3_0(99));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 101,
                                eval_occ_func_3_0(101));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 103,
                                eval_occ_func_3_0(103));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 105,
                                eval_occ_func_3_0(105));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 107,
                                eval_occ_func_3_0(107));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 109,
                                eval_occ_func_3_0(109));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 111,
                                eval_occ_func_3_0(111));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 113,
                                eval_occ_func_3_0(113));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 115,
                                eval_occ_func_3_0(115));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 117,
                                eval_occ_func_3_0(117));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 119,
                                eval_occ_func_3_0(119));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 121,
                                eval_occ_func_3_0(121));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 123,
                                eval_occ_func_3_0(123));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 125,
                                eval_occ_func_3_0(125));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 127,
                                eval_occ_func_3_0(127));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 129,
                                eval_occ_func_3_0(129));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 131,
                                eval_occ_func_3_0(131));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 133,
                                eval_occ_func_3_0(133));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 135,
                                eval_occ_func_3_0(135));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 137,
                                eval_occ_func_3_0(137));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 151,
                                eval_occ_func_3_0(151));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 155,
                                eval_occ_func_3_0(155));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 159,
                                eval_occ_func_3_0(159));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 163,
                                eval_occ_func_3_0(163));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 167,
                                eval_occ_func_3_0(167));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 171,
                                eval_occ_func_3_0(171));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 175,
                                eval_occ_func_3_0(175));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 179,
                                eval_occ_func_3_0(179));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 183,
                                eval_occ_func_3_0(183));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 187,
                                eval_occ_func_3_0(187));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 191,
                                eval_occ_func_3_0(191));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 195,
                                eval_occ_func_3_0(195));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 199,
                                eval_occ_func_3_0(199));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 203,
                                eval_occ_func_3_0(203));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 207,
                                eval_occ_func_3_0(207));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 211,
                                eval_occ_func_3_0(211));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 215,
                                eval_occ_func_3_0(215));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 219,
                                eval_occ_func_3_0(219));
    ParamPack::Val<Scalar>::set(m_params, m_occ_site_func_param_key, 0, 223,
                                eval_occ_func_3_0(223));
  }
}

// Basis functions for empty cluster:
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_0_0() const {
  return 1;
}

/**** Basis functions for orbit 1****
0.3333333 0.6666666 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_1_0() const {
  return (occ_func_2_0(0) + occ_func_2_0(1)) / 2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_1_0_at_0() const {
  return (occ_func_2_0(0)) / 2.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_1_0_at_1() const {
  return (occ_func_2_0(1)) / 2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_1_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) * (1) / 2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_1_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) * (1) / 2.;
}

/**** Basis functions for orbit 2****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_2_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16)) /
         2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_2_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) +
          occ_func_2_0(15) * occ_func_3_0(0)) /
         2.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_2_0_at_1() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16)) /
         2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_2_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(1) + occ_func_2_0(15)) / 2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_2_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(0) + occ_func_3_0(16)) / 2.;
}

/**** Basis functions for orbit 3****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_3_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) +
          occ_func_2_0(11) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(13) +
          occ_func_2_0(12) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(10) +
          occ_func_2_0(9) * occ_func_2_0(1)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_3_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(2) +
          occ_func_2_0(0) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_2_0(0)) /
         6.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_3_0_at_1() const {
  return (occ_func_2_0(11) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(5) +
          occ_func_2_0(1) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(7)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_3_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) + occ_func_2_0(6) + occ_func_2_0(12) +
          occ_func_2_0(2) + occ_func_2_0(10) + occ_func_2_0(4)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_3_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(11) + occ_func_2_0(5) + occ_func_2_0(13) +
          occ_func_2_0(3) + occ_func_2_0(9) + occ_func_2_0(7)) /
         6.;
}

/**** Basis functions for orbit 4****
0.3333333 0.6666666 0.2500000 Va  O

1.3333333 1.6666667 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_4_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(51) +
          occ_func_2_0(1) * occ_func_3_0(8) +
          occ_func_2_0(1) * occ_func_3_0(10) +
          occ_func_2_0(48) * occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(13) +
          occ_func_2_0(0) * occ_func_3_0(47) +
          occ_func_2_0(0) * occ_func_3_0(43) +
          occ_func_2_0(52) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(12) +
          occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(11) * occ_func_3_0(0)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_4_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(51) +
          occ_func_2_0(7) * occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(35) +
          occ_func_2_0(0) * occ_func_3_0(39) +
          occ_func_2_0(0) * occ_func_3_0(13) +
          occ_func_2_0(0) * occ_func_3_0(47) +
          occ_func_2_0(0) * occ_func_3_0(43) +
          occ_func_2_0(0) * occ_func_3_0(31) +
          occ_func_2_0(3) * occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(11) * occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_4_0_at_1() const {
  return (occ_func_2_0(32) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(8) +
          occ_func_2_0(1) * occ_func_3_0(10) +
          occ_func_2_0(48) * occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_3_0(1) +
          occ_func_2_0(36) * occ_func_3_0(1) +
          occ_func_2_0(40) * occ_func_3_0(1) +
          occ_func_2_0(52) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_3_0(6) +
          occ_func_2_0(1) * occ_func_3_0(4)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_4_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(51) + occ_func_2_0(7) + occ_func_2_0(5) +
          occ_func_3_0(35) + occ_func_3_0(39) + occ_func_3_0(13) +
          occ_func_3_0(47) + occ_func_3_0(43) + occ_func_3_0(31) +
          occ_func_2_0(3) + occ_func_2_0(9) + occ_func_2_0(11)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_4_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(32) + occ_func_3_0(8) + occ_func_3_0(10) +
          occ_func_2_0(48) + occ_func_2_0(44) + occ_func_2_0(2) +
          occ_func_2_0(36) + occ_func_2_0(40) + occ_func_2_0(52) +
          occ_func_3_0(12) + occ_func_3_0(6) + occ_func_3_0(4)) /
         12.;
}

/**** Basis functions for orbit 5****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 1.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_5_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(16) +
          occ_func_2_0(1) * occ_func_2_0(17)) /
         2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_5_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(16) +
          occ_func_2_0(14) * occ_func_2_0(0)) /
         2.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_5_0_at_1() const {
  return (occ_func_2_0(1) * occ_func_2_0(17) +
          occ_func_2_0(15) * occ_func_2_0(1)) /
         2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_5_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(16) + occ_func_2_0(14)) / 2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_5_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(17) + occ_func_2_0(15)) / 2.;
}

/**** Basis functions for orbit 6****
0.3333333 0.6666666 0.2500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_6_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_2_0(27) +
          occ_func_2_0(1) * occ_func_2_0(25) +
          occ_func_2_0(24) * occ_func_2_0(0) +
          occ_func_2_0(26) * occ_func_2_0(0) +
          occ_func_2_0(29) * occ_func_2_0(1)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_6_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_2_0(0) +
          occ_func_2_0(24) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(20)) /
         6.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_6_0_at_1() const {
  return (occ_func_2_0(1) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_2_0(1) +
          occ_func_2_0(29) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(19)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_6_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(28) + occ_func_2_0(18) + occ_func_2_0(24) +
          occ_func_2_0(22) + occ_func_2_0(26) + occ_func_2_0(20)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_6_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(27) + occ_func_2_0(21) + occ_func_2_0(25) +
          occ_func_2_0(23) + occ_func_2_0(29) + occ_func_2_0(19)) /
         6.;
}

/**** Basis functions for orbit 7****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_7_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) +
          occ_func_2_0(49) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(51) +
          occ_func_2_0(52) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(46) +
          occ_func_2_0(0) * occ_func_2_0(44) +
          occ_func_2_0(50) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(48) +
          occ_func_2_0(45) * occ_func_2_0(1) +
          occ_func_2_0(47) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(53) +
          occ_func_2_0(43) * occ_func_2_0(1)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_7_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) +
          occ_func_2_0(40) * occ_func_2_0(0) +
          occ_func_2_0(52) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(30) +
          occ_func_2_0(0) * occ_func_2_0(46) +
          occ_func_2_0(36) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(44) +
          occ_func_2_0(38) * occ_func_2_0(0) +
          occ_func_2_0(50) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(32) +
          occ_func_2_0(0) * occ_func_2_0(48) +
          occ_func_2_0(34) * occ_func_2_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_7_0_at_1() const {
  return (occ_func_2_0(49) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(35) +
          occ_func_2_0(1) * occ_func_2_0(51) +
          occ_func_2_0(33) * occ_func_2_0(1) +
          occ_func_2_0(45) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(39) +
          occ_func_2_0(47) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(37) +
          occ_func_2_0(1) * occ_func_2_0(53) +
          occ_func_2_0(31) * occ_func_2_0(1) +
          occ_func_2_0(43) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(41)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_7_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(42) + occ_func_2_0(40) + occ_func_2_0(52) +
          occ_func_2_0(30) + occ_func_2_0(46) + occ_func_2_0(36) +
          occ_func_2_0(44) + occ_func_2_0(38) + occ_func_2_0(50) +
          occ_func_2_0(32) + occ_func_2_0(48) + occ_func_2_0(34)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_7_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(49) + occ_func_2_0(35) + occ_func_2_0(51) +
          occ_func_2_0(33) + occ_func_2_0(45) + occ_func_2_0(39) +
          occ_func_2_0(47) + occ_func_2_0(37) + occ_func_2_0(53) +
          occ_func_2_0(31) + occ_func_2_0(43) + occ_func_2_0(41)) /
         12.;
}

/**** Basis functions for orbit 8****
0.3333333 0.6666666 0.2500000 Va  O

2.3333333 1.6666667 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_8_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(87) +
          occ_func_2_0(1) * occ_func_3_0(26) +
          occ_func_2_0(1) * occ_func_3_0(24) +
          occ_func_2_0(80) * occ_func_3_0(1) +
          occ_func_2_0(84) * occ_func_3_0(1) +
          occ_func_2_0(28) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(25) +
          occ_func_2_0(0) * occ_func_3_0(27) +
          occ_func_2_0(29) * occ_func_3_0(0) +
          occ_func_2_0(83) * occ_func_3_0(0) +
          occ_func_2_0(79) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(88)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_8_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(87) +
          occ_func_2_0(21) * occ_func_3_0(0) +
          occ_func_2_0(23) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(75) +
          occ_func_2_0(0) * occ_func_3_0(71) +
          occ_func_2_0(0) * occ_func_3_0(19) +
          occ_func_2_0(0) * occ_func_3_0(25) +
          occ_func_2_0(0) * occ_func_3_0(27) +
          occ_func_2_0(29) * occ_func_3_0(0) +
          occ_func_2_0(83) * occ_func_3_0(0) +
          occ_func_2_0(79) * occ_func_3_0(0) +
          occ_func_2_0(67) * occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_8_0_at_1() const {
  return (occ_func_2_0(68) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(26) +
          occ_func_2_0(1) * occ_func_3_0(24) +
          occ_func_2_0(80) * occ_func_3_0(1) +
          occ_func_2_0(84) * occ_func_3_0(1) +
          occ_func_2_0(28) * occ_func_3_0(1) +
          occ_func_2_0(22) * occ_func_3_0(1) +
          occ_func_2_0(20) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(18) +
          occ_func_2_0(1) * occ_func_3_0(72) +
          occ_func_2_0(1) * occ_func_3_0(76) +
          occ_func_2_0(1) * occ_func_3_0(88)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_8_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(87) + occ_func_2_0(21) + occ_func_2_0(23) +
          occ_func_3_0(75) + occ_func_3_0(71) + occ_func_3_0(19) +
          occ_func_3_0(25) + occ_func_3_0(27) + occ_func_2_0(29) +
          occ_func_2_0(83) + occ_func_2_0(79) + occ_func_2_0(67)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_8_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(68) + occ_func_3_0(26) + occ_func_3_0(24) +
          occ_func_2_0(80) + occ_func_2_0(84) + occ_func_2_0(28) +
          occ_func_2_0(22) + occ_func_2_0(20) + occ_func_3_0(18) +
          occ_func_3_0(72) + occ_func_3_0(76) + occ_func_3_0(88)) /
         12.;
}

/**** Basis functions for orbit 9****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 2.6666666 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_9_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(60) +
          occ_func_2_0(63) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(65) +
          occ_func_2_0(64) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(62) +
          occ_func_2_0(61) * occ_func_2_0(1)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_9_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(60) +
          occ_func_2_0(58) * occ_func_2_0(0) +
          occ_func_2_0(64) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(54) +
          occ_func_2_0(0) * occ_func_2_0(62) +
          occ_func_2_0(56) * occ_func_2_0(0)) /
         6.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_9_0_at_1() const {
  return (occ_func_2_0(63) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(57) +
          occ_func_2_0(1) * occ_func_2_0(65) +
          occ_func_2_0(55) * occ_func_2_0(1) +
          occ_func_2_0(61) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(59)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_9_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(60) + occ_func_2_0(58) + occ_func_2_0(64) +
          occ_func_2_0(54) + occ_func_2_0(62) + occ_func_2_0(56)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_9_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(63) + occ_func_2_0(57) + occ_func_2_0(65) +
          occ_func_2_0(55) + occ_func_2_0(61) + occ_func_2_0(59)) /
         6.;
}

/**** Basis functions for orbit 10****
0.3333333 0.6666666 0.2500000 Va  O

2.3333333 2.6666666 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_10_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(111) +
          occ_func_2_0(1) * occ_func_3_0(60) +
          occ_func_2_0(1) * occ_func_3_0(62) +
          occ_func_2_0(108) * occ_func_3_0(1) +
          occ_func_2_0(104) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(65) +
          occ_func_2_0(0) * occ_func_3_0(107) +
          occ_func_2_0(0) * occ_func_3_0(103) +
          occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(64) +
          occ_func_2_0(61) * occ_func_3_0(0) +
          occ_func_2_0(63) * occ_func_3_0(0)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_10_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(111) +
          occ_func_2_0(59) * occ_func_3_0(0) +
          occ_func_2_0(57) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(95) +
          occ_func_2_0(0) * occ_func_3_0(99) +
          occ_func_2_0(0) * occ_func_3_0(65) +
          occ_func_2_0(0) * occ_func_3_0(107) +
          occ_func_2_0(0) * occ_func_3_0(103) +
          occ_func_2_0(0) * occ_func_3_0(91) +
          occ_func_2_0(55) * occ_func_3_0(0) +
          occ_func_2_0(61) * occ_func_3_0(0) +
          occ_func_2_0(63) * occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_10_0_at_1() const {
  return (occ_func_2_0(92) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(60) +
          occ_func_2_0(1) * occ_func_3_0(62) +
          occ_func_2_0(108) * occ_func_3_0(1) +
          occ_func_2_0(104) * occ_func_3_0(1) +
          occ_func_2_0(54) * occ_func_3_0(1) +
          occ_func_2_0(96) * occ_func_3_0(1) +
          occ_func_2_0(100) * occ_func_3_0(1) +
          occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(64) +
          occ_func_2_0(1) * occ_func_3_0(58) +
          occ_func_2_0(1) * occ_func_3_0(56)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_10_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(111) + occ_func_2_0(59) + occ_func_2_0(57) +
          occ_func_3_0(95) + occ_func_3_0(99) + occ_func_3_0(65) +
          occ_func_3_0(107) + occ_func_3_0(103) + occ_func_3_0(91) +
          occ_func_2_0(55) + occ_func_2_0(61) + occ_func_2_0(63)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_10_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(92) + occ_func_3_0(60) + occ_func_3_0(62) +
          occ_func_2_0(108) + occ_func_2_0(104) + occ_func_2_0(54) +
          occ_func_2_0(96) + occ_func_2_0(100) + occ_func_2_0(112) +
          occ_func_3_0(64) + occ_func_3_0(58) + occ_func_3_0(56)) /
         12.;
}

/**** Basis functions for orbit 11****
0.3333333 0.6666666 0.2500000 Va  O

2.3333333 1.6666667 1.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_11_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(88) +
          occ_func_2_0(1) * occ_func_2_0(85) +
          occ_func_2_0(1) * occ_func_2_0(81) +
          occ_func_2_0(78) * occ_func_2_0(0) +
          occ_func_2_0(82) * occ_func_2_0(0) +
          occ_func_2_0(87) * occ_func_2_0(1)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_11_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(88) +
          occ_func_2_0(66) * occ_func_2_0(0) +
          occ_func_2_0(78) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(76) +
          occ_func_2_0(82) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(72)) /
         6.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_11_0_at_1() const {
  return (occ_func_2_0(1) * occ_func_2_0(85) +
          occ_func_2_0(71) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(81) +
          occ_func_2_0(75) * occ_func_2_0(1) +
          occ_func_2_0(87) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(69)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_11_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(88) + occ_func_2_0(66) + occ_func_2_0(78) +
          occ_func_2_0(76) + occ_func_2_0(82) + occ_func_2_0(72)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_11_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(85) + occ_func_2_0(71) + occ_func_2_0(81) +
          occ_func_2_0(75) + occ_func_2_0(87) + occ_func_2_0(69)) /
         6.;
}

/**** Basis functions for orbit 12****
0.3333333 0.6666666 0.2500000 Va  O

2.3333333 1.6666667 -0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_12_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(86) +
          occ_func_2_0(1) * occ_func_2_0(83) +
          occ_func_2_0(1) * occ_func_2_0(79) +
          occ_func_2_0(80) * occ_func_2_0(0) +
          occ_func_2_0(84) * occ_func_2_0(0) +
          occ_func_2_0(89) * occ_func_2_0(1)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_12_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(86) +
          occ_func_2_0(68) * occ_func_2_0(0) +
          occ_func_2_0(80) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(74) +
          occ_func_2_0(84) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(70)) /
         6.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_12_0_at_1() const {
  return (occ_func_2_0(1) * occ_func_2_0(83) +
          occ_func_2_0(73) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(79) +
          occ_func_2_0(77) * occ_func_2_0(1) +
          occ_func_2_0(89) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(67)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_12_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(86) + occ_func_2_0(68) + occ_func_2_0(80) +
          occ_func_2_0(74) + occ_func_2_0(84) + occ_func_2_0(70)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_12_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(83) + occ_func_2_0(73) + occ_func_2_0(79) +
          occ_func_2_0(77) + occ_func_2_0(89) + occ_func_2_0(67)) /
         6.;
}

/**** Basis functions for orbit 13****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 1.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_13_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(17) +
          occ_func_2_0(1) * occ_func_3_0(200)) /
         2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_13_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(17) +
          occ_func_2_0(199) * occ_func_3_0(0)) /
         2.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_13_0_at_1() const {
  return (occ_func_2_0(14) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(200)) /
         2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_13_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(17) + occ_func_2_0(199)) / 2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_13_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(14) + occ_func_3_0(200)) / 2.;
}

/**** Basis functions for orbit 14****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 2.6666666 -0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_14_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(102) +
          occ_func_2_0(109) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(111) +
          occ_func_2_0(112) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(106) +
          occ_func_2_0(0) * occ_func_2_0(104) +
          occ_func_2_0(110) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(108) +
          occ_func_2_0(105) * occ_func_2_0(1) +
          occ_func_2_0(107) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(113) +
          occ_func_2_0(103) * occ_func_2_0(1)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_14_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(102) +
          occ_func_2_0(100) * occ_func_2_0(0) +
          occ_func_2_0(112) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(90) +
          occ_func_2_0(0) * occ_func_2_0(106) +
          occ_func_2_0(96) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(104) +
          occ_func_2_0(98) * occ_func_2_0(0) +
          occ_func_2_0(110) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(92) +
          occ_func_2_0(0) * occ_func_2_0(108) +
          occ_func_2_0(94) * occ_func_2_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_14_0_at_1() const {
  return (occ_func_2_0(109) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(95) +
          occ_func_2_0(1) * occ_func_2_0(111) +
          occ_func_2_0(93) * occ_func_2_0(1) +
          occ_func_2_0(105) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(99) +
          occ_func_2_0(107) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(97) +
          occ_func_2_0(1) * occ_func_2_0(113) +
          occ_func_2_0(91) * occ_func_2_0(1) +
          occ_func_2_0(103) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(101)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_14_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(102) + occ_func_2_0(100) + occ_func_2_0(112) +
          occ_func_2_0(90) + occ_func_2_0(106) + occ_func_2_0(96) +
          occ_func_2_0(104) + occ_func_2_0(98) + occ_func_2_0(110) +
          occ_func_2_0(92) + occ_func_2_0(108) + occ_func_2_0(94)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_14_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(109) + occ_func_2_0(95) + occ_func_2_0(111) +
          occ_func_2_0(93) + occ_func_2_0(105) + occ_func_2_0(99) +
          occ_func_2_0(107) + occ_func_2_0(97) + occ_func_2_0(113) +
          occ_func_2_0(91) + occ_func_2_0(103) + occ_func_2_0(101)) /
         12.;
}

/**** Basis functions for orbit 15****
0.3333333 0.6666666 0.2500000 Va  O

1.3333333 1.6666667 -1.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_15_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(223) +
          occ_func_2_0(1) * occ_func_3_0(42) +
          occ_func_2_0(1) * occ_func_3_0(46) +
          occ_func_2_0(220) * occ_func_3_0(1) +
          occ_func_2_0(216) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(53) +
          occ_func_2_0(0) * occ_func_3_0(219) +
          occ_func_2_0(0) * occ_func_3_0(215) +
          occ_func_2_0(224) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(50) +
          occ_func_2_0(45) * occ_func_3_0(0) +
          occ_func_2_0(49) * occ_func_3_0(0)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_15_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(223) +
          occ_func_2_0(41) * occ_func_3_0(0) +
          occ_func_2_0(37) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(207) +
          occ_func_2_0(0) * occ_func_3_0(211) +
          occ_func_2_0(0) * occ_func_3_0(53) +
          occ_func_2_0(0) * occ_func_3_0(219) +
          occ_func_2_0(0) * occ_func_3_0(215) +
          occ_func_2_0(0) * occ_func_3_0(203) +
          occ_func_2_0(33) * occ_func_3_0(0) +
          occ_func_2_0(45) * occ_func_3_0(0) +
          occ_func_2_0(49) * occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_15_0_at_1() const {
  return (occ_func_2_0(204) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(42) +
          occ_func_2_0(1) * occ_func_3_0(46) +
          occ_func_2_0(220) * occ_func_3_0(1) +
          occ_func_2_0(216) * occ_func_3_0(1) +
          occ_func_2_0(30) * occ_func_3_0(1) +
          occ_func_2_0(208) * occ_func_3_0(1) +
          occ_func_2_0(212) * occ_func_3_0(1) +
          occ_func_2_0(224) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(50) +
          occ_func_2_0(1) * occ_func_3_0(38) +
          occ_func_2_0(1) * occ_func_3_0(34)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_15_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(223) + occ_func_2_0(41) + occ_func_2_0(37) +
          occ_func_3_0(207) + occ_func_3_0(211) + occ_func_3_0(53) +
          occ_func_3_0(219) + occ_func_3_0(215) + occ_func_3_0(203) +
          occ_func_2_0(33) + occ_func_2_0(45) + occ_func_2_0(49)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_15_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(204) + occ_func_3_0(42) + occ_func_3_0(46) +
          occ_func_2_0(220) + occ_func_2_0(216) + occ_func_2_0(30) +
          occ_func_2_0(208) + occ_func_2_0(212) + occ_func_2_0(224) +
          occ_func_3_0(50) + occ_func_3_0(38) + occ_func_3_0(34)) /
         12.;
}

/**** Basis functions for orbit 16****
0.3333333 0.6666666 0.2500000 Va  O

1.3333333 -1.3333333 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_16_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(126) +
          occ_func_2_0(1) * occ_func_2_0(135) +
          occ_func_2_0(133) * occ_func_2_0(1) +
          occ_func_2_0(0) * occ_func_2_0(132) +
          occ_func_2_0(134) * occ_func_2_0(0) +
          occ_func_2_0(128) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(136) +
          occ_func_2_0(130) * occ_func_2_0(0) +
          occ_func_2_0(127) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(131) +
          occ_func_2_0(137) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(129)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_16_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(126) +
          occ_func_2_0(124) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(132) +
          occ_func_2_0(118) * occ_func_2_0(0) +
          occ_func_2_0(134) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(116) +
          occ_func_2_0(128) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(122) +
          occ_func_2_0(0) * occ_func_2_0(136) +
          occ_func_2_0(114) * occ_func_2_0(0) +
          occ_func_2_0(130) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(120)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_16_0_at_1() const {
  return (occ_func_2_0(1) * occ_func_2_0(135) +
          occ_func_2_0(117) * occ_func_2_0(1) +
          occ_func_2_0(133) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(119) +
          occ_func_2_0(127) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(125) +
          occ_func_2_0(1) * occ_func_2_0(131) +
          occ_func_2_0(121) * occ_func_2_0(1) +
          occ_func_2_0(137) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(115) +
          occ_func_2_0(1) * occ_func_2_0(129) +
          occ_func_2_0(123) * occ_func_2_0(1)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_16_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(126) + occ_func_2_0(124) + occ_func_2_0(132) +
          occ_func_2_0(118) + occ_func_2_0(134) + occ_func_2_0(116) +
          occ_func_2_0(128) + occ_func_2_0(122) + occ_func_2_0(136) +
          occ_func_2_0(114) + occ_func_2_0(130) + occ_func_2_0(120)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_16_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(135) + occ_func_2_0(117) + occ_func_2_0(133) +
          occ_func_2_0(119) + occ_func_2_0(127) + occ_func_2_0(125) +
          occ_func_2_0(131) + occ_func_2_0(121) + occ_func_2_0(137) +
          occ_func_2_0(115) + occ_func_2_0(129) + occ_func_2_0(123)) /
         12.;
}

/**** Basis functions for orbit 17****
0.3333333 0.6666666 0.2500000 Va  O

1.3333333 -1.3333333 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_17_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(175) +
          occ_func_2_0(1) * occ_func_3_0(134) +
          occ_func_2_0(133) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(187) +
          occ_func_2_0(192) * occ_func_3_0(1) +
          occ_func_2_0(128) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(137) +
          occ_func_2_0(130) * occ_func_3_0(1) +
          occ_func_2_0(127) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(184) +
          occ_func_2_0(195) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(180) +
          occ_func_2_0(0) * occ_func_3_0(179) +
          occ_func_2_0(196) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(183) +
          occ_func_2_0(1) * occ_func_3_0(176) +
          occ_func_2_0(131) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(136) +
          occ_func_2_0(129) * occ_func_3_0(0) +
          occ_func_2_0(132) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(135) +
          occ_func_2_0(191) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(188) +
          occ_func_2_0(126) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_17_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(175) +
          occ_func_2_0(117) * occ_func_3_0(0) +
          occ_func_2_0(133) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(187) +
          occ_func_2_0(0) * occ_func_3_0(155) +
          occ_func_2_0(0) * occ_func_3_0(123) +
          occ_func_2_0(0) * occ_func_3_0(137) +
          occ_func_2_0(0) * occ_func_3_0(121) +
          occ_func_2_0(127) * occ_func_3_0(0) +
          occ_func_2_0(163) * occ_func_3_0(0) +
          occ_func_2_0(195) * occ_func_3_0(0) +
          occ_func_2_0(167) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(179) +
          occ_func_2_0(0) * occ_func_3_0(151) +
          occ_func_2_0(0) * occ_func_3_0(183) +
          occ_func_2_0(171) * occ_func_3_0(0) +
          occ_func_2_0(131) * occ_func_3_0(0) +
          occ_func_2_0(115) * occ_func_3_0(0) +
          occ_func_2_0(129) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(119) +
          occ_func_2_0(0) * occ_func_3_0(135) +
          occ_func_2_0(191) * occ_func_3_0(0) +
          occ_func_2_0(159) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(125)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_17_0_at_1() const {
  return (occ_func_2_0(172) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(134) +
          occ_func_2_0(1) * occ_func_3_0(118) +
          occ_func_2_0(160) * occ_func_3_0(1) +
          occ_func_2_0(192) * occ_func_3_0(1) +
          occ_func_2_0(128) * occ_func_3_0(1) +
          occ_func_2_0(114) * occ_func_3_0(1) +
          occ_func_2_0(130) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(124) +
          occ_func_2_0(1) * occ_func_3_0(184) +
          occ_func_2_0(1) * occ_func_3_0(152) +
          occ_func_2_0(1) * occ_func_3_0(180) +
          occ_func_2_0(168) * occ_func_3_0(1) +
          occ_func_2_0(196) * occ_func_3_0(1) +
          occ_func_2_0(164) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(176) +
          occ_func_2_0(1) * occ_func_3_0(120) +
          occ_func_2_0(1) * occ_func_3_0(136) +
          occ_func_2_0(1) * occ_func_3_0(122) +
          occ_func_2_0(132) * occ_func_3_0(1) +
          occ_func_2_0(116) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(156) +
          occ_func_2_0(1) * occ_func_3_0(188) +
          occ_func_2_0(126) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_17_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(175) + occ_func_2_0(117) + occ_func_2_0(133) +
          occ_func_3_0(187) + occ_func_3_0(155) + occ_func_3_0(123) +
          occ_func_3_0(137) + occ_func_3_0(121) + occ_func_2_0(127) +
          occ_func_2_0(163) + occ_func_2_0(195) + occ_func_2_0(167) +
          occ_func_3_0(179) + occ_func_3_0(151) + occ_func_3_0(183) +
          occ_func_2_0(171) + occ_func_2_0(131) + occ_func_2_0(115) +
          occ_func_2_0(129) + occ_func_3_0(119) + occ_func_3_0(135) +
          occ_func_2_0(191) + occ_func_2_0(159) + occ_func_3_0(125)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_17_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(172) + occ_func_3_0(134) + occ_func_3_0(118) +
          occ_func_2_0(160) + occ_func_2_0(192) + occ_func_2_0(128) +
          occ_func_2_0(114) + occ_func_2_0(130) + occ_func_3_0(124) +
          occ_func_3_0(184) + occ_func_3_0(152) + occ_func_3_0(180) +
          occ_func_2_0(168) + occ_func_2_0(196) + occ_func_2_0(164) +
          occ_func_3_0(176) + occ_func_3_0(120) + occ_func_3_0(136) +
          occ_func_3_0(122) + occ_func_2_0(132) + occ_func_2_0(116) +
          occ_func_3_0(156) + occ_func_3_0(188) + occ_func_2_0(126)) /
         24.;
}

/**** Basis functions for orbit 18****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

1.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_18_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(12) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(13) +
          occ_func_2_0(10) * occ_func_2_0(12) * occ_func_2_0(0) +
          occ_func_2_0(13) * occ_func_2_0(9) * occ_func_2_0(1)) /
         4.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_18_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(12) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(10) +
          occ_func_2_0(2) * occ_func_2_0(4) * occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_2_0(12) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(4) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(2)) /
         4.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_18_0_at_1() const {
  return (occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(13) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_2_0(3) * occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_2_0(9) * occ_func_2_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(7) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(3)) /
         4.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_18_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) * occ_func_2_0(12) +
          occ_func_2_0(6) * occ_func_2_0(10) +
          occ_func_2_0(2) * occ_func_2_0(4) +
          occ_func_2_0(10) * occ_func_2_0(12) +
          occ_func_2_0(8) * occ_func_2_0(4) +
          occ_func_2_0(6) * occ_func_2_0(2)) /
         4.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_18_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(11) * occ_func_2_0(13) +
          occ_func_2_0(5) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_2_0(3) +
          occ_func_2_0(13) * occ_func_2_0(9) +
          occ_func_2_0(11) * occ_func_2_0(7) +
          occ_func_2_0(5) * occ_func_2_0(3)) /
         4.;
}

/**** Basis functions for orbit 19****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 0.7500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_19_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(8) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(13) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(10) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(44) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_2_0(0) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(48) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_2_0(1) +
          occ_func_2_0(11) * occ_func_3_0(10) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(13) +
          occ_func_2_0(9) * occ_func_3_0(8) * occ_func_2_0(1) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(12) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(0) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(9) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(11) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(9) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(52) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_2_0(0) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(11) +
          occ_func_2_0(13) * occ_func_3_0(12) * occ_func_2_0(1) +
          occ_func_2_0(8) * occ_func_3_0(43) * occ_func_2_0(0)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_19_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_3_0(7) * occ_func_2_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(35) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(51) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(2) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_3_0(5) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_3_0(39) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(2) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_3_0(35) * occ_func_2_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(39) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(5) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(13) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(7) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(6) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(4) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(9) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(47) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(31) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(43) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(31) * occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(4) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(11) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(3) +
          occ_func_2_0(8) * occ_func_3_0(43) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(6)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_19_0_at_1() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(8) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(5) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(32) * occ_func_2_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(2) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(10) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(44) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(32) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(48) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(7) +
          occ_func_2_0(11) * occ_func_3_0(10) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(5) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(2) * occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_3_0(8) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(7) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(6) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(12) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(4) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(6) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(36) * occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(3) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(40) * occ_func_2_0(1) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(52) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(36) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(4) * occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_3_0(12) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(3) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(40)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_19_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(1) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_3_0(7) +
          occ_func_2_0(15) * occ_func_2_0(35) +
          occ_func_2_0(15) * occ_func_2_0(51) +
          occ_func_2_0(12) * occ_func_3_0(13) +
          occ_func_3_0(1) * occ_func_2_0(2) +
          occ_func_3_0(1) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_3_0(5) +
          occ_func_3_0(15) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_3_0(39) +
          occ_func_2_0(12) * occ_func_3_0(51) +
          occ_func_3_0(15) * occ_func_2_0(2) +
          occ_func_3_0(15) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_3_0(35) +
          occ_func_2_0(15) * occ_func_2_0(39) +
          occ_func_2_0(1) * occ_func_2_0(5) +
          occ_func_2_0(1) * occ_func_2_0(13) +
          occ_func_2_0(1) * occ_func_2_0(7) +
          occ_func_2_0(8) * occ_func_3_0(9) +
          occ_func_3_0(1) * occ_func_2_0(6) +
          occ_func_3_0(1) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(3) +
          occ_func_2_0(10) * occ_func_3_0(11) +
          occ_func_3_0(1) * occ_func_2_0(4) +
          occ_func_2_0(1) * occ_func_2_0(9) +
          occ_func_2_0(15) * occ_func_2_0(47) +
          occ_func_2_0(15) * occ_func_2_0(31) +
          occ_func_2_0(15) * occ_func_2_0(43) +
          occ_func_3_0(15) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(31) +
          occ_func_2_0(10) * occ_func_3_0(47) +
          occ_func_3_0(15) * occ_func_2_0(4) +
          occ_func_2_0(1) * occ_func_2_0(11) +
          occ_func_2_0(1) * occ_func_2_0(3) +
          occ_func_2_0(8) * occ_func_3_0(43) +
          occ_func_3_0(15) * occ_func_2_0(6)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_19_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(0) * occ_func_2_0(8) +
          occ_func_2_0(11) * occ_func_3_0(48) +
          occ_func_3_0(16) * occ_func_2_0(5) +
          occ_func_3_0(16) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(32) +
          occ_func_2_0(0) * occ_func_2_0(2) +
          occ_func_2_0(0) * occ_func_2_0(10) +
          occ_func_2_0(16) * occ_func_2_0(44) +
          occ_func_2_0(16) * occ_func_2_0(32) +
          occ_func_2_0(16) * occ_func_2_0(48) +
          occ_func_2_0(9) * occ_func_3_0(44) +
          occ_func_3_0(16) * occ_func_2_0(7) +
          occ_func_2_0(11) * occ_func_3_0(10) +
          occ_func_3_0(0) * occ_func_2_0(5) +
          occ_func_3_0(0) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(2) +
          occ_func_2_0(9) * occ_func_3_0(8) +
          occ_func_3_0(0) * occ_func_2_0(7) +
          occ_func_2_0(0) * occ_func_2_0(6) +
          occ_func_2_0(0) * occ_func_2_0(12) +
          occ_func_2_0(0) * occ_func_2_0(4) +
          occ_func_3_0(0) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(6) +
          occ_func_3_0(16) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(36) +
          occ_func_2_0(13) * occ_func_3_0(52) +
          occ_func_3_0(16) * occ_func_2_0(3) +
          occ_func_3_0(16) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(40) +
          occ_func_2_0(16) * occ_func_2_0(52) +
          occ_func_2_0(16) * occ_func_2_0(36) +
          occ_func_3_0(0) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(4) +
          occ_func_2_0(13) * occ_func_3_0(12) +
          occ_func_3_0(0) * occ_func_2_0(3) +
          occ_func_2_0(16) * occ_func_2_0(40)) /
         24.;
}

/**** Basis functions for orbit 20****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

1.3333333 1.6666667 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_20_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(51) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(10) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(43) +
          occ_func_2_0(44) * occ_func_2_0(52) * occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_2_0(12) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(11) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(13) +
          occ_func_2_0(13) * occ_func_2_0(11) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_2_0(43) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(44) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(52)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_20_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(51) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(47) +
          occ_func_2_0(7) * occ_func_2_0(3) * occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(43) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_3_0(35) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(39) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(31) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(5) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(3) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(11) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_3_0(7) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(13) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(9) +
          occ_func_2_0(13) * occ_func_2_0(11) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_2_0(43) * occ_func_3_0(0) +
          occ_func_2_0(39) * occ_func_2_0(47) * occ_func_3_0(0) +
          occ_func_2_0(35) * occ_func_2_0(31) * occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_20_0_at_1() const {
  return (occ_func_2_0(32) * occ_func_2_0(36) * occ_func_3_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_3_0(8) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(10) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_3_0(6) +
          occ_func_2_0(48) * occ_func_2_0(40) * occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_2_0(52) * occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_2_0(12) * occ_func_3_0(1) +
          occ_func_2_0(8) * occ_func_2_0(4) * occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_2_0(6) * occ_func_3_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(4) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_3_0(2) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(40) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_3_0(32) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(44) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_3_0(36) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_3_0(48)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_20_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) * occ_func_3_0(51) +
          occ_func_2_0(6) * occ_func_3_0(47) +
          occ_func_2_0(7) * occ_func_2_0(3) +
          occ_func_2_0(5) * occ_func_2_0(9) +
          occ_func_2_0(12) * occ_func_3_0(43) +
          occ_func_2_0(2) * occ_func_3_0(35) +
          occ_func_2_0(10) * occ_func_3_0(39) +
          occ_func_2_0(4) * occ_func_3_0(31) +
          occ_func_2_0(8) * occ_func_3_0(5) +
          occ_func_2_0(6) * occ_func_3_0(3) +
          occ_func_2_0(12) * occ_func_3_0(11) +
          occ_func_2_0(2) * occ_func_3_0(7) +
          occ_func_2_0(10) * occ_func_3_0(13) +
          occ_func_2_0(4) * occ_func_3_0(9) +
          occ_func_2_0(13) * occ_func_2_0(11) +
          occ_func_2_0(51) * occ_func_2_0(43) +
          occ_func_2_0(39) * occ_func_2_0(47) +
          occ_func_2_0(35) * occ_func_2_0(31)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_20_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(32) * occ_func_2_0(36) +
          occ_func_2_0(11) * occ_func_3_0(12) +
          occ_func_2_0(5) * occ_func_3_0(8) +
          occ_func_2_0(13) * occ_func_3_0(10) +
          occ_func_2_0(3) * occ_func_3_0(6) +
          occ_func_2_0(48) * occ_func_2_0(40) +
          occ_func_2_0(44) * occ_func_2_0(52) +
          occ_func_2_0(10) * occ_func_2_0(12) +
          occ_func_2_0(8) * occ_func_2_0(4) +
          occ_func_2_0(2) * occ_func_2_0(6) +
          occ_func_2_0(9) * occ_func_3_0(4) +
          occ_func_2_0(7) * occ_func_3_0(2) +
          occ_func_2_0(11) * occ_func_3_0(40) +
          occ_func_2_0(5) * occ_func_3_0(32) +
          occ_func_2_0(13) * occ_func_3_0(44) +
          occ_func_2_0(3) * occ_func_3_0(36) +
          occ_func_2_0(9) * occ_func_3_0(52) +
          occ_func_2_0(7) * occ_func_3_0(48)) /
         12.;
}

/**** Basis functions for orbit 21****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

1.3333333 1.6666667 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_21_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(13) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(48) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(9) +
          occ_func_2_0(8) * occ_func_2_0(12) * occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_2_0(52) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(47) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(51) +
          occ_func_2_0(51) * occ_func_2_0(47) * occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(8) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(12)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_21_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(13) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(11) +
          occ_func_2_0(39) * occ_func_2_0(31) * occ_func_3_0(0) +
          occ_func_2_0(35) * occ_func_2_0(43) * occ_func_3_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(9) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_3_0(5) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(7) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(3) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(35) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(31) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(47) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_3_0(39) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(51) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(43) +
          occ_func_2_0(51) * occ_func_2_0(47) * occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(7) * occ_func_2_0(11) * occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_2_0(3) * occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_21_0_at_1() const {
  return (occ_func_2_0(2) * occ_func_2_0(4) * occ_func_3_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(48) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_3_0(40) +
          occ_func_2_0(10) * occ_func_2_0(6) * occ_func_3_0(1) +
          occ_func_2_0(8) * occ_func_2_0(12) * occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_2_0(52) * occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_2_0(36) * occ_func_3_0(1) +
          occ_func_2_0(32) * occ_func_2_0(40) * occ_func_3_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(36) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_3_0(32) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(6) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_3_0(2) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(8) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_3_0(4) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_3_0(10)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_21_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) * occ_func_3_0(13) +
          occ_func_2_0(6) * occ_func_3_0(11) +
          occ_func_2_0(39) * occ_func_2_0(31) +
          occ_func_2_0(35) * occ_func_2_0(43) +
          occ_func_2_0(12) * occ_func_3_0(9) +
          occ_func_2_0(2) * occ_func_3_0(5) +
          occ_func_2_0(10) * occ_func_3_0(7) +
          occ_func_2_0(4) * occ_func_3_0(3) +
          occ_func_2_0(8) * occ_func_3_0(35) +
          occ_func_2_0(6) * occ_func_3_0(31) +
          occ_func_2_0(12) * occ_func_3_0(47) +
          occ_func_2_0(2) * occ_func_3_0(39) +
          occ_func_2_0(10) * occ_func_3_0(51) +
          occ_func_2_0(4) * occ_func_3_0(43) +
          occ_func_2_0(51) * occ_func_2_0(47) +
          occ_func_2_0(13) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_2_0(3)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_21_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(2) * occ_func_2_0(4) +
          occ_func_2_0(11) * occ_func_3_0(52) +
          occ_func_2_0(5) * occ_func_3_0(44) +
          occ_func_2_0(13) * occ_func_3_0(48) +
          occ_func_2_0(3) * occ_func_3_0(40) +
          occ_func_2_0(10) * occ_func_2_0(6) +
          occ_func_2_0(8) * occ_func_2_0(12) +
          occ_func_2_0(48) * occ_func_2_0(52) +
          occ_func_2_0(44) * occ_func_2_0(36) +
          occ_func_2_0(32) * occ_func_2_0(40) +
          occ_func_2_0(9) * occ_func_3_0(36) +
          occ_func_2_0(7) * occ_func_3_0(32) +
          occ_func_2_0(11) * occ_func_3_0(6) +
          occ_func_2_0(5) * occ_func_3_0(2) +
          occ_func_2_0(13) * occ_func_3_0(8) +
          occ_func_2_0(3) * occ_func_3_0(4) +
          occ_func_2_0(9) * occ_func_3_0(12) +
          occ_func_2_0(7) * occ_func_3_0(10)) /
         12.;
}

/**** Basis functions for orbit 22****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 0.7500000 Va  O

0.3333333 0.6666666 1.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_22_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(17)) /
         2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_22_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_2_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(1)) /
         2.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_22_0_at_1() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(17) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(1)) /
         2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_22_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(1) * occ_func_2_0(16) +
          occ_func_2_0(14) * occ_func_3_0(15) +
          occ_func_2_0(15) * occ_func_2_0(1)) /
         2.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_22_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(0) * occ_func_2_0(16) +
          occ_func_3_0(16) * occ_func_2_0(17) +
          occ_func_2_0(15) * occ_func_3_0(0)) /
         2.;
}

/**** Basis functions for orbit 23****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 1.2500000 Va  O

0.3333333 1.6666667 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_23_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(9) +
          occ_func_2_0(47) * occ_func_2_0(11) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(52) +
          occ_func_2_0(12) * occ_func_2_0(52) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(11) +
          occ_func_2_0(43) * occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(8) * occ_func_2_0(44) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(13) +
          occ_func_2_0(10) * occ_func_2_0(48) * occ_func_3_0(1) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(48) +
          occ_func_2_0(51) * occ_func_2_0(13) * occ_func_3_0(0)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_23_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(9) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(43) +
          occ_func_2_0(47) * occ_func_2_0(11) * occ_func_3_0(0) +
          occ_func_2_0(31) * occ_func_2_0(3) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(3) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(31) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(11) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(47) +
          occ_func_2_0(43) * occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(7) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(39) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(13) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(51) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(5) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(35) +
          occ_func_2_0(7) * occ_func_2_0(39) * occ_func_3_0(0) +
          occ_func_2_0(35) * occ_func_2_0(5) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_2_0(13) * occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_23_0_at_1() const {
  return (occ_func_2_0(6) * occ_func_2_0(40) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(36) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(4) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(52) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(12) +
          occ_func_2_0(12) * occ_func_2_0(52) * occ_func_3_0(1) +
          occ_func_2_0(4) * occ_func_2_0(36) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(40) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(6) +
          occ_func_2_0(8) * occ_func_2_0(44) * occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_2_0(32) * occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_2_0(48) * occ_func_3_0(1) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_3_0(8) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(48) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(10) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(32) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(2)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_23_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(16) * occ_func_3_0(9) +
          occ_func_2_0(14) * occ_func_3_0(43) +
          occ_func_2_0(47) * occ_func_2_0(11) +
          occ_func_2_0(31) * occ_func_2_0(3) +
          occ_func_2_0(16) * occ_func_3_0(3) +
          occ_func_2_0(14) * occ_func_3_0(31) +
          occ_func_2_0(16) * occ_func_3_0(11) +
          occ_func_2_0(14) * occ_func_3_0(47) +
          occ_func_2_0(43) * occ_func_2_0(9) +
          occ_func_2_0(16) * occ_func_3_0(7) +
          occ_func_2_0(14) * occ_func_3_0(39) +
          occ_func_2_0(16) * occ_func_3_0(13) +
          occ_func_2_0(14) * occ_func_3_0(51) +
          occ_func_2_0(16) * occ_func_3_0(5) +
          occ_func_2_0(14) * occ_func_3_0(35) +
          occ_func_2_0(7) * occ_func_2_0(39) +
          occ_func_2_0(35) * occ_func_2_0(5) +
          occ_func_2_0(51) * occ_func_2_0(13)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_23_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(6) * occ_func_2_0(40) +
          occ_func_2_0(17) * occ_func_3_0(36) +
          occ_func_2_0(15) * occ_func_3_0(4) +
          occ_func_2_0(17) * occ_func_3_0(52) +
          occ_func_2_0(15) * occ_func_3_0(12) +
          occ_func_2_0(12) * occ_func_2_0(52) +
          occ_func_2_0(4) * occ_func_2_0(36) +
          occ_func_2_0(17) * occ_func_3_0(40) +
          occ_func_2_0(15) * occ_func_3_0(6) +
          occ_func_2_0(8) * occ_func_2_0(44) +
          occ_func_2_0(2) * occ_func_2_0(32) +
          occ_func_2_0(10) * occ_func_2_0(48) +
          occ_func_2_0(17) * occ_func_3_0(44) +
          occ_func_2_0(15) * occ_func_3_0(8) +
          occ_func_2_0(17) * occ_func_3_0(48) +
          occ_func_2_0(15) * occ_func_3_0(10) +
          occ_func_2_0(17) * occ_func_3_0(32) +
          occ_func_2_0(15) * occ_func_3_0(2)) /
         12.;
}

/**** Basis functions for orbit 24****
0.3333333 0.6666666 0.2500000 Va  O

1.3333333 0.6666666 0.2500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_24_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(27) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(11) +
          occ_func_2_0(24) * occ_func_2_0(10) * occ_func_2_0(0) +
          occ_func_2_0(26) * occ_func_2_0(8) * occ_func_2_0(0) +
          occ_func_2_0(29) * occ_func_2_0(13) * occ_func_2_0(1) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_2_0(28) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_2_0(8) +
          occ_func_2_0(26) * occ_func_2_0(12) * occ_func_2_0(0) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_2_0(29) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_2_0(27) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_2_0(25)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_24_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(28) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_2_0(12) +
          occ_func_2_0(18) * occ_func_2_0(2) * occ_func_2_0(0) +
          occ_func_2_0(24) * occ_func_2_0(10) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(22) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(4) +
          occ_func_2_0(26) * occ_func_2_0(8) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(6) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(20) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_2_0(28) +
          occ_func_2_0(2) * occ_func_2_0(0) * occ_func_2_0(10) +
          occ_func_2_0(18) * occ_func_2_0(4) * occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_2_0(8) +
          occ_func_2_0(24) * occ_func_2_0(6) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(4) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_2_0(12) * occ_func_2_0(0) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_2_0(2) +
          occ_func_2_0(0) * occ_func_2_0(6) * occ_func_2_0(20)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_24_0_at_1() const {
  return (occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(27) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_2_0(9) +
          occ_func_2_0(21) * occ_func_2_0(7) * occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(11) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_2_0(5) * occ_func_2_0(1) +
          occ_func_2_0(29) * occ_func_2_0(13) * occ_func_2_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(3) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(19) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_2_0(29) +
          occ_func_2_0(5) * occ_func_2_0(1) * occ_func_2_0(13) +
          occ_func_2_0(19) * occ_func_2_0(3) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_2_0(27) +
          occ_func_2_0(7) * occ_func_2_0(1) * occ_func_2_0(13) +
          occ_func_2_0(21) * occ_func_2_0(3) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_2_0(9) * occ_func_2_0(1) +
          occ_func_2_0(5) * occ_func_2_0(1) * occ_func_2_0(7)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_24_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(10) * occ_func_2_0(28) +
          occ_func_2_0(4) * occ_func_2_0(12) +
          occ_func_2_0(18) * occ_func_2_0(2) +
          occ_func_2_0(24) * occ_func_2_0(10) +
          occ_func_2_0(8) * occ_func_2_0(22) +
          occ_func_2_0(6) * occ_func_2_0(4) +
          occ_func_2_0(26) * occ_func_2_0(8) +
          occ_func_2_0(12) * occ_func_2_0(6) +
          occ_func_2_0(2) * occ_func_2_0(20) +
          occ_func_2_0(12) * occ_func_2_0(28) +
          occ_func_2_0(2) * occ_func_2_0(10) +
          occ_func_2_0(18) * occ_func_2_0(4) +
          occ_func_2_0(10) * occ_func_2_0(8) +
          occ_func_2_0(24) * occ_func_2_0(6) +
          occ_func_2_0(4) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_2_0(12) +
          occ_func_2_0(8) * occ_func_2_0(2) +
          occ_func_2_0(6) * occ_func_2_0(20)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_24_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(13) * occ_func_2_0(27) +
          occ_func_2_0(3) * occ_func_2_0(9) +
          occ_func_2_0(21) * occ_func_2_0(7) +
          occ_func_2_0(9) * occ_func_2_0(11) +
          occ_func_2_0(7) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_2_0(5) +
          occ_func_2_0(29) * occ_func_2_0(13) +
          occ_func_2_0(11) * occ_func_2_0(3) +
          occ_func_2_0(5) * occ_func_2_0(19) +
          occ_func_2_0(11) * occ_func_2_0(29) +
          occ_func_2_0(5) * occ_func_2_0(13) +
          occ_func_2_0(19) * occ_func_2_0(3) +
          occ_func_2_0(9) * occ_func_2_0(27) +
          occ_func_2_0(7) * occ_func_2_0(13) +
          occ_func_2_0(21) * occ_func_2_0(3) +
          occ_func_2_0(11) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_2_0(9) +
          occ_func_2_0(5) * occ_func_2_0(7)) /
         12.;
}

/**** Basis functions for orbit 25****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.2500000 Va  O

1.3333333 0.6666666 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_25_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_3_0(47) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_3_0(28) +
          occ_func_2_0(9) * occ_func_3_0(26) * occ_func_3_0(0) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_3_0(27) +
          occ_func_2_0(52) * occ_func_3_0(29) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_3_0(27) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_3_0(29) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_3_0(9) +
          occ_func_2_0(11) * occ_func_3_0(24) * occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_3_0(24) +
          occ_func_2_0(47) * occ_func_3_0(28) * occ_func_3_0(0) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_3_0(26) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_3_0(27) +
          occ_func_2_0(48) * occ_func_3_0(29) * occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_3_0(25) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_3_0(48) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_3_0(8) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_3_0(28) +
          occ_func_2_0(13) * occ_func_3_0(26) * occ_func_3_0(0) +
          occ_func_2_0(8) * occ_func_3_0(27) * occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_3_0(29) +
          occ_func_2_0(51) * occ_func_3_0(28) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_3_0(26) +
          occ_func_2_0(10) * occ_func_3_0(25) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_25_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_3_0(47) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_3_0(28) +
          occ_func_2_0(3) * occ_func_3_0(18) * occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_3_0(26) * occ_func_3_0(0) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_3_0(20) +
          occ_func_2_0(0) * occ_func_3_0(31) * occ_func_3_0(43) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_3_0(31) +
          occ_func_2_0(0) * occ_func_3_0(9) * occ_func_3_0(3) +
          occ_func_2_0(0) * occ_func_3_0(3) * occ_func_3_0(11) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_3_0(9) +
          occ_func_2_0(11) * occ_func_3_0(24) * occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_3_0(22) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_3_0(24) +
          occ_func_2_0(43) * occ_func_3_0(22) * occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_3_0(28) * occ_func_3_0(0) +
          occ_func_2_0(31) * occ_func_3_0(0) * occ_func_3_0(18) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_3_0(26) +
          occ_func_2_0(31) * occ_func_3_0(20) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(39) * occ_func_3_0(51) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_3_0(35) +
          occ_func_2_0(0) * occ_func_3_0(35) * occ_func_3_0(39) +
          occ_func_2_0(39) * occ_func_3_0(0) * occ_func_3_0(24) +
          occ_func_2_0(35) * occ_func_3_0(22) * occ_func_3_0(0) +
          occ_func_2_0(7) * occ_func_3_0(24) * occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_3_0(0) * occ_func_3_0(22) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_3_0(28) +
          occ_func_2_0(5) * occ_func_3_0(18) * occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_3_0(26) * occ_func_3_0(0) +
          occ_func_2_0(7) * occ_func_3_0(0) * occ_func_3_0(20) +
          occ_func_2_0(0) * occ_func_3_0(13) * occ_func_3_0(7) +
          occ_func_2_0(0) * occ_func_3_0(5) * occ_func_3_0(13) +
          occ_func_2_0(51) * occ_func_3_0(28) * occ_func_3_0(0) +
          occ_func_2_0(35) * occ_func_3_0(0) * occ_func_3_0(18) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_3_0(26) +
          occ_func_2_0(39) * occ_func_3_0(20) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(7) * occ_func_3_0(5)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_25_0_at_1() const {
  return (occ_func_2_0(40) * occ_func_3_0(1) * occ_func_3_0(25) +
          occ_func_2_0(36) * occ_func_3_0(23) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(4) * occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_3_0(6) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_3_0(27) +
          occ_func_2_0(40) * occ_func_3_0(21) * occ_func_3_0(1) +
          occ_func_2_0(52) * occ_func_3_0(29) * occ_func_3_0(1) +
          occ_func_2_0(36) * occ_func_3_0(1) * occ_func_3_0(19) +
          occ_func_2_0(12) * occ_func_3_0(27) * occ_func_3_0(1) +
          occ_func_2_0(6) * occ_func_3_0(1) * occ_func_3_0(21) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_3_0(29) +
          occ_func_2_0(4) * occ_func_3_0(19) * occ_func_3_0(1) +
          occ_func_2_0(6) * occ_func_3_0(25) * occ_func_3_0(1) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_3_0(23) +
          occ_func_2_0(1) * occ_func_3_0(6) * occ_func_3_0(4) +
          occ_func_2_0(1) * occ_func_3_0(36) * occ_func_3_0(40) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_3_0(36) +
          occ_func_2_0(1) * occ_func_3_0(40) * occ_func_3_0(52) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_3_0(27) +
          occ_func_2_0(32) * occ_func_3_0(21) * occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_3_0(29) * occ_func_3_0(1) +
          occ_func_2_0(32) * occ_func_3_0(1) * occ_func_3_0(19) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_3_0(25) +
          occ_func_2_0(44) * occ_func_3_0(23) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_3_0(48) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_3_0(8) +
          occ_func_2_0(1) * occ_func_3_0(2) * occ_func_3_0(10) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_3_0(2) +
          occ_func_2_0(8) * occ_func_3_0(27) * occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_3_0(1) * occ_func_3_0(21) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_3_0(29) +
          occ_func_2_0(2) * occ_func_3_0(19) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_3_0(32) +
          occ_func_2_0(1) * occ_func_3_0(32) * occ_func_3_0(44) +
          occ_func_2_0(10) * occ_func_3_0(25) * occ_func_3_0(1) +
          occ_func_2_0(8) * occ_func_3_0(1) * occ_func_3_0(23)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_25_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(43) * occ_func_3_0(47) +
          occ_func_2_0(11) * occ_func_3_0(28) +
          occ_func_2_0(3) * occ_func_3_0(18) +
          occ_func_2_0(9) * occ_func_3_0(26) +
          occ_func_2_0(3) * occ_func_3_0(20) +
          occ_func_3_0(31) * occ_func_3_0(43) +
          occ_func_3_0(47) * occ_func_3_0(31) +
          occ_func_3_0(9) * occ_func_3_0(3) +
          occ_func_3_0(3) * occ_func_3_0(11) +
          occ_func_3_0(11) * occ_func_3_0(9) +
          occ_func_2_0(11) * occ_func_3_0(24) +
          occ_func_2_0(9) * occ_func_3_0(22) +
          occ_func_2_0(47) * occ_func_3_0(24) +
          occ_func_2_0(43) * occ_func_3_0(22) +
          occ_func_2_0(47) * occ_func_3_0(28) +
          occ_func_2_0(31) * occ_func_3_0(18) +
          occ_func_2_0(43) * occ_func_3_0(26) +
          occ_func_2_0(31) * occ_func_3_0(20) +
          occ_func_3_0(39) * occ_func_3_0(51) +
          occ_func_3_0(51) * occ_func_3_0(35) +
          occ_func_3_0(35) * occ_func_3_0(39) +
          occ_func_2_0(39) * occ_func_3_0(24) +
          occ_func_2_0(35) * occ_func_3_0(22) +
          occ_func_2_0(7) * occ_func_3_0(24) +
          occ_func_2_0(5) * occ_func_3_0(22) +
          occ_func_2_0(13) * occ_func_3_0(28) +
          occ_func_2_0(5) * occ_func_3_0(18) +
          occ_func_2_0(13) * occ_func_3_0(26) +
          occ_func_2_0(7) * occ_func_3_0(20) +
          occ_func_3_0(13) * occ_func_3_0(7) +
          occ_func_3_0(5) * occ_func_3_0(13) +
          occ_func_2_0(51) * occ_func_3_0(28) +
          occ_func_2_0(35) * occ_func_3_0(18) +
          occ_func_2_0(51) * occ_func_3_0(26) +
          occ_func_2_0(39) * occ_func_3_0(20) +
          occ_func_3_0(7) * occ_func_3_0(5)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_25_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(40) * occ_func_3_0(25) +
          occ_func_2_0(36) * occ_func_3_0(23) +
          occ_func_3_0(4) * occ_func_3_0(12) +
          occ_func_3_0(12) * occ_func_3_0(6) +
          occ_func_2_0(52) * occ_func_3_0(27) +
          occ_func_2_0(40) * occ_func_3_0(21) +
          occ_func_2_0(52) * occ_func_3_0(29) +
          occ_func_2_0(36) * occ_func_3_0(19) +
          occ_func_2_0(12) * occ_func_3_0(27) +
          occ_func_2_0(6) * occ_func_3_0(21) +
          occ_func_2_0(12) * occ_func_3_0(29) +
          occ_func_2_0(4) * occ_func_3_0(19) +
          occ_func_2_0(6) * occ_func_3_0(25) +
          occ_func_2_0(4) * occ_func_3_0(23) +
          occ_func_3_0(6) * occ_func_3_0(4) +
          occ_func_3_0(36) * occ_func_3_0(40) +
          occ_func_3_0(52) * occ_func_3_0(36) +
          occ_func_3_0(40) * occ_func_3_0(52) +
          occ_func_2_0(44) * occ_func_3_0(27) +
          occ_func_2_0(32) * occ_func_3_0(21) +
          occ_func_2_0(48) * occ_func_3_0(29) +
          occ_func_2_0(32) * occ_func_3_0(19) +
          occ_func_2_0(48) * occ_func_3_0(25) +
          occ_func_2_0(44) * occ_func_3_0(23) +
          occ_func_3_0(44) * occ_func_3_0(48) +
          occ_func_3_0(10) * occ_func_3_0(8) +
          occ_func_3_0(2) * occ_func_3_0(10) +
          occ_func_3_0(8) * occ_func_3_0(2) +
          occ_func_2_0(8) * occ_func_3_0(27) +
          occ_func_2_0(2) * occ_func_3_0(21) +
          occ_func_2_0(10) * occ_func_3_0(29) +
          occ_func_2_0(2) * occ_func_3_0(19) +
          occ_func_3_0(48) * occ_func_3_0(32) +
          occ_func_3_0(32) * occ_func_3_0(44) +
          occ_func_2_0(10) * occ_func_3_0(25) +
          occ_func_2_0(8) * occ_func_3_0(23)) /
         24.;
}

/**** Basis functions for orbit 26****
0.3333333 0.6666666 0.2500000 Va  O

1.3333333 -0.3333333 0.2500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_26_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(24) * occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_2_0(29) * occ_func_2_0(27) +
          occ_func_2_0(0) * occ_func_2_0(26) * occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_2_0(25) * occ_func_2_0(29)) /
         4.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_26_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(24) * occ_func_2_0(28) +
          occ_func_2_0(22) * occ_func_2_0(0) * occ_func_2_0(26) +
          occ_func_2_0(18) * occ_func_2_0(20) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(26) * occ_func_2_0(28) +
          occ_func_2_0(20) * occ_func_2_0(0) * occ_func_2_0(24) +
          occ_func_2_0(18) * occ_func_2_0(22) * occ_func_2_0(0)) /
         4.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_26_0_at_1() const {
  return (occ_func_2_0(1) * occ_func_2_0(29) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_2_0(25) * occ_func_2_0(1) +
          occ_func_2_0(19) * occ_func_2_0(1) * occ_func_2_0(23) +
          occ_func_2_0(1) * occ_func_2_0(25) * occ_func_2_0(29) +
          occ_func_2_0(23) * occ_func_2_0(1) * occ_func_2_0(27) +
          occ_func_2_0(19) * occ_func_2_0(21) * occ_func_2_0(1)) /
         4.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_26_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(24) * occ_func_2_0(28) +
          occ_func_2_0(22) * occ_func_2_0(26) +
          occ_func_2_0(18) * occ_func_2_0(20) +
          occ_func_2_0(26) * occ_func_2_0(28) +
          occ_func_2_0(20) * occ_func_2_0(24) +
          occ_func_2_0(18) * occ_func_2_0(22)) /
         4.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_26_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(29) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_2_0(25) +
          occ_func_2_0(19) * occ_func_2_0(23) +
          occ_func_2_0(25) * occ_func_2_0(29) +
          occ_func_2_0(23) * occ_func_2_0(27) +
          occ_func_2_0(19) * occ_func_2_0(21)) /
         4.;
}

/**** Basis functions for orbit 27****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 0.7500000 Va  O

0.3333333 1.6666667 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_27_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(43) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_3_0(12) +
          occ_func_2_0(52) * occ_func_3_0(53) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(47) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_3_0(45) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_3_0(1) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_3_0(49) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_3_0(46) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(52) +
          occ_func_2_0(43) * occ_func_3_0(42) * occ_func_3_0(0) +
          occ_func_2_0(44) * occ_func_3_0(45) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(51) +
          occ_func_2_0(48) * occ_func_3_0(49) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_3_0(10) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_3_0(8) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_3_0(53) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(48) +
          occ_func_2_0(51) * occ_func_3_0(50) * occ_func_3_0(0) +
          occ_func_2_0(8) * occ_func_3_0(43) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_27_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(43) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_3_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_3_0(34) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_3_0(50) +
          occ_func_2_0(3) * occ_func_3_0(32) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(31) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(47) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_3_0(9) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_3_0(3) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_3_0(11) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_3_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_3_0(38) +
          occ_func_2_0(47) * occ_func_3_0(46) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(36) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(52) +
          occ_func_2_0(31) * occ_func_3_0(30) * occ_func_3_0(0) +
          occ_func_2_0(43) * occ_func_3_0(42) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(40) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(39) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(51) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(35) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(44) +
          occ_func_2_0(39) * occ_func_3_0(38) * occ_func_3_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_3_0(46) +
          occ_func_2_0(5) * occ_func_3_0(36) * occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_3_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_3_0(30) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_3_0(42) +
          occ_func_2_0(7) * occ_func_3_0(40) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_3_0(13) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_3_0(5) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(48) +
          occ_func_2_0(35) * occ_func_3_0(34) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_3_0(50) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(32) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_3_0(7)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_27_0_at_1() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(43) +
          occ_func_2_0(40) * occ_func_3_0(41) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_3_0(4) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_3_0(12) +
          occ_func_2_0(52) * occ_func_3_0(53) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(31) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(47) +
          occ_func_2_0(36) * occ_func_3_0(37) * occ_func_3_0(1) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_3_0(45) +
          occ_func_2_0(6) * occ_func_3_0(39) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_3_0(1) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_3_0(33) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_3_0(49) +
          occ_func_2_0(4) * occ_func_3_0(35) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_3_0(6) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(36) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(40) +
          occ_func_2_0(44) * occ_func_3_0(45) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(39) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(51) +
          occ_func_2_0(32) * occ_func_3_0(33) * occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_3_0(49) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_3_0(35) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_3_0(10) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_3_0(2) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_3_0(8) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_3_0(53) +
          occ_func_2_0(2) * occ_func_3_0(31) * occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_3_0(1) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_3_0(37) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(48) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_3_0(32) +
          occ_func_2_0(8) * occ_func_3_0(43) * occ_func_3_0(1) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_3_0(41)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_27_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(1) * occ_func_3_0(43) +
          occ_func_2_0(11) * occ_func_3_0(48) +
          occ_func_2_0(15) * occ_func_3_0(34) +
          occ_func_2_0(15) * occ_func_3_0(50) +
          occ_func_2_0(3) * occ_func_3_0(32) +
          occ_func_3_0(1) * occ_func_3_0(31) +
          occ_func_3_0(1) * occ_func_3_0(47) +
          occ_func_3_0(15) * occ_func_3_0(9) +
          occ_func_3_0(15) * occ_func_3_0(3) +
          occ_func_3_0(15) * occ_func_3_0(11) +
          occ_func_2_0(9) * occ_func_3_0(44) +
          occ_func_2_0(15) * occ_func_3_0(38) +
          occ_func_2_0(47) * occ_func_3_0(46) +
          occ_func_2_0(1) * occ_func_3_0(36) +
          occ_func_2_0(1) * occ_func_3_0(52) +
          occ_func_2_0(31) * occ_func_3_0(30) +
          occ_func_2_0(43) * occ_func_3_0(42) +
          occ_func_2_0(1) * occ_func_3_0(40) +
          occ_func_3_0(1) * occ_func_3_0(39) +
          occ_func_3_0(1) * occ_func_3_0(51) +
          occ_func_3_0(1) * occ_func_3_0(35) +
          occ_func_2_0(1) * occ_func_3_0(44) +
          occ_func_2_0(39) * occ_func_3_0(38) +
          occ_func_2_0(15) * occ_func_3_0(46) +
          occ_func_2_0(5) * occ_func_3_0(36) +
          occ_func_2_0(13) * occ_func_3_0(52) +
          occ_func_2_0(15) * occ_func_3_0(30) +
          occ_func_2_0(15) * occ_func_3_0(42) +
          occ_func_2_0(7) * occ_func_3_0(40) +
          occ_func_3_0(15) * occ_func_3_0(13) +
          occ_func_3_0(15) * occ_func_3_0(5) +
          occ_func_2_0(1) * occ_func_3_0(48) +
          occ_func_2_0(35) * occ_func_3_0(34) +
          occ_func_2_0(51) * occ_func_3_0(50) +
          occ_func_2_0(1) * occ_func_3_0(32) +
          occ_func_3_0(15) * occ_func_3_0(7)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_27_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(0) * occ_func_3_0(43) +
          occ_func_2_0(40) * occ_func_3_0(41) +
          occ_func_3_0(16) * occ_func_3_0(4) +
          occ_func_3_0(16) * occ_func_3_0(12) +
          occ_func_2_0(52) * occ_func_3_0(53) +
          occ_func_2_0(0) * occ_func_3_0(31) +
          occ_func_2_0(0) * occ_func_3_0(47) +
          occ_func_2_0(36) * occ_func_3_0(37) +
          occ_func_2_0(16) * occ_func_3_0(45) +
          occ_func_2_0(6) * occ_func_3_0(39) +
          occ_func_2_0(12) * occ_func_3_0(51) +
          occ_func_2_0(16) * occ_func_3_0(33) +
          occ_func_2_0(16) * occ_func_3_0(49) +
          occ_func_2_0(4) * occ_func_3_0(35) +
          occ_func_3_0(16) * occ_func_3_0(6) +
          occ_func_3_0(0) * occ_func_3_0(36) +
          occ_func_3_0(0) * occ_func_3_0(52) +
          occ_func_3_0(0) * occ_func_3_0(40) +
          occ_func_2_0(44) * occ_func_3_0(45) +
          occ_func_2_0(0) * occ_func_3_0(39) +
          occ_func_2_0(0) * occ_func_3_0(51) +
          occ_func_2_0(32) * occ_func_3_0(33) +
          occ_func_2_0(48) * occ_func_3_0(49) +
          occ_func_2_0(0) * occ_func_3_0(35) +
          occ_func_3_0(0) * occ_func_3_0(44) +
          occ_func_3_0(16) * occ_func_3_0(10) +
          occ_func_3_0(16) * occ_func_3_0(2) +
          occ_func_3_0(16) * occ_func_3_0(8) +
          occ_func_2_0(16) * occ_func_3_0(53) +
          occ_func_2_0(2) * occ_func_3_0(31) +
          occ_func_2_0(10) * occ_func_3_0(47) +
          occ_func_2_0(16) * occ_func_3_0(37) +
          occ_func_3_0(0) * occ_func_3_0(48) +
          occ_func_3_0(0) * occ_func_3_0(32) +
          occ_func_2_0(8) * occ_func_3_0(43) +
          occ_func_2_0(16) * occ_func_3_0(41)) /
         24.;
}

/**** Basis functions for orbit 28****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.7500000 Va  O

1.3333333 1.6666667 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_28_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_3_0(51) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_3_0(10) +
          occ_func_2_0(52) * occ_func_2_0(0) * occ_func_3_0(9) +
          occ_func_2_0(44) * occ_func_2_0(12) * occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_2_0(52) * occ_func_3_0(1) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_3_0(47) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_3_0(13) +
          occ_func_2_0(13) * occ_func_2_0(47) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_3_0(44) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_3_0(12) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_3_0(13) +
          occ_func_2_0(0) * occ_func_2_0(50) * occ_func_3_0(43) +
          occ_func_2_0(52) * occ_func_2_0(8) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_3_0(12) +
          occ_func_2_0(53) * occ_func_2_0(1) * occ_func_3_0(48) +
          occ_func_2_0(11) * occ_func_2_0(51) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(52) * occ_func_3_0(11) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_3_0(51) +
          occ_func_2_0(43) * occ_func_2_0(13) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_2_0(1) * occ_func_3_0(8) +
          occ_func_2_0(12) * occ_func_2_0(48) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_28_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_3_0(51) +
          occ_func_2_0(40) * occ_func_2_0(0) * occ_func_3_0(11) +
          occ_func_2_0(7) * occ_func_2_0(31) * occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_2_0(43) * occ_func_3_0(0) +
          occ_func_2_0(52) * occ_func_2_0(0) * occ_func_3_0(9) +
          occ_func_2_0(0) * occ_func_2_0(30) * occ_func_3_0(35) +
          occ_func_2_0(0) * occ_func_2_0(46) * occ_func_3_0(39) +
          occ_func_2_0(36) * occ_func_2_0(0) * occ_func_3_0(3) +
          occ_func_2_0(0) * occ_func_2_0(44) * occ_func_3_0(5) +
          occ_func_2_0(38) * occ_func_2_0(0) * occ_func_3_0(31) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_3_0(47) +
          occ_func_2_0(0) * occ_func_2_0(32) * occ_func_3_0(7) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_3_0(13) +
          occ_func_2_0(34) * occ_func_2_0(0) * occ_func_3_0(43) +
          occ_func_2_0(13) * occ_func_2_0(47) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(39) * occ_func_2_0(11) * occ_func_3_0(0) +
          occ_func_2_0(35) * occ_func_2_0(3) * occ_func_3_0(0) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_3_0(13) +
          occ_func_2_0(0) * occ_func_2_0(38) * occ_func_3_0(47) +
          occ_func_2_0(0) * occ_func_2_0(50) * occ_func_3_0(43) +
          occ_func_2_0(32) * occ_func_2_0(0) * occ_func_3_0(5) +
          occ_func_2_0(48) * occ_func_2_0(0) * occ_func_3_0(7) +
          occ_func_2_0(0) * occ_func_2_0(34) * occ_func_3_0(31) +
          occ_func_2_0(31) * occ_func_2_0(5) * occ_func_3_0(0) +
          occ_func_2_0(3) * occ_func_2_0(39) * occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_2_0(35) * occ_func_3_0(0) +
          occ_func_2_0(11) * occ_func_2_0(51) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(52) * occ_func_3_0(11) +
          occ_func_2_0(30) * occ_func_2_0(0) * occ_func_3_0(39) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_3_0(51) +
          occ_func_2_0(0) * occ_func_2_0(36) * occ_func_3_0(9) +
          occ_func_2_0(43) * occ_func_2_0(13) * occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_2_0(7) * occ_func_3_0(0) +
          occ_func_2_0(42) * occ_func_2_0(0) * occ_func_3_0(35) +
          occ_func_2_0(0) * occ_func_2_0(40) * occ_func_3_0(3)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_28_0_at_1() const {
  return (occ_func_2_0(32) * occ_func_2_0(4) * occ_func_3_0(1) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_2_0(35) * occ_func_3_0(8) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_3_0(10) +
          occ_func_2_0(33) * occ_func_2_0(1) * occ_func_3_0(40) +
          occ_func_2_0(48) * occ_func_2_0(6) * occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_2_0(12) * occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_2_0(52) * occ_func_3_0(1) +
          occ_func_2_0(8) * occ_func_2_0(36) * occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_2_0(40) * occ_func_3_0(1) +
          occ_func_2_0(45) * occ_func_2_0(1) * occ_func_3_0(36) +
          occ_func_2_0(1) * occ_func_2_0(39) * occ_func_3_0(2) +
          occ_func_2_0(47) * occ_func_2_0(1) * occ_func_3_0(6) +
          occ_func_2_0(1) * occ_func_2_0(37) * occ_func_3_0(32) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_3_0(44) +
          occ_func_2_0(31) * occ_func_2_0(1) * occ_func_3_0(4) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_2_0(41) * occ_func_3_0(48) +
          occ_func_2_0(36) * occ_func_2_0(2) * occ_func_3_0(1) +
          occ_func_2_0(40) * occ_func_2_0(10) * occ_func_3_0(1) +
          occ_func_2_0(52) * occ_func_2_0(8) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_3_0(52) +
          occ_func_2_0(39) * occ_func_2_0(1) * occ_func_3_0(10) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_3_0(12) +
          occ_func_2_0(37) * occ_func_2_0(1) * occ_func_3_0(44) +
          occ_func_2_0(53) * occ_func_2_0(1) * occ_func_3_0(48) +
          occ_func_2_0(1) * occ_func_2_0(31) * occ_func_3_0(6) +
          occ_func_2_0(1) * occ_func_2_0(43) * occ_func_3_0(4) +
          occ_func_2_0(41) * occ_func_2_0(1) * occ_func_3_0(32) +
          occ_func_2_0(4) * occ_func_2_0(44) * occ_func_3_0(1) +
          occ_func_2_0(6) * occ_func_2_0(32) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(49) * occ_func_3_0(40) +
          occ_func_2_0(35) * occ_func_2_0(1) * occ_func_3_0(2) +
          occ_func_2_0(51) * occ_func_2_0(1) * occ_func_3_0(8) +
          occ_func_2_0(1) * occ_func_2_0(33) * occ_func_3_0(36) +
          occ_func_2_0(12) * occ_func_2_0(48) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_28_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(42) * occ_func_3_0(51) +
          occ_func_2_0(40) * occ_func_3_0(11) +
          occ_func_2_0(7) * occ_func_2_0(31) +
          occ_func_2_0(5) * occ_func_2_0(43) +
          occ_func_2_0(52) * occ_func_3_0(9) +
          occ_func_2_0(30) * occ_func_3_0(35) +
          occ_func_2_0(46) * occ_func_3_0(39) +
          occ_func_2_0(36) * occ_func_3_0(3) +
          occ_func_2_0(44) * occ_func_3_0(5) +
          occ_func_2_0(38) * occ_func_3_0(31) +
          occ_func_2_0(50) * occ_func_3_0(47) +
          occ_func_2_0(32) * occ_func_3_0(7) +
          occ_func_2_0(48) * occ_func_3_0(13) +
          occ_func_2_0(34) * occ_func_3_0(43) +
          occ_func_2_0(13) * occ_func_2_0(47) +
          occ_func_2_0(51) * occ_func_2_0(9) +
          occ_func_2_0(39) * occ_func_2_0(11) +
          occ_func_2_0(35) * occ_func_2_0(3) +
          occ_func_2_0(44) * occ_func_3_0(13) +
          occ_func_2_0(38) * occ_func_3_0(47) +
          occ_func_2_0(50) * occ_func_3_0(43) +
          occ_func_2_0(32) * occ_func_3_0(5) +
          occ_func_2_0(48) * occ_func_3_0(7) +
          occ_func_2_0(34) * occ_func_3_0(31) +
          occ_func_2_0(31) * occ_func_2_0(5) +
          occ_func_2_0(3) * occ_func_2_0(39) +
          occ_func_2_0(9) * occ_func_2_0(35) +
          occ_func_2_0(11) * occ_func_2_0(51) +
          occ_func_2_0(52) * occ_func_3_0(11) +
          occ_func_2_0(30) * occ_func_3_0(39) +
          occ_func_2_0(46) * occ_func_3_0(51) +
          occ_func_2_0(36) * occ_func_3_0(9) +
          occ_func_2_0(43) * occ_func_2_0(13) +
          occ_func_2_0(47) * occ_func_2_0(7) +
          occ_func_2_0(42) * occ_func_3_0(35) +
          occ_func_2_0(40) * occ_func_3_0(3)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_28_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(32) * occ_func_2_0(4) +
          occ_func_2_0(49) * occ_func_3_0(52) +
          occ_func_2_0(35) * occ_func_3_0(8) +
          occ_func_2_0(51) * occ_func_3_0(10) +
          occ_func_2_0(33) * occ_func_3_0(40) +
          occ_func_2_0(48) * occ_func_2_0(6) +
          occ_func_2_0(44) * occ_func_2_0(12) +
          occ_func_2_0(10) * occ_func_2_0(52) +
          occ_func_2_0(8) * occ_func_2_0(36) +
          occ_func_2_0(2) * occ_func_2_0(40) +
          occ_func_2_0(45) * occ_func_3_0(36) +
          occ_func_2_0(39) * occ_func_3_0(2) +
          occ_func_2_0(47) * occ_func_3_0(6) +
          occ_func_2_0(37) * occ_func_3_0(32) +
          occ_func_2_0(53) * occ_func_3_0(44) +
          occ_func_2_0(31) * occ_func_3_0(4) +
          occ_func_2_0(43) * occ_func_3_0(12) +
          occ_func_2_0(41) * occ_func_3_0(48) +
          occ_func_2_0(36) * occ_func_2_0(2) +
          occ_func_2_0(40) * occ_func_2_0(10) +
          occ_func_2_0(52) * occ_func_2_0(8) +
          occ_func_2_0(45) * occ_func_3_0(52) +
          occ_func_2_0(39) * occ_func_3_0(10) +
          occ_func_2_0(47) * occ_func_3_0(12) +
          occ_func_2_0(37) * occ_func_3_0(44) +
          occ_func_2_0(53) * occ_func_3_0(48) +
          occ_func_2_0(31) * occ_func_3_0(6) +
          occ_func_2_0(43) * occ_func_3_0(4) +
          occ_func_2_0(41) * occ_func_3_0(32) +
          occ_func_2_0(4) * occ_func_2_0(44) +
          occ_func_2_0(6) * occ_func_2_0(32) +
          occ_func_2_0(49) * occ_func_3_0(40) +
          occ_func_2_0(35) * occ_func_3_0(2) +
          occ_func_2_0(51) * occ_func_3_0(8) +
          occ_func_2_0(33) * occ_func_3_0(36) +
          occ_func_2_0(12) * occ_func_2_0(48)) /
         24.;
}

/**** Basis functions for orbit 29****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 1.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_29_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(16) * occ_func_2_0(8) +
          occ_func_2_0(11) * occ_func_2_0(49) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_2_0(13) +
          occ_func_2_0(12) * occ_func_2_0(52) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_2_0(10) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_2_0(44) +
          occ_func_2_0(12) * occ_func_2_0(50) * occ_func_2_0(0) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_2_0(48) +
          occ_func_2_0(9) * occ_func_2_0(45) * occ_func_2_0(1) +
          occ_func_2_0(11) * occ_func_2_0(47) * occ_func_2_0(1) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_2_0(53) +
          occ_func_2_0(9) * occ_func_2_0(43) * occ_func_2_0(1) +
          occ_func_2_0(8) * occ_func_2_0(44) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_2_0(12) +
          occ_func_2_0(10) * occ_func_2_0(48) * occ_func_2_0(0) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_2_0(45) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_2_0(11) +
          occ_func_2_0(13) * occ_func_2_0(53) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_2_0(9) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_2_0(52) +
          occ_func_2_0(10) * occ_func_2_0(46) * occ_func_2_0(0) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_2_0(49) +
          occ_func_2_0(13) * occ_func_2_0(51) * occ_func_2_0(1) +
          occ_func_2_0(8) * occ_func_2_0(42) * occ_func_2_0(0)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_29_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(16) * occ_func_2_0(8) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_2_0(42) +
          occ_func_2_0(6) * occ_func_2_0(40) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_2_0(52) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_2_0(2) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_2_0(30) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_2_0(10) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_2_0(46) +
          occ_func_2_0(4) * occ_func_2_0(36) * occ_func_2_0(0) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_2_0(44) +
          occ_func_2_0(0) * occ_func_2_0(14) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_2_0(38) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_2_0(50) * occ_func_2_0(0) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_2_0(32) +
          occ_func_2_0(0) * occ_func_2_0(14) * occ_func_2_0(2) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_2_0(48) +
          occ_func_2_0(0) * occ_func_2_0(14) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_2_0(34) * occ_func_2_0(0) +
          occ_func_2_0(8) * occ_func_2_0(44) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_2_0(6) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_2_0(38) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_2_0(12) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_2_0(50) +
          occ_func_2_0(2) * occ_func_2_0(32) * occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_2_0(48) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_2_0(4) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_2_0(34) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_2_0(52) +
          occ_func_2_0(0) * occ_func_2_0(14) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_2_0(30) * occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_2_0(46) * occ_func_2_0(0) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_2_0(36) +
          occ_func_2_0(0) * occ_func_2_0(14) * occ_func_2_0(4) +
          occ_func_2_0(8) * occ_func_2_0(42) * occ_func_2_0(0) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_2_0(40) +
          occ_func_2_0(0) * occ_func_2_0(14) * occ_func_2_0(6)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_29_0_at_1() const {
  return (occ_func_2_0(11) * occ_func_2_0(49) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_2_0(5) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_2_0(35) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_2_0(13) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_2_0(51) +
          occ_func_2_0(3) * occ_func_2_0(33) * occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_2_0(45) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_2_0(7) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_2_0(39) +
          occ_func_2_0(11) * occ_func_2_0(47) * occ_func_2_0(1) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_2_0(37) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_2_0(5) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_2_0(53) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_2_0(31) * occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_2_0(43) * occ_func_2_0(1) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_2_0(41) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_2_0(7) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_2_0(45) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_2_0(39) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_2_0(11) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_2_0(47) +
          occ_func_2_0(5) * occ_func_2_0(37) * occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_2_0(53) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_2_0(3) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_2_0(31) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_2_0(9) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_2_0(43) +
          occ_func_2_0(7) * occ_func_2_0(41) * occ_func_2_0(1) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_2_0(49) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_2_0(35) * occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_2_0(51) * occ_func_2_0(1) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_2_0(33) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_2_0(3)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_29_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(16) * occ_func_2_0(8) +
          occ_func_2_0(14) * occ_func_2_0(42) +
          occ_func_2_0(6) * occ_func_2_0(40) +
          occ_func_2_0(12) * occ_func_2_0(52) +
          occ_func_2_0(16) * occ_func_2_0(2) +
          occ_func_2_0(14) * occ_func_2_0(30) +
          occ_func_2_0(16) * occ_func_2_0(10) +
          occ_func_2_0(14) * occ_func_2_0(46) +
          occ_func_2_0(4) * occ_func_2_0(36) +
          occ_func_2_0(16) * occ_func_2_0(44) +
          occ_func_2_0(14) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_2_0(38) +
          occ_func_2_0(12) * occ_func_2_0(50) +
          occ_func_2_0(16) * occ_func_2_0(32) +
          occ_func_2_0(14) * occ_func_2_0(2) +
          occ_func_2_0(16) * occ_func_2_0(48) +
          occ_func_2_0(14) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_2_0(34) +
          occ_func_2_0(8) * occ_func_2_0(44) +
          occ_func_2_0(16) * occ_func_2_0(6) +
          occ_func_2_0(14) * occ_func_2_0(38) +
          occ_func_2_0(16) * occ_func_2_0(12) +
          occ_func_2_0(14) * occ_func_2_0(50) +
          occ_func_2_0(2) * occ_func_2_0(32) +
          occ_func_2_0(10) * occ_func_2_0(48) +
          occ_func_2_0(16) * occ_func_2_0(4) +
          occ_func_2_0(14) * occ_func_2_0(34) +
          occ_func_2_0(16) * occ_func_2_0(52) +
          occ_func_2_0(14) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_2_0(30) +
          occ_func_2_0(10) * occ_func_2_0(46) +
          occ_func_2_0(16) * occ_func_2_0(36) +
          occ_func_2_0(14) * occ_func_2_0(4) +
          occ_func_2_0(8) * occ_func_2_0(42) +
          occ_func_2_0(16) * occ_func_2_0(40) +
          occ_func_2_0(14) * occ_func_2_0(6)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_29_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(11) * occ_func_2_0(49) +
          occ_func_2_0(17) * occ_func_2_0(5) +
          occ_func_2_0(15) * occ_func_2_0(35) +
          occ_func_2_0(17) * occ_func_2_0(13) +
          occ_func_2_0(15) * occ_func_2_0(51) +
          occ_func_2_0(3) * occ_func_2_0(33) +
          occ_func_2_0(9) * occ_func_2_0(45) +
          occ_func_2_0(17) * occ_func_2_0(7) +
          occ_func_2_0(15) * occ_func_2_0(39) +
          occ_func_2_0(11) * occ_func_2_0(47) +
          occ_func_2_0(17) * occ_func_2_0(37) +
          occ_func_2_0(15) * occ_func_2_0(5) +
          occ_func_2_0(17) * occ_func_2_0(53) +
          occ_func_2_0(15) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_2_0(31) +
          occ_func_2_0(9) * occ_func_2_0(43) +
          occ_func_2_0(17) * occ_func_2_0(41) +
          occ_func_2_0(15) * occ_func_2_0(7) +
          occ_func_2_0(17) * occ_func_2_0(45) +
          occ_func_2_0(15) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_2_0(39) +
          occ_func_2_0(17) * occ_func_2_0(11) +
          occ_func_2_0(15) * occ_func_2_0(47) +
          occ_func_2_0(5) * occ_func_2_0(37) +
          occ_func_2_0(13) * occ_func_2_0(53) +
          occ_func_2_0(17) * occ_func_2_0(3) +
          occ_func_2_0(15) * occ_func_2_0(31) +
          occ_func_2_0(17) * occ_func_2_0(9) +
          occ_func_2_0(15) * occ_func_2_0(43) +
          occ_func_2_0(7) * occ_func_2_0(41) +
          occ_func_2_0(17) * occ_func_2_0(49) +
          occ_func_2_0(15) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_2_0(35) +
          occ_func_2_0(13) * occ_func_2_0(51) +
          occ_func_2_0(17) * occ_func_2_0(33) +
          occ_func_2_0(15) * occ_func_2_0(3)) /
         24.;
}

/**** Basis functions for orbit 30****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

1.3333333 1.6666667 1.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_30_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(52) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(53) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(49) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(44) +
          occ_func_2_0(42) * occ_func_2_0(50) * occ_func_2_0(0) +
          occ_func_2_0(48) * occ_func_2_0(52) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(46) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(50) +
          occ_func_2_0(51) * occ_func_2_0(47) * occ_func_2_0(1) +
          occ_func_2_0(53) * occ_func_2_0(45) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(43) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(51)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_30_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(52) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(48) +
          occ_func_2_0(30) * occ_func_2_0(34) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(44) +
          occ_func_2_0(46) * occ_func_2_0(38) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(36) +
          occ_func_2_0(42) * occ_func_2_0(50) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(40) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_2_0(32) +
          occ_func_2_0(48) * occ_func_2_0(52) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(34) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(30) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(46) +
          occ_func_2_0(44) * occ_func_2_0(36) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(38) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(50) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_2_0(42) +
          occ_func_2_0(32) * occ_func_2_0(40) * occ_func_2_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_30_0_at_1() const {
  return (occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(53) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(45) +
          occ_func_2_0(39) * occ_func_2_0(31) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(49) +
          occ_func_2_0(35) * occ_func_2_0(43) * occ_func_2_0(1) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_2_0(41) +
          occ_func_2_0(51) * occ_func_2_0(47) * occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(37) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(33) +
          occ_func_2_0(53) * occ_func_2_0(45) * occ_func_2_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(39) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(31) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(43) +
          occ_func_2_0(41) * occ_func_2_0(49) * occ_func_2_0(1) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_2_0(35) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(51) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(47) +
          occ_func_2_0(37) * occ_func_2_0(33) * occ_func_2_0(1)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_30_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) * occ_func_2_0(52) +
          occ_func_2_0(6) * occ_func_2_0(48) +
          occ_func_2_0(30) * occ_func_2_0(34) +
          occ_func_2_0(12) * occ_func_2_0(44) +
          occ_func_2_0(46) * occ_func_2_0(38) +
          occ_func_2_0(2) * occ_func_2_0(36) +
          occ_func_2_0(42) * occ_func_2_0(50) +
          occ_func_2_0(10) * occ_func_2_0(40) +
          occ_func_2_0(4) * occ_func_2_0(32) +
          occ_func_2_0(48) * occ_func_2_0(52) +
          occ_func_2_0(8) * occ_func_2_0(34) +
          occ_func_2_0(6) * occ_func_2_0(30) +
          occ_func_2_0(12) * occ_func_2_0(46) +
          occ_func_2_0(44) * occ_func_2_0(36) +
          occ_func_2_0(2) * occ_func_2_0(38) +
          occ_func_2_0(10) * occ_func_2_0(50) +
          occ_func_2_0(4) * occ_func_2_0(42) +
          occ_func_2_0(32) * occ_func_2_0(40)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_30_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(11) * occ_func_2_0(53) +
          occ_func_2_0(5) * occ_func_2_0(45) +
          occ_func_2_0(39) * occ_func_2_0(31) +
          occ_func_2_0(13) * occ_func_2_0(49) +
          occ_func_2_0(35) * occ_func_2_0(43) +
          occ_func_2_0(3) * occ_func_2_0(41) +
          occ_func_2_0(51) * occ_func_2_0(47) +
          occ_func_2_0(9) * occ_func_2_0(37) +
          occ_func_2_0(7) * occ_func_2_0(33) +
          occ_func_2_0(53) * occ_func_2_0(45) +
          occ_func_2_0(11) * occ_func_2_0(39) +
          occ_func_2_0(5) * occ_func_2_0(31) +
          occ_func_2_0(13) * occ_func_2_0(43) +
          occ_func_2_0(41) * occ_func_2_0(49) +
          occ_func_2_0(3) * occ_func_2_0(35) +
          occ_func_2_0(9) * occ_func_2_0(51) +
          occ_func_2_0(7) * occ_func_2_0(47) +
          occ_func_2_0(37) * occ_func_2_0(33)) /
         12.;
}

/**** Basis functions for orbit 31****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

1.3333333 1.6666667 -0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_31_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(50) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(51) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(47) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(42) +
          occ_func_2_0(44) * occ_func_2_0(52) * occ_func_2_0(0) +
          occ_func_2_0(46) * occ_func_2_0(50) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(48) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(52) +
          occ_func_2_0(53) * occ_func_2_0(49) * occ_func_2_0(1) +
          occ_func_2_0(51) * occ_func_2_0(43) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(45) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(53)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_31_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(50) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(46) +
          occ_func_2_0(32) * occ_func_2_0(36) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(42) +
          occ_func_2_0(48) * occ_func_2_0(40) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(34) +
          occ_func_2_0(44) * occ_func_2_0(52) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(38) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_2_0(30) +
          occ_func_2_0(46) * occ_func_2_0(50) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(36) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(32) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(48) +
          occ_func_2_0(42) * occ_func_2_0(34) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(40) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(52) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_2_0(44) +
          occ_func_2_0(30) * occ_func_2_0(38) * occ_func_2_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_31_0_at_1() const {
  return (occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(51) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(43) +
          occ_func_2_0(41) * occ_func_2_0(33) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(47) +
          occ_func_2_0(37) * occ_func_2_0(45) * occ_func_2_0(1) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_2_0(39) +
          occ_func_2_0(53) * occ_func_2_0(49) * occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(35) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(31) +
          occ_func_2_0(51) * occ_func_2_0(43) * occ_func_2_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(41) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(33) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(45) +
          occ_func_2_0(39) * occ_func_2_0(47) * occ_func_2_0(1) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_2_0(37) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(53) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(49) +
          occ_func_2_0(35) * occ_func_2_0(31) * occ_func_2_0(1)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_31_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) * occ_func_2_0(50) +
          occ_func_2_0(6) * occ_func_2_0(46) +
          occ_func_2_0(32) * occ_func_2_0(36) +
          occ_func_2_0(12) * occ_func_2_0(42) +
          occ_func_2_0(48) * occ_func_2_0(40) +
          occ_func_2_0(2) * occ_func_2_0(34) +
          occ_func_2_0(44) * occ_func_2_0(52) +
          occ_func_2_0(10) * occ_func_2_0(38) +
          occ_func_2_0(4) * occ_func_2_0(30) +
          occ_func_2_0(46) * occ_func_2_0(50) +
          occ_func_2_0(8) * occ_func_2_0(36) +
          occ_func_2_0(6) * occ_func_2_0(32) +
          occ_func_2_0(12) * occ_func_2_0(48) +
          occ_func_2_0(42) * occ_func_2_0(34) +
          occ_func_2_0(2) * occ_func_2_0(40) +
          occ_func_2_0(10) * occ_func_2_0(52) +
          occ_func_2_0(4) * occ_func_2_0(44) +
          occ_func_2_0(30) * occ_func_2_0(38)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_31_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(11) * occ_func_2_0(51) +
          occ_func_2_0(5) * occ_func_2_0(43) +
          occ_func_2_0(41) * occ_func_2_0(33) +
          occ_func_2_0(13) * occ_func_2_0(47) +
          occ_func_2_0(37) * occ_func_2_0(45) +
          occ_func_2_0(3) * occ_func_2_0(39) +
          occ_func_2_0(53) * occ_func_2_0(49) +
          occ_func_2_0(9) * occ_func_2_0(35) +
          occ_func_2_0(7) * occ_func_2_0(31) +
          occ_func_2_0(51) * occ_func_2_0(43) +
          occ_func_2_0(11) * occ_func_2_0(41) +
          occ_func_2_0(5) * occ_func_2_0(33) +
          occ_func_2_0(13) * occ_func_2_0(45) +
          occ_func_2_0(39) * occ_func_2_0(47) +
          occ_func_2_0(3) * occ_func_2_0(37) +
          occ_func_2_0(9) * occ_func_2_0(53) +
          occ_func_2_0(7) * occ_func_2_0(49) +
          occ_func_2_0(35) * occ_func_2_0(31)) /
         12.;
}

/**** Basis functions for orbit 32****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.7500000 Va  O

1.3333333 0.6666666 -0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_32_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_2_0(46) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_2_0(29) +
          occ_func_2_0(45) * occ_func_2_0(27) * occ_func_2_0(1) +
          occ_func_2_0(52) * occ_func_2_0(0) * occ_func_2_0(26) +
          occ_func_2_0(52) * occ_func_2_0(28) * occ_func_2_0(0) +
          occ_func_2_0(50) * occ_func_2_0(26) * occ_func_2_0(0) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_2_0(28) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_2_0(44) +
          occ_func_2_0(49) * occ_func_2_0(25) * occ_func_2_0(1) +
          occ_func_2_0(47) * occ_func_2_0(1) * occ_func_2_0(25) +
          occ_func_2_0(47) * occ_func_2_0(29) * occ_func_2_0(1) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_2_0(27) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_2_0(26) +
          occ_func_2_0(48) * occ_func_2_0(28) * occ_func_2_0(0) +
          occ_func_2_0(48) * occ_func_2_0(0) * occ_func_2_0(24) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_2_0(49) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_2_0(43) +
          occ_func_2_0(53) * occ_func_2_0(1) * occ_func_2_0(29) +
          occ_func_2_0(53) * occ_func_2_0(27) * occ_func_2_0(1) +
          occ_func_2_0(42) * occ_func_2_0(26) * occ_func_2_0(0) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_2_0(28) +
          occ_func_2_0(51) * occ_func_2_0(29) * occ_func_2_0(1) +
          occ_func_2_0(51) * occ_func_2_0(1) * occ_func_2_0(27) +
          occ_func_2_0(46) * occ_func_2_0(24) * occ_func_2_0(0)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_32_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_2_0(46) +
          occ_func_2_0(40) * occ_func_2_0(0) * occ_func_2_0(24) +
          occ_func_2_0(36) * occ_func_2_0(22) * occ_func_2_0(0) +
          occ_func_2_0(52) * occ_func_2_0(0) * occ_func_2_0(26) +
          occ_func_2_0(0) * occ_func_2_0(30) * occ_func_2_0(42) +
          occ_func_2_0(40) * occ_func_2_0(20) * occ_func_2_0(0) +
          occ_func_2_0(52) * occ_func_2_0(28) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(46) * occ_func_2_0(30) +
          occ_func_2_0(36) * occ_func_2_0(0) * occ_func_2_0(18) +
          occ_func_2_0(50) * occ_func_2_0(26) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(44) * occ_func_2_0(32) +
          occ_func_2_0(38) * occ_func_2_0(0) * occ_func_2_0(20) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_2_0(28) +
          occ_func_2_0(0) * occ_func_2_0(32) * occ_func_2_0(48) +
          occ_func_2_0(34) * occ_func_2_0(18) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_2_0(44) +
          occ_func_2_0(38) * occ_func_2_0(24) * occ_func_2_0(0) +
          occ_func_2_0(34) * occ_func_2_0(0) * occ_func_2_0(22) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_2_0(26) +
          occ_func_2_0(0) * occ_func_2_0(38) * occ_func_2_0(50) +
          occ_func_2_0(32) * occ_func_2_0(20) * occ_func_2_0(0) +
          occ_func_2_0(48) * occ_func_2_0(28) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(50) * occ_func_2_0(34) +
          occ_func_2_0(32) * occ_func_2_0(0) * occ_func_2_0(18) +
          occ_func_2_0(48) * occ_func_2_0(0) * occ_func_2_0(24) +
          occ_func_2_0(44) * occ_func_2_0(22) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(34) * occ_func_2_0(38) +
          occ_func_2_0(42) * occ_func_2_0(26) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(52) * occ_func_2_0(40) +
          occ_func_2_0(30) * occ_func_2_0(0) * occ_func_2_0(20) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_2_0(28) +
          occ_func_2_0(0) * occ_func_2_0(36) * occ_func_2_0(52) +
          occ_func_2_0(30) * occ_func_2_0(18) * occ_func_2_0(0) +
          occ_func_2_0(46) * occ_func_2_0(24) * occ_func_2_0(0) +
          occ_func_2_0(42) * occ_func_2_0(0) * occ_func_2_0(22) +
          occ_func_2_0(0) * occ_func_2_0(40) * occ_func_2_0(36)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_32_0_at_1() const {
  return (occ_func_2_0(49) * occ_func_2_0(1) * occ_func_2_0(29) +
          occ_func_2_0(1) * occ_func_2_0(35) * occ_func_2_0(51) +
          occ_func_2_0(33) * occ_func_2_0(19) * occ_func_2_0(1) +
          occ_func_2_0(45) * occ_func_2_0(27) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_2_0(39) +
          occ_func_2_0(33) * occ_func_2_0(1) * occ_func_2_0(21) +
          occ_func_2_0(49) * occ_func_2_0(25) * occ_func_2_0(1) +
          occ_func_2_0(45) * occ_func_2_0(1) * occ_func_2_0(23) +
          occ_func_2_0(1) * occ_func_2_0(39) * occ_func_2_0(35) +
          occ_func_2_0(47) * occ_func_2_0(1) * occ_func_2_0(25) +
          occ_func_2_0(43) * occ_func_2_0(23) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(37) * occ_func_2_0(41) +
          occ_func_2_0(47) * occ_func_2_0(29) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_2_0(37) +
          occ_func_2_0(31) * occ_func_2_0(1) * occ_func_2_0(19) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_2_0(27) +
          occ_func_2_0(1) * occ_func_2_0(41) * occ_func_2_0(53) +
          occ_func_2_0(31) * occ_func_2_0(21) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_2_0(49) +
          occ_func_2_0(39) * occ_func_2_0(1) * occ_func_2_0(25) +
          occ_func_2_0(35) * occ_func_2_0(23) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_2_0(43) +
          occ_func_2_0(41) * occ_func_2_0(25) * occ_func_2_0(1) +
          occ_func_2_0(37) * occ_func_2_0(1) * occ_func_2_0(23) +
          occ_func_2_0(53) * occ_func_2_0(1) * occ_func_2_0(29) +
          occ_func_2_0(1) * occ_func_2_0(31) * occ_func_2_0(47) +
          occ_func_2_0(37) * occ_func_2_0(19) * occ_func_2_0(1) +
          occ_func_2_0(53) * occ_func_2_0(27) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(43) * occ_func_2_0(31) +
          occ_func_2_0(41) * occ_func_2_0(1) * occ_func_2_0(21) +
          occ_func_2_0(51) * occ_func_2_0(29) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(49) * occ_func_2_0(33) +
          occ_func_2_0(35) * occ_func_2_0(1) * occ_func_2_0(19) +
          occ_func_2_0(51) * occ_func_2_0(1) * occ_func_2_0(27) +
          occ_func_2_0(1) * occ_func_2_0(33) * occ_func_2_0(45) +
          occ_func_2_0(39) * occ_func_2_0(21) * occ_func_2_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_32_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(42) * occ_func_2_0(46) +
          occ_func_2_0(40) * occ_func_2_0(24) +
          occ_func_2_0(36) * occ_func_2_0(22) +
          occ_func_2_0(52) * occ_func_2_0(26) +
          occ_func_2_0(30) * occ_func_2_0(42) +
          occ_func_2_0(40) * occ_func_2_0(20) +
          occ_func_2_0(52) * occ_func_2_0(28) +
          occ_func_2_0(46) * occ_func_2_0(30) +
          occ_func_2_0(36) * occ_func_2_0(18) +
          occ_func_2_0(50) * occ_func_2_0(26) +
          occ_func_2_0(44) * occ_func_2_0(32) +
          occ_func_2_0(38) * occ_func_2_0(20) +
          occ_func_2_0(50) * occ_func_2_0(28) +
          occ_func_2_0(32) * occ_func_2_0(48) +
          occ_func_2_0(34) * occ_func_2_0(18) +
          occ_func_2_0(48) * occ_func_2_0(44) +
          occ_func_2_0(38) * occ_func_2_0(24) +
          occ_func_2_0(34) * occ_func_2_0(22) +
          occ_func_2_0(44) * occ_func_2_0(26) +
          occ_func_2_0(38) * occ_func_2_0(50) +
          occ_func_2_0(32) * occ_func_2_0(20) +
          occ_func_2_0(48) * occ_func_2_0(28) +
          occ_func_2_0(50) * occ_func_2_0(34) +
          occ_func_2_0(32) * occ_func_2_0(18) +
          occ_func_2_0(48) * occ_func_2_0(24) +
          occ_func_2_0(44) * occ_func_2_0(22) +
          occ_func_2_0(34) * occ_func_2_0(38) +
          occ_func_2_0(42) * occ_func_2_0(26) +
          occ_func_2_0(52) * occ_func_2_0(40) +
          occ_func_2_0(30) * occ_func_2_0(20) +
          occ_func_2_0(46) * occ_func_2_0(28) +
          occ_func_2_0(36) * occ_func_2_0(52) +
          occ_func_2_0(30) * occ_func_2_0(18) +
          occ_func_2_0(46) * occ_func_2_0(24) +
          occ_func_2_0(42) * occ_func_2_0(22) +
          occ_func_2_0(40) * occ_func_2_0(36)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_32_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(49) * occ_func_2_0(29) +
          occ_func_2_0(35) * occ_func_2_0(51) +
          occ_func_2_0(33) * occ_func_2_0(19) +
          occ_func_2_0(45) * occ_func_2_0(27) +
          occ_func_2_0(51) * occ_func_2_0(39) +
          occ_func_2_0(33) * occ_func_2_0(21) +
          occ_func_2_0(49) * occ_func_2_0(25) +
          occ_func_2_0(45) * occ_func_2_0(23) +
          occ_func_2_0(39) * occ_func_2_0(35) +
          occ_func_2_0(47) * occ_func_2_0(25) +
          occ_func_2_0(43) * occ_func_2_0(23) +
          occ_func_2_0(37) * occ_func_2_0(41) +
          occ_func_2_0(47) * occ_func_2_0(29) +
          occ_func_2_0(53) * occ_func_2_0(37) +
          occ_func_2_0(31) * occ_func_2_0(19) +
          occ_func_2_0(43) * occ_func_2_0(27) +
          occ_func_2_0(41) * occ_func_2_0(53) +
          occ_func_2_0(31) * occ_func_2_0(21) +
          occ_func_2_0(45) * occ_func_2_0(49) +
          occ_func_2_0(39) * occ_func_2_0(25) +
          occ_func_2_0(35) * occ_func_2_0(23) +
          occ_func_2_0(47) * occ_func_2_0(43) +
          occ_func_2_0(41) * occ_func_2_0(25) +
          occ_func_2_0(37) * occ_func_2_0(23) +
          occ_func_2_0(53) * occ_func_2_0(29) +
          occ_func_2_0(31) * occ_func_2_0(47) +
          occ_func_2_0(37) * occ_func_2_0(19) +
          occ_func_2_0(53) * occ_func_2_0(27) +
          occ_func_2_0(43) * occ_func_2_0(31) +
          occ_func_2_0(41) * occ_func_2_0(21) +
          occ_func_2_0(51) * occ_func_2_0(29) +
          occ_func_2_0(49) * occ_func_2_0(33) +
          occ_func_2_0(35) * occ_func_2_0(19) +
          occ_func_2_0(51) * occ_func_2_0(27) +
          occ_func_2_0(33) * occ_func_2_0(45) +
          occ_func_2_0(39) * occ_func_2_0(21)) /
         24.;
}

/**** Basis functions for orbit 33****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.2500000 Va  O

1.3333333 0.6666666 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_33_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_2_0(10) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_2_0(29) +
          occ_func_2_0(9) * occ_func_3_0(26) * occ_func_2_0(1) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_2_0(84) +
          occ_func_2_0(12) * occ_func_3_0(87) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_3_0(27) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_2_0(28) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_2_0(8) +
          occ_func_2_0(11) * occ_func_3_0(24) * occ_func_2_0(1) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_2_0(79) +
          occ_func_2_0(11) * occ_func_3_0(88) * occ_func_2_0(1) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_2_0(83) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_2_0(84) +
          occ_func_2_0(10) * occ_func_3_0(87) * occ_func_2_0(0) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_2_0(80) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_2_0(11) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_2_0(9) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_2_0(29) +
          occ_func_2_0(13) * occ_func_3_0(26) * occ_func_2_0(1) +
          occ_func_2_0(8) * occ_func_3_0(27) * occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_2_0(28) +
          occ_func_2_0(13) * occ_func_3_0(88) * occ_func_2_0(1) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_2_0(83) +
          occ_func_2_0(10) * occ_func_3_0(25) * occ_func_2_0(0)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_33_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_3_0(75) * occ_func_2_0(0) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_2_0(29) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_2_0(21) +
          occ_func_2_0(0) * occ_func_3_0(31) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_3_0(71) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_3_0(87) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_2_0(2) +
          occ_func_2_0(12) * occ_func_3_0(27) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(9) * occ_func_2_0(2) +
          occ_func_2_0(0) * occ_func_3_0(3) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_3_0(19) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_3_0(25) * occ_func_2_0(0) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_2_0(23) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_2_0(79) +
          occ_func_2_0(31) * occ_func_3_0(0) * occ_func_2_0(67) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_2_0(83) +
          occ_func_2_0(0) * occ_func_3_0(39) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(71) * occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_3_0(87) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_2_0(4) +
          occ_func_2_0(8) * occ_func_3_0(75) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(35) * occ_func_2_0(6) +
          occ_func_2_0(39) * occ_func_3_0(0) * occ_func_2_0(79) +
          occ_func_2_0(5) * occ_func_3_0(0) * occ_func_2_0(23) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_2_0(29) +
          occ_func_2_0(7) * occ_func_3_0(0) * occ_func_2_0(21) +
          occ_func_2_0(8) * occ_func_3_0(27) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(13) * occ_func_2_0(6) +
          occ_func_2_0(0) * occ_func_3_0(5) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(19) * occ_func_2_0(0) +
          occ_func_2_0(35) * occ_func_3_0(0) * occ_func_2_0(67) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_2_0(83) +
          occ_func_2_0(10) * occ_func_3_0(25) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(7) * occ_func_2_0(4)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_33_0_at_1() const {
  return (occ_func_2_0(40) * occ_func_3_0(1) * occ_func_2_0(80) +
          occ_func_2_0(1) * occ_func_3_0(4) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(18) * occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_3_0(26) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_2_0(7) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_2_0(84) +
          occ_func_2_0(36) * occ_func_3_0(1) * occ_func_2_0(68) +
          occ_func_2_0(6) * occ_func_3_0(1) * occ_func_2_0(20) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_2_0(28) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_2_0(22) +
          occ_func_2_0(11) * occ_func_3_0(24) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(6) * occ_func_2_0(5) +
          occ_func_2_0(9) * occ_func_3_0(76) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(36) * occ_func_2_0(7) +
          occ_func_2_0(11) * occ_func_3_0(88) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_2_0(5) +
          occ_func_2_0(1) * occ_func_3_0(40) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(72) * occ_func_2_0(1) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_2_0(84) +
          occ_func_2_0(32) * occ_func_3_0(1) * occ_func_2_0(68) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_2_0(80) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(76) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(24) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(2) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(18) * occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_3_0(26) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_2_0(3) +
          occ_func_2_0(2) * occ_func_3_0(1) * occ_func_2_0(20) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_2_0(28) +
          occ_func_2_0(13) * occ_func_3_0(88) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_2_0(3) +
          occ_func_2_0(1) * occ_func_3_0(32) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(72) * occ_func_2_0(1) +
          occ_func_2_0(8) * occ_func_3_0(1) * occ_func_2_0(22)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_33_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(43) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_3_0(75) +
          occ_func_2_0(11) * occ_func_2_0(29) +
          occ_func_2_0(3) * occ_func_2_0(21) +
          occ_func_3_0(31) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_3_0(71) +
          occ_func_2_0(12) * occ_func_3_0(87) +
          occ_func_3_0(47) * occ_func_2_0(2) +
          occ_func_2_0(12) * occ_func_3_0(27) +
          occ_func_3_0(9) * occ_func_2_0(2) +
          occ_func_3_0(3) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_3_0(19) +
          occ_func_3_0(11) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_3_0(25) +
          occ_func_2_0(9) * occ_func_2_0(23) +
          occ_func_2_0(47) * occ_func_2_0(79) +
          occ_func_2_0(31) * occ_func_2_0(67) +
          occ_func_2_0(43) * occ_func_2_0(83) +
          occ_func_3_0(39) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(71) +
          occ_func_2_0(10) * occ_func_3_0(87) +
          occ_func_3_0(51) * occ_func_2_0(4) +
          occ_func_2_0(8) * occ_func_3_0(75) +
          occ_func_3_0(35) * occ_func_2_0(6) +
          occ_func_2_0(39) * occ_func_2_0(79) +
          occ_func_2_0(5) * occ_func_2_0(23) +
          occ_func_2_0(13) * occ_func_2_0(29) +
          occ_func_2_0(7) * occ_func_2_0(21) +
          occ_func_2_0(8) * occ_func_3_0(27) +
          occ_func_3_0(13) * occ_func_2_0(6) +
          occ_func_3_0(5) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(19) +
          occ_func_2_0(35) * occ_func_2_0(67) +
          occ_func_2_0(51) * occ_func_2_0(83) +
          occ_func_2_0(10) * occ_func_3_0(25) +
          occ_func_3_0(7) * occ_func_2_0(4)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_33_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(40) * occ_func_2_0(80) +
          occ_func_3_0(4) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(18) +
          occ_func_2_0(9) * occ_func_3_0(26) +
          occ_func_3_0(12) * occ_func_2_0(7) +
          occ_func_2_0(52) * occ_func_2_0(84) +
          occ_func_2_0(36) * occ_func_2_0(68) +
          occ_func_2_0(6) * occ_func_2_0(20) +
          occ_func_2_0(12) * occ_func_2_0(28) +
          occ_func_2_0(4) * occ_func_2_0(22) +
          occ_func_2_0(11) * occ_func_3_0(24) +
          occ_func_3_0(6) * occ_func_2_0(5) +
          occ_func_2_0(9) * occ_func_3_0(76) +
          occ_func_3_0(36) * occ_func_2_0(7) +
          occ_func_2_0(11) * occ_func_3_0(88) +
          occ_func_3_0(52) * occ_func_2_0(5) +
          occ_func_3_0(40) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(72) +
          occ_func_2_0(44) * occ_func_2_0(84) +
          occ_func_2_0(32) * occ_func_2_0(68) +
          occ_func_2_0(48) * occ_func_2_0(80) +
          occ_func_3_0(44) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(76) +
          occ_func_3_0(10) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(24) +
          occ_func_3_0(2) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(18) +
          occ_func_2_0(13) * occ_func_3_0(26) +
          occ_func_3_0(8) * occ_func_2_0(3) +
          occ_func_2_0(2) * occ_func_2_0(20) +
          occ_func_2_0(10) * occ_func_2_0(28) +
          occ_func_2_0(13) * occ_func_3_0(88) +
          occ_func_3_0(48) * occ_func_2_0(3) +
          occ_func_3_0(32) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(72) +
          occ_func_2_0(8) * occ_func_2_0(22)) /
         24.;
}

/**** Basis functions for orbit 34****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

1.3333333 0.6666666 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_34_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(47) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(28) +
          occ_func_2_0(9) * occ_func_2_0(27) * occ_func_3_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(83) +
          occ_func_2_0(52) * occ_func_2_0(88) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_2_0(26) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(29) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(9) +
          occ_func_2_0(11) * occ_func_2_0(25) * occ_func_3_0(0) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(80) +
          occ_func_2_0(47) * occ_func_2_0(87) * occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(84) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(83) +
          occ_func_2_0(48) * occ_func_2_0(88) * occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(79) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_3_0(48) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_3_0(8) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(28) +
          occ_func_2_0(13) * occ_func_2_0(27) * occ_func_3_0(0) +
          occ_func_2_0(8) * occ_func_2_0(26) * occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(29) +
          occ_func_2_0(51) * occ_func_2_0(87) * occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(84) +
          occ_func_2_0(10) * occ_func_2_0(24) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_34_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(47) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(79) +
          occ_func_2_0(3) * occ_func_2_0(19) * occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_2_0(27) * occ_func_3_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(83) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_3_0(43) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(31) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(67) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(3) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(21) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(29) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_3_0(11) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(9) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(23) +
          occ_func_2_0(11) * occ_func_2_0(25) * occ_func_3_0(0) +
          occ_func_2_0(43) * occ_func_2_0(75) * occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_2_0(87) * occ_func_3_0(0) +
          occ_func_2_0(31) * occ_func_2_0(71) * occ_func_3_0(0) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(83) +
          occ_func_2_0(0) * occ_func_2_0(6) * occ_func_3_0(51) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_3_0(35) +
          occ_func_2_0(2) * occ_func_2_0(0) * occ_func_3_0(67) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(79) +
          occ_func_2_0(0) * occ_func_2_0(4) * occ_func_3_0(39) +
          occ_func_2_0(35) * occ_func_2_0(75) * occ_func_3_0(0) +
          occ_func_2_0(7) * occ_func_2_0(25) * occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_2_0(19) * occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_2_0(27) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_3_0(7) +
          occ_func_2_0(2) * occ_func_2_0(0) * occ_func_3_0(21) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(29) +
          occ_func_2_0(0) * occ_func_2_0(4) * occ_func_3_0(13) +
          occ_func_2_0(51) * occ_func_2_0(87) * occ_func_3_0(0) +
          occ_func_2_0(39) * occ_func_2_0(71) * occ_func_3_0(0) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(23) +
          occ_func_2_0(0) * occ_func_2_0(6) * occ_func_3_0(5)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_34_0_at_1() const {
  return (occ_func_2_0(36) * occ_func_2_0(76) * occ_func_3_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(28) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(6) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_3_0(20) +
          occ_func_2_0(40) * occ_func_2_0(72) * occ_func_3_0(1) +
          occ_func_2_0(52) * occ_func_2_0(88) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_2_0(26) * occ_func_3_0(1) +
          occ_func_2_0(4) * occ_func_2_0(18) * occ_func_3_0(1) +
          occ_func_2_0(6) * occ_func_2_0(24) * occ_func_3_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(22) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_3_0(4) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(80) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_3_0(40) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(36) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_3_0(68) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(84) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_3_0(52) +
          occ_func_2_0(32) * occ_func_2_0(72) * occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_2_0(88) * occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_2_0(76) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_3_0(48) +
          occ_func_2_0(7) * occ_func_2_0(1) * occ_func_3_0(80) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_3_0(8) +
          occ_func_2_0(5) * occ_func_2_0(1) * occ_func_3_0(22) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(28) +
          occ_func_2_0(1) * occ_func_2_0(3) * occ_func_3_0(10) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_3_0(2) +
          occ_func_2_0(7) * occ_func_2_0(1) * occ_func_3_0(20) +
          occ_func_2_0(8) * occ_func_2_0(26) * occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_2_0(18) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_3_0(32) +
          occ_func_2_0(5) * occ_func_2_0(1) * occ_func_3_0(68) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(84) +
          occ_func_2_0(1) * occ_func_2_0(3) * occ_func_3_0(44) +
          occ_func_2_0(10) * occ_func_2_0(24) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_34_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) * occ_func_3_0(47) +
          occ_func_2_0(6) * occ_func_3_0(79) +
          occ_func_2_0(3) * occ_func_2_0(19) +
          occ_func_2_0(9) * occ_func_2_0(27) +
          occ_func_2_0(12) * occ_func_3_0(83) +
          occ_func_2_0(2) * occ_func_3_0(43) +
          occ_func_2_0(10) * occ_func_3_0(31) +
          occ_func_2_0(4) * occ_func_3_0(67) +
          occ_func_2_0(8) * occ_func_3_0(3) +
          occ_func_2_0(6) * occ_func_3_0(21) +
          occ_func_2_0(12) * occ_func_3_0(29) +
          occ_func_2_0(2) * occ_func_3_0(11) +
          occ_func_2_0(10) * occ_func_3_0(9) +
          occ_func_2_0(4) * occ_func_3_0(23) +
          occ_func_2_0(11) * occ_func_2_0(25) +
          occ_func_2_0(43) * occ_func_2_0(75) +
          occ_func_2_0(47) * occ_func_2_0(87) +
          occ_func_2_0(31) * occ_func_2_0(71) +
          occ_func_2_0(8) * occ_func_3_0(83) +
          occ_func_2_0(6) * occ_func_3_0(51) +
          occ_func_2_0(12) * occ_func_3_0(35) +
          occ_func_2_0(2) * occ_func_3_0(67) +
          occ_func_2_0(10) * occ_func_3_0(79) +
          occ_func_2_0(4) * occ_func_3_0(39) +
          occ_func_2_0(35) * occ_func_2_0(75) +
          occ_func_2_0(7) * occ_func_2_0(25) +
          occ_func_2_0(5) * occ_func_2_0(19) +
          occ_func_2_0(13) * occ_func_2_0(27) +
          occ_func_2_0(12) * occ_func_3_0(7) +
          occ_func_2_0(2) * occ_func_3_0(21) +
          occ_func_2_0(10) * occ_func_3_0(29) +
          occ_func_2_0(4) * occ_func_3_0(13) +
          occ_func_2_0(51) * occ_func_2_0(87) +
          occ_func_2_0(39) * occ_func_2_0(71) +
          occ_func_2_0(8) * occ_func_3_0(23) +
          occ_func_2_0(6) * occ_func_3_0(5)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_34_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(36) * occ_func_2_0(76) +
          occ_func_2_0(11) * occ_func_3_0(28) +
          occ_func_2_0(5) * occ_func_3_0(12) +
          occ_func_2_0(13) * occ_func_3_0(6) +
          occ_func_2_0(3) * occ_func_3_0(20) +
          occ_func_2_0(40) * occ_func_2_0(72) +
          occ_func_2_0(52) * occ_func_2_0(88) +
          occ_func_2_0(12) * occ_func_2_0(26) +
          occ_func_2_0(4) * occ_func_2_0(18) +
          occ_func_2_0(6) * occ_func_2_0(24) +
          occ_func_2_0(9) * occ_func_3_0(22) +
          occ_func_2_0(7) * occ_func_3_0(4) +
          occ_func_2_0(11) * occ_func_3_0(80) +
          occ_func_2_0(5) * occ_func_3_0(40) +
          occ_func_2_0(13) * occ_func_3_0(36) +
          occ_func_2_0(3) * occ_func_3_0(68) +
          occ_func_2_0(9) * occ_func_3_0(84) +
          occ_func_2_0(7) * occ_func_3_0(52) +
          occ_func_2_0(32) * occ_func_2_0(72) +
          occ_func_2_0(48) * occ_func_2_0(88) +
          occ_func_2_0(44) * occ_func_2_0(76) +
          occ_func_2_0(9) * occ_func_3_0(48) +
          occ_func_2_0(7) * occ_func_3_0(80) +
          occ_func_2_0(11) * occ_func_3_0(8) +
          occ_func_2_0(5) * occ_func_3_0(22) +
          occ_func_2_0(13) * occ_func_3_0(28) +
          occ_func_2_0(3) * occ_func_3_0(10) +
          occ_func_2_0(9) * occ_func_3_0(2) +
          occ_func_2_0(7) * occ_func_3_0(20) +
          occ_func_2_0(8) * occ_func_2_0(26) +
          occ_func_2_0(2) * occ_func_2_0(18) +
          occ_func_2_0(11) * occ_func_3_0(32) +
          occ_func_2_0(5) * occ_func_3_0(68) +
          occ_func_2_0(13) * occ_func_3_0(84) +
          occ_func_2_0(3) * occ_func_3_0(44) +
          occ_func_2_0(10) * occ_func_2_0(24)) /
         24.;
}

/**** Basis functions for orbit 35****
0.3333333 0.6666666 0.2500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

2.3333333 1.6666667 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_35_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(28) * occ_func_3_0(29) +
          occ_func_2_0(1) * occ_func_2_0(27) * occ_func_3_0(84) +
          occ_func_2_0(1) * occ_func_2_0(25) * occ_func_3_0(80) +
          occ_func_2_0(24) * occ_func_2_0(0) * occ_func_3_0(1) +
          occ_func_2_0(26) * occ_func_2_0(0) * occ_func_3_0(1) +
          occ_func_2_0(88) * occ_func_2_0(16) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(24) * occ_func_3_0(79) +
          occ_func_2_0(0) * occ_func_2_0(26) * occ_func_3_0(83) +
          occ_func_2_0(29) * occ_func_2_0(1) * occ_func_3_0(16) +
          occ_func_2_0(27) * occ_func_2_0(1) * occ_func_3_0(0) +
          occ_func_2_0(25) * occ_func_2_0(1) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(29) * occ_func_3_0(28)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_35_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(28) * occ_func_3_0(29) +
          occ_func_2_0(18) * occ_func_2_0(0) * occ_func_3_0(1) +
          occ_func_2_0(71) * occ_func_2_0(15) * occ_func_3_0(0) +
          occ_func_2_0(75) * occ_func_2_0(15) * occ_func_3_0(0) +
          occ_func_2_0(24) * occ_func_2_0(0) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(22) * occ_func_3_0(23) +
          occ_func_2_0(26) * occ_func_2_0(0) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(20) * occ_func_3_0(21) +
          occ_func_2_0(28) * occ_func_2_0(0) * occ_func_3_0(15) +
          occ_func_2_0(0) * occ_func_2_0(18) * occ_func_3_0(67) +
          occ_func_2_0(0) * occ_func_2_0(24) * occ_func_3_0(79) +
          occ_func_2_0(22) * occ_func_2_0(0) * occ_func_3_0(15) +
          occ_func_2_0(0) * occ_func_2_0(26) * occ_func_3_0(83) +
          occ_func_2_0(20) * occ_func_2_0(0) * occ_func_3_0(15) +
          occ_func_2_0(87) * occ_func_2_0(15) * occ_func_3_0(0) +
          occ_func_2_0(27) * occ_func_2_0(1) * occ_func_3_0(0) +
          occ_func_2_0(25) * occ_func_2_0(1) * occ_func_3_0(0) +
          occ_func_2_0(19) * occ_func_2_0(1) * occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_35_0_at_1() const {
  return (occ_func_2_0(18) * occ_func_2_0(0) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(27) * occ_func_3_0(84) +
          occ_func_2_0(21) * occ_func_2_0(1) * occ_func_3_0(16) +
          occ_func_2_0(1) * occ_func_2_0(25) * occ_func_3_0(80) +
          occ_func_2_0(23) * occ_func_2_0(1) * occ_func_3_0(16) +
          occ_func_2_0(24) * occ_func_2_0(0) * occ_func_3_0(1) +
          occ_func_2_0(26) * occ_func_2_0(0) * occ_func_3_0(1) +
          occ_func_2_0(88) * occ_func_2_0(16) * occ_func_3_0(1) +
          occ_func_2_0(76) * occ_func_2_0(16) * occ_func_3_0(1) +
          occ_func_2_0(72) * occ_func_2_0(16) * occ_func_3_0(1) +
          occ_func_2_0(29) * occ_func_2_0(1) * occ_func_3_0(16) +
          occ_func_2_0(1) * occ_func_2_0(19) * occ_func_3_0(68) +
          occ_func_2_0(27) * occ_func_2_0(1) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(21) * occ_func_3_0(20) +
          occ_func_2_0(25) * occ_func_2_0(1) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(23) * occ_func_3_0(22) +
          occ_func_2_0(1) * occ_func_2_0(29) * occ_func_3_0(28) +
          occ_func_2_0(19) * occ_func_2_0(1) * occ_func_3_0(0)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_35_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(28) * occ_func_3_0(29) +
          occ_func_2_0(18) * occ_func_3_0(1) +
          occ_func_2_0(71) * occ_func_2_0(15) +
          occ_func_2_0(75) * occ_func_2_0(15) +
          occ_func_2_0(24) * occ_func_3_0(1) +
          occ_func_2_0(22) * occ_func_3_0(23) +
          occ_func_2_0(26) * occ_func_3_0(1) +
          occ_func_2_0(20) * occ_func_3_0(21) +
          occ_func_2_0(28) * occ_func_3_0(15) +
          occ_func_2_0(18) * occ_func_3_0(67) +
          occ_func_2_0(24) * occ_func_3_0(79) +
          occ_func_2_0(22) * occ_func_3_0(15) +
          occ_func_2_0(26) * occ_func_3_0(83) +
          occ_func_2_0(20) * occ_func_3_0(15) +
          occ_func_2_0(87) * occ_func_2_0(15) +
          occ_func_2_0(27) * occ_func_2_0(1) +
          occ_func_2_0(25) * occ_func_2_0(1) +
          occ_func_2_0(19) * occ_func_2_0(1)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_35_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(18) * occ_func_2_0(0) +
          occ_func_2_0(27) * occ_func_3_0(84) +
          occ_func_2_0(21) * occ_func_3_0(16) +
          occ_func_2_0(25) * occ_func_3_0(80) +
          occ_func_2_0(23) * occ_func_3_0(16) +
          occ_func_2_0(24) * occ_func_2_0(0) +
          occ_func_2_0(26) * occ_func_2_0(0) +
          occ_func_2_0(88) * occ_func_2_0(16) +
          occ_func_2_0(76) * occ_func_2_0(16) +
          occ_func_2_0(72) * occ_func_2_0(16) +
          occ_func_2_0(29) * occ_func_3_0(16) +
          occ_func_2_0(19) * occ_func_3_0(68) +
          occ_func_2_0(27) * occ_func_3_0(0) +
          occ_func_2_0(21) * occ_func_3_0(20) +
          occ_func_2_0(25) * occ_func_3_0(0) +
          occ_func_2_0(23) * occ_func_3_0(22) +
          occ_func_2_0(29) * occ_func_3_0(28) +
          occ_func_2_0(19) * occ_func_3_0(0)) /
         12.;
}

/**** Basis functions for orbit 36****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 0.7500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_36_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(27) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(25) +
          occ_func_2_0(24) * occ_func_3_0(25) * occ_func_2_0(0) +
          occ_func_2_0(26) * occ_func_3_0(27) * occ_func_2_0(0) +
          occ_func_2_0(28) * occ_func_3_0(87) * occ_func_2_0(0) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(80) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(84) +
          occ_func_2_0(29) * occ_func_3_0(88) * occ_func_2_0(1) +
          occ_func_2_0(27) * occ_func_3_0(26) * occ_func_2_0(1) +
          occ_func_2_0(25) * occ_func_3_0(24) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(29)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_36_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(19) * occ_func_2_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(83) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(79) +
          occ_func_2_0(24) * occ_func_3_0(25) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(27) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(20) +
          occ_func_2_0(28) * occ_func_3_0(87) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(18) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_3_0(75) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_3_0(71) * occ_func_2_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(67) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(21) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(23) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(29)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_36_0_at_1() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(72) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(76) * occ_func_2_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(22) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(20) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(68) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(80) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(84) +
          occ_func_2_0(29) * occ_func_3_0(88) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(19) +
          occ_func_2_0(27) * occ_func_3_0(26) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_3_0(24) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(23) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(18) * occ_func_2_0(1)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_36_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(1) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(19) +
          occ_func_2_0(15) * occ_func_2_0(83) +
          occ_func_2_0(15) * occ_func_2_0(79) +
          occ_func_2_0(24) * occ_func_3_0(25) +
          occ_func_3_0(1) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(27) +
          occ_func_3_0(1) * occ_func_2_0(20) +
          occ_func_2_0(28) * occ_func_3_0(87) +
          occ_func_3_0(15) * occ_func_2_0(18) +
          occ_func_3_0(15) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_3_0(75) +
          occ_func_3_0(15) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_3_0(71) +
          occ_func_2_0(15) * occ_func_2_0(67) +
          occ_func_2_0(1) * occ_func_2_0(21) +
          occ_func_2_0(1) * occ_func_2_0(23) +
          occ_func_2_0(1) * occ_func_2_0(29)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_36_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(0) * occ_func_2_0(28) +
          occ_func_3_0(16) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(72) +
          occ_func_3_0(16) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(76) +
          occ_func_2_0(0) * occ_func_2_0(22) +
          occ_func_2_0(0) * occ_func_2_0(20) +
          occ_func_2_0(16) * occ_func_2_0(68) +
          occ_func_2_0(16) * occ_func_2_0(80) +
          occ_func_2_0(16) * occ_func_2_0(84) +
          occ_func_2_0(29) * occ_func_3_0(88) +
          occ_func_3_0(16) * occ_func_2_0(19) +
          occ_func_2_0(27) * occ_func_3_0(26) +
          occ_func_3_0(0) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_3_0(24) +
          occ_func_3_0(0) * occ_func_2_0(23) +
          occ_func_3_0(0) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(18)) /
         12.;
}

/**** Basis functions for orbit 37****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.7500000 Va  O

1.3333333 2.6666666 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_37_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_3_0(83) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_3_0(28) +
          occ_func_2_0(88) * occ_func_2_0(10) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(46) * occ_func_3_0(79) +
          occ_func_2_0(24) * occ_func_2_0(48) * occ_func_3_0(1) +
          occ_func_2_0(26) * occ_func_2_0(44) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_3_0(29) +
          occ_func_2_0(27) * occ_func_2_0(51) * occ_func_3_0(0) +
          occ_func_2_0(87) * occ_func_2_0(13) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_3_0(84) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_3_0(10) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_3_0(11) +
          occ_func_2_0(0) * occ_func_2_0(50) * occ_func_3_0(83) +
          occ_func_2_0(88) * occ_func_2_0(12) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_3_0(84) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_3_0(28) +
          occ_func_2_0(27) * occ_func_2_0(43) * occ_func_3_0(0) +
          occ_func_2_0(25) * occ_func_2_0(47) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(52) * occ_func_3_0(29) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_3_0(43) +
          occ_func_2_0(1) * occ_func_2_0(49) * occ_func_3_0(80) +
          occ_func_2_0(87) * occ_func_2_0(11) * occ_func_3_0(0) +
          occ_func_2_0(26) * occ_func_2_0(52) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_37_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_3_0(83) +
          occ_func_2_0(40) * occ_func_2_0(0) * occ_func_3_0(13) +
          occ_func_2_0(25) * occ_func_2_0(39) * occ_func_3_0(0) +
          occ_func_2_0(19) * occ_func_2_0(35) * occ_func_3_0(0) +
          occ_func_2_0(52) * occ_func_2_0(0) * occ_func_3_0(5) +
          occ_func_2_0(0) * occ_func_2_0(30) * occ_func_3_0(67) +
          occ_func_2_0(0) * occ_func_2_0(46) * occ_func_3_0(79) +
          occ_func_2_0(36) * occ_func_2_0(0) * occ_func_3_0(7) +
          occ_func_2_0(0) * occ_func_2_0(44) * occ_func_3_0(23) +
          occ_func_2_0(38) * occ_func_2_0(0) * occ_func_3_0(35) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_3_0(39) +
          occ_func_2_0(0) * occ_func_2_0(32) * occ_func_3_0(21) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_3_0(29) +
          occ_func_2_0(34) * occ_func_2_0(0) * occ_func_3_0(51) +
          occ_func_2_0(27) * occ_func_2_0(51) * occ_func_3_0(0) +
          occ_func_2_0(87) * occ_func_2_0(13) * occ_func_3_0(0) +
          occ_func_2_0(71) * occ_func_2_0(7) * occ_func_3_0(0) +
          occ_func_2_0(75) * occ_func_2_0(5) * occ_func_3_0(0) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_3_0(11) +
          occ_func_2_0(0) * occ_func_2_0(38) * occ_func_3_0(79) +
          occ_func_2_0(0) * occ_func_2_0(50) * occ_func_3_0(83) +
          occ_func_2_0(32) * occ_func_2_0(0) * occ_func_3_0(9) +
          occ_func_2_0(48) * occ_func_2_0(0) * occ_func_3_0(3) +
          occ_func_2_0(0) * occ_func_2_0(34) * occ_func_3_0(67) +
          occ_func_2_0(71) * occ_func_2_0(3) * occ_func_3_0(0) +
          occ_func_2_0(19) * occ_func_2_0(31) * occ_func_3_0(0) +
          occ_func_2_0(27) * occ_func_2_0(43) * occ_func_3_0(0) +
          occ_func_2_0(25) * occ_func_2_0(47) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(52) * occ_func_3_0(29) +
          occ_func_2_0(30) * occ_func_2_0(0) * occ_func_3_0(47) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_3_0(43) +
          occ_func_2_0(0) * occ_func_2_0(36) * occ_func_3_0(23) +
          occ_func_2_0(75) * occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(87) * occ_func_2_0(11) * occ_func_3_0(0) +
          occ_func_2_0(42) * occ_func_2_0(0) * occ_func_3_0(31) +
          occ_func_2_0(0) * occ_func_2_0(40) * occ_func_3_0(21)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_37_0_at_1() const {
  return (occ_func_2_0(72) * occ_func_2_0(2) * occ_func_3_0(1) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_2_0(35) * occ_func_3_0(22) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_3_0(28) +
          occ_func_2_0(33) * occ_func_2_0(1) * occ_func_3_0(48) +
          occ_func_2_0(88) * occ_func_2_0(10) * occ_func_3_0(1) +
          occ_func_2_0(76) * occ_func_2_0(8) * occ_func_3_0(1) +
          occ_func_2_0(24) * occ_func_2_0(48) * occ_func_3_0(1) +
          occ_func_2_0(26) * occ_func_2_0(44) * occ_func_3_0(1) +
          occ_func_2_0(18) * occ_func_2_0(32) * occ_func_3_0(1) +
          occ_func_2_0(45) * occ_func_2_0(1) * occ_func_3_0(32) +
          occ_func_2_0(1) * occ_func_2_0(39) * occ_func_3_0(20) +
          occ_func_2_0(47) * occ_func_2_0(1) * occ_func_3_0(2) +
          occ_func_2_0(1) * occ_func_2_0(37) * occ_func_3_0(68) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_3_0(84) +
          occ_func_2_0(31) * occ_func_2_0(1) * occ_func_3_0(8) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_3_0(10) +
          occ_func_2_0(1) * occ_func_2_0(41) * occ_func_3_0(80) +
          occ_func_2_0(76) * occ_func_2_0(4) * occ_func_3_0(1) +
          occ_func_2_0(72) * occ_func_2_0(6) * occ_func_3_0(1) +
          occ_func_2_0(88) * occ_func_2_0(12) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_3_0(84) +
          occ_func_2_0(39) * occ_func_2_0(1) * occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_3_0(28) +
          occ_func_2_0(37) * occ_func_2_0(1) * occ_func_3_0(52) +
          occ_func_2_0(53) * occ_func_2_0(1) * occ_func_3_0(40) +
          occ_func_2_0(1) * occ_func_2_0(31) * occ_func_3_0(20) +
          occ_func_2_0(1) * occ_func_2_0(43) * occ_func_3_0(22) +
          occ_func_2_0(41) * occ_func_2_0(1) * occ_func_3_0(36) +
          occ_func_2_0(18) * occ_func_2_0(36) * occ_func_3_0(1) +
          occ_func_2_0(24) * occ_func_2_0(40) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(49) * occ_func_3_0(80) +
          occ_func_2_0(35) * occ_func_2_0(1) * occ_func_3_0(6) +
          occ_func_2_0(51) * occ_func_2_0(1) * occ_func_3_0(4) +
          occ_func_2_0(1) * occ_func_2_0(33) * occ_func_3_0(68) +
          occ_func_2_0(26) * occ_func_2_0(52) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_37_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(42) * occ_func_3_0(83) +
          occ_func_2_0(40) * occ_func_3_0(13) +
          occ_func_2_0(25) * occ_func_2_0(39) +
          occ_func_2_0(19) * occ_func_2_0(35) +
          occ_func_2_0(52) * occ_func_3_0(5) +
          occ_func_2_0(30) * occ_func_3_0(67) +
          occ_func_2_0(46) * occ_func_3_0(79) +
          occ_func_2_0(36) * occ_func_3_0(7) +
          occ_func_2_0(44) * occ_func_3_0(23) +
          occ_func_2_0(38) * occ_func_3_0(35) +
          occ_func_2_0(50) * occ_func_3_0(39) +
          occ_func_2_0(32) * occ_func_3_0(21) +
          occ_func_2_0(48) * occ_func_3_0(29) +
          occ_func_2_0(34) * occ_func_3_0(51) +
          occ_func_2_0(27) * occ_func_2_0(51) +
          occ_func_2_0(87) * occ_func_2_0(13) +
          occ_func_2_0(71) * occ_func_2_0(7) +
          occ_func_2_0(75) * occ_func_2_0(5) +
          occ_func_2_0(44) * occ_func_3_0(11) +
          occ_func_2_0(38) * occ_func_3_0(79) +
          occ_func_2_0(50) * occ_func_3_0(83) +
          occ_func_2_0(32) * occ_func_3_0(9) +
          occ_func_2_0(48) * occ_func_3_0(3) +
          occ_func_2_0(34) * occ_func_3_0(67) +
          occ_func_2_0(71) * occ_func_2_0(3) +
          occ_func_2_0(19) * occ_func_2_0(31) +
          occ_func_2_0(27) * occ_func_2_0(43) +
          occ_func_2_0(25) * occ_func_2_0(47) +
          occ_func_2_0(52) * occ_func_3_0(29) +
          occ_func_2_0(30) * occ_func_3_0(47) +
          occ_func_2_0(46) * occ_func_3_0(43) +
          occ_func_2_0(36) * occ_func_3_0(23) +
          occ_func_2_0(75) * occ_func_2_0(9) +
          occ_func_2_0(87) * occ_func_2_0(11) +
          occ_func_2_0(42) * occ_func_3_0(31) +
          occ_func_2_0(40) * occ_func_3_0(21)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_37_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(72) * occ_func_2_0(2) +
          occ_func_2_0(49) * occ_func_3_0(44) +
          occ_func_2_0(35) * occ_func_3_0(22) +
          occ_func_2_0(51) * occ_func_3_0(28) +
          occ_func_2_0(33) * occ_func_3_0(48) +
          occ_func_2_0(88) * occ_func_2_0(10) +
          occ_func_2_0(76) * occ_func_2_0(8) +
          occ_func_2_0(24) * occ_func_2_0(48) +
          occ_func_2_0(26) * occ_func_2_0(44) +
          occ_func_2_0(18) * occ_func_2_0(32) +
          occ_func_2_0(45) * occ_func_3_0(32) +
          occ_func_2_0(39) * occ_func_3_0(20) +
          occ_func_2_0(47) * occ_func_3_0(2) +
          occ_func_2_0(37) * occ_func_3_0(68) +
          occ_func_2_0(53) * occ_func_3_0(84) +
          occ_func_2_0(31) * occ_func_3_0(8) +
          occ_func_2_0(43) * occ_func_3_0(10) +
          occ_func_2_0(41) * occ_func_3_0(80) +
          occ_func_2_0(76) * occ_func_2_0(4) +
          occ_func_2_0(72) * occ_func_2_0(6) +
          occ_func_2_0(88) * occ_func_2_0(12) +
          occ_func_2_0(45) * occ_func_3_0(84) +
          occ_func_2_0(39) * occ_func_3_0(12) +
          occ_func_2_0(47) * occ_func_3_0(28) +
          occ_func_2_0(37) * occ_func_3_0(52) +
          occ_func_2_0(53) * occ_func_3_0(40) +
          occ_func_2_0(31) * occ_func_3_0(20) +
          occ_func_2_0(43) * occ_func_3_0(22) +
          occ_func_2_0(41) * occ_func_3_0(36) +
          occ_func_2_0(18) * occ_func_2_0(36) +
          occ_func_2_0(24) * occ_func_2_0(40) +
          occ_func_2_0(49) * occ_func_3_0(80) +
          occ_func_2_0(35) * occ_func_3_0(6) +
          occ_func_2_0(51) * occ_func_3_0(4) +
          occ_func_2_0(33) * occ_func_3_0(68) +
          occ_func_2_0(26) * occ_func_2_0(52)) /
         24.;
}

/**** Basis functions for orbit 38****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.7500000 Va  O

1.3333333 0.6666666 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_38_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_3_0(47) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_3_0(88) +
          occ_func_2_0(9) * occ_func_2_0(83) * occ_func_3_0(0) +
          occ_func_2_0(52) * occ_func_2_0(0) * occ_func_3_0(27) +
          occ_func_2_0(52) * occ_func_2_0(28) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_2_0(84) * occ_func_3_0(1) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_3_0(87) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_3_0(9) +
          occ_func_2_0(11) * occ_func_2_0(79) * occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_2_0(1) * occ_func_3_0(24) +
          occ_func_2_0(47) * occ_func_2_0(29) * occ_func_3_0(0) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_3_0(26) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_3_0(27) +
          occ_func_2_0(48) * occ_func_2_0(28) * occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_2_0(0) * occ_func_3_0(25) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_3_0(48) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_3_0(8) +
          occ_func_2_0(53) * occ_func_2_0(1) * occ_func_3_0(88) +
          occ_func_2_0(13) * occ_func_2_0(83) * occ_func_3_0(0) +
          occ_func_2_0(8) * occ_func_2_0(84) * occ_func_3_0(1) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_3_0(87) +
          occ_func_2_0(51) * occ_func_2_0(29) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_2_0(1) * occ_func_3_0(26) +
          occ_func_2_0(10) * occ_func_2_0(80) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_38_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_3_0(47) +
          occ_func_2_0(40) * occ_func_2_0(0) * occ_func_3_0(25) +
          occ_func_2_0(3) * occ_func_2_0(67) * occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_2_0(83) * occ_func_3_0(0) +
          occ_func_2_0(52) * occ_func_2_0(0) * occ_func_3_0(27) +
          occ_func_2_0(0) * occ_func_2_0(30) * occ_func_3_0(43) +
          occ_func_2_0(0) * occ_func_2_0(46) * occ_func_3_0(31) +
          occ_func_2_0(36) * occ_func_2_0(0) * occ_func_3_0(19) +
          occ_func_2_0(0) * occ_func_2_0(44) * occ_func_3_0(3) +
          occ_func_2_0(38) * occ_func_2_0(0) * occ_func_3_0(71) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_3_0(87) +
          occ_func_2_0(0) * occ_func_2_0(32) * occ_func_3_0(11) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_3_0(9) +
          occ_func_2_0(34) * occ_func_2_0(0) * occ_func_3_0(75) +
          occ_func_2_0(11) * occ_func_2_0(79) * occ_func_3_0(0) +
          occ_func_2_0(43) * occ_func_2_0(23) * occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_2_0(29) * occ_func_3_0(0) +
          occ_func_2_0(31) * occ_func_2_0(21) * occ_func_3_0(0) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_3_0(27) +
          occ_func_2_0(0) * occ_func_2_0(38) * occ_func_3_0(51) +
          occ_func_2_0(0) * occ_func_2_0(50) * occ_func_3_0(35) +
          occ_func_2_0(32) * occ_func_2_0(0) * occ_func_3_0(19) +
          occ_func_2_0(48) * occ_func_2_0(0) * occ_func_3_0(25) +
          occ_func_2_0(0) * occ_func_2_0(34) * occ_func_3_0(39) +
          occ_func_2_0(35) * occ_func_2_0(23) * occ_func_3_0(0) +
          occ_func_2_0(7) * occ_func_2_0(79) * occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_2_0(67) * occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_2_0(83) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(52) * occ_func_3_0(7) +
          occ_func_2_0(30) * occ_func_2_0(0) * occ_func_3_0(71) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_3_0(87) +
          occ_func_2_0(0) * occ_func_2_0(36) * occ_func_3_0(13) +
          occ_func_2_0(51) * occ_func_2_0(29) * occ_func_3_0(0) +
          occ_func_2_0(39) * occ_func_2_0(21) * occ_func_3_0(0) +
          occ_func_2_0(42) * occ_func_2_0(0) * occ_func_3_0(75) +
          occ_func_2_0(0) * occ_func_2_0(40) * occ_func_3_0(5)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_38_0_at_1() const {
  return (occ_func_2_0(36) * occ_func_2_0(22) * occ_func_3_0(1) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_3_0(88) +
          occ_func_2_0(1) * occ_func_2_0(35) * occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_3_0(6) +
          occ_func_2_0(33) * occ_func_2_0(1) * occ_func_3_0(72) +
          occ_func_2_0(40) * occ_func_2_0(20) * occ_func_3_0(1) +
          occ_func_2_0(52) * occ_func_2_0(28) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_2_0(84) * occ_func_3_0(1) +
          occ_func_2_0(4) * occ_func_2_0(68) * occ_func_3_0(1) +
          occ_func_2_0(6) * occ_func_2_0(80) * occ_func_3_0(1) +
          occ_func_2_0(45) * occ_func_2_0(1) * occ_func_3_0(76) +
          occ_func_2_0(1) * occ_func_2_0(39) * occ_func_3_0(4) +
          occ_func_2_0(47) * occ_func_2_0(1) * occ_func_3_0(24) +
          occ_func_2_0(1) * occ_func_2_0(37) * occ_func_3_0(40) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_3_0(36) +
          occ_func_2_0(31) * occ_func_2_0(1) * occ_func_3_0(18) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_3_0(26) +
          occ_func_2_0(1) * occ_func_2_0(41) * occ_func_3_0(52) +
          occ_func_2_0(32) * occ_func_2_0(20) * occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_2_0(28) * occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_2_0(22) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_3_0(48) +
          occ_func_2_0(39) * occ_func_2_0(1) * occ_func_3_0(24) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_3_0(8) +
          occ_func_2_0(37) * occ_func_2_0(1) * occ_func_3_0(76) +
          occ_func_2_0(53) * occ_func_2_0(1) * occ_func_3_0(88) +
          occ_func_2_0(1) * occ_func_2_0(31) * occ_func_3_0(10) +
          occ_func_2_0(1) * occ_func_2_0(43) * occ_func_3_0(2) +
          occ_func_2_0(41) * occ_func_2_0(1) * occ_func_3_0(72) +
          occ_func_2_0(8) * occ_func_2_0(84) * occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_2_0(68) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(49) * occ_func_3_0(32) +
          occ_func_2_0(35) * occ_func_2_0(1) * occ_func_3_0(18) +
          occ_func_2_0(51) * occ_func_2_0(1) * occ_func_3_0(26) +
          occ_func_2_0(1) * occ_func_2_0(33) * occ_func_3_0(44) +
          occ_func_2_0(10) * occ_func_2_0(80) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_38_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(42) * occ_func_3_0(47) +
          occ_func_2_0(40) * occ_func_3_0(25) +
          occ_func_2_0(3) * occ_func_2_0(67) +
          occ_func_2_0(9) * occ_func_2_0(83) +
          occ_func_2_0(52) * occ_func_3_0(27) +
          occ_func_2_0(30) * occ_func_3_0(43) +
          occ_func_2_0(46) * occ_func_3_0(31) +
          occ_func_2_0(36) * occ_func_3_0(19) +
          occ_func_2_0(44) * occ_func_3_0(3) +
          occ_func_2_0(38) * occ_func_3_0(71) +
          occ_func_2_0(50) * occ_func_3_0(87) +
          occ_func_2_0(32) * occ_func_3_0(11) +
          occ_func_2_0(48) * occ_func_3_0(9) +
          occ_func_2_0(34) * occ_func_3_0(75) +
          occ_func_2_0(11) * occ_func_2_0(79) +
          occ_func_2_0(43) * occ_func_2_0(23) +
          occ_func_2_0(47) * occ_func_2_0(29) +
          occ_func_2_0(31) * occ_func_2_0(21) +
          occ_func_2_0(44) * occ_func_3_0(27) +
          occ_func_2_0(38) * occ_func_3_0(51) +
          occ_func_2_0(50) * occ_func_3_0(35) +
          occ_func_2_0(32) * occ_func_3_0(19) +
          occ_func_2_0(48) * occ_func_3_0(25) +
          occ_func_2_0(34) * occ_func_3_0(39) +
          occ_func_2_0(35) * occ_func_2_0(23) +
          occ_func_2_0(7) * occ_func_2_0(79) +
          occ_func_2_0(5) * occ_func_2_0(67) +
          occ_func_2_0(13) * occ_func_2_0(83) +
          occ_func_2_0(52) * occ_func_3_0(7) +
          occ_func_2_0(30) * occ_func_3_0(71) +
          occ_func_2_0(46) * occ_func_3_0(87) +
          occ_func_2_0(36) * occ_func_3_0(13) +
          occ_func_2_0(51) * occ_func_2_0(29) +
          occ_func_2_0(39) * occ_func_2_0(21) +
          occ_func_2_0(42) * occ_func_3_0(75) +
          occ_func_2_0(40) * occ_func_3_0(5)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_38_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(36) * occ_func_2_0(22) +
          occ_func_2_0(49) * occ_func_3_0(88) +
          occ_func_2_0(35) * occ_func_3_0(12) +
          occ_func_2_0(51) * occ_func_3_0(6) +
          occ_func_2_0(33) * occ_func_3_0(72) +
          occ_func_2_0(40) * occ_func_2_0(20) +
          occ_func_2_0(52) * occ_func_2_0(28) +
          occ_func_2_0(12) * occ_func_2_0(84) +
          occ_func_2_0(4) * occ_func_2_0(68) +
          occ_func_2_0(6) * occ_func_2_0(80) +
          occ_func_2_0(45) * occ_func_3_0(76) +
          occ_func_2_0(39) * occ_func_3_0(4) +
          occ_func_2_0(47) * occ_func_3_0(24) +
          occ_func_2_0(37) * occ_func_3_0(40) +
          occ_func_2_0(53) * occ_func_3_0(36) +
          occ_func_2_0(31) * occ_func_3_0(18) +
          occ_func_2_0(43) * occ_func_3_0(26) +
          occ_func_2_0(41) * occ_func_3_0(52) +
          occ_func_2_0(32) * occ_func_2_0(20) +
          occ_func_2_0(48) * occ_func_2_0(28) +
          occ_func_2_0(44) * occ_func_2_0(22) +
          occ_func_2_0(45) * occ_func_3_0(48) +
          occ_func_2_0(39) * occ_func_3_0(24) +
          occ_func_2_0(47) * occ_func_3_0(8) +
          occ_func_2_0(37) * occ_func_3_0(76) +
          occ_func_2_0(53) * occ_func_3_0(88) +
          occ_func_2_0(31) * occ_func_3_0(10) +
          occ_func_2_0(43) * occ_func_3_0(2) +
          occ_func_2_0(41) * occ_func_3_0(72) +
          occ_func_2_0(8) * occ_func_2_0(84) +
          occ_func_2_0(2) * occ_func_2_0(68) +
          occ_func_2_0(49) * occ_func_3_0(32) +
          occ_func_2_0(35) * occ_func_3_0(18) +
          occ_func_2_0(51) * occ_func_3_0(26) +
          occ_func_2_0(33) * occ_func_3_0(44) +
          occ_func_2_0(10) * occ_func_2_0(80)) /
         24.;
}

/**** Basis functions for orbit 39****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 1.2500000 Va  O

2.3333333 1.6666667 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_39_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(29) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(84) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(80) +
          occ_func_2_0(24) * occ_func_2_0(80) * occ_func_3_0(1) +
          occ_func_2_0(26) * occ_func_2_0(84) * occ_func_3_0(1) +
          occ_func_2_0(88) * occ_func_2_0(28) * occ_func_3_0(1) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_3_0(25) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_3_0(27) +
          occ_func_2_0(87) * occ_func_2_0(29) * occ_func_3_0(0) +
          occ_func_2_0(27) * occ_func_2_0(83) * occ_func_3_0(0) +
          occ_func_2_0(25) * occ_func_2_0(79) * occ_func_3_0(0) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_3_0(88)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_39_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(29) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(87) +
          occ_func_2_0(71) * occ_func_2_0(21) * occ_func_3_0(0) +
          occ_func_2_0(75) * occ_func_2_0(23) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(23) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(75) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(21) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(71) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_3_0(19) +
          occ_func_2_0(0) * occ_func_2_0(14) * occ_func_3_0(67) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_3_0(25) +
          occ_func_2_0(0) * occ_func_2_0(14) * occ_func_3_0(79) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_3_0(27) +
          occ_func_2_0(0) * occ_func_2_0(14) * occ_func_3_0(83) +
          occ_func_2_0(87) * occ_func_2_0(29) * occ_func_3_0(0) +
          occ_func_2_0(27) * occ_func_2_0(83) * occ_func_3_0(0) +
          occ_func_2_0(25) * occ_func_2_0(79) * occ_func_3_0(0) +
          occ_func_2_0(19) * occ_func_2_0(67) * occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_39_0_at_1() const {
  return (occ_func_2_0(18) * occ_func_2_0(68) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(84) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(26) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(80) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(24) +
          occ_func_2_0(24) * occ_func_2_0(80) * occ_func_3_0(1) +
          occ_func_2_0(26) * occ_func_2_0(84) * occ_func_3_0(1) +
          occ_func_2_0(88) * occ_func_2_0(28) * occ_func_3_0(1) +
          occ_func_2_0(76) * occ_func_2_0(22) * occ_func_3_0(1) +
          occ_func_2_0(72) * occ_func_2_0(20) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(68) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(18) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_3_0(72) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_3_0(20) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_3_0(76) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_3_0(22) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_3_0(88) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_3_0(28)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_39_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(16) * occ_func_3_0(29) +
          occ_func_2_0(14) * occ_func_3_0(87) +
          occ_func_2_0(71) * occ_func_2_0(21) +
          occ_func_2_0(75) * occ_func_2_0(23) +
          occ_func_2_0(16) * occ_func_3_0(23) +
          occ_func_2_0(14) * occ_func_3_0(75) +
          occ_func_2_0(16) * occ_func_3_0(21) +
          occ_func_2_0(14) * occ_func_3_0(71) +
          occ_func_2_0(16) * occ_func_3_0(19) +
          occ_func_2_0(14) * occ_func_3_0(67) +
          occ_func_2_0(16) * occ_func_3_0(25) +
          occ_func_2_0(14) * occ_func_3_0(79) +
          occ_func_2_0(16) * occ_func_3_0(27) +
          occ_func_2_0(14) * occ_func_3_0(83) +
          occ_func_2_0(87) * occ_func_2_0(29) +
          occ_func_2_0(27) * occ_func_2_0(83) +
          occ_func_2_0(25) * occ_func_2_0(79) +
          occ_func_2_0(19) * occ_func_2_0(67)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_39_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(18) * occ_func_2_0(68) +
          occ_func_2_0(17) * occ_func_3_0(84) +
          occ_func_2_0(15) * occ_func_3_0(26) +
          occ_func_2_0(17) * occ_func_3_0(80) +
          occ_func_2_0(15) * occ_func_3_0(24) +
          occ_func_2_0(24) * occ_func_2_0(80) +
          occ_func_2_0(26) * occ_func_2_0(84) +
          occ_func_2_0(88) * occ_func_2_0(28) +
          occ_func_2_0(76) * occ_func_2_0(22) +
          occ_func_2_0(72) * occ_func_2_0(20) +
          occ_func_2_0(17) * occ_func_3_0(68) +
          occ_func_2_0(15) * occ_func_3_0(18) +
          occ_func_2_0(17) * occ_func_3_0(72) +
          occ_func_2_0(15) * occ_func_3_0(20) +
          occ_func_2_0(17) * occ_func_3_0(76) +
          occ_func_2_0(15) * occ_func_3_0(22) +
          occ_func_2_0(17) * occ_func_3_0(88) +
          occ_func_2_0(15) * occ_func_3_0(28)) /
         12.;
}

/**** Basis functions for orbit 40****
0.3333333 0.6666666 0.2500000 Va  O

1.3333333 -0.3333333 -0.2500000 Va  O

2.3333333 1.6666667 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_40_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(79) * occ_func_3_0(87) +
          occ_func_2_0(1) * occ_func_3_0(28) * occ_func_3_0(26) +
          occ_func_2_0(27) * occ_func_3_0(0) * occ_func_3_0(28) +
          occ_func_2_0(80) * occ_func_3_0(29) * occ_func_3_0(1) +
          occ_func_2_0(88) * occ_func_3_0(1) * occ_func_3_0(25) +
          occ_func_2_0(28) * occ_func_3_0(25) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(29) * occ_func_3_0(25) +
          occ_func_2_0(24) * occ_func_3_0(1) * occ_func_3_0(29) +
          occ_func_2_0(29) * occ_func_3_0(26) * occ_func_3_0(0) +
          occ_func_2_0(83) * occ_func_3_0(28) * occ_func_3_0(0) +
          occ_func_2_0(87) * occ_func_3_0(0) * occ_func_3_0(26) +
          occ_func_2_0(1) * occ_func_3_0(84) * occ_func_3_0(88) +
          occ_func_2_0(0) * occ_func_3_0(83) * occ_func_3_0(87) +
          occ_func_2_0(88) * occ_func_3_0(1) * occ_func_3_0(27) +
          occ_func_2_0(84) * occ_func_3_0(29) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(80) * occ_func_3_0(88) +
          occ_func_2_0(25) * occ_func_3_0(0) * occ_func_3_0(28) +
          occ_func_2_0(1) * occ_func_3_0(28) * occ_func_3_0(24) +
          occ_func_2_0(29) * occ_func_3_0(24) * occ_func_3_0(0) +
          occ_func_2_0(26) * occ_func_3_0(1) * occ_func_3_0(29) +
          occ_func_2_0(0) * occ_func_3_0(29) * occ_func_3_0(27) +
          occ_func_2_0(87) * occ_func_3_0(0) * occ_func_3_0(24) +
          occ_func_2_0(79) * occ_func_3_0(28) * occ_func_3_0(0) +
          occ_func_2_0(28) * occ_func_3_0(27) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_40_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(79) * occ_func_3_0(87) +
          occ_func_2_0(21) * occ_func_3_0(24) * occ_func_3_0(0) +
          occ_func_2_0(19) * occ_func_3_0(0) * occ_func_3_0(22) +
          occ_func_2_0(27) * occ_func_3_0(0) * occ_func_3_0(28) +
          occ_func_2_0(23) * occ_func_3_0(18) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(83) * occ_func_3_0(75) +
          occ_func_2_0(0) * occ_func_3_0(67) * occ_func_3_0(71) +
          occ_func_2_0(0) * occ_func_3_0(21) * occ_func_3_0(19) +
          occ_func_2_0(0) * occ_func_3_0(29) * occ_func_3_0(25) +
          occ_func_2_0(0) * occ_func_3_0(23) * occ_func_3_0(27) +
          occ_func_2_0(29) * occ_func_3_0(26) * occ_func_3_0(0) +
          occ_func_2_0(25) * occ_func_3_0(0) * occ_func_3_0(20) +
          occ_func_2_0(83) * occ_func_3_0(28) * occ_func_3_0(0) +
          occ_func_2_0(75) * occ_func_3_0(0) * occ_func_3_0(18) +
          occ_func_2_0(87) * occ_func_3_0(0) * occ_func_3_0(26) +
          occ_func_2_0(79) * occ_func_3_0(20) * occ_func_3_0(0) +
          occ_func_2_0(71) * occ_func_3_0(0) * occ_func_3_0(24) +
          occ_func_2_0(67) * occ_func_3_0(22) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(83) * occ_func_3_0(87) +
          occ_func_2_0(0) * occ_func_3_0(67) * occ_func_3_0(75) +
          occ_func_2_0(0) * occ_func_3_0(79) * occ_func_3_0(71) +
          occ_func_2_0(75) * occ_func_3_0(0) * occ_func_3_0(26) +
          occ_func_2_0(67) * occ_func_3_0(20) * occ_func_3_0(0) +
          occ_func_2_0(25) * occ_func_3_0(0) * occ_func_3_0(28) +
          occ_func_2_0(21) * occ_func_3_0(18) * occ_func_3_0(0) +
          occ_func_2_0(23) * occ_func_3_0(26) * occ_func_3_0(0) +
          occ_func_2_0(19) * occ_func_3_0(0) * occ_func_3_0(20) +
          occ_func_2_0(29) * occ_func_3_0(24) * occ_func_3_0(0) +
          occ_func_2_0(27) * occ_func_3_0(0) * occ_func_3_0(22) +
          occ_func_2_0(0) * occ_func_3_0(21) * occ_func_3_0(25) +
          occ_func_2_0(0) * occ_func_3_0(29) * occ_func_3_0(27) +
          occ_func_2_0(87) * occ_func_3_0(0) * occ_func_3_0(24) +
          occ_func_2_0(83) * occ_func_3_0(22) * occ_func_3_0(0) +
          occ_func_2_0(79) * occ_func_3_0(28) * occ_func_3_0(0) +
          occ_func_2_0(71) * occ_func_3_0(0) * occ_func_3_0(18) +
          occ_func_2_0(0) * occ_func_3_0(23) * occ_func_3_0(19)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_40_0_at_1() const {
  return (occ_func_2_0(76) * occ_func_3_0(1) * occ_func_3_0(27) +
          occ_func_2_0(68) * occ_func_3_0(21) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(28) * occ_func_3_0(26) +
          occ_func_2_0(1) * occ_func_3_0(20) * occ_func_3_0(24) +
          occ_func_2_0(80) * occ_func_3_0(29) * occ_func_3_0(1) +
          occ_func_2_0(72) * occ_func_3_0(1) * occ_func_3_0(19) +
          occ_func_2_0(88) * occ_func_3_0(1) * occ_func_3_0(25) +
          occ_func_2_0(84) * occ_func_3_0(23) * occ_func_3_0(1) +
          occ_func_2_0(28) * occ_func_3_0(25) * occ_func_3_0(1) +
          occ_func_2_0(26) * occ_func_3_0(1) * occ_func_3_0(23) +
          occ_func_2_0(22) * occ_func_3_0(27) * occ_func_3_0(1) +
          occ_func_2_0(18) * occ_func_3_0(1) * occ_func_3_0(21) +
          occ_func_2_0(24) * occ_func_3_0(1) * occ_func_3_0(29) +
          occ_func_2_0(20) * occ_func_3_0(19) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(22) * occ_func_3_0(18) +
          occ_func_2_0(1) * occ_func_3_0(80) * occ_func_3_0(72) +
          occ_func_2_0(1) * occ_func_3_0(68) * occ_func_3_0(76) +
          occ_func_2_0(1) * occ_func_3_0(84) * occ_func_3_0(88) +
          occ_func_2_0(72) * occ_func_3_0(1) * occ_func_3_0(25) +
          occ_func_2_0(68) * occ_func_3_0(23) * occ_func_3_0(1) +
          occ_func_2_0(88) * occ_func_3_0(1) * occ_func_3_0(27) +
          occ_func_2_0(80) * occ_func_3_0(21) * occ_func_3_0(1) +
          occ_func_2_0(84) * occ_func_3_0(29) * occ_func_3_0(1) +
          occ_func_2_0(76) * occ_func_3_0(1) * occ_func_3_0(19) +
          occ_func_2_0(1) * occ_func_3_0(80) * occ_func_3_0(88) +
          occ_func_2_0(1) * occ_func_3_0(22) * occ_func_3_0(26) +
          occ_func_2_0(1) * occ_func_3_0(28) * occ_func_3_0(24) +
          occ_func_2_0(1) * occ_func_3_0(20) * occ_func_3_0(18) +
          occ_func_2_0(26) * occ_func_3_0(1) * occ_func_3_0(29) +
          occ_func_2_0(22) * occ_func_3_0(19) * occ_func_3_0(1) +
          occ_func_2_0(20) * occ_func_3_0(25) * occ_func_3_0(1) +
          occ_func_2_0(18) * occ_func_3_0(1) * occ_func_3_0(23) +
          occ_func_2_0(1) * occ_func_3_0(68) * occ_func_3_0(72) +
          occ_func_2_0(1) * occ_func_3_0(84) * occ_func_3_0(76) +
          occ_func_2_0(28) * occ_func_3_0(27) * occ_func_3_0(1) +
          occ_func_2_0(24) * occ_func_3_0(1) * occ_func_3_0(21)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_40_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(79) * occ_func_3_0(87) +
          occ_func_2_0(21) * occ_func_3_0(24) +
          occ_func_2_0(19) * occ_func_3_0(22) +
          occ_func_2_0(27) * occ_func_3_0(28) +
          occ_func_2_0(23) * occ_func_3_0(18) +
          occ_func_3_0(83) * occ_func_3_0(75) +
          occ_func_3_0(67) * occ_func_3_0(71) +
          occ_func_3_0(21) * occ_func_3_0(19) +
          occ_func_3_0(29) * occ_func_3_0(25) +
          occ_func_3_0(23) * occ_func_3_0(27) +
          occ_func_2_0(29) * occ_func_3_0(26) +
          occ_func_2_0(25) * occ_func_3_0(20) +
          occ_func_2_0(83) * occ_func_3_0(28) +
          occ_func_2_0(75) * occ_func_3_0(18) +
          occ_func_2_0(87) * occ_func_3_0(26) +
          occ_func_2_0(79) * occ_func_3_0(20) +
          occ_func_2_0(71) * occ_func_3_0(24) +
          occ_func_2_0(67) * occ_func_3_0(22) +
          occ_func_3_0(83) * occ_func_3_0(87) +
          occ_func_3_0(67) * occ_func_3_0(75) +
          occ_func_3_0(79) * occ_func_3_0(71) +
          occ_func_2_0(75) * occ_func_3_0(26) +
          occ_func_2_0(67) * occ_func_3_0(20) +
          occ_func_2_0(25) * occ_func_3_0(28) +
          occ_func_2_0(21) * occ_func_3_0(18) +
          occ_func_2_0(23) * occ_func_3_0(26) +
          occ_func_2_0(19) * occ_func_3_0(20) +
          occ_func_2_0(29) * occ_func_3_0(24) +
          occ_func_2_0(27) * occ_func_3_0(22) +
          occ_func_3_0(21) * occ_func_3_0(25) +
          occ_func_3_0(29) * occ_func_3_0(27) +
          occ_func_2_0(87) * occ_func_3_0(24) +
          occ_func_2_0(83) * occ_func_3_0(22) +
          occ_func_2_0(79) * occ_func_3_0(28) +
          occ_func_2_0(71) * occ_func_3_0(18) +
          occ_func_3_0(23) * occ_func_3_0(19)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_40_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(76) * occ_func_3_0(27) +
          occ_func_2_0(68) * occ_func_3_0(21) +
          occ_func_3_0(28) * occ_func_3_0(26) +
          occ_func_3_0(20) * occ_func_3_0(24) +
          occ_func_2_0(80) * occ_func_3_0(29) +
          occ_func_2_0(72) * occ_func_3_0(19) +
          occ_func_2_0(88) * occ_func_3_0(25) +
          occ_func_2_0(84) * occ_func_3_0(23) +
          occ_func_2_0(28) * occ_func_3_0(25) +
          occ_func_2_0(26) * occ_func_3_0(23) +
          occ_func_2_0(22) * occ_func_3_0(27) +
          occ_func_2_0(18) * occ_func_3_0(21) +
          occ_func_2_0(24) * occ_func_3_0(29) +
          occ_func_2_0(20) * occ_func_3_0(19) +
          occ_func_3_0(22) * occ_func_3_0(18) +
          occ_func_3_0(80) * occ_func_3_0(72) +
          occ_func_3_0(68) * occ_func_3_0(76) +
          occ_func_3_0(84) * occ_func_3_0(88) +
          occ_func_2_0(72) * occ_func_3_0(25) +
          occ_func_2_0(68) * occ_func_3_0(23) +
          occ_func_2_0(88) * occ_func_3_0(27) +
          occ_func_2_0(80) * occ_func_3_0(21) +
          occ_func_2_0(84) * occ_func_3_0(29) +
          occ_func_2_0(76) * occ_func_3_0(19) +
          occ_func_3_0(80) * occ_func_3_0(88) +
          occ_func_3_0(22) * occ_func_3_0(26) +
          occ_func_3_0(28) * occ_func_3_0(24) +
          occ_func_3_0(20) * occ_func_3_0(18) +
          occ_func_2_0(26) * occ_func_3_0(29) +
          occ_func_2_0(22) * occ_func_3_0(19) +
          occ_func_2_0(20) * occ_func_3_0(25) +
          occ_func_2_0(18) * occ_func_3_0(23) +
          occ_func_3_0(68) * occ_func_3_0(72) +
          occ_func_3_0(84) * occ_func_3_0(76) +
          occ_func_2_0(28) * occ_func_3_0(27) +
          occ_func_2_0(24) * occ_func_3_0(21)) /
         24.;
}

/**** Basis functions for orbit 41****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

0.3333333 2.6666666 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_41_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(60) +
          occ_func_2_0(63) * occ_func_2_0(11) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(65) +
          occ_func_2_0(64) * occ_func_2_0(12) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(62) +
          occ_func_2_0(61) * occ_func_2_0(9) * occ_func_2_0(1)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_41_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(60) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(8) +
          occ_func_2_0(58) * occ_func_2_0(6) * occ_func_2_0(0) +
          occ_func_2_0(64) * occ_func_2_0(12) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(2) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(54) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(62) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_2_0(10) +
          occ_func_2_0(56) * occ_func_2_0(4) * occ_func_2_0(0)) /
         6.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_41_0_at_1() const {
  return (occ_func_2_0(63) * occ_func_2_0(11) * occ_func_2_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(5) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(57) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(65) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_2_0(13) +
          occ_func_2_0(55) * occ_func_2_0(3) * occ_func_2_0(1) +
          occ_func_2_0(61) * occ_func_2_0(9) * occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(7) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(59)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_41_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) * occ_func_2_0(60) +
          occ_func_2_0(6) * occ_func_2_0(8) +
          occ_func_2_0(58) * occ_func_2_0(6) +
          occ_func_2_0(64) * occ_func_2_0(12) +
          occ_func_2_0(12) * occ_func_2_0(2) +
          occ_func_2_0(2) * occ_func_2_0(54) +
          occ_func_2_0(10) * occ_func_2_0(62) +
          occ_func_2_0(4) * occ_func_2_0(10) +
          occ_func_2_0(56) * occ_func_2_0(4)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_41_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(63) * occ_func_2_0(11) +
          occ_func_2_0(11) * occ_func_2_0(5) +
          occ_func_2_0(5) * occ_func_2_0(57) +
          occ_func_2_0(13) * occ_func_2_0(65) +
          occ_func_2_0(3) * occ_func_2_0(13) +
          occ_func_2_0(55) * occ_func_2_0(3) +
          occ_func_2_0(61) * occ_func_2_0(9) +
          occ_func_2_0(9) * occ_func_2_0(7) +
          occ_func_2_0(7) * occ_func_2_0(59)) /
         6.;
}

/**** Basis functions for orbit 42****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.2500000 Va  O

0.3333333 2.6666666 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_42_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_2_0(60) +
          occ_func_2_0(63) * occ_func_3_0(10) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_2_0(65) +
          occ_func_2_0(64) * occ_func_3_0(51) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_2_0(62) +
          occ_func_2_0(0) * occ_func_3_0(9) * occ_func_2_0(60) +
          occ_func_2_0(64) * occ_func_3_0(13) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_2_0(62) +
          occ_func_2_0(61) * occ_func_3_0(8) * occ_func_2_0(1) +
          occ_func_2_0(63) * occ_func_3_0(48) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_2_0(65) +
          occ_func_2_0(61) * occ_func_3_0(44) * occ_func_2_0(1)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_42_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_2_0(60) +
          occ_func_2_0(58) * occ_func_3_0(39) * occ_func_2_0(0) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_2_0(5) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_2_0(13) +
          occ_func_2_0(64) * occ_func_3_0(51) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(31) * occ_func_2_0(54) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_2_0(62) +
          occ_func_2_0(56) * occ_func_3_0(35) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(9) * occ_func_2_0(60) +
          occ_func_2_0(58) * occ_func_3_0(7) * occ_func_2_0(0) +
          occ_func_2_0(64) * occ_func_3_0(13) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(3) * occ_func_2_0(54) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_2_0(62) +
          occ_func_2_0(56) * occ_func_3_0(5) * occ_func_2_0(0) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_2_0(7) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_2_0(35) +
          occ_func_2_0(31) * occ_func_3_0(0) * occ_func_2_0(51) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_2_0(39)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_42_0_at_1() const {
  return (occ_func_2_0(40) * occ_func_3_0(1) * occ_func_2_0(44) +
          occ_func_2_0(63) * occ_func_3_0(10) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(4) * occ_func_2_0(57) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_2_0(65) +
          occ_func_2_0(55) * occ_func_3_0(2) * occ_func_2_0(1) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_2_0(32) +
          occ_func_2_0(36) * occ_func_3_0(1) * occ_func_2_0(48) +
          occ_func_2_0(6) * occ_func_3_0(1) * occ_func_2_0(8) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_2_0(2) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_2_0(10) +
          occ_func_2_0(61) * occ_func_3_0(8) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(6) * occ_func_2_0(59) +
          occ_func_2_0(63) * occ_func_3_0(48) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(36) * occ_func_2_0(57) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_2_0(65) +
          occ_func_2_0(55) * occ_func_3_0(32) * occ_func_2_0(1) +
          occ_func_2_0(61) * occ_func_3_0(44) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(40) * occ_func_2_0(59)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_42_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(43) * occ_func_2_0(60) +
          occ_func_2_0(58) * occ_func_3_0(39) +
          occ_func_2_0(11) * occ_func_2_0(5) +
          occ_func_2_0(3) * occ_func_2_0(13) +
          occ_func_2_0(64) * occ_func_3_0(51) +
          occ_func_3_0(31) * occ_func_2_0(54) +
          occ_func_3_0(47) * occ_func_2_0(62) +
          occ_func_2_0(56) * occ_func_3_0(35) +
          occ_func_3_0(9) * occ_func_2_0(60) +
          occ_func_2_0(58) * occ_func_3_0(7) +
          occ_func_2_0(64) * occ_func_3_0(13) +
          occ_func_3_0(3) * occ_func_2_0(54) +
          occ_func_3_0(11) * occ_func_2_0(62) +
          occ_func_2_0(56) * occ_func_3_0(5) +
          occ_func_2_0(9) * occ_func_2_0(7) +
          occ_func_2_0(47) * occ_func_2_0(35) +
          occ_func_2_0(31) * occ_func_2_0(51) +
          occ_func_2_0(43) * occ_func_2_0(39)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_42_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(40) * occ_func_2_0(44) +
          occ_func_2_0(63) * occ_func_3_0(10) +
          occ_func_3_0(4) * occ_func_2_0(57) +
          occ_func_3_0(12) * occ_func_2_0(65) +
          occ_func_2_0(55) * occ_func_3_0(2) +
          occ_func_2_0(52) * occ_func_2_0(32) +
          occ_func_2_0(36) * occ_func_2_0(48) +
          occ_func_2_0(6) * occ_func_2_0(8) +
          occ_func_2_0(12) * occ_func_2_0(2) +
          occ_func_2_0(4) * occ_func_2_0(10) +
          occ_func_2_0(61) * occ_func_3_0(8) +
          occ_func_3_0(6) * occ_func_2_0(59) +
          occ_func_2_0(63) * occ_func_3_0(48) +
          occ_func_3_0(36) * occ_func_2_0(57) +
          occ_func_3_0(52) * occ_func_2_0(65) +
          occ_func_2_0(55) * occ_func_3_0(32) +
          occ_func_2_0(61) * occ_func_3_0(44) +
          occ_func_3_0(40) * occ_func_2_0(59)) /
         12.;
}

/**** Basis functions for orbit 43****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_43_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(28) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(65) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(25) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(60) +
          occ_func_2_0(26) * occ_func_2_0(64) * occ_func_2_0(0) +
          occ_func_2_0(28) * occ_func_2_0(64) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(62) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(26) +
          occ_func_2_0(29) * occ_func_2_0(63) * occ_func_2_0(1) +
          occ_func_2_0(27) * occ_func_2_0(61) * occ_func_2_0(1) +
          occ_func_2_0(25) * occ_func_2_0(63) * occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(65) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_2_0(64) +
          occ_func_2_0(24) * occ_func_2_0(62) * occ_func_2_0(0) +
          occ_func_2_0(26) * occ_func_2_0(60) * occ_func_2_0(0) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_2_0(29) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_2_0(27) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_2_0(63) +
          occ_func_2_0(29) * occ_func_2_0(65) * occ_func_2_0(1) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_2_0(24) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_2_0(64) +
          occ_func_2_0(27) * occ_func_2_0(65) * occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_2_0(61) +
          occ_func_2_0(28) * occ_func_2_0(62) * occ_func_2_0(0)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_43_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(28) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(62) +
          occ_func_2_0(18) * occ_func_2_0(56) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(60) +
          occ_func_2_0(24) * occ_func_2_0(58) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_2_0(64) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(20) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_2_0(54) +
          occ_func_2_0(28) * occ_func_2_0(64) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(18) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(54) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(62) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_2_0(56) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(26) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_2_0(60) +
          occ_func_2_0(20) * occ_func_2_0(58) * occ_func_2_0(0) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_2_0(64) +
          occ_func_2_0(0) * occ_func_2_0(6) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_2_0(54) * occ_func_2_0(0) +
          occ_func_2_0(24) * occ_func_2_0(62) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_2_0(22) +
          occ_func_2_0(2) * occ_func_2_0(0) * occ_func_2_0(56) +
          occ_func_2_0(26) * occ_func_2_0(60) * occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_2_0(58) +
          occ_func_2_0(0) * occ_func_2_0(4) * occ_func_2_0(20) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_2_0(60) * occ_func_2_0(0) +
          occ_func_2_0(2) * occ_func_2_0(0) * occ_func_2_0(58) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_2_0(64) +
          occ_func_2_0(0) * occ_func_2_0(4) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_2_0(54) * occ_func_2_0(0) +
          occ_func_2_0(28) * occ_func_2_0(62) * occ_func_2_0(0) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_2_0(56) +
          occ_func_2_0(0) * occ_func_2_0(6) * occ_func_2_0(18)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_43_0_at_1() const {
  return (occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(65) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_2_0(55) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_2_0(61) * occ_func_2_0(1) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_2_0(59) +
          occ_func_2_0(29) * occ_func_2_0(63) * occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(57) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(19) +
          occ_func_2_0(27) * occ_func_2_0(61) * occ_func_2_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(59) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_2_0(63) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(23) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_2_0(57) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(65) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_2_0(55) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_2_0(29) +
          occ_func_2_0(7) * occ_func_2_0(1) * occ_func_2_0(63) +
          occ_func_2_0(19) * occ_func_2_0(57) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_2_0(27) +
          occ_func_2_0(5) * occ_func_2_0(1) * occ_func_2_0(61) +
          occ_func_2_0(21) * occ_func_2_0(59) * occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_2_0(63) +
          occ_func_2_0(1) * occ_func_2_0(3) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_2_0(57) * occ_func_2_0(1) +
          occ_func_2_0(29) * occ_func_2_0(65) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_2_0(19) +
          occ_func_2_0(7) * occ_func_2_0(1) * occ_func_2_0(55) +
          occ_func_2_0(27) * occ_func_2_0(65) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_2_0(21) +
          occ_func_2_0(5) * occ_func_2_0(1) * occ_func_2_0(55) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_2_0(61) +
          occ_func_2_0(25) * occ_func_2_0(59) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(3) * occ_func_2_0(23)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_43_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) * occ_func_2_0(28) +
          occ_func_2_0(6) * occ_func_2_0(62) +
          occ_func_2_0(18) * occ_func_2_0(56) +
          occ_func_2_0(12) * occ_func_2_0(60) +
          occ_func_2_0(24) * occ_func_2_0(58) +
          occ_func_2_0(2) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_2_0(64) +
          occ_func_2_0(10) * occ_func_2_0(20) +
          occ_func_2_0(4) * occ_func_2_0(54) +
          occ_func_2_0(28) * occ_func_2_0(64) +
          occ_func_2_0(8) * occ_func_2_0(18) +
          occ_func_2_0(6) * occ_func_2_0(54) +
          occ_func_2_0(12) * occ_func_2_0(62) +
          occ_func_2_0(2) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_2_0(56) +
          occ_func_2_0(10) * occ_func_2_0(26) +
          occ_func_2_0(4) * occ_func_2_0(60) +
          occ_func_2_0(20) * occ_func_2_0(58) +
          occ_func_2_0(8) * occ_func_2_0(64) +
          occ_func_2_0(6) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_2_0(54) +
          occ_func_2_0(24) * occ_func_2_0(62) +
          occ_func_2_0(12) * occ_func_2_0(22) +
          occ_func_2_0(2) * occ_func_2_0(56) +
          occ_func_2_0(26) * occ_func_2_0(60) +
          occ_func_2_0(10) * occ_func_2_0(58) +
          occ_func_2_0(4) * occ_func_2_0(20) +
          occ_func_2_0(12) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_2_0(60) +
          occ_func_2_0(2) * occ_func_2_0(58) +
          occ_func_2_0(10) * occ_func_2_0(64) +
          occ_func_2_0(4) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_2_0(54) +
          occ_func_2_0(28) * occ_func_2_0(62) +
          occ_func_2_0(8) * occ_func_2_0(56) +
          occ_func_2_0(6) * occ_func_2_0(18)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_43_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(11) * occ_func_2_0(65) +
          occ_func_2_0(5) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_2_0(55) +
          occ_func_2_0(13) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_2_0(61) +
          occ_func_2_0(3) * occ_func_2_0(59) +
          occ_func_2_0(29) * occ_func_2_0(63) +
          occ_func_2_0(9) * occ_func_2_0(57) +
          occ_func_2_0(7) * occ_func_2_0(19) +
          occ_func_2_0(27) * occ_func_2_0(61) +
          occ_func_2_0(11) * occ_func_2_0(59) +
          occ_func_2_0(5) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_2_0(63) +
          occ_func_2_0(13) * occ_func_2_0(23) +
          occ_func_2_0(3) * occ_func_2_0(57) +
          occ_func_2_0(9) * occ_func_2_0(65) +
          occ_func_2_0(7) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_2_0(55) +
          occ_func_2_0(9) * occ_func_2_0(29) +
          occ_func_2_0(7) * occ_func_2_0(63) +
          occ_func_2_0(19) * occ_func_2_0(57) +
          occ_func_2_0(11) * occ_func_2_0(27) +
          occ_func_2_0(5) * occ_func_2_0(61) +
          occ_func_2_0(21) * occ_func_2_0(59) +
          occ_func_2_0(13) * occ_func_2_0(63) +
          occ_func_2_0(3) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_2_0(57) +
          occ_func_2_0(29) * occ_func_2_0(65) +
          occ_func_2_0(9) * occ_func_2_0(19) +
          occ_func_2_0(7) * occ_func_2_0(55) +
          occ_func_2_0(27) * occ_func_2_0(65) +
          occ_func_2_0(11) * occ_func_2_0(21) +
          occ_func_2_0(5) * occ_func_2_0(55) +
          occ_func_2_0(13) * occ_func_2_0(61) +
          occ_func_2_0(25) * occ_func_2_0(59) +
          occ_func_2_0(3) * occ_func_2_0(23)) /
         24.;
}

/**** Basis functions for orbit 44****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.7500000 Va  O

0.3333333 2.6666666 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_44_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_2_0(60) +
          occ_func_2_0(63) * occ_func_2_0(47) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_2_0(65) +
          occ_func_2_0(64) * occ_func_2_0(50) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(46) * occ_func_2_0(62) +
          occ_func_2_0(0) * occ_func_2_0(44) * occ_func_2_0(60) +
          occ_func_2_0(64) * occ_func_2_0(52) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_2_0(62) +
          occ_func_2_0(61) * occ_func_2_0(43) * occ_func_2_0(1) +
          occ_func_2_0(63) * occ_func_2_0(49) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_2_0(65) +
          occ_func_2_0(61) * occ_func_2_0(45) * occ_func_2_0(1)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_44_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_2_0(60) +
          occ_func_2_0(40) * occ_func_2_0(0) * occ_func_2_0(44) +
          occ_func_2_0(58) * occ_func_2_0(38) * occ_func_2_0(0) +
          occ_func_2_0(64) * occ_func_2_0(50) * occ_func_2_0(0) +
          occ_func_2_0(52) * occ_func_2_0(0) * occ_func_2_0(32) +
          occ_func_2_0(0) * occ_func_2_0(30) * occ_func_2_0(54) +
          occ_func_2_0(0) * occ_func_2_0(46) * occ_func_2_0(62) +
          occ_func_2_0(36) * occ_func_2_0(0) * occ_func_2_0(48) +
          occ_func_2_0(56) * occ_func_2_0(34) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(44) * occ_func_2_0(60) +
          occ_func_2_0(38) * occ_func_2_0(0) * occ_func_2_0(42) +
          occ_func_2_0(58) * occ_func_2_0(40) * occ_func_2_0(0) +
          occ_func_2_0(64) * occ_func_2_0(52) * occ_func_2_0(0) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_2_0(30) +
          occ_func_2_0(0) * occ_func_2_0(32) * occ_func_2_0(54) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_2_0(62) +
          occ_func_2_0(34) * occ_func_2_0(0) * occ_func_2_0(46) +
          occ_func_2_0(56) * occ_func_2_0(36) * occ_func_2_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_44_0_at_1() const {
  return (occ_func_2_0(63) * occ_func_2_0(47) * occ_func_2_0(1) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_2_0(37) +
          occ_func_2_0(1) * occ_func_2_0(35) * occ_func_2_0(57) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_2_0(65) +
          occ_func_2_0(33) * occ_func_2_0(1) * occ_func_2_0(53) +
          occ_func_2_0(55) * occ_func_2_0(31) * occ_func_2_0(1) +
          occ_func_2_0(61) * occ_func_2_0(43) * occ_func_2_0(1) +
          occ_func_2_0(45) * occ_func_2_0(1) * occ_func_2_0(41) +
          occ_func_2_0(1) * occ_func_2_0(39) * occ_func_2_0(59) +
          occ_func_2_0(63) * occ_func_2_0(49) * occ_func_2_0(1) +
          occ_func_2_0(47) * occ_func_2_0(1) * occ_func_2_0(35) +
          occ_func_2_0(1) * occ_func_2_0(37) * occ_func_2_0(57) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_2_0(65) +
          occ_func_2_0(31) * occ_func_2_0(1) * occ_func_2_0(51) +
          occ_func_2_0(55) * occ_func_2_0(33) * occ_func_2_0(1) +
          occ_func_2_0(61) * occ_func_2_0(45) * occ_func_2_0(1) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_2_0(39) +
          occ_func_2_0(1) * occ_func_2_0(41) * occ_func_2_0(59)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_44_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(42) * occ_func_2_0(60) +
          occ_func_2_0(40) * occ_func_2_0(44) +
          occ_func_2_0(58) * occ_func_2_0(38) +
          occ_func_2_0(64) * occ_func_2_0(50) +
          occ_func_2_0(52) * occ_func_2_0(32) +
          occ_func_2_0(30) * occ_func_2_0(54) +
          occ_func_2_0(46) * occ_func_2_0(62) +
          occ_func_2_0(36) * occ_func_2_0(48) +
          occ_func_2_0(56) * occ_func_2_0(34) +
          occ_func_2_0(44) * occ_func_2_0(60) +
          occ_func_2_0(38) * occ_func_2_0(42) +
          occ_func_2_0(58) * occ_func_2_0(40) +
          occ_func_2_0(64) * occ_func_2_0(52) +
          occ_func_2_0(50) * occ_func_2_0(30) +
          occ_func_2_0(32) * occ_func_2_0(54) +
          occ_func_2_0(48) * occ_func_2_0(62) +
          occ_func_2_0(34) * occ_func_2_0(46) +
          occ_func_2_0(56) * occ_func_2_0(36)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_44_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(63) * occ_func_2_0(47) +
          occ_func_2_0(49) * occ_func_2_0(37) +
          occ_func_2_0(35) * occ_func_2_0(57) +
          occ_func_2_0(51) * occ_func_2_0(65) +
          occ_func_2_0(33) * occ_func_2_0(53) +
          occ_func_2_0(55) * occ_func_2_0(31) +
          occ_func_2_0(61) * occ_func_2_0(43) +
          occ_func_2_0(45) * occ_func_2_0(41) +
          occ_func_2_0(39) * occ_func_2_0(59) +
          occ_func_2_0(63) * occ_func_2_0(49) +
          occ_func_2_0(47) * occ_func_2_0(35) +
          occ_func_2_0(37) * occ_func_2_0(57) +
          occ_func_2_0(53) * occ_func_2_0(65) +
          occ_func_2_0(31) * occ_func_2_0(51) +
          occ_func_2_0(55) * occ_func_2_0(33) +
          occ_func_2_0(61) * occ_func_2_0(45) +
          occ_func_2_0(43) * occ_func_2_0(39) +
          occ_func_2_0(41) * occ_func_2_0(59)) /
         12.;
}

/**** Basis functions for orbit 45****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.2500000 Va  O

2.3333333 1.6666667 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_45_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_3_0(87) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_3_0(64) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_3_0(24) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_3_0(61) +
          occ_func_2_0(84) * occ_func_3_0(65) * occ_func_3_0(1) +
          occ_func_2_0(28) * occ_func_3_0(65) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_3_0(63) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_3_0(27) +
          occ_func_2_0(29) * occ_func_3_0(62) * occ_func_3_0(0) +
          occ_func_2_0(83) * occ_func_3_0(60) * occ_func_3_0(0) +
          occ_func_2_0(79) * occ_func_3_0(62) * occ_func_3_0(0) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_3_0(64) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_3_0(65) +
          occ_func_2_0(80) * occ_func_3_0(63) * occ_func_3_0(1) +
          occ_func_2_0(84) * occ_func_3_0(61) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_3_0(88) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_3_0(26) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_3_0(62) +
          occ_func_2_0(29) * occ_func_3_0(64) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(13) * occ_func_3_0(25) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_3_0(65) +
          occ_func_2_0(83) * occ_func_3_0(64) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_3_0(60) +
          occ_func_2_0(28) * occ_func_3_0(63) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_45_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_3_0(87) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_3_0(64) +
          occ_func_2_0(21) * occ_func_3_0(54) * occ_func_3_0(0) +
          occ_func_2_0(23) * occ_func_3_0(60) * occ_func_3_0(0) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_3_0(58) +
          occ_func_2_0(0) * occ_func_3_0(31) * occ_func_3_0(75) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_3_0(71) +
          occ_func_2_0(0) * occ_func_3_0(9) * occ_func_3_0(19) +
          occ_func_2_0(0) * occ_func_3_0(3) * occ_func_3_0(25) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_3_0(27) +
          occ_func_2_0(29) * occ_func_3_0(62) * occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_3_0(56) +
          occ_func_2_0(83) * occ_func_3_0(60) * occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_3_0(58) +
          occ_func_2_0(79) * occ_func_3_0(62) * occ_func_3_0(0) +
          occ_func_2_0(31) * occ_func_3_0(0) * occ_func_3_0(56) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_3_0(64) +
          occ_func_2_0(67) * occ_func_3_0(54) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(39) * occ_func_3_0(87) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_3_0(75) +
          occ_func_2_0(0) * occ_func_3_0(35) * occ_func_3_0(71) +
          occ_func_2_0(39) * occ_func_3_0(0) * occ_func_3_0(62) +
          occ_func_2_0(67) * occ_func_3_0(56) * occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_3_0(0) * occ_func_3_0(60) +
          occ_func_2_0(21) * occ_func_3_0(58) * occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_3_0(62) +
          occ_func_2_0(23) * occ_func_3_0(56) * occ_func_3_0(0) +
          occ_func_2_0(29) * occ_func_3_0(64) * occ_func_3_0(0) +
          occ_func_2_0(7) * occ_func_3_0(0) * occ_func_3_0(54) +
          occ_func_2_0(0) * occ_func_3_0(13) * occ_func_3_0(25) +
          occ_func_2_0(0) * occ_func_3_0(5) * occ_func_3_0(27) +
          occ_func_2_0(83) * occ_func_3_0(64) * occ_func_3_0(0) +
          occ_func_2_0(35) * occ_func_3_0(0) * occ_func_3_0(54) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_3_0(60) +
          occ_func_2_0(79) * occ_func_3_0(58) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(7) * occ_func_3_0(19)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_45_0_at_1() const {
  return (occ_func_2_0(40) * occ_func_3_0(1) * occ_func_3_0(63) +
          occ_func_2_0(68) * occ_func_3_0(57) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(4) * occ_func_3_0(26) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_3_0(24) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_3_0(61) +
          occ_func_2_0(80) * occ_func_3_0(59) * occ_func_3_0(1) +
          occ_func_2_0(84) * occ_func_3_0(65) * occ_func_3_0(1) +
          occ_func_2_0(36) * occ_func_3_0(1) * occ_func_3_0(55) +
          occ_func_2_0(28) * occ_func_3_0(65) * occ_func_3_0(1) +
          occ_func_2_0(6) * occ_func_3_0(1) * occ_func_3_0(55) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_3_0(63) +
          occ_func_2_0(22) * occ_func_3_0(57) * occ_func_3_0(1) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_3_0(61) +
          occ_func_2_0(20) * occ_func_3_0(59) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(6) * occ_func_3_0(18) +
          occ_func_2_0(1) * occ_func_3_0(36) * occ_func_3_0(72) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_3_0(76) +
          occ_func_2_0(1) * occ_func_3_0(40) * occ_func_3_0(88) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_3_0(65) +
          occ_func_2_0(68) * occ_func_3_0(55) * occ_func_3_0(1) +
          occ_func_2_0(80) * occ_func_3_0(63) * occ_func_3_0(1) +
          occ_func_2_0(32) * occ_func_3_0(1) * occ_func_3_0(57) +
          occ_func_2_0(84) * occ_func_3_0(61) * occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_3_0(59) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_3_0(88) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_3_0(26) +
          occ_func_2_0(1) * occ_func_3_0(2) * occ_func_3_0(24) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_3_0(18) +
          occ_func_2_0(22) * occ_func_3_0(61) * occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_3_0(1) * occ_func_3_0(59) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_3_0(65) +
          occ_func_2_0(20) * occ_func_3_0(55) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_3_0(72) +
          occ_func_2_0(1) * occ_func_3_0(32) * occ_func_3_0(76) +
          occ_func_2_0(28) * occ_func_3_0(63) * occ_func_3_0(1) +
          occ_func_2_0(8) * occ_func_3_0(1) * occ_func_3_0(57)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_45_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(43) * occ_func_3_0(87) +
          occ_func_2_0(11) * occ_func_3_0(64) +
          occ_func_2_0(21) * occ_func_3_0(54) +
          occ_func_2_0(23) * occ_func_3_0(60) +
          occ_func_2_0(3) * occ_func_3_0(58) +
          occ_func_3_0(31) * occ_func_3_0(75) +
          occ_func_3_0(47) * occ_func_3_0(71) +
          occ_func_3_0(9) * occ_func_3_0(19) +
          occ_func_3_0(3) * occ_func_3_0(25) +
          occ_func_3_0(11) * occ_func_3_0(27) +
          occ_func_2_0(29) * occ_func_3_0(62) +
          occ_func_2_0(9) * occ_func_3_0(56) +
          occ_func_2_0(83) * occ_func_3_0(60) +
          occ_func_2_0(47) * occ_func_3_0(58) +
          occ_func_2_0(79) * occ_func_3_0(62) +
          occ_func_2_0(31) * occ_func_3_0(56) +
          occ_func_2_0(43) * occ_func_3_0(64) +
          occ_func_2_0(67) * occ_func_3_0(54) +
          occ_func_3_0(39) * occ_func_3_0(87) +
          occ_func_3_0(51) * occ_func_3_0(75) +
          occ_func_3_0(35) * occ_func_3_0(71) +
          occ_func_2_0(39) * occ_func_3_0(62) +
          occ_func_2_0(67) * occ_func_3_0(56) +
          occ_func_2_0(5) * occ_func_3_0(60) +
          occ_func_2_0(21) * occ_func_3_0(58) +
          occ_func_2_0(13) * occ_func_3_0(62) +
          occ_func_2_0(23) * occ_func_3_0(56) +
          occ_func_2_0(29) * occ_func_3_0(64) +
          occ_func_2_0(7) * occ_func_3_0(54) +
          occ_func_3_0(13) * occ_func_3_0(25) +
          occ_func_3_0(5) * occ_func_3_0(27) +
          occ_func_2_0(83) * occ_func_3_0(64) +
          occ_func_2_0(35) * occ_func_3_0(54) +
          occ_func_2_0(51) * occ_func_3_0(60) +
          occ_func_2_0(79) * occ_func_3_0(58) +
          occ_func_3_0(7) * occ_func_3_0(19)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_45_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(40) * occ_func_3_0(63) +
          occ_func_2_0(68) * occ_func_3_0(57) +
          occ_func_3_0(4) * occ_func_3_0(26) +
          occ_func_3_0(12) * occ_func_3_0(24) +
          occ_func_2_0(52) * occ_func_3_0(61) +
          occ_func_2_0(80) * occ_func_3_0(59) +
          occ_func_2_0(84) * occ_func_3_0(65) +
          occ_func_2_0(36) * occ_func_3_0(55) +
          occ_func_2_0(28) * occ_func_3_0(65) +
          occ_func_2_0(6) * occ_func_3_0(55) +
          occ_func_2_0(12) * occ_func_3_0(63) +
          occ_func_2_0(22) * occ_func_3_0(57) +
          occ_func_2_0(4) * occ_func_3_0(61) +
          occ_func_2_0(20) * occ_func_3_0(59) +
          occ_func_3_0(6) * occ_func_3_0(18) +
          occ_func_3_0(36) * occ_func_3_0(72) +
          occ_func_3_0(52) * occ_func_3_0(76) +
          occ_func_3_0(40) * occ_func_3_0(88) +
          occ_func_2_0(44) * occ_func_3_0(65) +
          occ_func_2_0(68) * occ_func_3_0(55) +
          occ_func_2_0(80) * occ_func_3_0(63) +
          occ_func_2_0(32) * occ_func_3_0(57) +
          occ_func_2_0(84) * occ_func_3_0(61) +
          occ_func_2_0(48) * occ_func_3_0(59) +
          occ_func_3_0(44) * occ_func_3_0(88) +
          occ_func_3_0(10) * occ_func_3_0(26) +
          occ_func_3_0(2) * occ_func_3_0(24) +
          occ_func_3_0(8) * occ_func_3_0(18) +
          occ_func_2_0(22) * occ_func_3_0(61) +
          occ_func_2_0(2) * occ_func_3_0(59) +
          occ_func_2_0(10) * occ_func_3_0(65) +
          occ_func_2_0(20) * occ_func_3_0(55) +
          occ_func_3_0(48) * occ_func_3_0(72) +
          occ_func_3_0(32) * occ_func_3_0(76) +
          occ_func_2_0(28) * occ_func_3_0(63) +
          occ_func_2_0(8) * occ_func_3_0(57)) /
         24.;
}

/**** Basis functions for orbit 46****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.7500000 Va  O

2.3333333 1.6666667 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_46_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(9) * occ_func_3_0(29) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_3_0(64) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_3_0(80) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_3_0(61) +
          occ_func_2_0(26) * occ_func_3_0(65) * occ_func_3_0(1) +
          occ_func_2_0(88) * occ_func_3_0(65) * occ_func_3_0(1) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_3_0(63) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_3_0(83) +
          occ_func_2_0(87) * occ_func_3_0(62) * occ_func_3_0(0) +
          occ_func_2_0(27) * occ_func_3_0(60) * occ_func_3_0(0) +
          occ_func_2_0(25) * occ_func_3_0(62) * occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_3_0(64) +
          occ_func_2_0(8) * occ_func_3_0(1) * occ_func_3_0(65) +
          occ_func_2_0(24) * occ_func_3_0(63) * occ_func_3_0(1) +
          occ_func_2_0(26) * occ_func_3_0(61) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_3_0(28) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_3_0(84) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_3_0(62) +
          occ_func_2_0(87) * occ_func_3_0(64) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_3_0(79) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_3_0(65) +
          occ_func_2_0(27) * occ_func_3_0(64) * occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_3_0(60) +
          occ_func_2_0(88) * occ_func_3_0(63) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_46_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(9) * occ_func_3_0(29) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_3_0(64) +
          occ_func_2_0(71) * occ_func_3_0(54) * occ_func_3_0(0) +
          occ_func_2_0(75) * occ_func_3_0(60) * occ_func_3_0(0) +
          occ_func_2_0(31) * occ_func_3_0(0) * occ_func_3_0(58) +
          occ_func_2_0(0) * occ_func_3_0(3) * occ_func_3_0(23) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_3_0(21) +
          occ_func_2_0(0) * occ_func_3_0(43) * occ_func_3_0(67) +
          occ_func_2_0(0) * occ_func_3_0(31) * occ_func_3_0(79) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_3_0(83) +
          occ_func_2_0(87) * occ_func_3_0(62) * occ_func_3_0(0) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_3_0(56) +
          occ_func_2_0(27) * occ_func_3_0(60) * occ_func_3_0(0) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_3_0(58) +
          occ_func_2_0(25) * occ_func_3_0(62) * occ_func_3_0(0) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_3_0(56) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_3_0(64) +
          occ_func_2_0(19) * occ_func_3_0(54) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(7) * occ_func_3_0(29) +
          occ_func_2_0(0) * occ_func_3_0(13) * occ_func_3_0(23) +
          occ_func_2_0(0) * occ_func_3_0(5) * occ_func_3_0(21) +
          occ_func_2_0(7) * occ_func_3_0(0) * occ_func_3_0(62) +
          occ_func_2_0(19) * occ_func_3_0(56) * occ_func_3_0(0) +
          occ_func_2_0(35) * occ_func_3_0(0) * occ_func_3_0(60) +
          occ_func_2_0(71) * occ_func_3_0(58) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_3_0(62) +
          occ_func_2_0(75) * occ_func_3_0(56) * occ_func_3_0(0) +
          occ_func_2_0(87) * occ_func_3_0(64) * occ_func_3_0(0) +
          occ_func_2_0(39) * occ_func_3_0(0) * occ_func_3_0(54) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_3_0(79) +
          occ_func_2_0(0) * occ_func_3_0(35) * occ_func_3_0(83) +
          occ_func_2_0(27) * occ_func_3_0(64) * occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_3_0(0) * occ_func_3_0(54) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_3_0(60) +
          occ_func_2_0(25) * occ_func_3_0(58) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(39) * occ_func_3_0(67)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_46_0_at_1() const {
  return (occ_func_2_0(6) * occ_func_3_0(1) * occ_func_3_0(63) +
          occ_func_2_0(18) * occ_func_3_0(57) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(36) * occ_func_3_0(84) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_3_0(80) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_3_0(61) +
          occ_func_2_0(24) * occ_func_3_0(59) * occ_func_3_0(1) +
          occ_func_2_0(26) * occ_func_3_0(65) * occ_func_3_0(1) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_3_0(55) +
          occ_func_2_0(88) * occ_func_3_0(65) * occ_func_3_0(1) +
          occ_func_2_0(40) * occ_func_3_0(1) * occ_func_3_0(55) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_3_0(63) +
          occ_func_2_0(76) * occ_func_3_0(57) * occ_func_3_0(1) +
          occ_func_2_0(36) * occ_func_3_0(1) * occ_func_3_0(61) +
          occ_func_2_0(72) * occ_func_3_0(59) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(40) * occ_func_3_0(68) +
          occ_func_2_0(1) * occ_func_3_0(4) * occ_func_3_0(20) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_3_0(22) +
          occ_func_2_0(1) * occ_func_3_0(6) * occ_func_3_0(28) +
          occ_func_2_0(8) * occ_func_3_0(1) * occ_func_3_0(65) +
          occ_func_2_0(18) * occ_func_3_0(55) * occ_func_3_0(1) +
          occ_func_2_0(24) * occ_func_3_0(63) * occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_3_0(1) * occ_func_3_0(57) +
          occ_func_2_0(26) * occ_func_3_0(61) * occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_3_0(59) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_3_0(28) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_3_0(84) +
          occ_func_2_0(1) * occ_func_3_0(32) * occ_func_3_0(80) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_3_0(68) +
          occ_func_2_0(76) * occ_func_3_0(61) * occ_func_3_0(1) +
          occ_func_2_0(32) * occ_func_3_0(1) * occ_func_3_0(59) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_3_0(65) +
          occ_func_2_0(72) * occ_func_3_0(55) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_3_0(20) +
          occ_func_2_0(1) * occ_func_3_0(2) * occ_func_3_0(22) +
          occ_func_2_0(88) * occ_func_3_0(63) * occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_3_0(57)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_46_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(9) * occ_func_3_0(29) +
          occ_func_2_0(47) * occ_func_3_0(64) +
          occ_func_2_0(71) * occ_func_3_0(54) +
          occ_func_2_0(75) * occ_func_3_0(60) +
          occ_func_2_0(31) * occ_func_3_0(58) +
          occ_func_3_0(3) * occ_func_3_0(23) +
          occ_func_3_0(11) * occ_func_3_0(21) +
          occ_func_3_0(43) * occ_func_3_0(67) +
          occ_func_3_0(31) * occ_func_3_0(79) +
          occ_func_3_0(47) * occ_func_3_0(83) +
          occ_func_2_0(87) * occ_func_3_0(62) +
          occ_func_2_0(43) * occ_func_3_0(56) +
          occ_func_2_0(27) * occ_func_3_0(60) +
          occ_func_2_0(11) * occ_func_3_0(58) +
          occ_func_2_0(25) * occ_func_3_0(62) +
          occ_func_2_0(3) * occ_func_3_0(56) +
          occ_func_2_0(9) * occ_func_3_0(64) +
          occ_func_2_0(19) * occ_func_3_0(54) +
          occ_func_3_0(7) * occ_func_3_0(29) +
          occ_func_3_0(13) * occ_func_3_0(23) +
          occ_func_3_0(5) * occ_func_3_0(21) +
          occ_func_2_0(7) * occ_func_3_0(62) +
          occ_func_2_0(19) * occ_func_3_0(56) +
          occ_func_2_0(35) * occ_func_3_0(60) +
          occ_func_2_0(71) * occ_func_3_0(58) +
          occ_func_2_0(51) * occ_func_3_0(62) +
          occ_func_2_0(75) * occ_func_3_0(56) +
          occ_func_2_0(87) * occ_func_3_0(64) +
          occ_func_2_0(39) * occ_func_3_0(54) +
          occ_func_3_0(51) * occ_func_3_0(79) +
          occ_func_3_0(35) * occ_func_3_0(83) +
          occ_func_2_0(27) * occ_func_3_0(64) +
          occ_func_2_0(5) * occ_func_3_0(54) +
          occ_func_2_0(13) * occ_func_3_0(60) +
          occ_func_2_0(25) * occ_func_3_0(58) +
          occ_func_3_0(39) * occ_func_3_0(67)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_46_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(6) * occ_func_3_0(63) +
          occ_func_2_0(18) * occ_func_3_0(57) +
          occ_func_3_0(36) * occ_func_3_0(84) +
          occ_func_3_0(52) * occ_func_3_0(80) +
          occ_func_2_0(12) * occ_func_3_0(61) +
          occ_func_2_0(24) * occ_func_3_0(59) +
          occ_func_2_0(26) * occ_func_3_0(65) +
          occ_func_2_0(4) * occ_func_3_0(55) +
          occ_func_2_0(88) * occ_func_3_0(65) +
          occ_func_2_0(40) * occ_func_3_0(55) +
          occ_func_2_0(52) * occ_func_3_0(63) +
          occ_func_2_0(76) * occ_func_3_0(57) +
          occ_func_2_0(36) * occ_func_3_0(61) +
          occ_func_2_0(72) * occ_func_3_0(59) +
          occ_func_3_0(40) * occ_func_3_0(68) +
          occ_func_3_0(4) * occ_func_3_0(20) +
          occ_func_3_0(12) * occ_func_3_0(22) +
          occ_func_3_0(6) * occ_func_3_0(28) +
          occ_func_2_0(8) * occ_func_3_0(65) +
          occ_func_2_0(18) * occ_func_3_0(55) +
          occ_func_2_0(24) * occ_func_3_0(63) +
          occ_func_2_0(2) * occ_func_3_0(57) +
          occ_func_2_0(26) * occ_func_3_0(61) +
          occ_func_2_0(10) * occ_func_3_0(59) +
          occ_func_3_0(8) * occ_func_3_0(28) +
          occ_func_3_0(48) * occ_func_3_0(84) +
          occ_func_3_0(32) * occ_func_3_0(80) +
          occ_func_3_0(44) * occ_func_3_0(68) +
          occ_func_2_0(76) * occ_func_3_0(61) +
          occ_func_2_0(32) * occ_func_3_0(59) +
          occ_func_2_0(48) * occ_func_3_0(65) +
          occ_func_2_0(72) * occ_func_3_0(55) +
          occ_func_3_0(10) * occ_func_3_0(20) +
          occ_func_3_0(2) * occ_func_3_0(22) +
          occ_func_2_0(88) * occ_func_3_0(63) +
          occ_func_2_0(44) * occ_func_3_0(57)) /
         24.;
}

/**** Basis functions for orbit 47****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 2.6666666 0.2500000 Va  O

2.3333333 2.6666666 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_47_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(60) * occ_func_2_0(64) +
          occ_func_2_0(63) * occ_func_2_0(1) * occ_func_2_0(65) +
          occ_func_2_0(62) * occ_func_2_0(64) * occ_func_2_0(0) +
          occ_func_2_0(65) * occ_func_2_0(61) * occ_func_2_0(1)) /
         4.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_47_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(60) * occ_func_2_0(64) +
          occ_func_2_0(58) * occ_func_2_0(0) * occ_func_2_0(62) +
          occ_func_2_0(54) * occ_func_2_0(56) * occ_func_2_0(0) +
          occ_func_2_0(62) * occ_func_2_0(64) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(60) * occ_func_2_0(56) +
          occ_func_2_0(58) * occ_func_2_0(0) * occ_func_2_0(54)) /
         4.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_47_0_at_1() const {
  return (occ_func_2_0(63) * occ_func_2_0(1) * occ_func_2_0(65) +
          occ_func_2_0(1) * occ_func_2_0(57) * occ_func_2_0(61) +
          occ_func_2_0(59) * occ_func_2_0(55) * occ_func_2_0(1) +
          occ_func_2_0(65) * occ_func_2_0(61) * occ_func_2_0(1) +
          occ_func_2_0(63) * occ_func_2_0(1) * occ_func_2_0(59) +
          occ_func_2_0(1) * occ_func_2_0(57) * occ_func_2_0(55)) /
         4.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_47_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(60) * occ_func_2_0(64) +
          occ_func_2_0(58) * occ_func_2_0(62) +
          occ_func_2_0(54) * occ_func_2_0(56) +
          occ_func_2_0(62) * occ_func_2_0(64) +
          occ_func_2_0(60) * occ_func_2_0(56) +
          occ_func_2_0(58) * occ_func_2_0(54)) /
         4.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_47_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(63) * occ_func_2_0(65) +
          occ_func_2_0(57) * occ_func_2_0(61) +
          occ_func_2_0(59) * occ_func_2_0(55) +
          occ_func_2_0(65) * occ_func_2_0(61) +
          occ_func_2_0(63) * occ_func_2_0(59) +
          occ_func_2_0(57) * occ_func_2_0(55)) /
         4.;
}

/**** Basis functions for orbit 48****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.2500000 Va  O

0.3333333 2.6666666 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_48_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_3_0(103) +
          occ_func_2_0(63) * occ_func_3_0(10) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_3_0(64) +
          occ_func_2_0(112) * occ_func_3_0(13) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_3_0(107) +
          occ_func_2_0(0) * occ_func_3_0(9) * occ_func_3_0(61) +
          occ_func_2_0(64) * occ_func_3_0(13) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_3_0(63) +
          occ_func_2_0(61) * occ_func_3_0(8) * occ_func_3_0(0) +
          occ_func_2_0(107) * occ_func_3_0(10) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_3_0(112) +
          occ_func_2_0(103) * occ_func_3_0(8) * occ_func_3_0(0) +
          occ_func_2_0(104) * occ_func_3_0(9) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_3_0(111) +
          occ_func_2_0(108) * occ_func_3_0(11) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_3_0(104) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_3_0(62) +
          occ_func_2_0(65) * occ_func_3_0(12) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_3_0(60) +
          occ_func_2_0(0) * occ_func_3_0(13) * occ_func_3_0(65) +
          occ_func_2_0(62) * occ_func_3_0(11) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_3_0(108) +
          occ_func_2_0(111) * occ_func_3_0(12) * occ_func_3_0(0) +
          occ_func_2_0(60) * occ_func_3_0(9) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_48_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_3_0(103) +
          occ_func_2_0(63) * occ_func_3_0(10) * occ_func_3_0(0) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_3_0(4) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_3_0(12) +
          occ_func_2_0(55) * occ_func_3_0(2) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(31) * occ_func_3_0(91) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_3_0(107) +
          occ_func_2_0(0) * occ_func_3_0(9) * occ_func_3_0(61) +
          occ_func_2_0(0) * occ_func_3_0(3) * occ_func_3_0(55) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_3_0(63) +
          occ_func_2_0(61) * occ_func_3_0(8) * occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_3_0(6) +
          occ_func_2_0(107) * occ_func_3_0(10) * occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_3_0(4) +
          occ_func_2_0(31) * occ_func_3_0(0) * occ_func_3_0(12) +
          occ_func_2_0(91) * occ_func_3_0(2) * occ_func_3_0(0) +
          occ_func_2_0(103) * occ_func_3_0(8) * occ_func_3_0(0) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_3_0(6) +
          occ_func_2_0(0) * occ_func_3_0(39) * occ_func_3_0(99) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_3_0(111) +
          occ_func_2_0(0) * occ_func_3_0(35) * occ_func_3_0(95) +
          occ_func_2_0(39) * occ_func_3_0(0) * occ_func_3_0(8) +
          occ_func_2_0(99) * occ_func_3_0(6) * occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_3_0(0) * occ_func_3_0(10) +
          occ_func_2_0(57) * occ_func_3_0(4) * occ_func_3_0(0) +
          occ_func_2_0(65) * occ_func_3_0(12) * occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_3_0(2) +
          occ_func_2_0(7) * occ_func_3_0(0) * occ_func_3_0(8) +
          occ_func_2_0(59) * occ_func_3_0(6) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(13) * occ_func_3_0(65) +
          occ_func_2_0(0) * occ_func_3_0(5) * occ_func_3_0(57) +
          occ_func_2_0(35) * occ_func_3_0(0) * occ_func_3_0(10) +
          occ_func_2_0(95) * occ_func_3_0(4) * occ_func_3_0(0) +
          occ_func_2_0(111) * occ_func_3_0(12) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_3_0(2) +
          occ_func_2_0(0) * occ_func_3_0(7) * occ_func_3_0(59)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_48_0_at_1() const {
  return (occ_func_2_0(40) * occ_func_3_0(1) * occ_func_3_0(9) +
          occ_func_2_0(100) * occ_func_3_0(7) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(4) * occ_func_3_0(56) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_3_0(64) +
          occ_func_2_0(112) * occ_func_3_0(13) * occ_func_3_0(1) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_3_0(3) +
          occ_func_2_0(36) * occ_func_3_0(1) * occ_func_3_0(11) +
          occ_func_2_0(96) * occ_func_3_0(5) * occ_func_3_0(1) +
          occ_func_2_0(6) * occ_func_3_0(1) * occ_func_3_0(9) +
          occ_func_2_0(58) * occ_func_3_0(7) * occ_func_3_0(1) +
          occ_func_2_0(64) * occ_func_3_0(13) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_3_0(3) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_3_0(11) +
          occ_func_2_0(56) * occ_func_3_0(5) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(6) * occ_func_3_0(58) +
          occ_func_2_0(1) * occ_func_3_0(36) * occ_func_3_0(96) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_3_0(112) +
          occ_func_2_0(1) * occ_func_3_0(40) * occ_func_3_0(100) +
          occ_func_2_0(104) * occ_func_3_0(9) * occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_3_0(7) +
          occ_func_2_0(32) * occ_func_3_0(1) * occ_func_3_0(13) +
          occ_func_2_0(92) * occ_func_3_0(3) * occ_func_3_0(1) +
          occ_func_2_0(108) * occ_func_3_0(11) * occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_3_0(5) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_3_0(104) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_3_0(62) +
          occ_func_2_0(1) * occ_func_3_0(2) * occ_func_3_0(54) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_3_0(60) +
          occ_func_2_0(2) * occ_func_3_0(1) * occ_func_3_0(13) +
          occ_func_2_0(54) * occ_func_3_0(3) * occ_func_3_0(1) +
          occ_func_2_0(62) * occ_func_3_0(11) * occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_3_0(5) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_3_0(108) +
          occ_func_2_0(1) * occ_func_3_0(32) * occ_func_3_0(92) +
          occ_func_2_0(60) * occ_func_3_0(9) * occ_func_3_0(1) +
          occ_func_2_0(8) * occ_func_3_0(1) * occ_func_3_0(7)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_48_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(43) * occ_func_3_0(103) +
          occ_func_2_0(63) * occ_func_3_0(10) +
          occ_func_2_0(11) * occ_func_3_0(4) +
          occ_func_2_0(3) * occ_func_3_0(12) +
          occ_func_2_0(55) * occ_func_3_0(2) +
          occ_func_3_0(31) * occ_func_3_0(91) +
          occ_func_3_0(47) * occ_func_3_0(107) +
          occ_func_3_0(9) * occ_func_3_0(61) +
          occ_func_3_0(3) * occ_func_3_0(55) +
          occ_func_3_0(11) * occ_func_3_0(63) +
          occ_func_2_0(61) * occ_func_3_0(8) +
          occ_func_2_0(9) * occ_func_3_0(6) +
          occ_func_2_0(107) * occ_func_3_0(10) +
          occ_func_2_0(47) * occ_func_3_0(4) +
          occ_func_2_0(31) * occ_func_3_0(12) +
          occ_func_2_0(91) * occ_func_3_0(2) +
          occ_func_2_0(103) * occ_func_3_0(8) +
          occ_func_2_0(43) * occ_func_3_0(6) +
          occ_func_3_0(39) * occ_func_3_0(99) +
          occ_func_3_0(51) * occ_func_3_0(111) +
          occ_func_3_0(35) * occ_func_3_0(95) +
          occ_func_2_0(39) * occ_func_3_0(8) +
          occ_func_2_0(99) * occ_func_3_0(6) +
          occ_func_2_0(5) * occ_func_3_0(10) +
          occ_func_2_0(57) * occ_func_3_0(4) +
          occ_func_2_0(65) * occ_func_3_0(12) +
          occ_func_2_0(13) * occ_func_3_0(2) +
          occ_func_2_0(7) * occ_func_3_0(8) +
          occ_func_2_0(59) * occ_func_3_0(6) +
          occ_func_3_0(13) * occ_func_3_0(65) +
          occ_func_3_0(5) * occ_func_3_0(57) +
          occ_func_2_0(35) * occ_func_3_0(10) +
          occ_func_2_0(95) * occ_func_3_0(4) +
          occ_func_2_0(111) * occ_func_3_0(12) +
          occ_func_2_0(51) * occ_func_3_0(2) +
          occ_func_3_0(7) * occ_func_3_0(59)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_48_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(40) * occ_func_3_0(9) +
          occ_func_2_0(100) * occ_func_3_0(7) +
          occ_func_3_0(4) * occ_func_3_0(56) +
          occ_func_3_0(12) * occ_func_3_0(64) +
          occ_func_2_0(112) * occ_func_3_0(13) +
          occ_func_2_0(52) * occ_func_3_0(3) +
          occ_func_2_0(36) * occ_func_3_0(11) +
          occ_func_2_0(96) * occ_func_3_0(5) +
          occ_func_2_0(6) * occ_func_3_0(9) +
          occ_func_2_0(58) * occ_func_3_0(7) +
          occ_func_2_0(64) * occ_func_3_0(13) +
          occ_func_2_0(12) * occ_func_3_0(3) +
          occ_func_2_0(4) * occ_func_3_0(11) +
          occ_func_2_0(56) * occ_func_3_0(5) +
          occ_func_3_0(6) * occ_func_3_0(58) +
          occ_func_3_0(36) * occ_func_3_0(96) +
          occ_func_3_0(52) * occ_func_3_0(112) +
          occ_func_3_0(40) * occ_func_3_0(100) +
          occ_func_2_0(104) * occ_func_3_0(9) +
          occ_func_2_0(44) * occ_func_3_0(7) +
          occ_func_2_0(32) * occ_func_3_0(13) +
          occ_func_2_0(92) * occ_func_3_0(3) +
          occ_func_2_0(108) * occ_func_3_0(11) +
          occ_func_2_0(48) * occ_func_3_0(5) +
          occ_func_3_0(44) * occ_func_3_0(104) +
          occ_func_3_0(10) * occ_func_3_0(62) +
          occ_func_3_0(2) * occ_func_3_0(54) +
          occ_func_3_0(8) * occ_func_3_0(60) +
          occ_func_2_0(2) * occ_func_3_0(13) +
          occ_func_2_0(54) * occ_func_3_0(3) +
          occ_func_2_0(62) * occ_func_3_0(11) +
          occ_func_2_0(10) * occ_func_3_0(5) +
          occ_func_3_0(48) * occ_func_3_0(108) +
          occ_func_3_0(32) * occ_func_3_0(92) +
          occ_func_2_0(60) * occ_func_3_0(9) +
          occ_func_2_0(8) * occ_func_3_0(7)) /
         24.;
}

/**** Basis functions for orbit 49****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.2500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_49_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_2_0(28) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_2_0(65) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_2_0(25) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_2_0(104) +
          occ_func_2_0(26) * occ_func_3_0(111) * occ_func_2_0(0) +
          occ_func_2_0(28) * occ_func_3_0(65) * occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_2_0(62) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_2_0(26) +
          occ_func_2_0(29) * occ_func_3_0(62) * occ_func_2_0(1) +
          occ_func_2_0(27) * occ_func_3_0(104) * occ_func_2_0(1) +
          occ_func_2_0(25) * occ_func_3_0(108) * occ_func_2_0(1) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_2_0(111) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_2_0(112) +
          occ_func_2_0(24) * occ_func_3_0(107) * occ_func_2_0(0) +
          occ_func_2_0(26) * occ_func_3_0(103) * occ_func_2_0(0) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_2_0(29) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_2_0(27) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_2_0(63) +
          occ_func_2_0(29) * occ_func_3_0(64) * occ_func_2_0(1) +
          occ_func_2_0(0) * occ_func_3_0(13) * occ_func_2_0(24) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_2_0(64) +
          occ_func_2_0(27) * occ_func_3_0(112) * occ_func_2_0(1) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_2_0(103) +
          occ_func_2_0(28) * occ_func_3_0(63) * occ_func_2_0(0)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_49_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(95) * occ_func_2_0(0) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_2_0(65) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_2_0(59) +
          occ_func_2_0(24) * occ_func_3_0(99) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(31) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(111) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_2_0(20) +
          occ_func_2_0(28) * occ_func_3_0(65) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(9) * occ_func_2_0(18) +
          occ_func_2_0(0) * occ_func_3_0(3) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_3_0(57) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_3_0(59) * occ_func_2_0(0) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_2_0(57) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_2_0(99) +
          occ_func_2_0(31) * occ_func_3_0(0) * occ_func_2_0(95) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_2_0(111) +
          occ_func_2_0(0) * occ_func_3_0(39) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(91) * occ_func_2_0(0) +
          occ_func_2_0(24) * occ_func_3_0(107) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(103) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(35) * occ_func_2_0(20) +
          occ_func_2_0(39) * occ_func_3_0(0) * occ_func_2_0(107) +
          occ_func_2_0(5) * occ_func_3_0(0) * occ_func_2_0(61) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_2_0(63) +
          occ_func_2_0(7) * occ_func_3_0(0) * occ_func_2_0(55) +
          occ_func_2_0(0) * occ_func_3_0(13) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_3_0(61) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(5) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_3_0(55) * occ_func_2_0(0) +
          occ_func_2_0(35) * occ_func_3_0(0) * occ_func_2_0(91) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_2_0(103) +
          occ_func_2_0(28) * occ_func_3_0(63) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(7) * occ_func_2_0(18)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_49_0_at_1() const {
  return (occ_func_2_0(40) * occ_func_3_0(1) * occ_func_2_0(108) +
          occ_func_2_0(1) * occ_func_3_0(4) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(54) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(60) * occ_func_2_0(1) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_2_0(104) +
          occ_func_2_0(36) * occ_func_3_0(1) * occ_func_2_0(92) +
          occ_func_2_0(6) * occ_func_3_0(1) * occ_func_2_0(54) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_2_0(62) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_2_0(60) +
          occ_func_2_0(29) * occ_func_3_0(62) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(6) * occ_func_2_0(19) +
          occ_func_2_0(27) * occ_func_3_0(104) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(36) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_3_0(108) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_2_0(23) +
          occ_func_2_0(1) * occ_func_3_0(40) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(92) * occ_func_2_0(1) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_2_0(112) +
          occ_func_2_0(32) * occ_func_3_0(1) * occ_func_2_0(96) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_2_0(100) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(96) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(58) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(2) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(56) * occ_func_2_0(1) +
          occ_func_2_0(29) * occ_func_3_0(64) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_2_0(19) +
          occ_func_2_0(2) * occ_func_3_0(1) * occ_func_2_0(58) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_2_0(64) +
          occ_func_2_0(27) * occ_func_3_0(112) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_3_0(100) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(32) * occ_func_2_0(23) +
          occ_func_2_0(8) * occ_func_3_0(1) * occ_func_2_0(56)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_49_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(43) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(95) +
          occ_func_2_0(11) * occ_func_2_0(65) +
          occ_func_2_0(3) * occ_func_2_0(59) +
          occ_func_2_0(24) * occ_func_3_0(99) +
          occ_func_3_0(31) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(111) +
          occ_func_3_0(47) * occ_func_2_0(20) +
          occ_func_2_0(28) * occ_func_3_0(65) +
          occ_func_3_0(9) * occ_func_2_0(18) +
          occ_func_3_0(3) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_3_0(57) +
          occ_func_3_0(11) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_3_0(59) +
          occ_func_2_0(9) * occ_func_2_0(57) +
          occ_func_2_0(47) * occ_func_2_0(99) +
          occ_func_2_0(31) * occ_func_2_0(95) +
          occ_func_2_0(43) * occ_func_2_0(111) +
          occ_func_3_0(39) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(91) +
          occ_func_2_0(24) * occ_func_3_0(107) +
          occ_func_3_0(51) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(103) +
          occ_func_3_0(35) * occ_func_2_0(20) +
          occ_func_2_0(39) * occ_func_2_0(107) +
          occ_func_2_0(5) * occ_func_2_0(61) +
          occ_func_2_0(13) * occ_func_2_0(63) +
          occ_func_2_0(7) * occ_func_2_0(55) +
          occ_func_3_0(13) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_3_0(61) +
          occ_func_3_0(5) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_3_0(55) +
          occ_func_2_0(35) * occ_func_2_0(91) +
          occ_func_2_0(51) * occ_func_2_0(103) +
          occ_func_2_0(28) * occ_func_3_0(63) +
          occ_func_3_0(7) * occ_func_2_0(18)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_49_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(40) * occ_func_2_0(108) +
          occ_func_3_0(4) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(54) +
          occ_func_3_0(12) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(60) +
          occ_func_2_0(52) * occ_func_2_0(104) +
          occ_func_2_0(36) * occ_func_2_0(92) +
          occ_func_2_0(6) * occ_func_2_0(54) +
          occ_func_2_0(12) * occ_func_2_0(62) +
          occ_func_2_0(4) * occ_func_2_0(60) +
          occ_func_2_0(29) * occ_func_3_0(62) +
          occ_func_3_0(6) * occ_func_2_0(19) +
          occ_func_2_0(27) * occ_func_3_0(104) +
          occ_func_3_0(36) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_3_0(108) +
          occ_func_3_0(52) * occ_func_2_0(23) +
          occ_func_3_0(40) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(92) +
          occ_func_2_0(44) * occ_func_2_0(112) +
          occ_func_2_0(32) * occ_func_2_0(96) +
          occ_func_2_0(48) * occ_func_2_0(100) +
          occ_func_3_0(44) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(96) +
          occ_func_3_0(10) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(58) +
          occ_func_3_0(2) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(56) +
          occ_func_2_0(29) * occ_func_3_0(64) +
          occ_func_3_0(8) * occ_func_2_0(19) +
          occ_func_2_0(2) * occ_func_2_0(58) +
          occ_func_2_0(10) * occ_func_2_0(64) +
          occ_func_2_0(27) * occ_func_3_0(112) +
          occ_func_3_0(48) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_3_0(100) +
          occ_func_3_0(32) * occ_func_2_0(23) +
          occ_func_2_0(8) * occ_func_2_0(56)) /
         24.;
}

/**** Basis functions for orbit 50****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.7500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_50_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(9) * occ_func_2_0(28) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_2_0(111) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_2_0(25) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_2_0(60) +
          occ_func_2_0(26) * occ_func_3_0(65) * occ_func_2_0(0) +
          occ_func_2_0(28) * occ_func_3_0(111) * occ_func_2_0(0) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_2_0(108) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_2_0(26) +
          occ_func_2_0(29) * occ_func_3_0(108) * occ_func_2_0(1) +
          occ_func_2_0(27) * occ_func_3_0(60) * occ_func_2_0(1) +
          occ_func_2_0(25) * occ_func_3_0(62) * occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_2_0(65) +
          occ_func_2_0(8) * occ_func_3_0(1) * occ_func_2_0(64) +
          occ_func_2_0(24) * occ_func_3_0(63) * occ_func_2_0(0) +
          occ_func_2_0(26) * occ_func_3_0(61) * occ_func_2_0(0) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_2_0(29) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_2_0(27) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_2_0(107) +
          occ_func_2_0(29) * occ_func_3_0(112) * occ_func_2_0(1) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_2_0(24) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_2_0(112) +
          occ_func_2_0(27) * occ_func_3_0(64) * occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_2_0(61) +
          occ_func_2_0(28) * occ_func_3_0(107) * occ_func_2_0(0)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_50_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(9) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(57) * occ_func_2_0(0) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_2_0(111) +
          occ_func_2_0(31) * occ_func_3_0(0) * occ_func_2_0(99) +
          occ_func_2_0(24) * occ_func_3_0(59) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(3) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(65) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_2_0(20) +
          occ_func_2_0(28) * occ_func_3_0(111) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(43) * occ_func_2_0(18) +
          occ_func_2_0(0) * occ_func_3_0(31) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_3_0(95) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_3_0(99) * occ_func_2_0(0) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_2_0(95) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_2_0(59) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_2_0(57) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_2_0(65) +
          occ_func_2_0(0) * occ_func_3_0(7) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(55) * occ_func_2_0(0) +
          occ_func_2_0(24) * occ_func_3_0(63) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(13) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(61) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(5) * occ_func_2_0(20) +
          occ_func_2_0(7) * occ_func_3_0(0) * occ_func_2_0(63) +
          occ_func_2_0(35) * occ_func_3_0(0) * occ_func_2_0(103) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_2_0(107) +
          occ_func_2_0(39) * occ_func_3_0(0) * occ_func_2_0(91) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_3_0(103) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(35) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_3_0(91) * occ_func_2_0(0) +
          occ_func_2_0(5) * occ_func_3_0(0) * occ_func_2_0(55) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_2_0(61) +
          occ_func_2_0(28) * occ_func_3_0(107) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(39) * occ_func_2_0(18)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_50_0_at_1() const {
  return (occ_func_2_0(6) * occ_func_3_0(1) * occ_func_2_0(62) +
          occ_func_2_0(1) * occ_func_3_0(36) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(92) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(104) * occ_func_2_0(1) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_2_0(60) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_2_0(54) +
          occ_func_2_0(40) * occ_func_3_0(1) * occ_func_2_0(92) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_2_0(108) +
          occ_func_2_0(36) * occ_func_3_0(1) * occ_func_2_0(104) +
          occ_func_2_0(29) * occ_func_3_0(108) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(40) * occ_func_2_0(19) +
          occ_func_2_0(27) * occ_func_3_0(60) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(4) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_3_0(62) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_2_0(23) +
          occ_func_2_0(1) * occ_func_3_0(6) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(54) * occ_func_2_0(1) +
          occ_func_2_0(8) * occ_func_3_0(1) * occ_func_2_0(64) +
          occ_func_2_0(2) * occ_func_3_0(1) * occ_func_2_0(56) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_2_0(58) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(56) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(100) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(32) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(96) * occ_func_2_0(1) +
          occ_func_2_0(29) * occ_func_3_0(112) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_2_0(19) +
          occ_func_2_0(32) * occ_func_3_0(1) * occ_func_2_0(100) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_2_0(112) +
          occ_func_2_0(27) * occ_func_3_0(64) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_3_0(58) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(2) * occ_func_2_0(23) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_2_0(96)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_50_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(9) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(57) +
          occ_func_2_0(47) * occ_func_2_0(111) +
          occ_func_2_0(31) * occ_func_2_0(99) +
          occ_func_2_0(24) * occ_func_3_0(59) +
          occ_func_3_0(3) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(65) +
          occ_func_3_0(11) * occ_func_2_0(20) +
          occ_func_2_0(28) * occ_func_3_0(111) +
          occ_func_3_0(43) * occ_func_2_0(18) +
          occ_func_3_0(31) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_3_0(95) +
          occ_func_3_0(47) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_3_0(99) +
          occ_func_2_0(43) * occ_func_2_0(95) +
          occ_func_2_0(11) * occ_func_2_0(59) +
          occ_func_2_0(3) * occ_func_2_0(57) +
          occ_func_2_0(9) * occ_func_2_0(65) +
          occ_func_3_0(7) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(55) +
          occ_func_2_0(24) * occ_func_3_0(63) +
          occ_func_3_0(13) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(61) +
          occ_func_3_0(5) * occ_func_2_0(20) +
          occ_func_2_0(7) * occ_func_2_0(63) +
          occ_func_2_0(35) * occ_func_2_0(103) +
          occ_func_2_0(51) * occ_func_2_0(107) +
          occ_func_2_0(39) * occ_func_2_0(91) +
          occ_func_3_0(51) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_3_0(103) +
          occ_func_3_0(35) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_3_0(91) +
          occ_func_2_0(5) * occ_func_2_0(55) +
          occ_func_2_0(13) * occ_func_2_0(61) +
          occ_func_2_0(28) * occ_func_3_0(107) +
          occ_func_3_0(39) * occ_func_2_0(18)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_50_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(6) * occ_func_2_0(62) +
          occ_func_3_0(36) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(92) +
          occ_func_3_0(52) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(104) +
          occ_func_2_0(12) * occ_func_2_0(60) +
          occ_func_2_0(4) * occ_func_2_0(54) +
          occ_func_2_0(40) * occ_func_2_0(92) +
          occ_func_2_0(52) * occ_func_2_0(108) +
          occ_func_2_0(36) * occ_func_2_0(104) +
          occ_func_2_0(29) * occ_func_3_0(108) +
          occ_func_3_0(40) * occ_func_2_0(19) +
          occ_func_2_0(27) * occ_func_3_0(60) +
          occ_func_3_0(4) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_3_0(62) +
          occ_func_3_0(12) * occ_func_2_0(23) +
          occ_func_3_0(6) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(54) +
          occ_func_2_0(8) * occ_func_2_0(64) +
          occ_func_2_0(2) * occ_func_2_0(56) +
          occ_func_2_0(10) * occ_func_2_0(58) +
          occ_func_3_0(8) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(56) +
          occ_func_3_0(48) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(100) +
          occ_func_3_0(32) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(96) +
          occ_func_2_0(29) * occ_func_3_0(112) +
          occ_func_3_0(44) * occ_func_2_0(19) +
          occ_func_2_0(32) * occ_func_2_0(100) +
          occ_func_2_0(48) * occ_func_2_0(112) +
          occ_func_2_0(27) * occ_func_3_0(64) +
          occ_func_3_0(10) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_3_0(58) +
          occ_func_3_0(2) * occ_func_2_0(23) +
          occ_func_2_0(44) * occ_func_2_0(96)) /
         24.;
}

/**** Basis functions for orbit 51****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.7500000 Va  O

0.3333333 2.6666666 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_51_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_3_0(103) +
          occ_func_2_0(63) * occ_func_2_0(47) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_3_0(64) +
          occ_func_2_0(112) * occ_func_2_0(12) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(46) * occ_func_3_0(107) +
          occ_func_2_0(0) * occ_func_2_0(44) * occ_func_3_0(61) +
          occ_func_2_0(64) * occ_func_2_0(52) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_3_0(63) +
          occ_func_2_0(61) * occ_func_2_0(43) * occ_func_3_0(0) +
          occ_func_2_0(107) * occ_func_2_0(11) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_3_0(112) +
          occ_func_2_0(103) * occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(104) * occ_func_2_0(8) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(50) * occ_func_3_0(111) +
          occ_func_2_0(108) * occ_func_2_0(10) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_3_0(104) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_3_0(62) +
          occ_func_2_0(65) * occ_func_2_0(51) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(43) * occ_func_3_0(60) +
          occ_func_2_0(0) * occ_func_2_0(52) * occ_func_3_0(65) +
          occ_func_2_0(62) * occ_func_2_0(48) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(49) * occ_func_3_0(108) +
          occ_func_2_0(111) * occ_func_2_0(13) * occ_func_3_0(0) +
          occ_func_2_0(60) * occ_func_2_0(44) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_51_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_3_0(103) +
          occ_func_2_0(40) * occ_func_2_0(0) * occ_func_3_0(9) +
          occ_func_2_0(63) * occ_func_2_0(47) * occ_func_3_0(0) +
          occ_func_2_0(55) * occ_func_2_0(31) * occ_func_3_0(0) +
          occ_func_2_0(52) * occ_func_2_0(0) * occ_func_3_0(3) +
          occ_func_2_0(0) * occ_func_2_0(30) * occ_func_3_0(91) +
          occ_func_2_0(0) * occ_func_2_0(46) * occ_func_3_0(107) +
          occ_func_2_0(36) * occ_func_2_0(0) * occ_func_3_0(11) +
          occ_func_2_0(0) * occ_func_2_0(44) * occ_func_3_0(61) +
          occ_func_2_0(38) * occ_func_2_0(0) * occ_func_3_0(43) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_3_0(31) +
          occ_func_2_0(0) * occ_func_2_0(32) * occ_func_3_0(55) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_3_0(63) +
          occ_func_2_0(34) * occ_func_2_0(0) * occ_func_3_0(47) +
          occ_func_2_0(61) * occ_func_2_0(43) * occ_func_3_0(0) +
          occ_func_2_0(107) * occ_func_2_0(11) * occ_func_3_0(0) +
          occ_func_2_0(91) * occ_func_2_0(3) * occ_func_3_0(0) +
          occ_func_2_0(103) * occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_3_0(7) +
          occ_func_2_0(0) * occ_func_2_0(38) * occ_func_3_0(99) +
          occ_func_2_0(0) * occ_func_2_0(50) * occ_func_3_0(111) +
          occ_func_2_0(32) * occ_func_2_0(0) * occ_func_3_0(13) +
          occ_func_2_0(48) * occ_func_2_0(0) * occ_func_3_0(5) +
          occ_func_2_0(0) * occ_func_2_0(34) * occ_func_3_0(95) +
          occ_func_2_0(99) * occ_func_2_0(7) * occ_func_3_0(0) +
          occ_func_2_0(57) * occ_func_2_0(35) * occ_func_3_0(0) +
          occ_func_2_0(65) * occ_func_2_0(51) * occ_func_3_0(0) +
          occ_func_2_0(59) * occ_func_2_0(39) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(52) * occ_func_3_0(65) +
          occ_func_2_0(30) * occ_func_2_0(0) * occ_func_3_0(51) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_3_0(35) +
          occ_func_2_0(0) * occ_func_2_0(36) * occ_func_3_0(57) +
          occ_func_2_0(95) * occ_func_2_0(5) * occ_func_3_0(0) +
          occ_func_2_0(111) * occ_func_2_0(13) * occ_func_3_0(0) +
          occ_func_2_0(42) * occ_func_2_0(0) * occ_func_3_0(39) +
          occ_func_2_0(0) * occ_func_2_0(40) * occ_func_3_0(59)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_51_0_at_1() const {
  return (occ_func_2_0(100) * occ_func_2_0(6) * occ_func_3_0(1) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_3_0(36) +
          occ_func_2_0(1) * occ_func_2_0(35) * occ_func_3_0(56) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_3_0(64) +
          occ_func_2_0(33) * occ_func_2_0(1) * occ_func_3_0(52) +
          occ_func_2_0(112) * occ_func_2_0(12) * occ_func_3_0(1) +
          occ_func_2_0(96) * occ_func_2_0(4) * occ_func_3_0(1) +
          occ_func_2_0(58) * occ_func_2_0(40) * occ_func_3_0(1) +
          occ_func_2_0(64) * occ_func_2_0(52) * occ_func_3_0(1) +
          occ_func_2_0(56) * occ_func_2_0(36) * occ_func_3_0(1) +
          occ_func_2_0(45) * occ_func_2_0(1) * occ_func_3_0(40) +
          occ_func_2_0(1) * occ_func_2_0(39) * occ_func_3_0(58) +
          occ_func_2_0(47) * occ_func_2_0(1) * occ_func_3_0(4) +
          occ_func_2_0(1) * occ_func_2_0(37) * occ_func_3_0(96) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_3_0(112) +
          occ_func_2_0(31) * occ_func_2_0(1) * occ_func_3_0(12) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_3_0(6) +
          occ_func_2_0(1) * occ_func_2_0(41) * occ_func_3_0(100) +
          occ_func_2_0(104) * occ_func_2_0(8) * occ_func_3_0(1) +
          occ_func_2_0(92) * occ_func_2_0(2) * occ_func_3_0(1) +
          occ_func_2_0(108) * occ_func_2_0(10) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_3_0(104) +
          occ_func_2_0(39) * occ_func_2_0(1) * occ_func_3_0(8) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_3_0(62) +
          occ_func_2_0(37) * occ_func_2_0(1) * occ_func_3_0(48) +
          occ_func_2_0(53) * occ_func_2_0(1) * occ_func_3_0(32) +
          occ_func_2_0(1) * occ_func_2_0(31) * occ_func_3_0(54) +
          occ_func_2_0(1) * occ_func_2_0(43) * occ_func_3_0(60) +
          occ_func_2_0(41) * occ_func_2_0(1) * occ_func_3_0(44) +
          occ_func_2_0(54) * occ_func_2_0(32) * occ_func_3_0(1) +
          occ_func_2_0(62) * occ_func_2_0(48) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(49) * occ_func_3_0(108) +
          occ_func_2_0(35) * occ_func_2_0(1) * occ_func_3_0(10) +
          occ_func_2_0(51) * occ_func_2_0(1) * occ_func_3_0(2) +
          occ_func_2_0(1) * occ_func_2_0(33) * occ_func_3_0(92) +
          occ_func_2_0(60) * occ_func_2_0(44) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_51_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(42) * occ_func_3_0(103) +
          occ_func_2_0(40) * occ_func_3_0(9) +
          occ_func_2_0(63) * occ_func_2_0(47) +
          occ_func_2_0(55) * occ_func_2_0(31) +
          occ_func_2_0(52) * occ_func_3_0(3) +
          occ_func_2_0(30) * occ_func_3_0(91) +
          occ_func_2_0(46) * occ_func_3_0(107) +
          occ_func_2_0(36) * occ_func_3_0(11) +
          occ_func_2_0(44) * occ_func_3_0(61) +
          occ_func_2_0(38) * occ_func_3_0(43) +
          occ_func_2_0(50) * occ_func_3_0(31) +
          occ_func_2_0(32) * occ_func_3_0(55) +
          occ_func_2_0(48) * occ_func_3_0(63) +
          occ_func_2_0(34) * occ_func_3_0(47) +
          occ_func_2_0(61) * occ_func_2_0(43) +
          occ_func_2_0(107) * occ_func_2_0(11) +
          occ_func_2_0(91) * occ_func_2_0(3) +
          occ_func_2_0(103) * occ_func_2_0(9) +
          occ_func_2_0(44) * occ_func_3_0(7) +
          occ_func_2_0(38) * occ_func_3_0(99) +
          occ_func_2_0(50) * occ_func_3_0(111) +
          occ_func_2_0(32) * occ_func_3_0(13) +
          occ_func_2_0(48) * occ_func_3_0(5) +
          occ_func_2_0(34) * occ_func_3_0(95) +
          occ_func_2_0(99) * occ_func_2_0(7) +
          occ_func_2_0(57) * occ_func_2_0(35) +
          occ_func_2_0(65) * occ_func_2_0(51) +
          occ_func_2_0(59) * occ_func_2_0(39) +
          occ_func_2_0(52) * occ_func_3_0(65) +
          occ_func_2_0(30) * occ_func_3_0(51) +
          occ_func_2_0(46) * occ_func_3_0(35) +
          occ_func_2_0(36) * occ_func_3_0(57) +
          occ_func_2_0(95) * occ_func_2_0(5) +
          occ_func_2_0(111) * occ_func_2_0(13) +
          occ_func_2_0(42) * occ_func_3_0(39) +
          occ_func_2_0(40) * occ_func_3_0(59)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_51_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(100) * occ_func_2_0(6) +
          occ_func_2_0(49) * occ_func_3_0(36) +
          occ_func_2_0(35) * occ_func_3_0(56) +
          occ_func_2_0(51) * occ_func_3_0(64) +
          occ_func_2_0(33) * occ_func_3_0(52) +
          occ_func_2_0(112) * occ_func_2_0(12) +
          occ_func_2_0(96) * occ_func_2_0(4) +
          occ_func_2_0(58) * occ_func_2_0(40) +
          occ_func_2_0(64) * occ_func_2_0(52) +
          occ_func_2_0(56) * occ_func_2_0(36) +
          occ_func_2_0(45) * occ_func_3_0(40) +
          occ_func_2_0(39) * occ_func_3_0(58) +
          occ_func_2_0(47) * occ_func_3_0(4) +
          occ_func_2_0(37) * occ_func_3_0(96) +
          occ_func_2_0(53) * occ_func_3_0(112) +
          occ_func_2_0(31) * occ_func_3_0(12) +
          occ_func_2_0(43) * occ_func_3_0(6) +
          occ_func_2_0(41) * occ_func_3_0(100) +
          occ_func_2_0(104) * occ_func_2_0(8) +
          occ_func_2_0(92) * occ_func_2_0(2) +
          occ_func_2_0(108) * occ_func_2_0(10) +
          occ_func_2_0(45) * occ_func_3_0(104) +
          occ_func_2_0(39) * occ_func_3_0(8) +
          occ_func_2_0(47) * occ_func_3_0(62) +
          occ_func_2_0(37) * occ_func_3_0(48) +
          occ_func_2_0(53) * occ_func_3_0(32) +
          occ_func_2_0(31) * occ_func_3_0(54) +
          occ_func_2_0(43) * occ_func_3_0(60) +
          occ_func_2_0(41) * occ_func_3_0(44) +
          occ_func_2_0(54) * occ_func_2_0(32) +
          occ_func_2_0(62) * occ_func_2_0(48) +
          occ_func_2_0(49) * occ_func_3_0(108) +
          occ_func_2_0(35) * occ_func_3_0(10) +
          occ_func_2_0(51) * occ_func_3_0(2) +
          occ_func_2_0(33) * occ_func_3_0(92) +
          occ_func_2_0(60) * occ_func_2_0(44)) /
         24.;
}

/**** Basis functions for orbit 52****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

2.3333333 1.6666667 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_52_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(87) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(64) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(24) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(103) +
          occ_func_2_0(84) * occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(28) * occ_func_2_0(64) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(63) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(27) +
          occ_func_2_0(29) * occ_func_2_0(63) * occ_func_3_0(0) +
          occ_func_2_0(83) * occ_func_2_0(103) * occ_func_3_0(0) +
          occ_func_2_0(79) * occ_func_2_0(107) * occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(112) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(111) +
          occ_func_2_0(80) * occ_func_2_0(108) * occ_func_3_0(1) +
          occ_func_2_0(84) * occ_func_2_0(104) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_3_0(88) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_3_0(26) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(62) +
          occ_func_2_0(29) * occ_func_2_0(65) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_3_0(25) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(65) +
          occ_func_2_0(83) * occ_func_2_0(111) * occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(104) +
          occ_func_2_0(28) * occ_func_2_0(62) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_52_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(87) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(107) +
          occ_func_2_0(21) * occ_func_2_0(55) * occ_func_3_0(0) +
          occ_func_2_0(23) * occ_func_2_0(61) * occ_func_3_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(103) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_3_0(75) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(71) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(91) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(19) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(55) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(63) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_3_0(25) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(27) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(61) +
          occ_func_2_0(29) * occ_func_2_0(63) * occ_func_3_0(0) +
          occ_func_2_0(83) * occ_func_2_0(103) * occ_func_3_0(0) +
          occ_func_2_0(79) * occ_func_2_0(107) * occ_func_3_0(0) +
          occ_func_2_0(67) * occ_func_2_0(91) * occ_func_3_0(0) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(111) +
          occ_func_2_0(0) * occ_func_2_0(6) * occ_func_3_0(87) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_3_0(75) +
          occ_func_2_0(2) * occ_func_2_0(0) * occ_func_3_0(95) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(99) +
          occ_func_2_0(0) * occ_func_2_0(4) * occ_func_3_0(71) +
          occ_func_2_0(67) * occ_func_2_0(95) * occ_func_3_0(0) +
          occ_func_2_0(21) * occ_func_2_0(59) * occ_func_3_0(0) +
          occ_func_2_0(23) * occ_func_2_0(57) * occ_func_3_0(0) +
          occ_func_2_0(29) * occ_func_2_0(65) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_3_0(25) +
          occ_func_2_0(2) * occ_func_2_0(0) * occ_func_3_0(59) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(65) +
          occ_func_2_0(0) * occ_func_2_0(4) * occ_func_3_0(27) +
          occ_func_2_0(83) * occ_func_2_0(111) * occ_func_3_0(0) +
          occ_func_2_0(79) * occ_func_2_0(99) * occ_func_3_0(0) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(57) +
          occ_func_2_0(0) * occ_func_2_0(6) * occ_func_3_0(19)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_52_0_at_1() const {
  return (occ_func_2_0(68) * occ_func_2_0(96) * occ_func_3_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(64) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_3_0(26) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(24) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_3_0(58) +
          occ_func_2_0(80) * occ_func_2_0(100) * occ_func_3_0(1) +
          occ_func_2_0(84) * occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(28) * occ_func_2_0(64) * occ_func_3_0(1) +
          occ_func_2_0(22) * occ_func_2_0(56) * occ_func_3_0(1) +
          occ_func_2_0(20) * occ_func_2_0(58) * occ_func_3_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(56) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_3_0(18) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(100) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_3_0(72) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(76) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_3_0(96) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(112) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_3_0(88) +
          occ_func_2_0(68) * occ_func_2_0(92) * occ_func_3_0(1) +
          occ_func_2_0(80) * occ_func_2_0(108) * occ_func_3_0(1) +
          occ_func_2_0(84) * occ_func_2_0(104) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_3_0(88) +
          occ_func_2_0(7) * occ_func_2_0(1) * occ_func_3_0(108) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_3_0(26) +
          occ_func_2_0(5) * occ_func_2_0(1) * occ_func_3_0(60) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(62) +
          occ_func_2_0(1) * occ_func_2_0(3) * occ_func_3_0(24) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_3_0(18) +
          occ_func_2_0(7) * occ_func_2_0(1) * occ_func_3_0(54) +
          occ_func_2_0(22) * occ_func_2_0(60) * occ_func_3_0(1) +
          occ_func_2_0(20) * occ_func_2_0(54) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_3_0(72) +
          occ_func_2_0(5) * occ_func_2_0(1) * occ_func_3_0(92) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(104) +
          occ_func_2_0(1) * occ_func_2_0(3) * occ_func_3_0(76) +
          occ_func_2_0(28) * occ_func_2_0(62) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_52_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) * occ_func_3_0(87) +
          occ_func_2_0(6) * occ_func_3_0(107) +
          occ_func_2_0(21) * occ_func_2_0(55) +
          occ_func_2_0(23) * occ_func_2_0(61) +
          occ_func_2_0(12) * occ_func_3_0(103) +
          occ_func_2_0(2) * occ_func_3_0(75) +
          occ_func_2_0(10) * occ_func_3_0(71) +
          occ_func_2_0(4) * occ_func_3_0(91) +
          occ_func_2_0(8) * occ_func_3_0(19) +
          occ_func_2_0(6) * occ_func_3_0(55) +
          occ_func_2_0(12) * occ_func_3_0(63) +
          occ_func_2_0(2) * occ_func_3_0(25) +
          occ_func_2_0(10) * occ_func_3_0(27) +
          occ_func_2_0(4) * occ_func_3_0(61) +
          occ_func_2_0(29) * occ_func_2_0(63) +
          occ_func_2_0(83) * occ_func_2_0(103) +
          occ_func_2_0(79) * occ_func_2_0(107) +
          occ_func_2_0(67) * occ_func_2_0(91) +
          occ_func_2_0(8) * occ_func_3_0(111) +
          occ_func_2_0(6) * occ_func_3_0(87) +
          occ_func_2_0(12) * occ_func_3_0(75) +
          occ_func_2_0(2) * occ_func_3_0(95) +
          occ_func_2_0(10) * occ_func_3_0(99) +
          occ_func_2_0(4) * occ_func_3_0(71) +
          occ_func_2_0(67) * occ_func_2_0(95) +
          occ_func_2_0(21) * occ_func_2_0(59) +
          occ_func_2_0(23) * occ_func_2_0(57) +
          occ_func_2_0(29) * occ_func_2_0(65) +
          occ_func_2_0(12) * occ_func_3_0(25) +
          occ_func_2_0(2) * occ_func_3_0(59) +
          occ_func_2_0(10) * occ_func_3_0(65) +
          occ_func_2_0(4) * occ_func_3_0(27) +
          occ_func_2_0(83) * occ_func_2_0(111) +
          occ_func_2_0(79) * occ_func_2_0(99) +
          occ_func_2_0(8) * occ_func_3_0(57) +
          occ_func_2_0(6) * occ_func_3_0(19)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_52_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(68) * occ_func_2_0(96) +
          occ_func_2_0(11) * occ_func_3_0(64) +
          occ_func_2_0(5) * occ_func_3_0(26) +
          occ_func_2_0(13) * occ_func_3_0(24) +
          occ_func_2_0(3) * occ_func_3_0(58) +
          occ_func_2_0(80) * occ_func_2_0(100) +
          occ_func_2_0(84) * occ_func_2_0(112) +
          occ_func_2_0(28) * occ_func_2_0(64) +
          occ_func_2_0(22) * occ_func_2_0(56) +
          occ_func_2_0(20) * occ_func_2_0(58) +
          occ_func_2_0(9) * occ_func_3_0(56) +
          occ_func_2_0(7) * occ_func_3_0(18) +
          occ_func_2_0(11) * occ_func_3_0(100) +
          occ_func_2_0(5) * occ_func_3_0(72) +
          occ_func_2_0(13) * occ_func_3_0(76) +
          occ_func_2_0(3) * occ_func_3_0(96) +
          occ_func_2_0(9) * occ_func_3_0(112) +
          occ_func_2_0(7) * occ_func_3_0(88) +
          occ_func_2_0(68) * occ_func_2_0(92) +
          occ_func_2_0(80) * occ_func_2_0(108) +
          occ_func_2_0(84) * occ_func_2_0(104) +
          occ_func_2_0(9) * occ_func_3_0(88) +
          occ_func_2_0(7) * occ_func_3_0(108) +
          occ_func_2_0(11) * occ_func_3_0(26) +
          occ_func_2_0(5) * occ_func_3_0(60) +
          occ_func_2_0(13) * occ_func_3_0(62) +
          occ_func_2_0(3) * occ_func_3_0(24) +
          occ_func_2_0(9) * occ_func_3_0(18) +
          occ_func_2_0(7) * occ_func_3_0(54) +
          occ_func_2_0(22) * occ_func_2_0(60) +
          occ_func_2_0(20) * occ_func_2_0(54) +
          occ_func_2_0(11) * occ_func_3_0(72) +
          occ_func_2_0(5) * occ_func_3_0(92) +
          occ_func_2_0(13) * occ_func_3_0(104) +
          occ_func_2_0(3) * occ_func_3_0(76) +
          occ_func_2_0(28) * occ_func_2_0(62)) /
         24.;
}

/**** Basis functions for orbit 53****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

2.3333333 1.6666667 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_53_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(29) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(112) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(80) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(61) +
          occ_func_2_0(26) * occ_func_2_0(64) * occ_func_3_0(1) +
          occ_func_2_0(88) * occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(107) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(83) +
          occ_func_2_0(87) * occ_func_2_0(107) * occ_func_3_0(0) +
          occ_func_2_0(27) * occ_func_2_0(61) * occ_func_3_0(0) +
          occ_func_2_0(25) * occ_func_2_0(63) * occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(64) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(65) +
          occ_func_2_0(24) * occ_func_2_0(62) * occ_func_3_0(1) +
          occ_func_2_0(26) * occ_func_2_0(60) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_3_0(28) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_3_0(84) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(108) +
          occ_func_2_0(87) * occ_func_2_0(111) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_3_0(79) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(111) +
          occ_func_2_0(27) * occ_func_2_0(65) * occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(60) +
          occ_func_2_0(88) * occ_func_2_0(108) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_53_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(29) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(63) +
          occ_func_2_0(71) * occ_func_2_0(91) * occ_func_3_0(0) +
          occ_func_2_0(75) * occ_func_2_0(103) * occ_func_3_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(61) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_3_0(23) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(21) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(55) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(67) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(91) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(107) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_3_0(79) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(83) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(103) +
          occ_func_2_0(87) * occ_func_2_0(107) * occ_func_3_0(0) +
          occ_func_2_0(27) * occ_func_2_0(61) * occ_func_3_0(0) +
          occ_func_2_0(25) * occ_func_2_0(63) * occ_func_3_0(0) +
          occ_func_2_0(19) * occ_func_2_0(55) * occ_func_3_0(0) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(65) +
          occ_func_2_0(0) * occ_func_2_0(6) * occ_func_3_0(29) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_3_0(23) +
          occ_func_2_0(2) * occ_func_2_0(0) * occ_func_3_0(57) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(59) +
          occ_func_2_0(0) * occ_func_2_0(4) * occ_func_3_0(21) +
          occ_func_2_0(19) * occ_func_2_0(57) * occ_func_3_0(0) +
          occ_func_2_0(71) * occ_func_2_0(99) * occ_func_3_0(0) +
          occ_func_2_0(75) * occ_func_2_0(95) * occ_func_3_0(0) +
          occ_func_2_0(87) * occ_func_2_0(111) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_3_0(79) +
          occ_func_2_0(2) * occ_func_2_0(0) * occ_func_3_0(99) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(111) +
          occ_func_2_0(0) * occ_func_2_0(4) * occ_func_3_0(83) +
          occ_func_2_0(27) * occ_func_2_0(65) * occ_func_3_0(0) +
          occ_func_2_0(25) * occ_func_2_0(59) * occ_func_3_0(0) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(95) +
          occ_func_2_0(0) * occ_func_2_0(6) * occ_func_3_0(67)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_53_0_at_1() const {
  return (occ_func_2_0(18) * occ_func_2_0(56) * occ_func_3_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(112) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_3_0(84) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(80) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_3_0(100) +
          occ_func_2_0(24) * occ_func_2_0(58) * occ_func_3_0(1) +
          occ_func_2_0(26) * occ_func_2_0(64) * occ_func_3_0(1) +
          occ_func_2_0(88) * occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(76) * occ_func_2_0(96) * occ_func_3_0(1) +
          occ_func_2_0(72) * occ_func_2_0(100) * occ_func_3_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(96) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_3_0(68) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(58) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_3_0(20) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(22) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_3_0(56) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_3_0(64) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_3_0(28) +
          occ_func_2_0(18) * occ_func_2_0(54) * occ_func_3_0(1) +
          occ_func_2_0(24) * occ_func_2_0(62) * occ_func_3_0(1) +
          occ_func_2_0(26) * occ_func_2_0(60) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_3_0(28) +
          occ_func_2_0(7) * occ_func_2_0(1) * occ_func_3_0(62) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_3_0(84) +
          occ_func_2_0(5) * occ_func_2_0(1) * occ_func_3_0(104) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(108) +
          occ_func_2_0(1) * occ_func_2_0(3) * occ_func_3_0(80) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_3_0(68) +
          occ_func_2_0(7) * occ_func_2_0(1) * occ_func_3_0(92) +
          occ_func_2_0(76) * occ_func_2_0(104) * occ_func_3_0(1) +
          occ_func_2_0(72) * occ_func_2_0(92) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_3_0(20) +
          occ_func_2_0(5) * occ_func_2_0(1) * occ_func_3_0(54) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(60) +
          occ_func_2_0(1) * occ_func_2_0(3) * occ_func_3_0(22) +
          occ_func_2_0(88) * occ_func_2_0(108) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_53_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) * occ_func_3_0(29) +
          occ_func_2_0(6) * occ_func_3_0(63) +
          occ_func_2_0(71) * occ_func_2_0(91) +
          occ_func_2_0(75) * occ_func_2_0(103) +
          occ_func_2_0(12) * occ_func_3_0(61) +
          occ_func_2_0(2) * occ_func_3_0(23) +
          occ_func_2_0(10) * occ_func_3_0(21) +
          occ_func_2_0(4) * occ_func_3_0(55) +
          occ_func_2_0(8) * occ_func_3_0(67) +
          occ_func_2_0(6) * occ_func_3_0(91) +
          occ_func_2_0(12) * occ_func_3_0(107) +
          occ_func_2_0(2) * occ_func_3_0(79) +
          occ_func_2_0(10) * occ_func_3_0(83) +
          occ_func_2_0(4) * occ_func_3_0(103) +
          occ_func_2_0(87) * occ_func_2_0(107) +
          occ_func_2_0(27) * occ_func_2_0(61) +
          occ_func_2_0(25) * occ_func_2_0(63) +
          occ_func_2_0(19) * occ_func_2_0(55) +
          occ_func_2_0(8) * occ_func_3_0(65) +
          occ_func_2_0(6) * occ_func_3_0(29) +
          occ_func_2_0(12) * occ_func_3_0(23) +
          occ_func_2_0(2) * occ_func_3_0(57) +
          occ_func_2_0(10) * occ_func_3_0(59) +
          occ_func_2_0(4) * occ_func_3_0(21) +
          occ_func_2_0(19) * occ_func_2_0(57) +
          occ_func_2_0(71) * occ_func_2_0(99) +
          occ_func_2_0(75) * occ_func_2_0(95) +
          occ_func_2_0(87) * occ_func_2_0(111) +
          occ_func_2_0(12) * occ_func_3_0(79) +
          occ_func_2_0(2) * occ_func_3_0(99) +
          occ_func_2_0(10) * occ_func_3_0(111) +
          occ_func_2_0(4) * occ_func_3_0(83) +
          occ_func_2_0(27) * occ_func_2_0(65) +
          occ_func_2_0(25) * occ_func_2_0(59) +
          occ_func_2_0(8) * occ_func_3_0(95) +
          occ_func_2_0(6) * occ_func_3_0(67)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_53_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(18) * occ_func_2_0(56) +
          occ_func_2_0(11) * occ_func_3_0(112) +
          occ_func_2_0(5) * occ_func_3_0(84) +
          occ_func_2_0(13) * occ_func_3_0(80) +
          occ_func_2_0(3) * occ_func_3_0(100) +
          occ_func_2_0(24) * occ_func_2_0(58) +
          occ_func_2_0(26) * occ_func_2_0(64) +
          occ_func_2_0(88) * occ_func_2_0(112) +
          occ_func_2_0(76) * occ_func_2_0(96) +
          occ_func_2_0(72) * occ_func_2_0(100) +
          occ_func_2_0(9) * occ_func_3_0(96) +
          occ_func_2_0(7) * occ_func_3_0(68) +
          occ_func_2_0(11) * occ_func_3_0(58) +
          occ_func_2_0(5) * occ_func_3_0(20) +
          occ_func_2_0(13) * occ_func_3_0(22) +
          occ_func_2_0(3) * occ_func_3_0(56) +
          occ_func_2_0(9) * occ_func_3_0(64) +
          occ_func_2_0(7) * occ_func_3_0(28) +
          occ_func_2_0(18) * occ_func_2_0(54) +
          occ_func_2_0(24) * occ_func_2_0(62) +
          occ_func_2_0(26) * occ_func_2_0(60) +
          occ_func_2_0(9) * occ_func_3_0(28) +
          occ_func_2_0(7) * occ_func_3_0(62) +
          occ_func_2_0(11) * occ_func_3_0(84) +
          occ_func_2_0(5) * occ_func_3_0(104) +
          occ_func_2_0(13) * occ_func_3_0(108) +
          occ_func_2_0(3) * occ_func_3_0(80) +
          occ_func_2_0(9) * occ_func_3_0(68) +
          occ_func_2_0(7) * occ_func_3_0(92) +
          occ_func_2_0(76) * occ_func_2_0(104) +
          occ_func_2_0(72) * occ_func_2_0(92) +
          occ_func_2_0(11) * occ_func_3_0(20) +
          occ_func_2_0(5) * occ_func_3_0(54) +
          occ_func_2_0(13) * occ_func_3_0(60) +
          occ_func_2_0(3) * occ_func_3_0(22) +
          occ_func_2_0(88) * occ_func_2_0(108)) /
         24.;
}

/**** Basis functions for orbit 54****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.7500000 Va  O

2.3333333 1.6666667 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_54_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_3_0(87) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_3_0(112) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_3_0(24) +
          occ_func_2_0(52) * occ_func_2_0(0) * occ_func_3_0(61) +
          occ_func_2_0(84) * occ_func_2_0(64) * occ_func_3_0(1) +
          occ_func_2_0(28) * occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_3_0(107) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_3_0(27) +
          occ_func_2_0(29) * occ_func_2_0(107) * occ_func_3_0(0) +
          occ_func_2_0(83) * occ_func_2_0(61) * occ_func_3_0(0) +
          occ_func_2_0(79) * occ_func_2_0(63) * occ_func_3_0(0) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_3_0(64) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_3_0(65) +
          occ_func_2_0(80) * occ_func_2_0(62) * occ_func_3_0(1) +
          occ_func_2_0(84) * occ_func_2_0(60) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_3_0(88) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_3_0(26) +
          occ_func_2_0(53) * occ_func_2_0(1) * occ_func_3_0(108) +
          occ_func_2_0(29) * occ_func_2_0(111) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(52) * occ_func_3_0(25) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_3_0(111) +
          occ_func_2_0(83) * occ_func_2_0(65) * occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_2_0(1) * occ_func_3_0(60) +
          occ_func_2_0(28) * occ_func_2_0(108) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_54_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_3_0(87) +
          occ_func_2_0(40) * occ_func_2_0(0) * occ_func_3_0(63) +
          occ_func_2_0(21) * occ_func_2_0(91) * occ_func_3_0(0) +
          occ_func_2_0(23) * occ_func_2_0(103) * occ_func_3_0(0) +
          occ_func_2_0(52) * occ_func_2_0(0) * occ_func_3_0(61) +
          occ_func_2_0(0) * occ_func_2_0(30) * occ_func_3_0(75) +
          occ_func_2_0(0) * occ_func_2_0(46) * occ_func_3_0(71) +
          occ_func_2_0(36) * occ_func_2_0(0) * occ_func_3_0(55) +
          occ_func_2_0(0) * occ_func_2_0(44) * occ_func_3_0(19) +
          occ_func_2_0(38) * occ_func_2_0(0) * occ_func_3_0(91) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_3_0(107) +
          occ_func_2_0(0) * occ_func_2_0(32) * occ_func_3_0(25) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_3_0(27) +
          occ_func_2_0(34) * occ_func_2_0(0) * occ_func_3_0(103) +
          occ_func_2_0(29) * occ_func_2_0(107) * occ_func_3_0(0) +
          occ_func_2_0(83) * occ_func_2_0(61) * occ_func_3_0(0) +
          occ_func_2_0(79) * occ_func_2_0(63) * occ_func_3_0(0) +
          occ_func_2_0(67) * occ_func_2_0(55) * occ_func_3_0(0) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_3_0(65) +
          occ_func_2_0(0) * occ_func_2_0(38) * occ_func_3_0(87) +
          occ_func_2_0(0) * occ_func_2_0(50) * occ_func_3_0(75) +
          occ_func_2_0(32) * occ_func_2_0(0) * occ_func_3_0(57) +
          occ_func_2_0(48) * occ_func_2_0(0) * occ_func_3_0(59) +
          occ_func_2_0(0) * occ_func_2_0(34) * occ_func_3_0(71) +
          occ_func_2_0(67) * occ_func_2_0(57) * occ_func_3_0(0) +
          occ_func_2_0(21) * occ_func_2_0(99) * occ_func_3_0(0) +
          occ_func_2_0(23) * occ_func_2_0(95) * occ_func_3_0(0) +
          occ_func_2_0(29) * occ_func_2_0(111) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(52) * occ_func_3_0(25) +
          occ_func_2_0(30) * occ_func_2_0(0) * occ_func_3_0(99) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_3_0(111) +
          occ_func_2_0(0) * occ_func_2_0(36) * occ_func_3_0(27) +
          occ_func_2_0(83) * occ_func_2_0(65) * occ_func_3_0(0) +
          occ_func_2_0(79) * occ_func_2_0(59) * occ_func_3_0(0) +
          occ_func_2_0(42) * occ_func_2_0(0) * occ_func_3_0(95) +
          occ_func_2_0(0) * occ_func_2_0(40) * occ_func_3_0(19)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_54_0_at_1() const {
  return (occ_func_2_0(68) * occ_func_2_0(56) * occ_func_3_0(1) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_3_0(112) +
          occ_func_2_0(1) * occ_func_2_0(35) * occ_func_3_0(26) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_3_0(24) +
          occ_func_2_0(33) * occ_func_2_0(1) * occ_func_3_0(100) +
          occ_func_2_0(80) * occ_func_2_0(58) * occ_func_3_0(1) +
          occ_func_2_0(84) * occ_func_2_0(64) * occ_func_3_0(1) +
          occ_func_2_0(28) * occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(22) * occ_func_2_0(96) * occ_func_3_0(1) +
          occ_func_2_0(20) * occ_func_2_0(100) * occ_func_3_0(1) +
          occ_func_2_0(45) * occ_func_2_0(1) * occ_func_3_0(96) +
          occ_func_2_0(1) * occ_func_2_0(39) * occ_func_3_0(18) +
          occ_func_2_0(47) * occ_func_2_0(1) * occ_func_3_0(58) +
          occ_func_2_0(1) * occ_func_2_0(37) * occ_func_3_0(72) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_3_0(76) +
          occ_func_2_0(31) * occ_func_2_0(1) * occ_func_3_0(56) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_3_0(64) +
          occ_func_2_0(1) * occ_func_2_0(41) * occ_func_3_0(88) +
          occ_func_2_0(68) * occ_func_2_0(54) * occ_func_3_0(1) +
          occ_func_2_0(80) * occ_func_2_0(62) * occ_func_3_0(1) +
          occ_func_2_0(84) * occ_func_2_0(60) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_3_0(88) +
          occ_func_2_0(39) * occ_func_2_0(1) * occ_func_3_0(62) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_3_0(26) +
          occ_func_2_0(37) * occ_func_2_0(1) * occ_func_3_0(104) +
          occ_func_2_0(53) * occ_func_2_0(1) * occ_func_3_0(108) +
          occ_func_2_0(1) * occ_func_2_0(31) * occ_func_3_0(24) +
          occ_func_2_0(1) * occ_func_2_0(43) * occ_func_3_0(18) +
          occ_func_2_0(41) * occ_func_2_0(1) * occ_func_3_0(92) +
          occ_func_2_0(22) * occ_func_2_0(104) * occ_func_3_0(1) +
          occ_func_2_0(20) * occ_func_2_0(92) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(49) * occ_func_3_0(72) +
          occ_func_2_0(35) * occ_func_2_0(1) * occ_func_3_0(54) +
          occ_func_2_0(51) * occ_func_2_0(1) * occ_func_3_0(60) +
          occ_func_2_0(1) * occ_func_2_0(33) * occ_func_3_0(76) +
          occ_func_2_0(28) * occ_func_2_0(108) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_54_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(42) * occ_func_3_0(87) +
          occ_func_2_0(40) * occ_func_3_0(63) +
          occ_func_2_0(21) * occ_func_2_0(91) +
          occ_func_2_0(23) * occ_func_2_0(103) +
          occ_func_2_0(52) * occ_func_3_0(61) +
          occ_func_2_0(30) * occ_func_3_0(75) +
          occ_func_2_0(46) * occ_func_3_0(71) +
          occ_func_2_0(36) * occ_func_3_0(55) +
          occ_func_2_0(44) * occ_func_3_0(19) +
          occ_func_2_0(38) * occ_func_3_0(91) +
          occ_func_2_0(50) * occ_func_3_0(107) +
          occ_func_2_0(32) * occ_func_3_0(25) +
          occ_func_2_0(48) * occ_func_3_0(27) +
          occ_func_2_0(34) * occ_func_3_0(103) +
          occ_func_2_0(29) * occ_func_2_0(107) +
          occ_func_2_0(83) * occ_func_2_0(61) +
          occ_func_2_0(79) * occ_func_2_0(63) +
          occ_func_2_0(67) * occ_func_2_0(55) +
          occ_func_2_0(44) * occ_func_3_0(65) +
          occ_func_2_0(38) * occ_func_3_0(87) +
          occ_func_2_0(50) * occ_func_3_0(75) +
          occ_func_2_0(32) * occ_func_3_0(57) +
          occ_func_2_0(48) * occ_func_3_0(59) +
          occ_func_2_0(34) * occ_func_3_0(71) +
          occ_func_2_0(67) * occ_func_2_0(57) +
          occ_func_2_0(21) * occ_func_2_0(99) +
          occ_func_2_0(23) * occ_func_2_0(95) +
          occ_func_2_0(29) * occ_func_2_0(111) +
          occ_func_2_0(52) * occ_func_3_0(25) +
          occ_func_2_0(30) * occ_func_3_0(99) +
          occ_func_2_0(46) * occ_func_3_0(111) +
          occ_func_2_0(36) * occ_func_3_0(27) +
          occ_func_2_0(83) * occ_func_2_0(65) +
          occ_func_2_0(79) * occ_func_2_0(59) +
          occ_func_2_0(42) * occ_func_3_0(95) +
          occ_func_2_0(40) * occ_func_3_0(19)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_54_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(68) * occ_func_2_0(56) +
          occ_func_2_0(49) * occ_func_3_0(112) +
          occ_func_2_0(35) * occ_func_3_0(26) +
          occ_func_2_0(51) * occ_func_3_0(24) +
          occ_func_2_0(33) * occ_func_3_0(100) +
          occ_func_2_0(80) * occ_func_2_0(58) +
          occ_func_2_0(84) * occ_func_2_0(64) +
          occ_func_2_0(28) * occ_func_2_0(112) +
          occ_func_2_0(22) * occ_func_2_0(96) +
          occ_func_2_0(20) * occ_func_2_0(100) +
          occ_func_2_0(45) * occ_func_3_0(96) +
          occ_func_2_0(39) * occ_func_3_0(18) +
          occ_func_2_0(47) * occ_func_3_0(58) +
          occ_func_2_0(37) * occ_func_3_0(72) +
          occ_func_2_0(53) * occ_func_3_0(76) +
          occ_func_2_0(31) * occ_func_3_0(56) +
          occ_func_2_0(43) * occ_func_3_0(64) +
          occ_func_2_0(41) * occ_func_3_0(88) +
          occ_func_2_0(68) * occ_func_2_0(54) +
          occ_func_2_0(80) * occ_func_2_0(62) +
          occ_func_2_0(84) * occ_func_2_0(60) +
          occ_func_2_0(45) * occ_func_3_0(88) +
          occ_func_2_0(39) * occ_func_3_0(62) +
          occ_func_2_0(47) * occ_func_3_0(26) +
          occ_func_2_0(37) * occ_func_3_0(104) +
          occ_func_2_0(53) * occ_func_3_0(108) +
          occ_func_2_0(31) * occ_func_3_0(24) +
          occ_func_2_0(43) * occ_func_3_0(18) +
          occ_func_2_0(41) * occ_func_3_0(92) +
          occ_func_2_0(22) * occ_func_2_0(104) +
          occ_func_2_0(20) * occ_func_2_0(92) +
          occ_func_2_0(49) * occ_func_3_0(72) +
          occ_func_2_0(35) * occ_func_3_0(54) +
          occ_func_2_0(51) * occ_func_3_0(60) +
          occ_func_2_0(33) * occ_func_3_0(76) +
          occ_func_2_0(28) * occ_func_2_0(108)) /
         24.;
}

/**** Basis functions for orbit 55****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.7500000 Va  O

2.3333333 2.6666666 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_55_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_3_0(111) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_3_0(84) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_3_0(62) +
          occ_func_2_0(108) * occ_func_2_0(24) * occ_func_3_0(1) +
          occ_func_2_0(104) * occ_func_2_0(26) * occ_func_3_0(1) +
          occ_func_2_0(62) * occ_func_2_0(88) * occ_func_3_0(1) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_3_0(79) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_3_0(65) +
          occ_func_2_0(65) * occ_func_2_0(87) * occ_func_3_0(0) +
          occ_func_2_0(111) * occ_func_2_0(27) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_3_0(104) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_3_0(28) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_3_0(29) +
          occ_func_2_0(0) * occ_func_2_0(50) * occ_func_3_0(103) +
          occ_func_2_0(112) * occ_func_2_0(26) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_3_0(112) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_3_0(64) +
          occ_func_2_0(53) * occ_func_2_0(1) * occ_func_3_0(80) +
          occ_func_2_0(63) * occ_func_2_0(87) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(52) * occ_func_3_0(63) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_3_0(83) +
          occ_func_2_0(103) * occ_func_2_0(27) * occ_func_3_0(0) +
          occ_func_2_0(107) * occ_func_2_0(25) * occ_func_3_0(0) +
          occ_func_2_0(64) * occ_func_2_0(88) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_55_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(42) * occ_func_3_0(111) +
          occ_func_2_0(40) * occ_func_2_0(0) * occ_func_3_0(29) +
          occ_func_2_0(59) * occ_func_2_0(71) * occ_func_3_0(0) +
          occ_func_2_0(57) * occ_func_2_0(75) * occ_func_3_0(0) +
          occ_func_2_0(52) * occ_func_2_0(0) * occ_func_3_0(23) +
          occ_func_2_0(0) * occ_func_2_0(30) * occ_func_3_0(95) +
          occ_func_2_0(0) * occ_func_2_0(46) * occ_func_3_0(99) +
          occ_func_2_0(36) * occ_func_2_0(0) * occ_func_3_0(21) +
          occ_func_2_0(0) * occ_func_2_0(44) * occ_func_3_0(57) +
          occ_func_2_0(38) * occ_func_2_0(0) * occ_func_3_0(67) +
          occ_func_2_0(50) * occ_func_2_0(0) * occ_func_3_0(79) +
          occ_func_2_0(0) * occ_func_2_0(32) * occ_func_3_0(59) +
          occ_func_2_0(0) * occ_func_2_0(48) * occ_func_3_0(65) +
          occ_func_2_0(34) * occ_func_2_0(0) * occ_func_3_0(83) +
          occ_func_2_0(65) * occ_func_2_0(87) * occ_func_3_0(0) +
          occ_func_2_0(111) * occ_func_2_0(27) * occ_func_3_0(0) +
          occ_func_2_0(99) * occ_func_2_0(25) * occ_func_3_0(0) +
          occ_func_2_0(95) * occ_func_2_0(19) * occ_func_3_0(0) +
          occ_func_2_0(44) * occ_func_2_0(0) * occ_func_3_0(29) +
          occ_func_2_0(0) * occ_func_2_0(38) * occ_func_3_0(107) +
          occ_func_2_0(0) * occ_func_2_0(50) * occ_func_3_0(103) +
          occ_func_2_0(32) * occ_func_2_0(0) * occ_func_3_0(23) +
          occ_func_2_0(48) * occ_func_2_0(0) * occ_func_3_0(21) +
          occ_func_2_0(0) * occ_func_2_0(34) * occ_func_3_0(91) +
          occ_func_2_0(91) * occ_func_2_0(19) * occ_func_3_0(0) +
          occ_func_2_0(55) * occ_func_2_0(71) * occ_func_3_0(0) +
          occ_func_2_0(61) * occ_func_2_0(75) * occ_func_3_0(0) +
          occ_func_2_0(63) * occ_func_2_0(87) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(52) * occ_func_3_0(63) +
          occ_func_2_0(30) * occ_func_2_0(0) * occ_func_3_0(79) +
          occ_func_2_0(46) * occ_func_2_0(0) * occ_func_3_0(83) +
          occ_func_2_0(0) * occ_func_2_0(36) * occ_func_3_0(61) +
          occ_func_2_0(103) * occ_func_2_0(27) * occ_func_3_0(0) +
          occ_func_2_0(107) * occ_func_2_0(25) * occ_func_3_0(0) +
          occ_func_2_0(42) * occ_func_2_0(0) * occ_func_3_0(67) +
          occ_func_2_0(0) * occ_func_2_0(40) * occ_func_3_0(55)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_55_0_at_1() const {
  return (occ_func_2_0(92) * occ_func_2_0(18) * occ_func_3_0(1) +
          occ_func_2_0(49) * occ_func_2_0(1) * occ_func_3_0(84) +
          occ_func_2_0(1) * occ_func_2_0(35) * occ_func_3_0(60) +
          occ_func_2_0(1) * occ_func_2_0(51) * occ_func_3_0(62) +
          occ_func_2_0(33) * occ_func_2_0(1) * occ_func_3_0(80) +
          occ_func_2_0(108) * occ_func_2_0(24) * occ_func_3_0(1) +
          occ_func_2_0(104) * occ_func_2_0(26) * occ_func_3_0(1) +
          occ_func_2_0(62) * occ_func_2_0(88) * occ_func_3_0(1) +
          occ_func_2_0(60) * occ_func_2_0(76) * occ_func_3_0(1) +
          occ_func_2_0(54) * occ_func_2_0(72) * occ_func_3_0(1) +
          occ_func_2_0(45) * occ_func_2_0(1) * occ_func_3_0(68) +
          occ_func_2_0(1) * occ_func_2_0(39) * occ_func_3_0(54) +
          occ_func_2_0(47) * occ_func_2_0(1) * occ_func_3_0(20) +
          occ_func_2_0(1) * occ_func_2_0(37) * occ_func_3_0(92) +
          occ_func_2_0(1) * occ_func_2_0(53) * occ_func_3_0(104) +
          occ_func_2_0(31) * occ_func_2_0(1) * occ_func_3_0(22) +
          occ_func_2_0(43) * occ_func_2_0(1) * occ_func_3_0(28) +
          occ_func_2_0(1) * occ_func_2_0(41) * occ_func_3_0(108) +
          occ_func_2_0(96) * occ_func_2_0(18) * occ_func_3_0(1) +
          occ_func_2_0(100) * occ_func_2_0(24) * occ_func_3_0(1) +
          occ_func_2_0(112) * occ_func_2_0(26) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(45) * occ_func_3_0(112) +
          occ_func_2_0(39) * occ_func_2_0(1) * occ_func_3_0(28) +
          occ_func_2_0(1) * occ_func_2_0(47) * occ_func_3_0(64) +
          occ_func_2_0(37) * occ_func_2_0(1) * occ_func_3_0(84) +
          occ_func_2_0(53) * occ_func_2_0(1) * occ_func_3_0(80) +
          occ_func_2_0(1) * occ_func_2_0(31) * occ_func_3_0(58) +
          occ_func_2_0(1) * occ_func_2_0(43) * occ_func_3_0(56) +
          occ_func_2_0(41) * occ_func_2_0(1) * occ_func_3_0(68) +
          occ_func_2_0(56) * occ_func_2_0(76) * occ_func_3_0(1) +
          occ_func_2_0(58) * occ_func_2_0(72) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(49) * occ_func_3_0(100) +
          occ_func_2_0(35) * occ_func_2_0(1) * occ_func_3_0(20) +
          occ_func_2_0(51) * occ_func_2_0(1) * occ_func_3_0(22) +
          occ_func_2_0(1) * occ_func_2_0(33) * occ_func_3_0(96) +
          occ_func_2_0(64) * occ_func_2_0(88) * occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_55_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(42) * occ_func_3_0(111) +
          occ_func_2_0(40) * occ_func_3_0(29) +
          occ_func_2_0(59) * occ_func_2_0(71) +
          occ_func_2_0(57) * occ_func_2_0(75) +
          occ_func_2_0(52) * occ_func_3_0(23) +
          occ_func_2_0(30) * occ_func_3_0(95) +
          occ_func_2_0(46) * occ_func_3_0(99) +
          occ_func_2_0(36) * occ_func_3_0(21) +
          occ_func_2_0(44) * occ_func_3_0(57) +
          occ_func_2_0(38) * occ_func_3_0(67) +
          occ_func_2_0(50) * occ_func_3_0(79) +
          occ_func_2_0(32) * occ_func_3_0(59) +
          occ_func_2_0(48) * occ_func_3_0(65) +
          occ_func_2_0(34) * occ_func_3_0(83) +
          occ_func_2_0(65) * occ_func_2_0(87) +
          occ_func_2_0(111) * occ_func_2_0(27) +
          occ_func_2_0(99) * occ_func_2_0(25) +
          occ_func_2_0(95) * occ_func_2_0(19) +
          occ_func_2_0(44) * occ_func_3_0(29) +
          occ_func_2_0(38) * occ_func_3_0(107) +
          occ_func_2_0(50) * occ_func_3_0(103) +
          occ_func_2_0(32) * occ_func_3_0(23) +
          occ_func_2_0(48) * occ_func_3_0(21) +
          occ_func_2_0(34) * occ_func_3_0(91) +
          occ_func_2_0(91) * occ_func_2_0(19) +
          occ_func_2_0(55) * occ_func_2_0(71) +
          occ_func_2_0(61) * occ_func_2_0(75) +
          occ_func_2_0(63) * occ_func_2_0(87) +
          occ_func_2_0(52) * occ_func_3_0(63) +
          occ_func_2_0(30) * occ_func_3_0(79) +
          occ_func_2_0(46) * occ_func_3_0(83) +
          occ_func_2_0(36) * occ_func_3_0(61) +
          occ_func_2_0(103) * occ_func_2_0(27) +
          occ_func_2_0(107) * occ_func_2_0(25) +
          occ_func_2_0(42) * occ_func_3_0(67) +
          occ_func_2_0(40) * occ_func_3_0(55)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_55_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(92) * occ_func_2_0(18) +
          occ_func_2_0(49) * occ_func_3_0(84) +
          occ_func_2_0(35) * occ_func_3_0(60) +
          occ_func_2_0(51) * occ_func_3_0(62) +
          occ_func_2_0(33) * occ_func_3_0(80) +
          occ_func_2_0(108) * occ_func_2_0(24) +
          occ_func_2_0(104) * occ_func_2_0(26) +
          occ_func_2_0(62) * occ_func_2_0(88) +
          occ_func_2_0(60) * occ_func_2_0(76) +
          occ_func_2_0(54) * occ_func_2_0(72) +
          occ_func_2_0(45) * occ_func_3_0(68) +
          occ_func_2_0(39) * occ_func_3_0(54) +
          occ_func_2_0(47) * occ_func_3_0(20) +
          occ_func_2_0(37) * occ_func_3_0(92) +
          occ_func_2_0(53) * occ_func_3_0(104) +
          occ_func_2_0(31) * occ_func_3_0(22) +
          occ_func_2_0(43) * occ_func_3_0(28) +
          occ_func_2_0(41) * occ_func_3_0(108) +
          occ_func_2_0(96) * occ_func_2_0(18) +
          occ_func_2_0(100) * occ_func_2_0(24) +
          occ_func_2_0(112) * occ_func_2_0(26) +
          occ_func_2_0(45) * occ_func_3_0(112) +
          occ_func_2_0(39) * occ_func_3_0(28) +
          occ_func_2_0(47) * occ_func_3_0(64) +
          occ_func_2_0(37) * occ_func_3_0(84) +
          occ_func_2_0(53) * occ_func_3_0(80) +
          occ_func_2_0(31) * occ_func_3_0(58) +
          occ_func_2_0(43) * occ_func_3_0(56) +
          occ_func_2_0(41) * occ_func_3_0(68) +
          occ_func_2_0(56) * occ_func_2_0(76) +
          occ_func_2_0(58) * occ_func_2_0(72) +
          occ_func_2_0(49) * occ_func_3_0(100) +
          occ_func_2_0(35) * occ_func_3_0(20) +
          occ_func_2_0(51) * occ_func_3_0(22) +
          occ_func_2_0(33) * occ_func_3_0(96) +
          occ_func_2_0(64) * occ_func_2_0(88)) /
         24.;
}

/**** Basis functions for orbit 56****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 0.7500000 Va  O

0.3333333 2.6666666 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_56_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(60) +
          occ_func_2_0(63) * occ_func_3_0(108) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(65) +
          occ_func_2_0(64) * occ_func_3_0(65) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(62) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(104) +
          occ_func_2_0(64) * occ_func_3_0(111) * occ_func_2_0(0) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(108) +
          occ_func_2_0(61) * occ_func_3_0(104) * occ_func_2_0(1) +
          occ_func_2_0(63) * occ_func_3_0(62) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(65) +
          occ_func_2_0(61) * occ_func_3_0(60) * occ_func_2_0(1) +
          occ_func_2_0(60) * occ_func_3_0(61) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(64) +
          occ_func_2_0(62) * occ_func_3_0(63) * occ_func_2_0(0) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(61) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(63) +
          occ_func_2_0(65) * occ_func_3_0(112) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(61) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(112) +
          occ_func_2_0(62) * occ_func_3_0(107) * occ_func_2_0(0) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(63) +
          occ_func_2_0(65) * occ_func_3_0(64) * occ_func_2_0(1) +
          occ_func_2_0(60) * occ_func_3_0(103) * occ_func_2_0(0)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_56_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(60) +
          occ_func_2_0(58) * occ_func_3_0(59) * occ_func_2_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(95) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(111) +
          occ_func_2_0(64) * occ_func_3_0(65) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(54) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(62) +
          occ_func_2_0(56) * occ_func_3_0(57) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(60) +
          occ_func_2_0(58) * occ_func_3_0(99) * occ_func_2_0(0) +
          occ_func_2_0(64) * occ_func_3_0(111) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(54) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(62) +
          occ_func_2_0(56) * occ_func_3_0(95) * occ_func_2_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(99) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(57) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(65) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(59) +
          occ_func_2_0(60) * occ_func_3_0(61) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(58) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(64) +
          occ_func_2_0(54) * occ_func_3_0(55) * occ_func_2_0(0) +
          occ_func_2_0(62) * occ_func_3_0(63) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(56) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(61) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(107) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(91) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(103) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(64) +
          occ_func_2_0(54) * occ_func_3_0(91) * occ_func_2_0(0) +
          occ_func_2_0(62) * occ_func_3_0(107) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(56) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(63) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(55) +
          occ_func_2_0(60) * occ_func_3_0(103) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(58)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_56_0_at_1() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(60) +
          occ_func_2_0(63) * occ_func_3_0(108) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(57) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(65) +
          occ_func_2_0(55) * occ_func_3_0(92) * occ_func_2_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(54) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(62) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(104) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(92) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(108) +
          occ_func_2_0(61) * occ_func_3_0(104) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(59) +
          occ_func_2_0(63) * occ_func_3_0(62) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(57) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(65) +
          occ_func_2_0(55) * occ_func_3_0(54) * occ_func_2_0(1) +
          occ_func_2_0(61) * occ_func_3_0(60) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(59) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(58) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(64) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(56) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(61) +
          occ_func_2_0(59) * occ_func_3_0(58) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(63) +
          occ_func_2_0(57) * occ_func_3_0(96) * occ_func_2_0(1) +
          occ_func_2_0(65) * occ_func_3_0(112) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(55) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(61) +
          occ_func_2_0(59) * occ_func_3_0(100) * occ_func_2_0(1) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(112) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(96) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(63) +
          occ_func_2_0(57) * occ_func_3_0(56) * occ_func_2_0(1) +
          occ_func_2_0(65) * occ_func_3_0(64) * occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(55) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(100)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_56_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(1) * occ_func_2_0(60) +
          occ_func_2_0(58) * occ_func_3_0(59) +
          occ_func_2_0(15) * occ_func_2_0(95) +
          occ_func_2_0(15) * occ_func_2_0(111) +
          occ_func_2_0(64) * occ_func_3_0(65) +
          occ_func_3_0(1) * occ_func_2_0(54) +
          occ_func_3_0(1) * occ_func_2_0(62) +
          occ_func_2_0(56) * occ_func_3_0(57) +
          occ_func_3_0(15) * occ_func_2_0(60) +
          occ_func_2_0(58) * occ_func_3_0(99) +
          occ_func_2_0(64) * occ_func_3_0(111) +
          occ_func_3_0(15) * occ_func_2_0(54) +
          occ_func_3_0(15) * occ_func_2_0(62) +
          occ_func_2_0(56) * occ_func_3_0(95) +
          occ_func_2_0(15) * occ_func_2_0(99) +
          occ_func_2_0(1) * occ_func_2_0(57) +
          occ_func_2_0(1) * occ_func_2_0(65) +
          occ_func_2_0(1) * occ_func_2_0(59) +
          occ_func_2_0(60) * occ_func_3_0(61) +
          occ_func_3_0(1) * occ_func_2_0(58) +
          occ_func_3_0(1) * occ_func_2_0(64) +
          occ_func_2_0(54) * occ_func_3_0(55) +
          occ_func_2_0(62) * occ_func_3_0(63) +
          occ_func_3_0(1) * occ_func_2_0(56) +
          occ_func_2_0(1) * occ_func_2_0(61) +
          occ_func_2_0(15) * occ_func_2_0(107) +
          occ_func_2_0(15) * occ_func_2_0(91) +
          occ_func_2_0(15) * occ_func_2_0(103) +
          occ_func_3_0(15) * occ_func_2_0(64) +
          occ_func_2_0(54) * occ_func_3_0(91) +
          occ_func_2_0(62) * occ_func_3_0(107) +
          occ_func_3_0(15) * occ_func_2_0(56) +
          occ_func_2_0(1) * occ_func_2_0(63) +
          occ_func_2_0(1) * occ_func_2_0(55) +
          occ_func_2_0(60) * occ_func_3_0(103) +
          occ_func_3_0(15) * occ_func_2_0(58)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_56_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(0) * occ_func_2_0(60) +
          occ_func_2_0(63) * occ_func_3_0(108) +
          occ_func_3_0(16) * occ_func_2_0(57) +
          occ_func_3_0(16) * occ_func_2_0(65) +
          occ_func_2_0(55) * occ_func_3_0(92) +
          occ_func_2_0(0) * occ_func_2_0(54) +
          occ_func_2_0(0) * occ_func_2_0(62) +
          occ_func_2_0(16) * occ_func_2_0(104) +
          occ_func_2_0(16) * occ_func_2_0(92) +
          occ_func_2_0(16) * occ_func_2_0(108) +
          occ_func_2_0(61) * occ_func_3_0(104) +
          occ_func_3_0(16) * occ_func_2_0(59) +
          occ_func_2_0(63) * occ_func_3_0(62) +
          occ_func_3_0(0) * occ_func_2_0(57) +
          occ_func_3_0(0) * occ_func_2_0(65) +
          occ_func_2_0(55) * occ_func_3_0(54) +
          occ_func_2_0(61) * occ_func_3_0(60) +
          occ_func_3_0(0) * occ_func_2_0(59) +
          occ_func_2_0(0) * occ_func_2_0(58) +
          occ_func_2_0(0) * occ_func_2_0(64) +
          occ_func_2_0(0) * occ_func_2_0(56) +
          occ_func_3_0(0) * occ_func_2_0(61) +
          occ_func_2_0(59) * occ_func_3_0(58) +
          occ_func_3_0(16) * occ_func_2_0(63) +
          occ_func_2_0(57) * occ_func_3_0(96) +
          occ_func_2_0(65) * occ_func_3_0(112) +
          occ_func_3_0(16) * occ_func_2_0(55) +
          occ_func_3_0(16) * occ_func_2_0(61) +
          occ_func_2_0(59) * occ_func_3_0(100) +
          occ_func_2_0(16) * occ_func_2_0(112) +
          occ_func_2_0(16) * occ_func_2_0(96) +
          occ_func_3_0(0) * occ_func_2_0(63) +
          occ_func_2_0(57) * occ_func_3_0(56) +
          occ_func_2_0(65) * occ_func_3_0(64) +
          occ_func_3_0(0) * occ_func_2_0(55) +
          occ_func_2_0(16) * occ_func_2_0(100)) /
         24.;
}

/**** Basis functions for orbit 57****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 1.2500000 Va  O

0.3333333 2.6666666 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_57_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(61) +
          occ_func_2_0(107) * occ_func_2_0(63) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(112) +
          occ_func_2_0(64) * occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(63) +
          occ_func_2_0(103) * occ_func_2_0(61) * occ_func_3_0(0) +
          occ_func_2_0(60) * occ_func_2_0(104) * occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(65) +
          occ_func_2_0(62) * occ_func_2_0(108) * occ_func_3_0(1) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_3_0(104) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(108) +
          occ_func_2_0(111) * occ_func_2_0(65) * occ_func_3_0(0)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_57_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(61) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(103) +
          occ_func_2_0(107) * occ_func_2_0(63) * occ_func_3_0(0) +
          occ_func_2_0(91) * occ_func_2_0(55) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(55) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(91) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(63) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(107) +
          occ_func_2_0(103) * occ_func_2_0(61) * occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(59) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(99) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(65) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(111) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(57) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(95) +
          occ_func_2_0(59) * occ_func_2_0(99) * occ_func_3_0(0) +
          occ_func_2_0(95) * occ_func_2_0(57) * occ_func_3_0(0) +
          occ_func_2_0(111) * occ_func_2_0(65) * occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_57_0_at_1() const {
  return (occ_func_2_0(58) * occ_func_2_0(100) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(96) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(56) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(112) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(64) +
          occ_func_2_0(64) * occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(56) * occ_func_2_0(96) * occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(100) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(58) +
          occ_func_2_0(60) * occ_func_2_0(104) * occ_func_3_0(1) +
          occ_func_2_0(54) * occ_func_2_0(92) * occ_func_3_0(1) +
          occ_func_2_0(62) * occ_func_2_0(108) * occ_func_3_0(1) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_3_0(104) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_3_0(60) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(108) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(62) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(92) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(54)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_57_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(16) * occ_func_3_0(61) +
          occ_func_2_0(14) * occ_func_3_0(103) +
          occ_func_2_0(107) * occ_func_2_0(63) +
          occ_func_2_0(91) * occ_func_2_0(55) +
          occ_func_2_0(16) * occ_func_3_0(55) +
          occ_func_2_0(14) * occ_func_3_0(91) +
          occ_func_2_0(16) * occ_func_3_0(63) +
          occ_func_2_0(14) * occ_func_3_0(107) +
          occ_func_2_0(103) * occ_func_2_0(61) +
          occ_func_2_0(16) * occ_func_3_0(59) +
          occ_func_2_0(14) * occ_func_3_0(99) +
          occ_func_2_0(16) * occ_func_3_0(65) +
          occ_func_2_0(14) * occ_func_3_0(111) +
          occ_func_2_0(16) * occ_func_3_0(57) +
          occ_func_2_0(14) * occ_func_3_0(95) +
          occ_func_2_0(59) * occ_func_2_0(99) +
          occ_func_2_0(95) * occ_func_2_0(57) +
          occ_func_2_0(111) * occ_func_2_0(65)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_57_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(58) * occ_func_2_0(100) +
          occ_func_2_0(17) * occ_func_3_0(96) +
          occ_func_2_0(15) * occ_func_3_0(56) +
          occ_func_2_0(17) * occ_func_3_0(112) +
          occ_func_2_0(15) * occ_func_3_0(64) +
          occ_func_2_0(64) * occ_func_2_0(112) +
          occ_func_2_0(56) * occ_func_2_0(96) +
          occ_func_2_0(17) * occ_func_3_0(100) +
          occ_func_2_0(15) * occ_func_3_0(58) +
          occ_func_2_0(60) * occ_func_2_0(104) +
          occ_func_2_0(54) * occ_func_2_0(92) +
          occ_func_2_0(62) * occ_func_2_0(108) +
          occ_func_2_0(17) * occ_func_3_0(104) +
          occ_func_2_0(15) * occ_func_3_0(60) +
          occ_func_2_0(17) * occ_func_3_0(108) +
          occ_func_2_0(15) * occ_func_3_0(62) +
          occ_func_2_0(17) * occ_func_3_0(92) +
          occ_func_2_0(15) * occ_func_3_0(54)) /
         12.;
}

/**** Basis functions for orbit 58****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 2.6666666 0.2500000 Va  O

2.3333333 2.6666666 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_58_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(60) * occ_func_3_0(111) +
          occ_func_2_0(63) * occ_func_2_0(1) * occ_func_3_0(64) +
          occ_func_2_0(1) * occ_func_2_0(65) * occ_func_3_0(62) +
          occ_func_2_0(64) * occ_func_2_0(0) * occ_func_3_0(103) +
          occ_func_2_0(104) * occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(62) * occ_func_2_0(64) * occ_func_3_0(1) +
          occ_func_2_0(64) * occ_func_2_0(0) * occ_func_3_0(63) +
          occ_func_2_0(0) * occ_func_2_0(62) * occ_func_3_0(65) +
          occ_func_2_0(65) * occ_func_2_0(63) * occ_func_3_0(0) +
          occ_func_2_0(111) * occ_func_2_0(103) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(65) * occ_func_3_0(104) +
          occ_func_2_0(61) * occ_func_2_0(1) * occ_func_3_0(112)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_58_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(60) * occ_func_3_0(111) +
          occ_func_2_0(58) * occ_func_2_0(0) * occ_func_3_0(107) +
          occ_func_2_0(59) * occ_func_2_0(55) * occ_func_3_0(0) +
          occ_func_2_0(57) * occ_func_2_0(61) * occ_func_3_0(0) +
          occ_func_2_0(64) * occ_func_2_0(0) * occ_func_3_0(103) +
          occ_func_2_0(0) * occ_func_2_0(54) * occ_func_3_0(95) +
          occ_func_2_0(0) * occ_func_2_0(62) * occ_func_3_0(99) +
          occ_func_2_0(56) * occ_func_2_0(0) * occ_func_3_0(91) +
          occ_func_2_0(0) * occ_func_2_0(60) * occ_func_3_0(57) +
          occ_func_2_0(58) * occ_func_2_0(0) * occ_func_3_0(55) +
          occ_func_2_0(64) * occ_func_2_0(0) * occ_func_3_0(63) +
          occ_func_2_0(0) * occ_func_2_0(54) * occ_func_3_0(59) +
          occ_func_2_0(0) * occ_func_2_0(62) * occ_func_3_0(65) +
          occ_func_2_0(56) * occ_func_2_0(0) * occ_func_3_0(61) +
          occ_func_2_0(65) * occ_func_2_0(63) * occ_func_3_0(0) +
          occ_func_2_0(111) * occ_func_2_0(103) * occ_func_3_0(0) +
          occ_func_2_0(99) * occ_func_2_0(107) * occ_func_3_0(0) +
          occ_func_2_0(95) * occ_func_2_0(91) * occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_58_0_at_1() const {
  return (occ_func_2_0(92) * occ_func_2_0(96) * occ_func_3_0(1) +
          occ_func_2_0(63) * occ_func_2_0(1) * occ_func_3_0(64) +
          occ_func_2_0(1) * occ_func_2_0(57) * occ_func_3_0(60) +
          occ_func_2_0(1) * occ_func_2_0(65) * occ_func_3_0(62) +
          occ_func_2_0(55) * occ_func_2_0(1) * occ_func_3_0(58) +
          occ_func_2_0(108) * occ_func_2_0(100) * occ_func_3_0(1) +
          occ_func_2_0(104) * occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(62) * occ_func_2_0(64) * occ_func_3_0(1) +
          occ_func_2_0(60) * occ_func_2_0(56) * occ_func_3_0(1) +
          occ_func_2_0(54) * occ_func_2_0(58) * occ_func_3_0(1) +
          occ_func_2_0(61) * occ_func_2_0(1) * occ_func_3_0(56) +
          occ_func_2_0(1) * occ_func_2_0(59) * occ_func_3_0(54) +
          occ_func_2_0(63) * occ_func_2_0(1) * occ_func_3_0(100) +
          occ_func_2_0(1) * occ_func_2_0(57) * occ_func_3_0(92) +
          occ_func_2_0(1) * occ_func_2_0(65) * occ_func_3_0(104) +
          occ_func_2_0(55) * occ_func_2_0(1) * occ_func_3_0(96) +
          occ_func_2_0(61) * occ_func_2_0(1) * occ_func_3_0(112) +
          occ_func_2_0(1) * occ_func_2_0(59) * occ_func_3_0(108)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_58_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(60) * occ_func_3_0(111) +
          occ_func_2_0(58) * occ_func_3_0(107) +
          occ_func_2_0(59) * occ_func_2_0(55) +
          occ_func_2_0(57) * occ_func_2_0(61) +
          occ_func_2_0(64) * occ_func_3_0(103) +
          occ_func_2_0(54) * occ_func_3_0(95) +
          occ_func_2_0(62) * occ_func_3_0(99) +
          occ_func_2_0(56) * occ_func_3_0(91) +
          occ_func_2_0(60) * occ_func_3_0(57) +
          occ_func_2_0(58) * occ_func_3_0(55) +
          occ_func_2_0(64) * occ_func_3_0(63) +
          occ_func_2_0(54) * occ_func_3_0(59) +
          occ_func_2_0(62) * occ_func_3_0(65) +
          occ_func_2_0(56) * occ_func_3_0(61) +
          occ_func_2_0(65) * occ_func_2_0(63) +
          occ_func_2_0(111) * occ_func_2_0(103) +
          occ_func_2_0(99) * occ_func_2_0(107) +
          occ_func_2_0(95) * occ_func_2_0(91)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_58_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(92) * occ_func_2_0(96) +
          occ_func_2_0(63) * occ_func_3_0(64) +
          occ_func_2_0(57) * occ_func_3_0(60) +
          occ_func_2_0(65) * occ_func_3_0(62) +
          occ_func_2_0(55) * occ_func_3_0(58) +
          occ_func_2_0(108) * occ_func_2_0(100) +
          occ_func_2_0(104) * occ_func_2_0(112) +
          occ_func_2_0(62) * occ_func_2_0(64) +
          occ_func_2_0(60) * occ_func_2_0(56) +
          occ_func_2_0(54) * occ_func_2_0(58) +
          occ_func_2_0(61) * occ_func_3_0(56) +
          occ_func_2_0(59) * occ_func_3_0(54) +
          occ_func_2_0(63) * occ_func_3_0(100) +
          occ_func_2_0(57) * occ_func_3_0(92) +
          occ_func_2_0(65) * occ_func_3_0(104) +
          occ_func_2_0(55) * occ_func_3_0(96) +
          occ_func_2_0(61) * occ_func_3_0(112) +
          occ_func_2_0(59) * occ_func_3_0(108)) /
         12.;
}

/**** Basis functions for orbit 59****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 2.6666666 0.2500000 Va  O

2.3333333 2.6666666 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_59_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(60) * occ_func_3_0(65) +
          occ_func_2_0(63) * occ_func_2_0(1) * occ_func_3_0(112) +
          occ_func_2_0(1) * occ_func_2_0(65) * occ_func_3_0(108) +
          occ_func_2_0(64) * occ_func_2_0(0) * occ_func_3_0(61) +
          occ_func_2_0(60) * occ_func_2_0(64) * occ_func_3_0(1) +
          occ_func_2_0(108) * occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(64) * occ_func_2_0(0) * occ_func_3_0(107) +
          occ_func_2_0(0) * occ_func_2_0(62) * occ_func_3_0(111) +
          occ_func_2_0(111) * occ_func_2_0(107) * occ_func_3_0(0) +
          occ_func_2_0(65) * occ_func_2_0(61) * occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(65) * occ_func_3_0(60) +
          occ_func_2_0(61) * occ_func_2_0(1) * occ_func_3_0(64)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_59_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(60) * occ_func_3_0(65) +
          occ_func_2_0(58) * occ_func_2_0(0) * occ_func_3_0(63) +
          occ_func_2_0(99) * occ_func_2_0(91) * occ_func_3_0(0) +
          occ_func_2_0(95) * occ_func_2_0(103) * occ_func_3_0(0) +
          occ_func_2_0(64) * occ_func_2_0(0) * occ_func_3_0(61) +
          occ_func_2_0(0) * occ_func_2_0(54) * occ_func_3_0(57) +
          occ_func_2_0(0) * occ_func_2_0(62) * occ_func_3_0(59) +
          occ_func_2_0(56) * occ_func_2_0(0) * occ_func_3_0(55) +
          occ_func_2_0(0) * occ_func_2_0(60) * occ_func_3_0(95) +
          occ_func_2_0(58) * occ_func_2_0(0) * occ_func_3_0(91) +
          occ_func_2_0(64) * occ_func_2_0(0) * occ_func_3_0(107) +
          occ_func_2_0(0) * occ_func_2_0(54) * occ_func_3_0(99) +
          occ_func_2_0(0) * occ_func_2_0(62) * occ_func_3_0(111) +
          occ_func_2_0(56) * occ_func_2_0(0) * occ_func_3_0(103) +
          occ_func_2_0(111) * occ_func_2_0(107) * occ_func_3_0(0) +
          occ_func_2_0(65) * occ_func_2_0(61) * occ_func_3_0(0) +
          occ_func_2_0(59) * occ_func_2_0(63) * occ_func_3_0(0) +
          occ_func_2_0(57) * occ_func_2_0(55) * occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_59_0_at_1() const {
  return (occ_func_2_0(54) * occ_func_2_0(56) * occ_func_3_0(1) +
          occ_func_2_0(63) * occ_func_2_0(1) * occ_func_3_0(112) +
          occ_func_2_0(1) * occ_func_2_0(57) * occ_func_3_0(104) +
          occ_func_2_0(1) * occ_func_2_0(65) * occ_func_3_0(108) +
          occ_func_2_0(55) * occ_func_2_0(1) * occ_func_3_0(100) +
          occ_func_2_0(62) * occ_func_2_0(58) * occ_func_3_0(1) +
          occ_func_2_0(60) * occ_func_2_0(64) * occ_func_3_0(1) +
          occ_func_2_0(108) * occ_func_2_0(112) * occ_func_3_0(1) +
          occ_func_2_0(104) * occ_func_2_0(96) * occ_func_3_0(1) +
          occ_func_2_0(92) * occ_func_2_0(100) * occ_func_3_0(1) +
          occ_func_2_0(61) * occ_func_2_0(1) * occ_func_3_0(96) +
          occ_func_2_0(1) * occ_func_2_0(59) * occ_func_3_0(92) +
          occ_func_2_0(63) * occ_func_2_0(1) * occ_func_3_0(58) +
          occ_func_2_0(1) * occ_func_2_0(57) * occ_func_3_0(54) +
          occ_func_2_0(1) * occ_func_2_0(65) * occ_func_3_0(60) +
          occ_func_2_0(55) * occ_func_2_0(1) * occ_func_3_0(56) +
          occ_func_2_0(61) * occ_func_2_0(1) * occ_func_3_0(64) +
          occ_func_2_0(1) * occ_func_2_0(59) * occ_func_3_0(62)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_59_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(60) * occ_func_3_0(65) +
          occ_func_2_0(58) * occ_func_3_0(63) +
          occ_func_2_0(99) * occ_func_2_0(91) +
          occ_func_2_0(95) * occ_func_2_0(103) +
          occ_func_2_0(64) * occ_func_3_0(61) +
          occ_func_2_0(54) * occ_func_3_0(57) +
          occ_func_2_0(62) * occ_func_3_0(59) +
          occ_func_2_0(56) * occ_func_3_0(55) +
          occ_func_2_0(60) * occ_func_3_0(95) +
          occ_func_2_0(58) * occ_func_3_0(91) +
          occ_func_2_0(64) * occ_func_3_0(107) +
          occ_func_2_0(54) * occ_func_3_0(99) +
          occ_func_2_0(62) * occ_func_3_0(111) +
          occ_func_2_0(56) * occ_func_3_0(103) +
          occ_func_2_0(111) * occ_func_2_0(107) +
          occ_func_2_0(65) * occ_func_2_0(61) +
          occ_func_2_0(59) * occ_func_2_0(63) +
          occ_func_2_0(57) * occ_func_2_0(55)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_59_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(54) * occ_func_2_0(56) +
          occ_func_2_0(63) * occ_func_3_0(112) +
          occ_func_2_0(57) * occ_func_3_0(104) +
          occ_func_2_0(65) * occ_func_3_0(108) +
          occ_func_2_0(55) * occ_func_3_0(100) +
          occ_func_2_0(62) * occ_func_2_0(58) +
          occ_func_2_0(60) * occ_func_2_0(64) +
          occ_func_2_0(108) * occ_func_2_0(112) +
          occ_func_2_0(104) * occ_func_2_0(96) +
          occ_func_2_0(92) * occ_func_2_0(100) +
          occ_func_2_0(61) * occ_func_3_0(96) +
          occ_func_2_0(59) * occ_func_3_0(92) +
          occ_func_2_0(63) * occ_func_3_0(58) +
          occ_func_2_0(57) * occ_func_3_0(54) +
          occ_func_2_0(65) * occ_func_3_0(60) +
          occ_func_2_0(55) * occ_func_3_0(56) +
          occ_func_2_0(61) * occ_func_3_0(64) +
          occ_func_2_0(59) * occ_func_3_0(62)) /
         12.;
}

/**** Basis functions for orbit 60****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 0.7500000 Va  O

1.3333333 1.6666667 0.2500000 Va  O

1.3333333 1.6666667 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_60_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(12) *
              occ_func_3_0(13) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(9) *
              occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(11) *
              occ_func_3_0(48) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(52) *
              occ_func_3_0(13)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_60_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(12) *
              occ_func_3_0(13) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(43) *
              occ_func_3_0(8) +
          occ_func_2_0(39) * occ_func_3_0(6) * occ_func_2_0(15) *
              occ_func_3_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(47) *
              occ_func_3_0(10) +
          occ_func_2_0(35) * occ_func_3_0(4) * occ_func_2_0(15) *
              occ_func_3_0(0) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(4) *
              occ_func_3_0(5) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(6) *
              occ_func_3_0(7) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(12) *
              occ_func_3_0(51) +
          occ_func_2_0(2) * occ_func_3_0(31) * occ_func_2_0(0) *
              occ_func_3_0(15)) /
         6.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_60_0_at_1() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(12) *
              occ_func_3_0(13) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(9) *
              occ_func_3_0(44) +
          occ_func_2_0(7) * occ_func_3_0(40) * occ_func_2_0(1) *
              occ_func_3_0(16) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(11) *
              occ_func_3_0(48) +
          occ_func_2_0(5) * occ_func_3_0(36) * occ_func_2_0(1) *
              occ_func_3_0(16) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(4) *
              occ_func_3_0(5) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(6) *
              occ_func_3_0(7) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(52) *
              occ_func_3_0(13) +
          occ_func_2_0(32) * occ_func_3_0(3) * occ_func_2_0(16) *
              occ_func_3_0(1)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_60_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(1) * occ_func_2_0(12) * occ_func_3_0(13) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_3_0(1) +
          occ_func_2_0(15) * occ_func_2_0(43) * occ_func_3_0(8) +
          occ_func_2_0(39) * occ_func_3_0(6) * occ_func_2_0(15) +
          occ_func_2_0(15) * occ_func_2_0(47) * occ_func_3_0(10) +
          occ_func_2_0(35) * occ_func_3_0(4) * occ_func_2_0(15) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_3_0(1) +
          occ_func_3_0(1) * occ_func_2_0(4) * occ_func_3_0(5) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_3_0(1) +
          occ_func_3_0(1) * occ_func_2_0(6) * occ_func_3_0(7) +
          occ_func_3_0(15) * occ_func_2_0(12) * occ_func_3_0(51) +
          occ_func_2_0(2) * occ_func_3_0(31) * occ_func_3_0(15)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_60_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(0) * occ_func_2_0(12) * occ_func_3_0(13) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_2_0(0) +
          occ_func_3_0(16) * occ_func_2_0(9) * occ_func_3_0(44) +
          occ_func_2_0(7) * occ_func_3_0(40) * occ_func_3_0(16) +
          occ_func_3_0(16) * occ_func_2_0(11) * occ_func_3_0(48) +
          occ_func_2_0(5) * occ_func_3_0(36) * occ_func_3_0(16) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(4) * occ_func_3_0(5) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(6) * occ_func_3_0(7) +
          occ_func_2_0(16) * occ_func_2_0(52) * occ_func_3_0(13) +
          occ_func_2_0(32) * occ_func_3_0(3) * occ_func_2_0(16)) /
         6.;
}

/**** Basis functions for orbit 61****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 0.7500000 Va  O

1.3333333 0.6666666 0.2500000 Va  O

1.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_61_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(10) *
              occ_func_2_0(12) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(13) *
              occ_func_2_0(9) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_2_0(1) *
              occ_func_2_0(13) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(12) *
              occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(0) *
              occ_func_2_0(10) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_2_0(0) *
              occ_func_2_0(8) +
          occ_func_2_0(8) * occ_func_3_0(43) * occ_func_2_0(12) *
              occ_func_2_0(0) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(44) *
              occ_func_2_0(52) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_2_0(9) *
              occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_3_0(12) * occ_func_2_0(11) *
              occ_func_2_0(1) +
          occ_func_2_0(11) * occ_func_3_0(10) * occ_func_2_0(1) *
              occ_func_2_0(13) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(13) *
              occ_func_2_0(11)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_61_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(10) *
              occ_func_2_0(12) +
          occ_func_2_0(4) * occ_func_3_0(5) * occ_func_2_0(0) *
              occ_func_2_0(8) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_2_0(6) *
              occ_func_2_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(51) *
              occ_func_2_0(43) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(39) *
              occ_func_2_0(47) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(12) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(8) *
              occ_func_2_0(4) +
          occ_func_2_0(6) * occ_func_3_0(7) * occ_func_2_0(0) *
              occ_func_2_0(2) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(0) *
              occ_func_2_0(10) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(4) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(2) *
              occ_func_2_0(6) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_2_0(0) *
              occ_func_2_0(8) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_2_0(6) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(2) *
              occ_func_2_0(4) +
          occ_func_2_0(8) * occ_func_3_0(43) * occ_func_2_0(12) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(10) *
              occ_func_2_0(6) +
          occ_func_2_0(4) * occ_func_3_0(35) * occ_func_2_0(0) *
              occ_func_2_0(2) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(8) *
              occ_func_2_0(12) +
          occ_func_2_0(6) * occ_func_3_0(39) * occ_func_2_0(0) *
              occ_func_2_0(10) +
          occ_func_2_0(2) * occ_func_3_0(31) * occ_func_2_0(4) *
              occ_func_2_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(35) *
              occ_func_2_0(31) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(7) *
              occ_func_2_0(3) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(5) *
              occ_func_2_0(9) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(13) *
              occ_func_2_0(11)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_61_0_at_1() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(10) *
              occ_func_2_0(12) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(13) *
              occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(40) * occ_func_2_0(11) *
              occ_func_2_0(1) +
          occ_func_2_0(3) * occ_func_3_0(32) * occ_func_2_0(1) *
              occ_func_2_0(5) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_2_0(1) *
              occ_func_2_0(13) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(7) *
              occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(36) * occ_func_2_0(3) *
              occ_func_2_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(8) *
              occ_func_2_0(4) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(2) *
              occ_func_2_0(6) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(32) *
              occ_func_2_0(36) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(48) *
              occ_func_2_0(40) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(44) *
              occ_func_2_0(52) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_2_0(9) *
              occ_func_2_0(1) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_2_0(1) *
              occ_func_2_0(7) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(5) *
              occ_func_2_0(3) +
          occ_func_2_0(13) * occ_func_3_0(12) * occ_func_2_0(11) *
              occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_3_0(8) * occ_func_2_0(1) *
              occ_func_2_0(5) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(7) *
              occ_func_2_0(3) +
          occ_func_2_0(11) * occ_func_3_0(10) * occ_func_2_0(1) *
              occ_func_2_0(13) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(5) *
              occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(6) * occ_func_2_0(3) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(13) *
              occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(4) * occ_func_2_0(9) *
              occ_func_2_0(1) +
          occ_func_2_0(3) * occ_func_3_0(2) * occ_func_2_0(1) *
              occ_func_2_0(7)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_61_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(1) * occ_func_2_0(10) * occ_func_2_0(12) +
          occ_func_2_0(4) * occ_func_3_0(5) * occ_func_2_0(8) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_2_0(6) +
          occ_func_2_0(15) * occ_func_2_0(51) * occ_func_2_0(43) +
          occ_func_2_0(15) * occ_func_2_0(39) * occ_func_2_0(47) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(12) +
          occ_func_3_0(1) * occ_func_2_0(8) * occ_func_2_0(4) +
          occ_func_2_0(6) * occ_func_3_0(7) * occ_func_2_0(2) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(10) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(4) +
          occ_func_3_0(1) * occ_func_2_0(2) * occ_func_2_0(6) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_2_0(8) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_2_0(6) +
          occ_func_3_0(15) * occ_func_2_0(2) * occ_func_2_0(4) +
          occ_func_2_0(8) * occ_func_3_0(43) * occ_func_2_0(12) +
          occ_func_3_0(15) * occ_func_2_0(10) * occ_func_2_0(6) +
          occ_func_2_0(4) * occ_func_3_0(35) * occ_func_2_0(2) +
          occ_func_3_0(15) * occ_func_2_0(8) * occ_func_2_0(12) +
          occ_func_2_0(6) * occ_func_3_0(39) * occ_func_2_0(10) +
          occ_func_2_0(2) * occ_func_3_0(31) * occ_func_2_0(4) +
          occ_func_2_0(15) * occ_func_2_0(35) * occ_func_2_0(31) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(3) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(9) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(11)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_61_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(12) +
          occ_func_3_0(16) * occ_func_2_0(13) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(40) * occ_func_2_0(11) +
          occ_func_2_0(3) * occ_func_3_0(32) * occ_func_2_0(5) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_2_0(13) +
          occ_func_3_0(16) * occ_func_2_0(7) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(36) * occ_func_2_0(3) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(4) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(6) +
          occ_func_2_0(16) * occ_func_2_0(32) * occ_func_2_0(36) +
          occ_func_2_0(16) * occ_func_2_0(48) * occ_func_2_0(40) +
          occ_func_2_0(16) * occ_func_2_0(44) * occ_func_2_0(52) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_2_0(9) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_2_0(7) +
          occ_func_3_0(16) * occ_func_2_0(5) * occ_func_2_0(3) +
          occ_func_2_0(13) * occ_func_3_0(12) * occ_func_2_0(11) +
          occ_func_2_0(9) * occ_func_3_0(8) * occ_func_2_0(5) +
          occ_func_3_0(0) * occ_func_2_0(7) * occ_func_2_0(3) +
          occ_func_2_0(11) * occ_func_3_0(10) * occ_func_2_0(13) +
          occ_func_3_0(0) * occ_func_2_0(5) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(6) * occ_func_2_0(3) +
          occ_func_3_0(0) * occ_func_2_0(13) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(4) * occ_func_2_0(9) +
          occ_func_2_0(3) * occ_func_3_0(2) * occ_func_2_0(7)) /
         12.;
}

/**** Basis functions for orbit 62****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

1.3333333 1.6666667 0.2500000 Va  O

1.3333333 1.6666667 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_62_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(12) *
              occ_func_3_0(13) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(13) *
              occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(11) *
              occ_func_3_0(48) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(8) *
              occ_func_3_0(9) +
          occ_func_2_0(8) * occ_func_2_0(12) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_2_0(52) * occ_func_2_0(16) *
              occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(10) *
              occ_func_3_0(47) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(12) *
              occ_func_3_0(51) +
          occ_func_2_0(13) * occ_func_2_0(11) * occ_func_2_0(1) *
              occ_func_3_0(16) +
          occ_func_2_0(13) * occ_func_2_0(9) * occ_func_2_0(1) *
              occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(9) *
              occ_func_3_0(8) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(13) *
              occ_func_3_0(12)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_62_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(12) *
              occ_func_3_0(13) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(10) *
              occ_func_3_0(11) +
          occ_func_2_0(2) * occ_func_2_0(4) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(39) * occ_func_2_0(31) * occ_func_2_0(15) *
              occ_func_3_0(0) +
          occ_func_2_0(35) * occ_func_2_0(43) * occ_func_2_0(15) *
              occ_func_3_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(8) *
              occ_func_3_0(9) +
          occ_func_2_0(10) * occ_func_2_0(6) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(4) *
              occ_func_3_0(5) +
          occ_func_2_0(8) * occ_func_2_0(12) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(6) *
              occ_func_3_0(7) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_2_0(2) *
              occ_func_3_0(3) +
          occ_func_2_0(10) * occ_func_2_0(12) * occ_func_2_0(0) *
              occ_func_3_0(15) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(4) *
              occ_func_3_0(35) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(2) *
              occ_func_3_0(31) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(10) *
              occ_func_3_0(47) +
          occ_func_2_0(8) * occ_func_2_0(4) * occ_func_2_0(0) *
              occ_func_3_0(15) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(6) *
              occ_func_3_0(39) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(12) *
              occ_func_3_0(51) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_2_0(8) *
              occ_func_3_0(43) +
          occ_func_2_0(2) * occ_func_2_0(6) * occ_func_2_0(0) *
              occ_func_3_0(15) +
          occ_func_2_0(51) * occ_func_2_0(47) * occ_func_2_0(15) *
              occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_2_0(9) * occ_func_2_0(1) *
              occ_func_3_0(0) +
          occ_func_2_0(7) * occ_func_2_0(11) * occ_func_2_0(1) *
              occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_2_0(3) * occ_func_2_0(1) *
              occ_func_3_0(0)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_62_0_at_1() const {
  return (occ_func_2_0(2) * occ_func_2_0(4) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(13) *
              occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(9) *
              occ_func_3_0(44) +
          occ_func_2_0(7) * occ_func_2_0(3) * occ_func_2_0(1) *
              occ_func_3_0(16) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(11) *
              occ_func_3_0(48) +
          occ_func_2_0(5) * occ_func_2_0(9) * occ_func_2_0(1) *
              occ_func_3_0(16) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_2_0(7) *
              occ_func_3_0(40) +
          occ_func_2_0(10) * occ_func_2_0(6) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(8) * occ_func_2_0(12) * occ_func_2_0(0) *
              occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_2_0(52) * occ_func_2_0(16) *
              occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_2_0(36) * occ_func_2_0(16) *
              occ_func_3_0(1) +
          occ_func_2_0(32) * occ_func_2_0(40) * occ_func_2_0(16) *
              occ_func_3_0(1) +
          occ_func_2_0(13) * occ_func_2_0(11) * occ_func_2_0(1) *
              occ_func_3_0(16) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(5) *
              occ_func_3_0(36) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(3) *
              occ_func_3_0(32) +
          occ_func_2_0(13) * occ_func_2_0(9) * occ_func_2_0(1) *
              occ_func_3_0(0) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(7) *
              occ_func_3_0(6) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(3) *
              occ_func_3_0(2) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(9) *
              occ_func_3_0(8) +
          occ_func_2_0(7) * occ_func_2_0(11) * occ_func_2_0(1) *
              occ_func_3_0(0) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_2_0(5) *
              occ_func_3_0(4) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(13) *
              occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(11) *
              occ_func_3_0(10) +
          occ_func_2_0(5) * occ_func_2_0(3) * occ_func_2_0(1) *
              occ_func_3_0(0)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_62_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) * occ_func_2_0(12) * occ_func_3_0(13) +
          occ_func_2_0(6) * occ_func_2_0(10) * occ_func_3_0(11) +
          occ_func_2_0(2) * occ_func_2_0(4) * occ_func_3_0(1) +
          occ_func_2_0(39) * occ_func_2_0(31) * occ_func_2_0(15) +
          occ_func_2_0(35) * occ_func_2_0(43) * occ_func_2_0(15) +
          occ_func_2_0(12) * occ_func_2_0(8) * occ_func_3_0(9) +
          occ_func_2_0(10) * occ_func_2_0(6) * occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_2_0(4) * occ_func_3_0(5) +
          occ_func_2_0(8) * occ_func_2_0(12) * occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_2_0(6) * occ_func_3_0(7) +
          occ_func_2_0(4) * occ_func_2_0(2) * occ_func_3_0(3) +
          occ_func_2_0(10) * occ_func_2_0(12) * occ_func_3_0(15) +
          occ_func_2_0(8) * occ_func_2_0(4) * occ_func_3_0(35) +
          occ_func_2_0(6) * occ_func_2_0(2) * occ_func_3_0(31) +
          occ_func_2_0(12) * occ_func_2_0(10) * occ_func_3_0(47) +
          occ_func_2_0(8) * occ_func_2_0(4) * occ_func_3_0(15) +
          occ_func_2_0(2) * occ_func_2_0(6) * occ_func_3_0(39) +
          occ_func_2_0(10) * occ_func_2_0(12) * occ_func_3_0(51) +
          occ_func_2_0(4) * occ_func_2_0(8) * occ_func_3_0(43) +
          occ_func_2_0(2) * occ_func_2_0(6) * occ_func_3_0(15) +
          occ_func_2_0(51) * occ_func_2_0(47) * occ_func_2_0(15) +
          occ_func_2_0(13) * occ_func_2_0(9) * occ_func_2_0(1) +
          occ_func_2_0(7) * occ_func_2_0(11) * occ_func_2_0(1) +
          occ_func_2_0(5) * occ_func_2_0(3) * occ_func_2_0(1)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_62_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(2) * occ_func_2_0(4) * occ_func_2_0(0) +
          occ_func_2_0(11) * occ_func_2_0(13) * occ_func_3_0(52) +
          occ_func_2_0(5) * occ_func_2_0(9) * occ_func_3_0(44) +
          occ_func_2_0(7) * occ_func_2_0(3) * occ_func_3_0(16) +
          occ_func_2_0(13) * occ_func_2_0(11) * occ_func_3_0(48) +
          occ_func_2_0(5) * occ_func_2_0(9) * occ_func_3_0(16) +
          occ_func_2_0(3) * occ_func_2_0(7) * occ_func_3_0(40) +
          occ_func_2_0(10) * occ_func_2_0(6) * occ_func_2_0(0) +
          occ_func_2_0(8) * occ_func_2_0(12) * occ_func_2_0(0) +
          occ_func_2_0(48) * occ_func_2_0(52) * occ_func_2_0(16) +
          occ_func_2_0(44) * occ_func_2_0(36) * occ_func_2_0(16) +
          occ_func_2_0(32) * occ_func_2_0(40) * occ_func_2_0(16) +
          occ_func_2_0(13) * occ_func_2_0(11) * occ_func_3_0(16) +
          occ_func_2_0(9) * occ_func_2_0(5) * occ_func_3_0(36) +
          occ_func_2_0(7) * occ_func_2_0(3) * occ_func_3_0(32) +
          occ_func_2_0(13) * occ_func_2_0(9) * occ_func_3_0(0) +
          occ_func_2_0(11) * occ_func_2_0(7) * occ_func_3_0(6) +
          occ_func_2_0(5) * occ_func_2_0(3) * occ_func_3_0(2) +
          occ_func_2_0(13) * occ_func_2_0(9) * occ_func_3_0(8) +
          occ_func_2_0(7) * occ_func_2_0(11) * occ_func_3_0(0) +
          occ_func_2_0(3) * occ_func_2_0(5) * occ_func_3_0(4) +
          occ_func_2_0(9) * occ_func_2_0(13) * occ_func_3_0(12) +
          occ_func_2_0(7) * occ_func_2_0(11) * occ_func_3_0(10) +
          occ_func_2_0(5) * occ_func_2_0(3) * occ_func_3_0(0)) /
         12.;
}

/**** Basis functions for orbit 63****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

1.3333333 1.6666667 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_63_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_2_0(8) *
              occ_func_3_0(51) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_2_0(13) *
              occ_func_3_0(10) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(9) +
          occ_func_2_0(44) * occ_func_3_0(13) * occ_func_2_0(52) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_2_0(10) *
              occ_func_3_0(13) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(13) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_2_0(12) *
              occ_func_3_0(43) +
          occ_func_2_0(52) * occ_func_3_0(9) * occ_func_2_0(44) *
              occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_2_0(11) *
              occ_func_3_0(12) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(10) +
          occ_func_2_0(11) * occ_func_3_0(12) * occ_func_2_0(13) *
              occ_func_3_0(0)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_63_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_2_0(8) *
              occ_func_3_0(51) +
          occ_func_2_0(6) * occ_func_3_0(15) * occ_func_2_0(0) *
              occ_func_3_0(47) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(12) +
          occ_func_2_0(7) * occ_func_3_0(2) * occ_func_2_0(3) *
              occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_3_0(8) * occ_func_2_0(9) *
              occ_func_3_0(0) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(6) +
          occ_func_2_0(12) * occ_func_3_0(15) * occ_func_2_0(0) *
              occ_func_3_0(43) +
          occ_func_2_0(0) * occ_func_3_0(31) * occ_func_2_0(2) *
              occ_func_3_0(35) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_2_0(10) *
              occ_func_3_0(39) +
          occ_func_2_0(4) * occ_func_3_0(15) * occ_func_2_0(0) *
              occ_func_3_0(31) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_2_0(10) *
              occ_func_3_0(13) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_2_0(0) *
              occ_func_3_0(9) +
          occ_func_2_0(8) * occ_func_3_0(15) * occ_func_2_0(0) *
              occ_func_3_0(51) +
          occ_func_2_0(0) * occ_func_3_0(39) * occ_func_2_0(6) *
              occ_func_3_0(47) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_2_0(12) *
              occ_func_3_0(43) +
          occ_func_2_0(2) * occ_func_3_0(15) * occ_func_2_0(0) *
              occ_func_3_0(35) +
          occ_func_2_0(10) * occ_func_3_0(15) * occ_func_2_0(0) *
              occ_func_3_0(39) +
          occ_func_2_0(0) * occ_func_3_0(35) * occ_func_2_0(4) *
              occ_func_3_0(31) +
          occ_func_2_0(5) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(8) +
          occ_func_2_0(3) * occ_func_3_0(6) * occ_func_2_0(7) *
              occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(10) +
          occ_func_2_0(9) * occ_func_3_0(4) * occ_func_2_0(5) *
              occ_func_3_0(0) +
          occ_func_2_0(11) * occ_func_3_0(12) * occ_func_2_0(13) *
              occ_func_3_0(0) +
          occ_func_2_0(7) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(2)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_63_0_at_1() const {
  return (occ_func_2_0(40) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(11) +
          occ_func_2_0(32) * occ_func_3_0(5) * occ_func_2_0(36) *
              occ_func_3_0(1) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_3_0(4) * occ_func_2_0(5) *
              occ_func_3_0(8) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_2_0(13) *
              occ_func_3_0(10) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(6) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(9) +
          occ_func_2_0(48) * occ_func_3_0(7) * occ_func_2_0(40) *
              occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_3_0(13) * occ_func_2_0(52) *
              occ_func_3_0(1) +
          occ_func_2_0(36) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(3) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_2_0(0) *
              occ_func_3_0(9) +
          occ_func_2_0(2) * occ_func_3_0(7) * occ_func_2_0(6) *
              occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(13) +
          occ_func_2_0(36) * occ_func_3_0(3) * occ_func_2_0(32) *
              occ_func_3_0(1) +
          occ_func_2_0(40) * occ_func_3_0(11) * occ_func_2_0(48) *
              occ_func_3_0(1) +
          occ_func_2_0(32) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(5) +
          occ_func_2_0(52) * occ_func_3_0(9) * occ_func_2_0(44) *
              occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(7) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_2_0(11) *
              occ_func_3_0(12) +
          occ_func_2_0(5) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(8) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(10) +
          occ_func_2_0(1) * occ_func_3_0(2) * occ_func_2_0(3) *
              occ_func_3_0(6) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_2_0(9) *
              occ_func_3_0(4) +
          occ_func_2_0(7) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(2)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_63_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(43) * occ_func_2_0(8) * occ_func_3_0(51) +
          occ_func_2_0(6) * occ_func_3_0(15) * occ_func_3_0(47) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(12) +
          occ_func_2_0(7) * occ_func_3_0(2) * occ_func_2_0(3) +
          occ_func_2_0(5) * occ_func_3_0(8) * occ_func_2_0(9) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_3_0(6) +
          occ_func_2_0(12) * occ_func_3_0(15) * occ_func_3_0(43) +
          occ_func_3_0(31) * occ_func_2_0(2) * occ_func_3_0(35) +
          occ_func_3_0(47) * occ_func_2_0(10) * occ_func_3_0(39) +
          occ_func_2_0(4) * occ_func_3_0(15) * occ_func_3_0(31) +
          occ_func_3_0(11) * occ_func_2_0(10) * occ_func_3_0(13) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_3_0(9) +
          occ_func_2_0(8) * occ_func_3_0(15) * occ_func_3_0(51) +
          occ_func_3_0(39) * occ_func_2_0(6) * occ_func_3_0(47) +
          occ_func_3_0(51) * occ_func_2_0(12) * occ_func_3_0(43) +
          occ_func_2_0(2) * occ_func_3_0(15) * occ_func_3_0(35) +
          occ_func_2_0(10) * occ_func_3_0(15) * occ_func_3_0(39) +
          occ_func_3_0(35) * occ_func_2_0(4) * occ_func_3_0(31) +
          occ_func_2_0(5) * occ_func_2_0(1) * occ_func_3_0(8) +
          occ_func_2_0(3) * occ_func_3_0(6) * occ_func_2_0(7) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(10) +
          occ_func_2_0(9) * occ_func_3_0(4) * occ_func_2_0(5) +
          occ_func_2_0(11) * occ_func_3_0(12) * occ_func_2_0(13) +
          occ_func_2_0(7) * occ_func_2_0(1) * occ_func_3_0(2)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_63_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(40) * occ_func_2_0(16) * occ_func_3_0(11) +
          occ_func_2_0(32) * occ_func_3_0(5) * occ_func_2_0(36) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_3_0(12) +
          occ_func_3_0(4) * occ_func_2_0(5) * occ_func_3_0(8) +
          occ_func_3_0(12) * occ_func_2_0(13) * occ_func_3_0(10) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_3_0(6) +
          occ_func_2_0(52) * occ_func_2_0(16) * occ_func_3_0(9) +
          occ_func_2_0(48) * occ_func_3_0(7) * occ_func_2_0(40) +
          occ_func_2_0(44) * occ_func_3_0(13) * occ_func_2_0(52) +
          occ_func_2_0(36) * occ_func_2_0(16) * occ_func_3_0(3) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(9) +
          occ_func_2_0(2) * occ_func_3_0(7) * occ_func_2_0(6) +
          occ_func_2_0(44) * occ_func_2_0(16) * occ_func_3_0(13) +
          occ_func_2_0(36) * occ_func_3_0(3) * occ_func_2_0(32) +
          occ_func_2_0(40) * occ_func_3_0(11) * occ_func_2_0(48) +
          occ_func_2_0(32) * occ_func_2_0(16) * occ_func_3_0(5) +
          occ_func_2_0(52) * occ_func_3_0(9) * occ_func_2_0(44) +
          occ_func_2_0(48) * occ_func_2_0(16) * occ_func_3_0(7) +
          occ_func_3_0(10) * occ_func_2_0(11) * occ_func_3_0(12) +
          occ_func_2_0(5) * occ_func_3_0(0) * occ_func_3_0(8) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_3_0(10) +
          occ_func_3_0(2) * occ_func_2_0(3) * occ_func_3_0(6) +
          occ_func_3_0(8) * occ_func_2_0(9) * occ_func_3_0(4) +
          occ_func_2_0(7) * occ_func_3_0(0) * occ_func_3_0(2)) /
         12.;
}

/**** Basis functions for orbit 64****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

0.3333333 1.6666667 0.7500000 Va  O

1.3333333 1.6666667 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_64_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(9) *
              occ_func_3_0(13) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(16) *
              occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(52) *
              occ_func_3_0(48) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(9) +
          occ_func_2_0(8) * occ_func_2_0(12) * occ_func_3_0(13) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(47) *
              occ_func_3_0(51) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(13) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_3_0(13) *
              occ_func_3_0(9) +
          occ_func_2_0(12) * occ_func_2_0(8) * occ_func_3_0(9) *
              occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_3_0(48) *
              occ_func_3_0(52) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(16) *
              occ_func_3_0(48) +
          occ_func_2_0(47) * occ_func_2_0(51) * occ_func_3_0(12) *
              occ_func_3_0(0)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_64_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_3_0(9) *
              occ_func_3_0(13) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(11) +
          occ_func_2_0(47) * occ_func_2_0(15) * occ_func_3_0(0) *
              occ_func_3_0(12) +
          occ_func_2_0(39) * occ_func_2_0(31) * occ_func_3_0(2) *
              occ_func_3_0(0) +
          occ_func_2_0(35) * occ_func_2_0(43) * occ_func_3_0(8) *
              occ_func_3_0(0) +
          occ_func_2_0(31) * occ_func_2_0(15) * occ_func_3_0(0) *
              occ_func_3_0(6) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(9) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_3_0(3) *
              occ_func_3_0(5) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(11) *
              occ_func_3_0(7) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(3) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_3_0(47) *
              occ_func_3_0(51) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(15) *
              occ_func_3_0(43) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(13) +
          occ_func_2_0(0) * occ_func_2_0(6) * occ_func_3_0(7) *
              occ_func_3_0(11) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_3_0(13) *
              occ_func_3_0(9) +
          occ_func_2_0(2) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(5) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(7) +
          occ_func_2_0(0) * occ_func_2_0(4) * occ_func_3_0(5) *
              occ_func_3_0(3) +
          occ_func_2_0(35) * occ_func_2_0(15) * occ_func_3_0(0) *
              occ_func_3_0(8) +
          occ_func_2_0(31) * occ_func_2_0(39) * occ_func_3_0(6) *
              occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_2_0(15) * occ_func_3_0(0) *
              occ_func_3_0(10) +
          occ_func_2_0(43) * occ_func_2_0(35) * occ_func_3_0(4) *
              occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_2_0(51) * occ_func_3_0(12) *
              occ_func_3_0(0) +
          occ_func_2_0(39) * occ_func_2_0(15) * occ_func_3_0(0) *
              occ_func_3_0(2)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_64_0_at_1() const {
  return (occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(11) +
          occ_func_2_0(2) * occ_func_2_0(4) * occ_func_3_0(5) *
              occ_func_3_0(1) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_3_0(16) *
              occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_3_0(36) *
              occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_3_0(52) *
              occ_func_3_0(48) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_3_0(16) *
              occ_func_3_0(40) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(9) +
          occ_func_2_0(10) * occ_func_2_0(6) * occ_func_3_0(7) *
              occ_func_3_0(1) +
          occ_func_2_0(8) * occ_func_2_0(12) * occ_func_3_0(13) *
              occ_func_3_0(1) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(3) +
          occ_func_2_0(36) * occ_func_2_0(16) * occ_func_3_0(1) *
              occ_func_3_0(9) +
          occ_func_2_0(32) * occ_func_2_0(40) * occ_func_3_0(7) *
              occ_func_3_0(1) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(13) +
          occ_func_2_0(4) * occ_func_2_0(2) * occ_func_3_0(3) *
              occ_func_3_0(1) +
          occ_func_2_0(6) * occ_func_2_0(10) * occ_func_3_0(11) *
              occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(5) +
          occ_func_2_0(12) * occ_func_2_0(8) * occ_func_3_0(9) *
              occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(1) *
              occ_func_3_0(7) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_3_0(48) *
              occ_func_3_0(52) +
          occ_func_2_0(5) * occ_func_2_0(1) * occ_func_3_0(16) *
              occ_func_3_0(44) +
          occ_func_2_0(13) * occ_func_2_0(1) * occ_func_3_0(16) *
              occ_func_3_0(48) +
          occ_func_2_0(1) * occ_func_2_0(3) * occ_func_3_0(32) *
              occ_func_3_0(40) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_3_0(44) *
              occ_func_3_0(36) +
          occ_func_2_0(7) * occ_func_2_0(1) * occ_func_3_0(16) *
              occ_func_3_0(32)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_64_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(8) * occ_func_3_0(9) * occ_func_3_0(13) +
          occ_func_2_0(6) * occ_func_3_0(1) * occ_func_3_0(11) +
          occ_func_2_0(47) * occ_func_2_0(15) * occ_func_3_0(12) +
          occ_func_2_0(39) * occ_func_2_0(31) * occ_func_3_0(2) +
          occ_func_2_0(35) * occ_func_2_0(43) * occ_func_3_0(8) +
          occ_func_2_0(31) * occ_func_2_0(15) * occ_func_3_0(6) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_3_0(9) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_3_0(5) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_3_0(7) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_3_0(3) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_3_0(51) +
          occ_func_2_0(4) * occ_func_3_0(15) * occ_func_3_0(43) +
          occ_func_2_0(8) * occ_func_3_0(1) * occ_func_3_0(13) +
          occ_func_2_0(6) * occ_func_3_0(7) * occ_func_3_0(11) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_3_0(9) +
          occ_func_2_0(2) * occ_func_3_0(1) * occ_func_3_0(5) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_3_0(7) +
          occ_func_2_0(4) * occ_func_3_0(5) * occ_func_3_0(3) +
          occ_func_2_0(35) * occ_func_2_0(15) * occ_func_3_0(8) +
          occ_func_2_0(31) * occ_func_2_0(39) * occ_func_3_0(6) +
          occ_func_2_0(51) * occ_func_2_0(15) * occ_func_3_0(10) +
          occ_func_2_0(43) * occ_func_2_0(35) * occ_func_3_0(4) +
          occ_func_2_0(47) * occ_func_2_0(51) * occ_func_3_0(12) +
          occ_func_2_0(39) * occ_func_2_0(15) * occ_func_3_0(2)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_64_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(6) * occ_func_2_0(0) * occ_func_3_0(11) +
          occ_func_2_0(2) * occ_func_2_0(4) * occ_func_3_0(5) +
          occ_func_2_0(11) * occ_func_3_0(16) * occ_func_3_0(52) +
          occ_func_2_0(5) * occ_func_3_0(36) * occ_func_3_0(44) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_3_0(48) +
          occ_func_2_0(3) * occ_func_3_0(16) * occ_func_3_0(40) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_3_0(9) +
          occ_func_2_0(10) * occ_func_2_0(6) * occ_func_3_0(7) +
          occ_func_2_0(8) * occ_func_2_0(12) * occ_func_3_0(13) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_3_0(3) +
          occ_func_2_0(36) * occ_func_2_0(16) * occ_func_3_0(9) +
          occ_func_2_0(32) * occ_func_2_0(40) * occ_func_3_0(7) +
          occ_func_2_0(8) * occ_func_2_0(0) * occ_func_3_0(13) +
          occ_func_2_0(4) * occ_func_2_0(2) * occ_func_3_0(3) +
          occ_func_2_0(6) * occ_func_2_0(10) * occ_func_3_0(11) +
          occ_func_2_0(2) * occ_func_2_0(0) * occ_func_3_0(5) +
          occ_func_2_0(12) * occ_func_2_0(8) * occ_func_3_0(9) +
          occ_func_2_0(10) * occ_func_2_0(0) * occ_func_3_0(7) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_3_0(52) +
          occ_func_2_0(5) * occ_func_3_0(16) * occ_func_3_0(44) +
          occ_func_2_0(13) * occ_func_3_0(16) * occ_func_3_0(48) +
          occ_func_2_0(3) * occ_func_3_0(32) * occ_func_3_0(40) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_3_0(36) +
          occ_func_2_0(7) * occ_func_3_0(16) * occ_func_3_0(32)) /
         12.;
}

/**** Basis functions for orbit 65****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 0.7500000 Va  O

0.3333333 0.6666666 1.2500000 Va  O

0.3333333 1.6666667 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_65_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(9) +
          occ_func_2_0(47) * occ_func_3_0(10) * occ_func_2_0(11) *
              occ_func_3_0(0) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(17) *
              occ_func_3_0(52) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(52) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(11) +
          occ_func_2_0(43) * occ_func_3_0(8) * occ_func_2_0(9) *
              occ_func_3_0(0) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(44) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(13) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(48) *
              occ_func_3_0(1) +
          occ_func_2_0(17) * occ_func_3_0(16) * occ_func_2_0(1) *
              occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(17) *
              occ_func_3_0(48) +
          occ_func_2_0(51) * occ_func_3_0(12) * occ_func_2_0(13) *
              occ_func_3_0(0)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_65_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(9) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_2_0(0) *
              occ_func_3_0(43) +
          occ_func_2_0(47) * occ_func_3_0(10) * occ_func_2_0(11) *
              occ_func_3_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(4) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(12) +
          occ_func_2_0(31) * occ_func_3_0(2) * occ_func_2_0(3) *
              occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(3) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_2_0(0) *
              occ_func_3_0(31) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(11) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_2_0(0) *
              occ_func_3_0(47) +
          occ_func_2_0(43) * occ_func_3_0(8) * occ_func_2_0(9) *
              occ_func_3_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(6) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(7) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_2_0(0) *
              occ_func_3_0(39) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(13) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_2_0(0) *
              occ_func_3_0(51) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(5) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_2_0(0) *
              occ_func_3_0(35) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(15) *
              occ_func_3_0(8) +
          occ_func_2_0(7) * occ_func_3_0(6) * occ_func_2_0(39) *
              occ_func_3_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(10) +
          occ_func_2_0(35) * occ_func_3_0(4) * occ_func_2_0(5) *
              occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_3_0(12) * occ_func_2_0(13) *
              occ_func_3_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(2)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_65_0_at_1() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(9) +
          occ_func_2_0(6) * occ_func_3_0(7) * occ_func_2_0(40) *
              occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(17) *
              occ_func_3_0(36) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(4) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(17) *
              occ_func_3_0(52) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(12) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(52) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(3) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(11) +
          occ_func_2_0(4) * occ_func_3_0(5) * occ_func_2_0(36) *
              occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(17) *
              occ_func_3_0(40) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(6) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(44) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(7) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(13) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_2_0(32) *
              occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(48) *
              occ_func_3_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(16) *
              occ_func_3_0(5) +
          occ_func_2_0(17) * occ_func_3_0(16) * occ_func_2_0(1) *
              occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(15) *
              occ_func_3_0(8) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(17) *
              occ_func_3_0(48) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(10) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(17) *
              occ_func_3_0(32) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(1) *
              occ_func_3_0(2)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_65_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(1) * occ_func_2_0(16) * occ_func_3_0(9) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_3_0(43) +
          occ_func_2_0(47) * occ_func_3_0(10) * occ_func_2_0(11) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(4) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(12) +
          occ_func_2_0(31) * occ_func_3_0(2) * occ_func_2_0(3) +
          occ_func_3_0(1) * occ_func_2_0(16) * occ_func_3_0(3) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_3_0(31) +
          occ_func_3_0(1) * occ_func_2_0(16) * occ_func_3_0(11) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_3_0(47) +
          occ_func_2_0(43) * occ_func_3_0(8) * occ_func_2_0(9) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(6) +
          occ_func_3_0(1) * occ_func_2_0(16) * occ_func_3_0(7) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_3_0(39) +
          occ_func_3_0(1) * occ_func_2_0(16) * occ_func_3_0(13) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_3_0(51) +
          occ_func_3_0(1) * occ_func_2_0(16) * occ_func_3_0(5) +
          occ_func_2_0(14) * occ_func_3_0(15) * occ_func_3_0(35) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_3_0(8) +
          occ_func_2_0(7) * occ_func_3_0(6) * occ_func_2_0(39) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(10) +
          occ_func_2_0(35) * occ_func_3_0(4) * occ_func_2_0(5) +
          occ_func_2_0(51) * occ_func_3_0(12) * occ_func_2_0(13) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(2)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_65_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(9) +
          occ_func_2_0(6) * occ_func_3_0(7) * occ_func_2_0(40) +
          occ_func_3_0(16) * occ_func_2_0(17) * occ_func_3_0(36) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_3_0(4) +
          occ_func_3_0(16) * occ_func_2_0(17) * occ_func_3_0(52) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_3_0(12) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(52) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(3) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(11) +
          occ_func_2_0(4) * occ_func_3_0(5) * occ_func_2_0(36) +
          occ_func_3_0(16) * occ_func_2_0(17) * occ_func_3_0(40) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_3_0(6) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(44) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(7) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(13) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_2_0(32) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(48) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(5) +
          occ_func_2_0(17) * occ_func_3_0(16) * occ_func_3_0(44) +
          occ_func_3_0(0) * occ_func_2_0(15) * occ_func_3_0(8) +
          occ_func_3_0(16) * occ_func_2_0(17) * occ_func_3_0(48) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_3_0(10) +
          occ_func_3_0(16) * occ_func_2_0(17) * occ_func_3_0(32) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_3_0(2)) /
         12.;
}

/**** Basis functions for orbit 66****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 1.2500000 Va  O

1.3333333 0.6666666 0.7500000 Va  O

1.3333333 1.6666667 0.7500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_66_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(11) *
              occ_func_3_0(13) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(52) *
              occ_func_3_0(44) +
          occ_func_2_0(43) * occ_func_2_0(9) * occ_func_3_0(0) *
              occ_func_3_0(12) +
          occ_func_2_0(10) * occ_func_2_0(48) * occ_func_3_0(13) *
              occ_func_3_0(1) +
          occ_func_2_0(12) * occ_func_2_0(52) * occ_func_3_0(1) *
              occ_func_3_0(11) +
          occ_func_2_0(52) * occ_func_2_0(12) * occ_func_3_0(1) *
              occ_func_3_0(9) +
          occ_func_2_0(44) * occ_func_2_0(8) * occ_func_3_0(13) *
              occ_func_3_0(1) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_3_0(9) *
              occ_func_3_0(13) +
          occ_func_2_0(51) * occ_func_2_0(13) * occ_func_3_0(8) *
              occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_2_0(51) * occ_func_3_0(10) *
              occ_func_3_0(0) +
          occ_func_2_0(11) * occ_func_2_0(47) * occ_func_3_0(0) *
              occ_func_3_0(12) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_3_0(52) *
              occ_func_3_0(48)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_66_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(11) *
              occ_func_3_0(13) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(47) *
              occ_func_3_0(51) +
          occ_func_2_0(39) * occ_func_2_0(7) * occ_func_3_0(10) *
              occ_func_3_0(0) +
          occ_func_2_0(31) * occ_func_2_0(3) * occ_func_3_0(0) *
              occ_func_3_0(4) +
          occ_func_2_0(43) * occ_func_2_0(9) * occ_func_3_0(0) *
              occ_func_3_0(12) +
          occ_func_2_0(35) * occ_func_2_0(5) * occ_func_3_0(2) *
              occ_func_3_0(0) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(9) *
              occ_func_3_0(5) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(43) *
              occ_func_3_0(35) +
          occ_func_2_0(0) * occ_func_2_0(16) * occ_func_3_0(3) *
              occ_func_3_0(7) +
          occ_func_2_0(14) * occ_func_2_0(0) * occ_func_3_0(31) *
              occ_func_3_0(39) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_3_0(3) *
              occ_func_3_0(5) +
          occ_func_2_0(0) * occ_func_2_0(14) * occ_func_3_0(31) *
              occ_func_3_0(35) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_3_0(11) *
              occ_func_3_0(7) +
          occ_func_2_0(0) * occ_func_2_0(14) * occ_func_3_0(47) *
              occ_func_3_0(39) +
          occ_func_2_0(16) * occ_func_2_0(0) * occ_func_3_0(9) *
              occ_func_3_0(13) +
          occ_func_2_0(0) * occ_func_2_0(14) * occ_func_3_0(43) *
              occ_func_3_0(51) +
          occ_func_2_0(51) * occ_func_2_0(13) * occ_func_3_0(8) *
              occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_2_0(11) * occ_func_3_0(0) *
              occ_func_3_0(6) +
          occ_func_2_0(13) * occ_func_2_0(51) * occ_func_3_0(10) *
              occ_func_3_0(0) +
          occ_func_2_0(9) * occ_func_2_0(43) * occ_func_3_0(0) *
              occ_func_3_0(4) +
          occ_func_2_0(11) * occ_func_2_0(47) * occ_func_3_0(0) *
              occ_func_3_0(12) +
          occ_func_2_0(7) * occ_func_2_0(39) * occ_func_3_0(2) *
              occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_2_0(35) * occ_func_3_0(8) *
              occ_func_3_0(0) +
          occ_func_2_0(3) * occ_func_2_0(31) * occ_func_3_0(0) *
              occ_func_3_0(6)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_66_0_at_1() const {
  return (occ_func_2_0(4) * occ_func_2_0(36) * occ_func_3_0(1) *
              occ_func_3_0(9) +
          occ_func_2_0(2) * occ_func_2_0(32) * occ_func_3_0(7) *
              occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(52) *
              occ_func_3_0(44) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(12) *
              occ_func_3_0(8) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(40) *
              occ_func_3_0(48) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(6) *
              occ_func_3_0(10) +
          occ_func_2_0(10) * occ_func_2_0(48) * occ_func_3_0(13) *
              occ_func_3_0(1) +
          occ_func_2_0(6) * occ_func_2_0(40) * occ_func_3_0(1) *
              occ_func_3_0(3) +
          occ_func_2_0(12) * occ_func_2_0(52) * occ_func_3_0(1) *
              occ_func_3_0(11) +
          occ_func_2_0(8) * occ_func_2_0(44) * occ_func_3_0(5) *
              occ_func_3_0(1) +
          occ_func_2_0(52) * occ_func_2_0(12) * occ_func_3_0(1) *
              occ_func_3_0(9) +
          occ_func_2_0(48) * occ_func_2_0(10) * occ_func_3_0(7) *
              occ_func_3_0(1) +
          occ_func_2_0(44) * occ_func_2_0(8) * occ_func_3_0(13) *
              occ_func_3_0(1) +
          occ_func_2_0(36) * occ_func_2_0(4) * occ_func_3_0(1) *
              occ_func_3_0(3) +
          occ_func_2_0(40) * occ_func_2_0(6) * occ_func_3_0(1) *
              occ_func_3_0(11) +
          occ_func_2_0(32) * occ_func_2_0(2) * occ_func_3_0(5) *
              occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_2_0(17) * occ_func_3_0(36) *
              occ_func_3_0(32) +
          occ_func_2_0(15) * occ_func_2_0(1) * occ_func_3_0(4) *
              occ_func_3_0(2) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_3_0(40) *
              occ_func_3_0(32) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_3_0(6) *
              occ_func_3_0(2) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_3_0(36) *
              occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_3_0(4) *
              occ_func_3_0(8) +
          occ_func_2_0(17) * occ_func_2_0(1) * occ_func_3_0(52) *
              occ_func_3_0(48) +
          occ_func_2_0(1) * occ_func_2_0(15) * occ_func_3_0(12) *
              occ_func_3_0(10)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_66_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(16) * occ_func_3_0(11) * occ_func_3_0(13) +
          occ_func_2_0(14) * occ_func_3_0(47) * occ_func_3_0(51) +
          occ_func_2_0(39) * occ_func_2_0(7) * occ_func_3_0(10) +
          occ_func_2_0(31) * occ_func_2_0(3) * occ_func_3_0(4) +
          occ_func_2_0(43) * occ_func_2_0(9) * occ_func_3_0(12) +
          occ_func_2_0(35) * occ_func_2_0(5) * occ_func_3_0(2) +
          occ_func_2_0(16) * occ_func_3_0(9) * occ_func_3_0(5) +
          occ_func_2_0(14) * occ_func_3_0(43) * occ_func_3_0(35) +
          occ_func_2_0(16) * occ_func_3_0(3) * occ_func_3_0(7) +
          occ_func_2_0(14) * occ_func_3_0(31) * occ_func_3_0(39) +
          occ_func_2_0(16) * occ_func_3_0(3) * occ_func_3_0(5) +
          occ_func_2_0(14) * occ_func_3_0(31) * occ_func_3_0(35) +
          occ_func_2_0(16) * occ_func_3_0(11) * occ_func_3_0(7) +
          occ_func_2_0(14) * occ_func_3_0(47) * occ_func_3_0(39) +
          occ_func_2_0(16) * occ_func_3_0(9) * occ_func_3_0(13) +
          occ_func_2_0(14) * occ_func_3_0(43) * occ_func_3_0(51) +
          occ_func_2_0(51) * occ_func_2_0(13) * occ_func_3_0(8) +
          occ_func_2_0(47) * occ_func_2_0(11) * occ_func_3_0(6) +
          occ_func_2_0(13) * occ_func_2_0(51) * occ_func_3_0(10) +
          occ_func_2_0(9) * occ_func_2_0(43) * occ_func_3_0(4) +
          occ_func_2_0(11) * occ_func_2_0(47) * occ_func_3_0(12) +
          occ_func_2_0(7) * occ_func_2_0(39) * occ_func_3_0(2) +
          occ_func_2_0(5) * occ_func_2_0(35) * occ_func_3_0(8) +
          occ_func_2_0(3) * occ_func_2_0(31) * occ_func_3_0(6)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_66_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(4) * occ_func_2_0(36) * occ_func_3_0(9) +
          occ_func_2_0(2) * occ_func_2_0(32) * occ_func_3_0(7) +
          occ_func_2_0(17) * occ_func_3_0(52) * occ_func_3_0(44) +
          occ_func_2_0(15) * occ_func_3_0(12) * occ_func_3_0(8) +
          occ_func_2_0(17) * occ_func_3_0(40) * occ_func_3_0(48) +
          occ_func_2_0(15) * occ_func_3_0(6) * occ_func_3_0(10) +
          occ_func_2_0(10) * occ_func_2_0(48) * occ_func_3_0(13) +
          occ_func_2_0(6) * occ_func_2_0(40) * occ_func_3_0(3) +
          occ_func_2_0(12) * occ_func_2_0(52) * occ_func_3_0(11) +
          occ_func_2_0(8) * occ_func_2_0(44) * occ_func_3_0(5) +
          occ_func_2_0(52) * occ_func_2_0(12) * occ_func_3_0(9) +
          occ_func_2_0(48) * occ_func_2_0(10) * occ_func_3_0(7) +
          occ_func_2_0(44) * occ_func_2_0(8) * occ_func_3_0(13) +
          occ_func_2_0(36) * occ_func_2_0(4) * occ_func_3_0(3) +
          occ_func_2_0(40) * occ_func_2_0(6) * occ_func_3_0(11) +
          occ_func_2_0(32) * occ_func_2_0(2) * occ_func_3_0(5) +
          occ_func_2_0(17) * occ_func_3_0(36) * occ_func_3_0(32) +
          occ_func_2_0(15) * occ_func_3_0(4) * occ_func_3_0(2) +
          occ_func_2_0(17) * occ_func_3_0(40) * occ_func_3_0(32) +
          occ_func_2_0(15) * occ_func_3_0(6) * occ_func_3_0(2) +
          occ_func_2_0(17) * occ_func_3_0(36) * occ_func_3_0(44) +
          occ_func_2_0(15) * occ_func_3_0(4) * occ_func_3_0(8) +
          occ_func_2_0(17) * occ_func_3_0(52) * occ_func_3_0(48) +
          occ_func_2_0(15) * occ_func_3_0(12) * occ_func_3_0(10)) /
         12.;
}

/**** Basis functions for orbit 67****
0.3333333 0.6666666 0.2500000 Va  O

1.3333333 0.6666666 0.2500000 Va  O

1.3333333 1.6666667 0.2500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_67_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(12) *
              occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(9) *
              occ_func_2_0(27) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(13) *
              occ_func_2_0(11) +
          occ_func_2_0(10) * occ_func_2_0(12) * occ_func_2_0(0) *
              occ_func_2_0(8) +
          occ_func_2_0(26) * occ_func_2_0(8) * occ_func_2_0(12) *
              occ_func_2_0(0) +
          occ_func_2_0(29) * occ_func_2_0(13) * occ_func_2_0(11) *
              occ_func_2_0(1)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_67_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(12) *
              occ_func_2_0(28) +
          occ_func_2_0(4) * occ_func_2_0(0) * occ_func_2_0(8) *
              occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_2_0(6) * occ_func_2_0(0) *
              occ_func_2_0(10) +
          occ_func_2_0(18) * occ_func_2_0(2) * occ_func_2_0(4) *
              occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_2_0(12) * occ_func_2_0(0) *
              occ_func_2_0(8) +
          occ_func_2_0(24) * occ_func_2_0(10) * occ_func_2_0(6) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(4) *
              occ_func_2_0(22) +
          occ_func_2_0(6) * occ_func_2_0(0) * occ_func_2_0(2) *
              occ_func_2_0(4) +
          occ_func_2_0(26) * occ_func_2_0(8) * occ_func_2_0(12) *
              occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_2_0(0) * occ_func_2_0(10) *
              occ_func_2_0(6) +
          occ_func_2_0(8) * occ_func_2_0(4) * occ_func_2_0(0) *
              occ_func_2_0(2) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(6) *
              occ_func_2_0(20)) /
         6.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_67_0_at_1() const {
  return (occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(9) *
              occ_func_2_0(27) +
          occ_func_2_0(7) * occ_func_2_0(11) * occ_func_2_0(1) *
              occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_2_0(1) * occ_func_2_0(5) *
              occ_func_2_0(9) +
          occ_func_2_0(21) * occ_func_2_0(7) * occ_func_2_0(3) *
              occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_2_0(1) * occ_func_2_0(13) *
              occ_func_2_0(11) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(11) *
              occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_2_0(5) * occ_func_2_0(9) *
              occ_func_2_0(1) +
          occ_func_2_0(5) * occ_func_2_0(3) * occ_func_2_0(1) *
              occ_func_2_0(7) +
          occ_func_2_0(29) * occ_func_2_0(13) * occ_func_2_0(11) *
              occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_2_0(9) * occ_func_2_0(1) *
              occ_func_2_0(5) +
          occ_func_2_0(11) * occ_func_2_0(1) * occ_func_2_0(7) *
              occ_func_2_0(3) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(3) *
              occ_func_2_0(19)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_67_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(10) * occ_func_2_0(12) * occ_func_2_0(28) +
          occ_func_2_0(4) * occ_func_2_0(8) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_2_0(6) * occ_func_2_0(10) +
          occ_func_2_0(18) * occ_func_2_0(2) * occ_func_2_0(4) +
          occ_func_2_0(10) * occ_func_2_0(12) * occ_func_2_0(8) +
          occ_func_2_0(24) * occ_func_2_0(10) * occ_func_2_0(6) +
          occ_func_2_0(8) * occ_func_2_0(4) * occ_func_2_0(22) +
          occ_func_2_0(6) * occ_func_2_0(2) * occ_func_2_0(4) +
          occ_func_2_0(26) * occ_func_2_0(8) * occ_func_2_0(12) +
          occ_func_2_0(12) * occ_func_2_0(10) * occ_func_2_0(6) +
          occ_func_2_0(8) * occ_func_2_0(4) * occ_func_2_0(2) +
          occ_func_2_0(2) * occ_func_2_0(6) * occ_func_2_0(20)) /
         6.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_67_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(13) * occ_func_2_0(9) * occ_func_2_0(27) +
          occ_func_2_0(7) * occ_func_2_0(11) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_2_0(5) * occ_func_2_0(9) +
          occ_func_2_0(21) * occ_func_2_0(7) * occ_func_2_0(3) +
          occ_func_2_0(9) * occ_func_2_0(13) * occ_func_2_0(11) +
          occ_func_2_0(7) * occ_func_2_0(11) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_2_0(5) * occ_func_2_0(9) +
          occ_func_2_0(5) * occ_func_2_0(3) * occ_func_2_0(7) +
          occ_func_2_0(29) * occ_func_2_0(13) * occ_func_2_0(11) +
          occ_func_2_0(13) * occ_func_2_0(9) * occ_func_2_0(5) +
          occ_func_2_0(11) * occ_func_2_0(7) * occ_func_2_0(3) +
          occ_func_2_0(5) * occ_func_2_0(3) * occ_func_2_0(19)) /
         6.;
}

/**** Basis functions for orbit 68****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 0.6666666 0.7500000 Va  O

0.3333333 1.6666667 0.2500000 Va  O

1.3333333 0.6666666 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_68_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(8) *
              occ_func_2_0(10) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_2_0(1) *
              occ_func_2_0(29) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_2_0(27) *
              occ_func_2_0(1) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(0) *
              occ_func_2_0(26) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(28) *
              occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_2_0(26) *
              occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_2_0(0) *
              occ_func_2_0(28) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(48) *
              occ_func_2_0(44) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_2_0(25) *
              occ_func_2_0(1) +
          occ_func_2_0(11) * occ_func_3_0(10) * occ_func_2_0(1) *
              occ_func_2_0(25) +
          occ_func_2_0(11) * occ_func_3_0(10) * occ_func_2_0(29) *
              occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_3_0(8) * occ_func_2_0(1) *
              occ_func_2_0(27) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(0) *
              occ_func_2_0(26) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(28) *
              occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(0) *
              occ_func_2_0(24) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(9) *
              occ_func_2_0(11) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(11) *
              occ_func_2_0(9) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_2_0(1) *
              occ_func_2_0(29) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_2_0(27) *
              occ_func_2_0(1) +
          occ_func_2_0(8) * occ_func_3_0(43) * occ_func_2_0(26) *
              occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_2_0(0) *
              occ_func_2_0(28) +
          occ_func_2_0(13) * occ_func_3_0(12) * occ_func_2_0(29) *
              occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_3_0(12) * occ_func_2_0(1) *
              occ_func_2_0(27) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_2_0(24) *
              occ_func_2_0(0)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_68_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(8) *
              occ_func_2_0(10) +
          occ_func_2_0(6) * occ_func_3_0(7) * occ_func_2_0(0) *
              occ_func_2_0(24) +
          occ_func_2_0(4) * occ_func_3_0(5) * occ_func_2_0(22) *
              occ_func_2_0(0) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(35) *
              occ_func_2_0(51) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(51) *
              occ_func_2_0(39) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(0) *
              occ_func_2_0(26) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(2) *
              occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_3_0(7) * occ_func_2_0(20) *
              occ_func_2_0(0) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(28) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(10) *
              occ_func_2_0(2) +
          occ_func_2_0(4) * occ_func_3_0(5) * occ_func_2_0(0) *
              occ_func_2_0(18) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_2_0(26) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(8) *
              occ_func_2_0(2) +
          occ_func_2_0(6) * occ_func_3_0(39) * occ_func_2_0(0) *
              occ_func_2_0(20) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_2_0(0) *
              occ_func_2_0(28) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(2) *
              occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_3_0(35) * occ_func_2_0(18) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(10) *
              occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_3_0(39) * occ_func_2_0(24) *
              occ_func_2_0(0) +
          occ_func_2_0(4) * occ_func_3_0(35) * occ_func_2_0(0) *
              occ_func_2_0(22) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(39) *
              occ_func_2_0(35) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(5) *
              occ_func_2_0(7) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(13) *
              occ_func_2_0(5) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(7) *
              occ_func_2_0(13) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(0) *
              occ_func_2_0(26) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(6) *
              occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_2_0(20) *
              occ_func_2_0(0) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(28) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(12) *
              occ_func_2_0(4) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_2_0(0) *
              occ_func_2_0(18) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(0) *
              occ_func_2_0(24) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(22) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(4) *
              occ_func_2_0(6) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(9) *
              occ_func_2_0(11) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(47) *
              occ_func_2_0(43) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(31) *
              occ_func_2_0(47) +
          occ_func_2_0(15) * occ_func_3_0(0) * occ_func_2_0(43) *
              occ_func_2_0(31) +
          occ_func_2_0(8) * occ_func_3_0(43) * occ_func_2_0(26) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(12) *
              occ_func_2_0(6) +
          occ_func_2_0(2) * occ_func_3_0(31) * occ_func_2_0(0) *
              occ_func_2_0(20) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_2_0(0) *
              occ_func_2_0(28) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(4) *
              occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(31) * occ_func_2_0(18) *
              occ_func_2_0(0) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(11) *
              occ_func_2_0(3) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(3) *
              occ_func_2_0(9) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_2_0(24) *
              occ_func_2_0(0) +
          occ_func_2_0(8) * occ_func_3_0(43) * occ_func_2_0(0) *
              occ_func_2_0(22) +
          occ_func_2_0(0) * occ_func_3_0(15) * occ_func_2_0(6) *
              occ_func_2_0(4)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_68_0_at_1() const {
  return (occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(8) *
              occ_func_2_0(10) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_2_0(1) *
              occ_func_2_0(29) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(5) *
              occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(32) * occ_func_2_0(19) *
              occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_2_0(27) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(13) *
              occ_func_2_0(7) +
          occ_func_2_0(3) * occ_func_3_0(32) * occ_func_2_0(1) *
              occ_func_2_0(21) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(2) *
              occ_func_2_0(8) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(10) *
              occ_func_2_0(2) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(44) *
              occ_func_2_0(32) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(32) *
              occ_func_2_0(48) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(48) *
              occ_func_2_0(44) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_2_0(25) *
              occ_func_2_0(1) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_2_0(1) *
              occ_func_2_0(23) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(7) *
              occ_func_2_0(5) +
          occ_func_2_0(11) * occ_func_3_0(10) * occ_func_2_0(1) *
              occ_func_2_0(25) +
          occ_func_2_0(9) * occ_func_3_0(8) * occ_func_2_0(23) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(5) *
              occ_func_2_0(7) +
          occ_func_2_0(11) * occ_func_3_0(10) * occ_func_2_0(29) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(13) *
              occ_func_2_0(5) +
          occ_func_2_0(3) * occ_func_3_0(2) * occ_func_2_0(1) *
              occ_func_2_0(19) +
          occ_func_2_0(9) * occ_func_3_0(8) * occ_func_2_0(1) *
              occ_func_2_0(27) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(7) *
              occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(2) * occ_func_2_0(21) *
              occ_func_2_0(1) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(6) *
              occ_func_2_0(12) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(12) *
              occ_func_2_0(4) +
          occ_func_2_0(0) * occ_func_3_0(1) * occ_func_2_0(4) *
              occ_func_2_0(6) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(9) *
              occ_func_2_0(11) +
          occ_func_2_0(7) * occ_func_3_0(6) * occ_func_2_0(1) *
              occ_func_2_0(25) +
          occ_func_2_0(5) * occ_func_3_0(4) * occ_func_2_0(23) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(11) *
              occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(40) * occ_func_2_0(25) *
              occ_func_2_0(1) +
          occ_func_2_0(5) * occ_func_3_0(36) * occ_func_2_0(1) *
              occ_func_2_0(23) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_2_0(1) *
              occ_func_2_0(29) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(3) *
              occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(36) * occ_func_2_0(19) *
              occ_func_2_0(1) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_2_0(27) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(16) * occ_func_2_0(9) *
              occ_func_2_0(3) +
          occ_func_2_0(7) * occ_func_3_0(40) * occ_func_2_0(1) *
              occ_func_2_0(21) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(52) *
              occ_func_2_0(40) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(36) *
              occ_func_2_0(52) +
          occ_func_2_0(13) * occ_func_3_0(12) * occ_func_2_0(29) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(11) *
              occ_func_2_0(3) +
          occ_func_2_0(5) * occ_func_3_0(4) * occ_func_2_0(1) *
              occ_func_2_0(19) +
          occ_func_2_0(13) * occ_func_3_0(12) * occ_func_2_0(1) *
              occ_func_2_0(27) +
          occ_func_2_0(1) * occ_func_3_0(0) * occ_func_2_0(3) *
              occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(6) * occ_func_2_0(21) *
              occ_func_2_0(1) +
          occ_func_2_0(16) * occ_func_3_0(1) * occ_func_2_0(40) *
              occ_func_2_0(36)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_68_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(1) * occ_func_2_0(8) * occ_func_2_0(10) +
          occ_func_2_0(6) * occ_func_3_0(7) * occ_func_2_0(24) +
          occ_func_2_0(4) * occ_func_3_0(5) * occ_func_2_0(22) +
          occ_func_2_0(15) * occ_func_2_0(35) * occ_func_2_0(51) +
          occ_func_2_0(15) * occ_func_2_0(51) * occ_func_2_0(39) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(26) +
          occ_func_3_0(1) * occ_func_2_0(2) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_3_0(7) * occ_func_2_0(20) +
          occ_func_2_0(12) * occ_func_3_0(13) * occ_func_2_0(28) +
          occ_func_3_0(1) * occ_func_2_0(10) * occ_func_2_0(2) +
          occ_func_2_0(4) * occ_func_3_0(5) * occ_func_2_0(18) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_2_0(26) +
          occ_func_3_0(15) * occ_func_2_0(8) * occ_func_2_0(2) +
          occ_func_2_0(6) * occ_func_3_0(39) * occ_func_2_0(20) +
          occ_func_2_0(12) * occ_func_3_0(51) * occ_func_2_0(28) +
          occ_func_3_0(15) * occ_func_2_0(2) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_3_0(35) * occ_func_2_0(18) +
          occ_func_3_0(15) * occ_func_2_0(10) * occ_func_2_0(8) +
          occ_func_2_0(6) * occ_func_3_0(39) * occ_func_2_0(24) +
          occ_func_2_0(4) * occ_func_3_0(35) * occ_func_2_0(22) +
          occ_func_2_0(15) * occ_func_2_0(39) * occ_func_2_0(35) +
          occ_func_2_0(1) * occ_func_2_0(5) * occ_func_2_0(7) +
          occ_func_2_0(1) * occ_func_2_0(13) * occ_func_2_0(5) +
          occ_func_2_0(1) * occ_func_2_0(7) * occ_func_2_0(13) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(26) +
          occ_func_3_0(1) * occ_func_2_0(6) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_2_0(20) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(28) +
          occ_func_3_0(1) * occ_func_2_0(12) * occ_func_2_0(4) +
          occ_func_2_0(2) * occ_func_3_0(3) * occ_func_2_0(18) +
          occ_func_2_0(10) * occ_func_3_0(11) * occ_func_2_0(24) +
          occ_func_2_0(8) * occ_func_3_0(9) * occ_func_2_0(22) +
          occ_func_3_0(1) * occ_func_2_0(4) * occ_func_2_0(6) +
          occ_func_2_0(1) * occ_func_2_0(9) * occ_func_2_0(11) +
          occ_func_2_0(15) * occ_func_2_0(47) * occ_func_2_0(43) +
          occ_func_2_0(15) * occ_func_2_0(31) * occ_func_2_0(47) +
          occ_func_2_0(15) * occ_func_2_0(43) * occ_func_2_0(31) +
          occ_func_2_0(8) * occ_func_3_0(43) * occ_func_2_0(26) +
          occ_func_3_0(15) * occ_func_2_0(12) * occ_func_2_0(6) +
          occ_func_2_0(2) * occ_func_3_0(31) * occ_func_2_0(20) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_2_0(28) +
          occ_func_3_0(15) * occ_func_2_0(4) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(31) * occ_func_2_0(18) +
          occ_func_2_0(1) * occ_func_2_0(11) * occ_func_2_0(3) +
          occ_func_2_0(1) * occ_func_2_0(3) * occ_func_2_0(9) +
          occ_func_2_0(10) * occ_func_3_0(47) * occ_func_2_0(24) +
          occ_func_2_0(8) * occ_func_3_0(43) * occ_func_2_0(22) +
          occ_func_3_0(15) * occ_func_2_0(6) * occ_func_2_0(4)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_68_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(0) * occ_func_2_0(8) * occ_func_2_0(10) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_2_0(29) +
          occ_func_3_0(16) * occ_func_2_0(5) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(32) * occ_func_2_0(19) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_2_0(27) +
          occ_func_3_0(16) * occ_func_2_0(13) * occ_func_2_0(7) +
          occ_func_2_0(3) * occ_func_3_0(32) * occ_func_2_0(21) +
          occ_func_2_0(0) * occ_func_2_0(2) * occ_func_2_0(8) +
          occ_func_2_0(0) * occ_func_2_0(10) * occ_func_2_0(2) +
          occ_func_2_0(16) * occ_func_2_0(44) * occ_func_2_0(32) +
          occ_func_2_0(16) * occ_func_2_0(32) * occ_func_2_0(48) +
          occ_func_2_0(16) * occ_func_2_0(48) * occ_func_2_0(44) +
          occ_func_2_0(11) * occ_func_3_0(48) * occ_func_2_0(25) +
          occ_func_2_0(9) * occ_func_3_0(44) * occ_func_2_0(23) +
          occ_func_3_0(16) * occ_func_2_0(7) * occ_func_2_0(5) +
          occ_func_2_0(11) * occ_func_3_0(10) * occ_func_2_0(25) +
          occ_func_2_0(9) * occ_func_3_0(8) * occ_func_2_0(23) +
          occ_func_3_0(0) * occ_func_2_0(5) * occ_func_2_0(7) +
          occ_func_2_0(11) * occ_func_3_0(10) * occ_func_2_0(29) +
          occ_func_3_0(0) * occ_func_2_0(13) * occ_func_2_0(5) +
          occ_func_2_0(3) * occ_func_3_0(2) * occ_func_2_0(19) +
          occ_func_2_0(9) * occ_func_3_0(8) * occ_func_2_0(27) +
          occ_func_3_0(0) * occ_func_2_0(7) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(2) * occ_func_2_0(21) +
          occ_func_2_0(0) * occ_func_2_0(6) * occ_func_2_0(12) +
          occ_func_2_0(0) * occ_func_2_0(12) * occ_func_2_0(4) +
          occ_func_2_0(0) * occ_func_2_0(4) * occ_func_2_0(6) +
          occ_func_3_0(0) * occ_func_2_0(9) * occ_func_2_0(11) +
          occ_func_2_0(7) * occ_func_3_0(6) * occ_func_2_0(25) +
          occ_func_2_0(5) * occ_func_3_0(4) * occ_func_2_0(23) +
          occ_func_3_0(16) * occ_func_2_0(11) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(40) * occ_func_2_0(25) +
          occ_func_2_0(5) * occ_func_3_0(36) * occ_func_2_0(23) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_2_0(29) +
          occ_func_3_0(16) * occ_func_2_0(3) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(36) * occ_func_2_0(19) +
          occ_func_2_0(13) * occ_func_3_0(52) * occ_func_2_0(27) +
          occ_func_3_0(16) * occ_func_2_0(9) * occ_func_2_0(3) +
          occ_func_2_0(7) * occ_func_3_0(40) * occ_func_2_0(21) +
          occ_func_2_0(16) * occ_func_2_0(52) * occ_func_2_0(40) +
          occ_func_2_0(16) * occ_func_2_0(36) * occ_func_2_0(52) +
          occ_func_2_0(13) * occ_func_3_0(12) * occ_func_2_0(29) +
          occ_func_3_0(0) * occ_func_2_0(11) * occ_func_2_0(3) +
          occ_func_2_0(5) * occ_func_3_0(4) * occ_func_2_0(19) +
          occ_func_2_0(13) * occ_func_3_0(12) * occ_func_2_0(27) +
          occ_func_3_0(0) * occ_func_2_0(3) * occ_func_2_0(9) +
          occ_func_2_0(7) * occ_func_3_0(6) * occ_func_2_0(21) +
          occ_func_2_0(16) * occ_func_2_0(40) * occ_func_2_0(36)) /
         24.;
}

/**** Basis functions for orbit 69****
0.3333333 0.6666666 0.2500000 Va  O

0.3333333 1.6666667 -0.2500000 Va  O

1.3333333 0.6666666 -0.2500000 Va  O

1.3333333 1.6666667 -0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_69_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_3_0(47) *
              occ_func_3_0(51) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_3_0(28) *
              occ_func_3_0(12) +
          occ_func_2_0(9) * occ_func_3_0(26) * occ_func_3_0(0) *
              occ_func_3_0(12) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_3_0(27) *
              occ_func_3_0(9) +
          occ_func_2_0(52) * occ_func_3_0(29) * occ_func_3_0(1) *
              occ_func_3_0(11) +
          occ_func_2_0(12) * occ_func_3_0(27) * occ_func_3_0(1) *
              occ_func_3_0(9) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_3_0(29) *
              occ_func_3_0(11) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_3_0(9) *
              occ_func_3_0(13) +
          occ_func_2_0(13) * occ_func_3_0(10) * occ_func_3_0(8) *
              occ_func_3_0(0) +
          occ_func_2_0(51) * occ_func_3_0(8) * occ_func_3_0(10) *
              occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_3_0(28) * occ_func_3_0(0) *
              occ_func_3_0(12) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_3_0(26) *
              occ_func_3_0(12) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_3_0(27) *
              occ_func_3_0(13) +
          occ_func_2_0(48) * occ_func_3_0(29) * occ_func_3_0(1) *
              occ_func_3_0(13) +
          occ_func_2_0(52) * occ_func_3_0(9) * occ_func_3_0(11) *
              occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_3_0(48) *
              occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_3_0(8) *
              occ_func_3_0(12) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_3_0(28) *
              occ_func_3_0(10) +
          occ_func_2_0(13) * occ_func_3_0(26) * occ_func_3_0(0) *
              occ_func_3_0(8) +
          occ_func_2_0(8) * occ_func_3_0(27) * occ_func_3_0(1) *
              occ_func_3_0(13) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_3_0(29) *
              occ_func_3_0(13) +
          occ_func_2_0(51) * occ_func_3_0(28) * occ_func_3_0(0) *
              occ_func_3_0(10) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_3_0(26) *
              occ_func_3_0(8) +
          occ_func_2_0(12) * occ_func_3_0(11) * occ_func_3_0(9) *
              occ_func_3_0(1)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_69_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(43) * occ_func_3_0(47) *
              occ_func_3_0(51) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_3_0(28) *
              occ_func_3_0(12) +
          occ_func_2_0(7) * occ_func_3_0(2) * occ_func_3_0(10) *
              occ_func_3_0(0) +
          occ_func_2_0(3) * occ_func_3_0(18) * occ_func_3_0(0) *
              occ_func_3_0(4) +
          occ_func_2_0(9) * occ_func_3_0(26) * occ_func_3_0(0) *
              occ_func_3_0(12) +
          occ_func_2_0(5) * occ_func_3_0(8) * occ_func_3_0(2) *
              occ_func_3_0(0) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_3_0(20) *
              occ_func_3_0(6) +
          occ_func_2_0(0) * occ_func_3_0(31) * occ_func_3_0(43) *
              occ_func_3_0(35) +
          occ_func_2_0(0) * occ_func_3_0(47) * occ_func_3_0(31) *
              occ_func_3_0(39) +
          occ_func_2_0(0) * occ_func_3_0(9) * occ_func_3_0(3) *
              occ_func_3_0(5) +
          occ_func_2_0(0) * occ_func_3_0(3) * occ_func_3_0(11) *
              occ_func_3_0(7) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_3_0(9) *
              occ_func_3_0(13) +
          occ_func_2_0(13) * occ_func_3_0(10) * occ_func_3_0(8) *
              occ_func_3_0(0) +
          occ_func_2_0(11) * occ_func_3_0(24) * occ_func_3_0(0) *
              occ_func_3_0(6) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_3_0(22) *
              occ_func_3_0(4) +
          occ_func_2_0(51) * occ_func_3_0(8) * occ_func_3_0(10) *
              occ_func_3_0(0) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_3_0(24) *
              occ_func_3_0(6) +
          occ_func_2_0(43) * occ_func_3_0(22) * occ_func_3_0(0) *
              occ_func_3_0(4) +
          occ_func_2_0(47) * occ_func_3_0(28) * occ_func_3_0(0) *
              occ_func_3_0(12) +
          occ_func_2_0(39) * occ_func_3_0(10) * occ_func_3_0(2) *
              occ_func_3_0(0) +
          occ_func_2_0(31) * occ_func_3_0(0) * occ_func_3_0(18) *
              occ_func_3_0(4) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_3_0(26) *
              occ_func_3_0(12) +
          occ_func_2_0(35) * occ_func_3_0(2) * occ_func_3_0(8) *
              occ_func_3_0(0) +
          occ_func_2_0(31) * occ_func_3_0(20) * occ_func_3_0(0) *
              occ_func_3_0(6) +
          occ_func_2_0(0) * occ_func_3_0(39) * occ_func_3_0(51) *
              occ_func_3_0(47) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_3_0(35) *
              occ_func_3_0(43) +
          occ_func_2_0(0) * occ_func_3_0(35) * occ_func_3_0(39) *
              occ_func_3_0(31) +
          occ_func_2_0(39) * occ_func_3_0(0) * occ_func_3_0(24) *
              occ_func_3_0(10) +
          occ_func_2_0(35) * occ_func_3_0(22) * occ_func_3_0(0) *
              occ_func_3_0(8) +
          occ_func_2_0(31) * occ_func_3_0(4) * occ_func_3_0(6) *
              occ_func_3_0(0) +
          occ_func_2_0(7) * occ_func_3_0(24) * occ_func_3_0(0) *
              occ_func_3_0(10) +
          occ_func_2_0(5) * occ_func_3_0(0) * occ_func_3_0(22) *
              occ_func_3_0(8) +
          occ_func_2_0(3) * occ_func_3_0(6) * occ_func_3_0(4) *
              occ_func_3_0(0) +
          occ_func_2_0(13) * occ_func_3_0(0) * occ_func_3_0(28) *
              occ_func_3_0(10) +
          occ_func_2_0(9) * occ_func_3_0(4) * occ_func_3_0(12) *
              occ_func_3_0(0) +
          occ_func_2_0(5) * occ_func_3_0(18) * occ_func_3_0(0) *
              occ_func_3_0(2) +
          occ_func_2_0(13) * occ_func_3_0(26) * occ_func_3_0(0) *
              occ_func_3_0(8) +
          occ_func_2_0(11) * occ_func_3_0(12) * occ_func_3_0(6) *
              occ_func_3_0(0) +
          occ_func_2_0(7) * occ_func_3_0(0) * occ_func_3_0(20) *
              occ_func_3_0(2) +
          occ_func_2_0(0) * occ_func_3_0(13) * occ_func_3_0(7) *
              occ_func_3_0(11) +
          occ_func_2_0(0) * occ_func_3_0(5) * occ_func_3_0(13) *
              occ_func_3_0(9) +
          occ_func_2_0(51) * occ_func_3_0(28) * occ_func_3_0(0) *
              occ_func_3_0(10) +
          occ_func_2_0(43) * occ_func_3_0(12) * occ_func_3_0(4) *
              occ_func_3_0(0) +
          occ_func_2_0(35) * occ_func_3_0(0) * occ_func_3_0(18) *
              occ_func_3_0(2) +
          occ_func_2_0(51) * occ_func_3_0(0) * occ_func_3_0(26) *
              occ_func_3_0(8) +
          occ_func_2_0(47) * occ_func_3_0(6) * occ_func_3_0(12) *
              occ_func_3_0(0) +
          occ_func_2_0(39) * occ_func_3_0(20) * occ_func_3_0(0) *
              occ_func_3_0(2) +
          occ_func_2_0(0) * occ_func_3_0(7) * occ_func_3_0(5) *
              occ_func_3_0(3)) /
         24.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_69_0_at_1() const {
  return (occ_func_2_0(40) * occ_func_3_0(1) * occ_func_3_0(25) *
              occ_func_3_0(11) +
          occ_func_2_0(36) * occ_func_3_0(23) * occ_func_3_0(1) *
              occ_func_3_0(9) +
          occ_func_2_0(32) * occ_func_3_0(5) * occ_func_3_0(7) *
              occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(4) * occ_func_3_0(12) *
              occ_func_3_0(8) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_3_0(6) *
              occ_func_3_0(10) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_3_0(27) *
              occ_func_3_0(9) +
          occ_func_2_0(48) * occ_func_3_0(7) * occ_func_3_0(13) *
              occ_func_3_0(1) +
          occ_func_2_0(40) * occ_func_3_0(21) * occ_func_3_0(1) *
              occ_func_3_0(3) +
          occ_func_2_0(52) * occ_func_3_0(29) * occ_func_3_0(1) *
              occ_func_3_0(11) +
          occ_func_2_0(44) * occ_func_3_0(13) * occ_func_3_0(5) *
              occ_func_3_0(1) +
          occ_func_2_0(36) * occ_func_3_0(1) * occ_func_3_0(19) *
              occ_func_3_0(3) +
          occ_func_2_0(12) * occ_func_3_0(27) * occ_func_3_0(1) *
              occ_func_3_0(9) +
          occ_func_2_0(10) * occ_func_3_0(13) * occ_func_3_0(7) *
              occ_func_3_0(1) +
          occ_func_2_0(6) * occ_func_3_0(1) * occ_func_3_0(21) *
              occ_func_3_0(3) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_3_0(29) *
              occ_func_3_0(11) +
          occ_func_2_0(8) * occ_func_3_0(5) * occ_func_3_0(13) *
              occ_func_3_0(1) +
          occ_func_2_0(4) * occ_func_3_0(19) * occ_func_3_0(1) *
              occ_func_3_0(3) +
          occ_func_2_0(6) * occ_func_3_0(25) * occ_func_3_0(1) *
              occ_func_3_0(11) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_3_0(23) *
              occ_func_3_0(9) +
          occ_func_2_0(2) * occ_func_3_0(7) * occ_func_3_0(5) *
              occ_func_3_0(1) +
          occ_func_2_0(1) * occ_func_3_0(6) * occ_func_3_0(4) *
              occ_func_3_0(2) +
          occ_func_2_0(1) * occ_func_3_0(36) * occ_func_3_0(40) *
              occ_func_3_0(32) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_3_0(36) *
              occ_func_3_0(44) +
          occ_func_2_0(1) * occ_func_3_0(40) * occ_func_3_0(52) *
              occ_func_3_0(48) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_3_0(27) *
              occ_func_3_0(13) +
          occ_func_2_0(36) * occ_func_3_0(3) * occ_func_3_0(9) *
              occ_func_3_0(1) +
          occ_func_2_0(32) * occ_func_3_0(21) * occ_func_3_0(1) *
              occ_func_3_0(7) +
          occ_func_2_0(48) * occ_func_3_0(29) * occ_func_3_0(1) *
              occ_func_3_0(13) +
          occ_func_2_0(40) * occ_func_3_0(11) * occ_func_3_0(3) *
              occ_func_3_0(1) +
          occ_func_2_0(32) * occ_func_3_0(1) * occ_func_3_0(19) *
              occ_func_3_0(5) +
          occ_func_2_0(52) * occ_func_3_0(9) * occ_func_3_0(11) *
              occ_func_3_0(1) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_3_0(25) *
              occ_func_3_0(7) +
          occ_func_2_0(44) * occ_func_3_0(23) * occ_func_3_0(1) *
              occ_func_3_0(5) +
          occ_func_2_0(1) * occ_func_3_0(44) * occ_func_3_0(48) *
              occ_func_3_0(52) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_3_0(8) *
              occ_func_3_0(12) +
          occ_func_2_0(1) * occ_func_3_0(2) * occ_func_3_0(10) *
              occ_func_3_0(6) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_3_0(2) *
              occ_func_3_0(4) +
          occ_func_2_0(8) * occ_func_3_0(27) * occ_func_3_0(1) *
              occ_func_3_0(13) +
          occ_func_2_0(4) * occ_func_3_0(9) * occ_func_3_0(3) *
              occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_3_0(1) * occ_func_3_0(21) *
              occ_func_3_0(7) +
          occ_func_2_0(10) * occ_func_3_0(1) * occ_func_3_0(29) *
              occ_func_3_0(13) +
          occ_func_2_0(6) * occ_func_3_0(3) * occ_func_3_0(11) *
              occ_func_3_0(1) +
          occ_func_2_0(2) * occ_func_3_0(19) * occ_func_3_0(1) *
              occ_func_3_0(5) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_3_0(32) *
              occ_func_3_0(40) +
          occ_func_2_0(1) * occ_func_3_0(32) * occ_func_3_0(44) *
              occ_func_3_0(36) +
          occ_func_2_0(12) * occ_func_3_0(11) * occ_func_3_0(9) *
              occ_func_3_0(1) +
          occ_func_2_0(10) * occ_func_3_0(25) * occ_func_3_0(1) *
              occ_func_3_0(7) +
          occ_func_2_0(8) * occ_func_3_0(1) * occ_func_3_0(23) *
              occ_func_3_0(5)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_69_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(43) * occ_func_3_0(47) * occ_func_3_0(51) +
          occ_func_2_0(11) * occ_func_3_0(28) * occ_func_3_0(12) +
          occ_func_2_0(7) * occ_func_3_0(2) * occ_func_3_0(10) +
          occ_func_2_0(3) * occ_func_3_0(18) * occ_func_3_0(4) +
          occ_func_2_0(9) * occ_func_3_0(26) * occ_func_3_0(12) +
          occ_func_2_0(5) * occ_func_3_0(8) * occ_func_3_0(2) +
          occ_func_2_0(3) * occ_func_3_0(20) * occ_func_3_0(6) +
          occ_func_3_0(31) * occ_func_3_0(43) * occ_func_3_0(35) +
          occ_func_3_0(47) * occ_func_3_0(31) * occ_func_3_0(39) +
          occ_func_3_0(9) * occ_func_3_0(3) * occ_func_3_0(5) +
          occ_func_3_0(3) * occ_func_3_0(11) * occ_func_3_0(7) +
          occ_func_3_0(11) * occ_func_3_0(9) * occ_func_3_0(13) +
          occ_func_2_0(13) * occ_func_3_0(10) * occ_func_3_0(8) +
          occ_func_2_0(11) * occ_func_3_0(24) * occ_func_3_0(6) +
          occ_func_2_0(9) * occ_func_3_0(22) * occ_func_3_0(4) +
          occ_func_2_0(51) * occ_func_3_0(8) * occ_func_3_0(10) +
          occ_func_2_0(47) * occ_func_3_0(24) * occ_func_3_0(6) +
          occ_func_2_0(43) * occ_func_3_0(22) * occ_func_3_0(4) +
          occ_func_2_0(47) * occ_func_3_0(28) * occ_func_3_0(12) +
          occ_func_2_0(39) * occ_func_3_0(10) * occ_func_3_0(2) +
          occ_func_2_0(31) * occ_func_3_0(18) * occ_func_3_0(4) +
          occ_func_2_0(43) * occ_func_3_0(26) * occ_func_3_0(12) +
          occ_func_2_0(35) * occ_func_3_0(2) * occ_func_3_0(8) +
          occ_func_2_0(31) * occ_func_3_0(20) * occ_func_3_0(6) +
          occ_func_3_0(39) * occ_func_3_0(51) * occ_func_3_0(47) +
          occ_func_3_0(51) * occ_func_3_0(35) * occ_func_3_0(43) +
          occ_func_3_0(35) * occ_func_3_0(39) * occ_func_3_0(31) +
          occ_func_2_0(39) * occ_func_3_0(24) * occ_func_3_0(10) +
          occ_func_2_0(35) * occ_func_3_0(22) * occ_func_3_0(8) +
          occ_func_2_0(31) * occ_func_3_0(4) * occ_func_3_0(6) +
          occ_func_2_0(7) * occ_func_3_0(24) * occ_func_3_0(10) +
          occ_func_2_0(5) * occ_func_3_0(22) * occ_func_3_0(8) +
          occ_func_2_0(3) * occ_func_3_0(6) * occ_func_3_0(4) +
          occ_func_2_0(13) * occ_func_3_0(28) * occ_func_3_0(10) +
          occ_func_2_0(9) * occ_func_3_0(4) * occ_func_3_0(12) +
          occ_func_2_0(5) * occ_func_3_0(18) * occ_func_3_0(2) +
          occ_func_2_0(13) * occ_func_3_0(26) * occ_func_3_0(8) +
          occ_func_2_0(11) * occ_func_3_0(12) * occ_func_3_0(6) +
          occ_func_2_0(7) * occ_func_3_0(20) * occ_func_3_0(2) +
          occ_func_3_0(13) * occ_func_3_0(7) * occ_func_3_0(11) +
          occ_func_3_0(5) * occ_func_3_0(13) * occ_func_3_0(9) +
          occ_func_2_0(51) * occ_func_3_0(28) * occ_func_3_0(10) +
          occ_func_2_0(43) * occ_func_3_0(12) * occ_func_3_0(4) +
          occ_func_2_0(35) * occ_func_3_0(18) * occ_func_3_0(2) +
          occ_func_2_0(51) * occ_func_3_0(26) * occ_func_3_0(8) +
          occ_func_2_0(47) * occ_func_3_0(6) * occ_func_3_0(12) +
          occ_func_2_0(39) * occ_func_3_0(20) * occ_func_3_0(2) +
          occ_func_3_0(7) * occ_func_3_0(5) * occ_func_3_0(3)) /
         24.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_69_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(40) * occ_func_3_0(25) * occ_func_3_0(11) +
          occ_func_2_0(36) * occ_func_3_0(23) * occ_func_3_0(9) +
          occ_func_2_0(32) * occ_func_3_0(5) * occ_func_3_0(7) +
          occ_func_3_0(4) * occ_func_3_0(12) * occ_func_3_0(8) +
          occ_func_3_0(12) * occ_func_3_0(6) * occ_func_3_0(10) +
          occ_func_2_0(52) * occ_func_3_0(27) * occ_func_3_0(9) +
          occ_func_2_0(48) * occ_func_3_0(7) * occ_func_3_0(13) +
          occ_func_2_0(40) * occ_func_3_0(21) * occ_func_3_0(3) +
          occ_func_2_0(52) * occ_func_3_0(29) * occ_func_3_0(11) +
          occ_func_2_0(44) * occ_func_3_0(13) * occ_func_3_0(5) +
          occ_func_2_0(36) * occ_func_3_0(19) * occ_func_3_0(3) +
          occ_func_2_0(12) * occ_func_3_0(27) * occ_func_3_0(9) +
          occ_func_2_0(10) * occ_func_3_0(13) * occ_func_3_0(7) +
          occ_func_2_0(6) * occ_func_3_0(21) * occ_func_3_0(3) +
          occ_func_2_0(12) * occ_func_3_0(29) * occ_func_3_0(11) +
          occ_func_2_0(8) * occ_func_3_0(5) * occ_func_3_0(13) +
          occ_func_2_0(4) * occ_func_3_0(19) * occ_func_3_0(3) +
          occ_func_2_0(6) * occ_func_3_0(25) * occ_func_3_0(11) +
          occ_func_2_0(4) * occ_func_3_0(23) * occ_func_3_0(9) +
          occ_func_2_0(2) * occ_func_3_0(7) * occ_func_3_0(5) +
          occ_func_3_0(6) * occ_func_3_0(4) * occ_func_3_0(2) +
          occ_func_3_0(36) * occ_func_3_0(40) * occ_func_3_0(32) +
          occ_func_3_0(52) * occ_func_3_0(36) * occ_func_3_0(44) +
          occ_func_3_0(40) * occ_func_3_0(52) * occ_func_3_0(48) +
          occ_func_2_0(44) * occ_func_3_0(27) * occ_func_3_0(13) +
          occ_func_2_0(36) * occ_func_3_0(3) * occ_func_3_0(9) +
          occ_func_2_0(32) * occ_func_3_0(21) * occ_func_3_0(7) +
          occ_func_2_0(48) * occ_func_3_0(29) * occ_func_3_0(13) +
          occ_func_2_0(40) * occ_func_3_0(11) * occ_func_3_0(3) +
          occ_func_2_0(32) * occ_func_3_0(19) * occ_func_3_0(5) +
          occ_func_2_0(52) * occ_func_3_0(9) * occ_func_3_0(11) +
          occ_func_2_0(48) * occ_func_3_0(25) * occ_func_3_0(7) +
          occ_func_2_0(44) * occ_func_3_0(23) * occ_func_3_0(5) +
          occ_func_3_0(44) * occ_func_3_0(48) * occ_func_3_0(52) +
          occ_func_3_0(10) * occ_func_3_0(8) * occ_func_3_0(12) +
          occ_func_3_0(2) * occ_func_3_0(10) * occ_func_3_0(6) +
          occ_func_3_0(8) * occ_func_3_0(2) * occ_func_3_0(4) +
          occ_func_2_0(8) * occ_func_3_0(27) * occ_func_3_0(13) +
          occ_func_2_0(4) * occ_func_3_0(9) * occ_func_3_0(3) +
          occ_func_2_0(2) * occ_func_3_0(21) * occ_func_3_0(7) +
          occ_func_2_0(10) * occ_func_3_0(29) * occ_func_3_0(13) +
          occ_func_2_0(6) * occ_func_3_0(3) * occ_func_3_0(11) +
          occ_func_2_0(2) * occ_func_3_0(19) * occ_func_3_0(5) +
          occ_func_3_0(48) * occ_func_3_0(32) * occ_func_3_0(40) +
          occ_func_3_0(32) * occ_func_3_0(44) * occ_func_3_0(36) +
          occ_func_2_0(12) * occ_func_3_0(11) * occ_func_3_0(9) +
          occ_func_2_0(10) * occ_func_3_0(25) * occ_func_3_0(7) +
          occ_func_2_0(8) * occ_func_3_0(23) * occ_func_3_0(5)) /
         24.;
}

/**** Basis functions for orbit 70****
0.3333333 0.6666666 0.2500000 Va  O

1.3333333 0.6666666 -0.2500000 Va  O

1.3333333 1.6666667 -0.2500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_70_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(47) * occ_func_3_0(51) *
              occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_3_0(8) *
              occ_func_2_0(27) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_3_0(12) *
              occ_func_2_0(11) +
          occ_func_2_0(48) * occ_func_3_0(13) * occ_func_3_0(1) *
              occ_func_2_0(44) +
          occ_func_2_0(26) * occ_func_3_0(43) * occ_func_3_0(51) *
              occ_func_2_0(0) +
          occ_func_2_0(28) * occ_func_3_0(11) * occ_func_3_0(13) *
              occ_func_2_0(0) +
          occ_func_2_0(8) * occ_func_3_0(13) * occ_func_3_0(1) *
              occ_func_2_0(10) +
          occ_func_2_0(0) * occ_func_3_0(9) * occ_func_3_0(13) *
              occ_func_2_0(26) +
          occ_func_2_0(29) * occ_func_3_0(12) * occ_func_3_0(10) *
              occ_func_2_0(1) +
          occ_func_2_0(27) * occ_func_3_0(52) * occ_func_3_0(44) *
              occ_func_2_0(1) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_3_0(12) *
              occ_func_2_0(43) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_3_0(48) *
              occ_func_2_0(29)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_70_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(47) * occ_func_3_0(51) *
              occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(31) * occ_func_3_0(35) *
              occ_func_2_0(0) +
          occ_func_2_0(7) * occ_func_3_0(10) * occ_func_3_0(0) *
              occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_3_0(4) *
              occ_func_2_0(9) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_3_0(12) *
              occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(2) * occ_func_3_0(0) *
              occ_func_2_0(7) +
          occ_func_2_0(24) * occ_func_3_0(47) * occ_func_3_0(39) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(43) * occ_func_3_0(35) *
              occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(43) * occ_func_3_0(51) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(31) * occ_func_3_0(39) *
              occ_func_2_0(20) +
          occ_func_2_0(28) * occ_func_3_0(11) * occ_func_3_0(13) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(3) * occ_func_3_0(5) *
              occ_func_2_0(18) +
          occ_func_2_0(0) * occ_func_3_0(11) * occ_func_3_0(7) *
              occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_3_0(9) * occ_func_3_0(5) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(9) * occ_func_3_0(13) *
              occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_3_0(3) * occ_func_3_0(7) *
              occ_func_2_0(0) +
          occ_func_2_0(13) * occ_func_3_0(8) * occ_func_3_0(0) *
              occ_func_2_0(5) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_3_0(6) *
              occ_func_2_0(3) +
          occ_func_2_0(51) * occ_func_3_0(10) * occ_func_3_0(0) *
              occ_func_2_0(39) +
          occ_func_2_0(43) * occ_func_3_0(0) * occ_func_3_0(4) *
              occ_func_2_0(31) +
          occ_func_2_0(47) * occ_func_3_0(0) * occ_func_3_0(12) *
              occ_func_2_0(43) +
          occ_func_2_0(39) * occ_func_3_0(2) * occ_func_3_0(0) *
              occ_func_2_0(35) +
          occ_func_2_0(35) * occ_func_3_0(8) * occ_func_3_0(0) *
              occ_func_2_0(51) +
          occ_func_2_0(31) * occ_func_3_0(0) * occ_func_3_0(6) *
              occ_func_2_0(47)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_70_0_at_1() const {
  return (occ_func_2_0(36) * occ_func_3_0(1) * occ_func_3_0(9) *
              occ_func_2_0(52) +
          occ_func_2_0(32) * occ_func_3_0(7) * occ_func_3_0(1) *
              occ_func_2_0(48) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_3_0(8) *
              occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(6) * occ_func_3_0(2) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(6) * occ_func_3_0(10) *
              occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(4) * occ_func_3_0(8) *
              occ_func_2_0(1) +
          occ_func_2_0(48) * occ_func_3_0(13) * occ_func_3_0(1) *
              occ_func_2_0(44) +
          occ_func_2_0(40) * occ_func_3_0(1) * occ_func_3_0(3) *
              occ_func_2_0(36) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_3_0(11) *
              occ_func_2_0(40) +
          occ_func_2_0(44) * occ_func_3_0(5) * occ_func_3_0(1) *
              occ_func_2_0(32) +
          occ_func_2_0(12) * occ_func_3_0(1) * occ_func_3_0(9) *
              occ_func_2_0(4) +
          occ_func_2_0(10) * occ_func_3_0(7) * occ_func_3_0(1) *
              occ_func_2_0(2) +
          occ_func_2_0(8) * occ_func_3_0(13) * occ_func_3_0(1) *
              occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_3_0(1) * occ_func_3_0(3) *
              occ_func_2_0(6) +
          occ_func_2_0(6) * occ_func_3_0(1) * occ_func_3_0(11) *
              occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(5) * occ_func_3_0(1) *
              occ_func_2_0(8) +
          occ_func_2_0(29) * occ_func_3_0(12) * occ_func_3_0(10) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(4) * occ_func_3_0(2) *
              occ_func_2_0(19) +
          occ_func_2_0(27) * occ_func_3_0(52) * occ_func_3_0(44) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(40) * occ_func_3_0(32) *
              occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_3_0(40) * occ_func_3_0(48) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(36) * occ_func_3_0(44) *
              occ_func_2_0(23) +
          occ_func_2_0(1) * occ_func_3_0(52) * occ_func_3_0(48) *
              occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(36) * occ_func_3_0(32) *
              occ_func_2_0(1)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_70_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(47) * occ_func_3_0(51) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(31) * occ_func_3_0(35) +
          occ_func_2_0(7) * occ_func_3_0(10) * occ_func_2_0(13) +
          occ_func_2_0(3) * occ_func_3_0(4) * occ_func_2_0(9) +
          occ_func_2_0(9) * occ_func_3_0(12) * occ_func_2_0(11) +
          occ_func_2_0(5) * occ_func_3_0(2) * occ_func_2_0(7) +
          occ_func_2_0(24) * occ_func_3_0(47) * occ_func_3_0(39) +
          occ_func_3_0(43) * occ_func_3_0(35) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(43) * occ_func_3_0(51) +
          occ_func_3_0(31) * occ_func_3_0(39) * occ_func_2_0(20) +
          occ_func_2_0(28) * occ_func_3_0(11) * occ_func_3_0(13) +
          occ_func_3_0(3) * occ_func_3_0(5) * occ_func_2_0(18) +
          occ_func_3_0(11) * occ_func_3_0(7) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_3_0(9) * occ_func_3_0(5) +
          occ_func_3_0(9) * occ_func_3_0(13) * occ_func_2_0(26) +
          occ_func_2_0(20) * occ_func_3_0(3) * occ_func_3_0(7) +
          occ_func_2_0(13) * occ_func_3_0(8) * occ_func_2_0(5) +
          occ_func_2_0(11) * occ_func_3_0(6) * occ_func_2_0(3) +
          occ_func_2_0(51) * occ_func_3_0(10) * occ_func_2_0(39) +
          occ_func_2_0(43) * occ_func_3_0(4) * occ_func_2_0(31) +
          occ_func_2_0(47) * occ_func_3_0(12) * occ_func_2_0(43) +
          occ_func_2_0(39) * occ_func_3_0(2) * occ_func_2_0(35) +
          occ_func_2_0(35) * occ_func_3_0(8) * occ_func_2_0(51) +
          occ_func_2_0(31) * occ_func_3_0(6) * occ_func_2_0(47)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_70_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(36) * occ_func_3_0(9) * occ_func_2_0(52) +
          occ_func_2_0(32) * occ_func_3_0(7) * occ_func_2_0(48) +
          occ_func_3_0(12) * occ_func_3_0(8) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(6) * occ_func_3_0(2) +
          occ_func_3_0(6) * occ_func_3_0(10) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(4) * occ_func_3_0(8) +
          occ_func_2_0(48) * occ_func_3_0(13) * occ_func_2_0(44) +
          occ_func_2_0(40) * occ_func_3_0(3) * occ_func_2_0(36) +
          occ_func_2_0(52) * occ_func_3_0(11) * occ_func_2_0(40) +
          occ_func_2_0(44) * occ_func_3_0(5) * occ_func_2_0(32) +
          occ_func_2_0(12) * occ_func_3_0(9) * occ_func_2_0(4) +
          occ_func_2_0(10) * occ_func_3_0(7) * occ_func_2_0(2) +
          occ_func_2_0(8) * occ_func_3_0(13) * occ_func_2_0(10) +
          occ_func_2_0(4) * occ_func_3_0(3) * occ_func_2_0(6) +
          occ_func_2_0(6) * occ_func_3_0(11) * occ_func_2_0(12) +
          occ_func_2_0(2) * occ_func_3_0(5) * occ_func_2_0(8) +
          occ_func_2_0(29) * occ_func_3_0(12) * occ_func_3_0(10) +
          occ_func_3_0(4) * occ_func_3_0(2) * occ_func_2_0(19) +
          occ_func_2_0(27) * occ_func_3_0(52) * occ_func_3_0(44) +
          occ_func_3_0(40) * occ_func_3_0(32) * occ_func_2_0(21) +
          occ_func_2_0(25) * occ_func_3_0(40) * occ_func_3_0(48) +
          occ_func_3_0(36) * occ_func_3_0(44) * occ_func_2_0(23) +
          occ_func_3_0(52) * occ_func_3_0(48) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(36) * occ_func_3_0(32)) /
         12.;
}

/**** Basis functions for orbit 71****
0.3333333 0.6666666 0.2500000 Va  O

1.3333333 0.6666666 -0.2500000 Va  O

1.3333333 0.6666666 0.7500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_71_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(47) * occ_func_3_0(11) *
              occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_3_0(52) *
              occ_func_2_0(27) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_3_0(16) *
              occ_func_2_0(11) +
          occ_func_2_0(24) * occ_func_3_0(47) * occ_func_3_0(11) *
              occ_func_2_0(0) +
          occ_func_2_0(26) * occ_func_3_0(43) * occ_func_3_0(9) *
              occ_func_2_0(0) +
          occ_func_2_0(29) * occ_func_3_0(12) * occ_func_3_0(52) *
              occ_func_2_0(1) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_3_0(13) *
              occ_func_2_0(28) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_3_0(17) *
              occ_func_2_0(44) +
          occ_func_2_0(26) * occ_func_3_0(51) * occ_func_3_0(13) *
              occ_func_2_0(0) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_3_0(10) *
              occ_func_2_0(29) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_3_0(44) *
              occ_func_2_0(27) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_3_0(48) *
              occ_func_2_0(25)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_71_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_3_0(47) * occ_func_3_0(11) *
              occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(31) * occ_func_3_0(3) *
              occ_func_2_0(0) +
          occ_func_2_0(3) * occ_func_3_0(0) * occ_func_3_0(16) *
              occ_func_2_0(9) +
          occ_func_2_0(31) * occ_func_3_0(14) * occ_func_3_0(0) *
              occ_func_2_0(43) +
          occ_func_2_0(9) * occ_func_3_0(0) * occ_func_3_0(16) *
              occ_func_2_0(11) +
          occ_func_2_0(43) * occ_func_3_0(14) * occ_func_3_0(0) *
              occ_func_2_0(47) +
          occ_func_2_0(24) * occ_func_3_0(47) * occ_func_3_0(11) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(43) * occ_func_3_0(9) *
              occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(43) * occ_func_3_0(9) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(31) * occ_func_3_0(3) *
              occ_func_2_0(20) +
          occ_func_2_0(11) * occ_func_3_0(0) * occ_func_3_0(16) *
              occ_func_2_0(3) +
          occ_func_2_0(47) * occ_func_3_0(14) * occ_func_3_0(0) *
              occ_func_2_0(31) +
          occ_func_2_0(0) * occ_func_3_0(51) * occ_func_3_0(13) *
              occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(35) * occ_func_3_0(5) *
              occ_func_2_0(0) +
          occ_func_2_0(24) * occ_func_3_0(39) * occ_func_3_0(7) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(35) * occ_func_3_0(5) *
              occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(51) * occ_func_3_0(13) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_3_0(39) * occ_func_3_0(7) *
              occ_func_2_0(20) +
          occ_func_2_0(5) * occ_func_3_0(16) * occ_func_3_0(0) *
              occ_func_2_0(13) +
          occ_func_2_0(35) * occ_func_3_0(0) * occ_func_3_0(14) *
              occ_func_2_0(51) +
          occ_func_2_0(7) * occ_func_3_0(0) * occ_func_3_0(16) *
              occ_func_2_0(13) +
          occ_func_2_0(39) * occ_func_3_0(14) * occ_func_3_0(0) *
              occ_func_2_0(51) +
          occ_func_2_0(5) * occ_func_3_0(0) * occ_func_3_0(16) *
              occ_func_2_0(7) +
          occ_func_2_0(35) * occ_func_3_0(14) * occ_func_3_0(0) *
              occ_func_2_0(39)) /
         12.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_71_0_at_1() const {
  return (occ_func_2_0(36) * occ_func_3_0(1) * occ_func_3_0(17) *
              occ_func_2_0(52) +
          occ_func_2_0(4) * occ_func_3_0(15) * occ_func_3_0(1) *
              occ_func_2_0(12) +
          occ_func_2_0(1) * occ_func_3_0(12) * occ_func_3_0(52) *
              occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(6) * occ_func_3_0(40) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(6) * occ_func_3_0(40) *
              occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(4) * occ_func_3_0(36) *
              occ_func_2_0(1) +
          occ_func_2_0(40) * occ_func_3_0(1) * occ_func_3_0(17) *
              occ_func_2_0(36) +
          occ_func_2_0(6) * occ_func_3_0(15) * occ_func_3_0(1) *
              occ_func_2_0(4) +
          occ_func_2_0(52) * occ_func_3_0(1) * occ_func_3_0(17) *
              occ_func_2_0(40) +
          occ_func_2_0(12) * occ_func_3_0(15) * occ_func_3_0(1) *
              occ_func_2_0(6) +
          occ_func_2_0(29) * occ_func_3_0(12) * occ_func_3_0(52) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(4) * occ_func_3_0(36) *
              occ_func_2_0(19) +
          occ_func_2_0(32) * occ_func_3_0(1) * occ_func_3_0(17) *
              occ_func_2_0(48) +
          occ_func_2_0(2) * occ_func_3_0(15) * occ_func_3_0(1) *
              occ_func_2_0(10) +
          occ_func_2_0(48) * occ_func_3_0(1) * occ_func_3_0(17) *
              occ_func_2_0(44) +
          occ_func_2_0(10) * occ_func_3_0(15) * occ_func_3_0(1) *
              occ_func_2_0(8) +
          occ_func_2_0(44) * occ_func_3_0(1) * occ_func_3_0(17) *
              occ_func_2_0(32) +
          occ_func_2_0(8) * occ_func_3_0(15) * occ_func_3_0(1) *
              occ_func_2_0(2) +
          occ_func_2_0(1) * occ_func_3_0(48) * occ_func_3_0(10) *
              occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(32) * occ_func_3_0(2) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(8) * occ_func_3_0(44) *
              occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(2) * occ_func_3_0(32) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_3_0(10) * occ_func_3_0(48) *
              occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(8) * occ_func_3_0(44) *
              occ_func_2_0(1)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_71_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_3_0(47) * occ_func_3_0(11) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(31) * occ_func_3_0(3) +
          occ_func_2_0(3) * occ_func_3_0(16) * occ_func_2_0(9) +
          occ_func_2_0(31) * occ_func_3_0(14) * occ_func_2_0(43) +
          occ_func_2_0(9) * occ_func_3_0(16) * occ_func_2_0(11) +
          occ_func_2_0(43) * occ_func_3_0(14) * occ_func_2_0(47) +
          occ_func_2_0(24) * occ_func_3_0(47) * occ_func_3_0(11) +
          occ_func_3_0(43) * occ_func_3_0(9) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(43) * occ_func_3_0(9) +
          occ_func_3_0(31) * occ_func_3_0(3) * occ_func_2_0(20) +
          occ_func_2_0(11) * occ_func_3_0(16) * occ_func_2_0(3) +
          occ_func_2_0(47) * occ_func_3_0(14) * occ_func_2_0(31) +
          occ_func_3_0(51) * occ_func_3_0(13) * occ_func_2_0(28) +
          occ_func_2_0(18) * occ_func_3_0(35) * occ_func_3_0(5) +
          occ_func_2_0(24) * occ_func_3_0(39) * occ_func_3_0(7) +
          occ_func_3_0(35) * occ_func_3_0(5) * occ_func_2_0(22) +
          occ_func_2_0(26) * occ_func_3_0(51) * occ_func_3_0(13) +
          occ_func_3_0(39) * occ_func_3_0(7) * occ_func_2_0(20) +
          occ_func_2_0(5) * occ_func_3_0(16) * occ_func_2_0(13) +
          occ_func_2_0(35) * occ_func_3_0(14) * occ_func_2_0(51) +
          occ_func_2_0(7) * occ_func_3_0(16) * occ_func_2_0(13) +
          occ_func_2_0(39) * occ_func_3_0(14) * occ_func_2_0(51) +
          occ_func_2_0(5) * occ_func_3_0(16) * occ_func_2_0(7) +
          occ_func_2_0(35) * occ_func_3_0(14) * occ_func_2_0(39)) /
         12.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_71_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(36) * occ_func_3_0(17) * occ_func_2_0(52) +
          occ_func_2_0(4) * occ_func_3_0(15) * occ_func_2_0(12) +
          occ_func_3_0(12) * occ_func_3_0(52) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(6) * occ_func_3_0(40) +
          occ_func_3_0(6) * occ_func_3_0(40) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(4) * occ_func_3_0(36) +
          occ_func_2_0(40) * occ_func_3_0(17) * occ_func_2_0(36) +
          occ_func_2_0(6) * occ_func_3_0(15) * occ_func_2_0(4) +
          occ_func_2_0(52) * occ_func_3_0(17) * occ_func_2_0(40) +
          occ_func_2_0(12) * occ_func_3_0(15) * occ_func_2_0(6) +
          occ_func_2_0(29) * occ_func_3_0(12) * occ_func_3_0(52) +
          occ_func_3_0(4) * occ_func_3_0(36) * occ_func_2_0(19) +
          occ_func_2_0(32) * occ_func_3_0(17) * occ_func_2_0(48) +
          occ_func_2_0(2) * occ_func_3_0(15) * occ_func_2_0(10) +
          occ_func_2_0(48) * occ_func_3_0(17) * occ_func_2_0(44) +
          occ_func_2_0(10) * occ_func_3_0(15) * occ_func_2_0(8) +
          occ_func_2_0(44) * occ_func_3_0(17) * occ_func_2_0(32) +
          occ_func_2_0(8) * occ_func_3_0(15) * occ_func_2_0(2) +
          occ_func_3_0(48) * occ_func_3_0(10) * occ_func_2_0(29) +
          occ_func_2_0(19) * occ_func_3_0(32) * occ_func_3_0(2) +
          occ_func_3_0(8) * occ_func_3_0(44) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_3_0(2) * occ_func_3_0(32) +
          occ_func_3_0(10) * occ_func_3_0(48) * occ_func_2_0(25) +
          occ_func_2_0(23) * occ_func_3_0(8) * occ_func_3_0(44)) /
         12.;
}

/**** Basis functions for orbit 72****
0.3333333 0.6666666 0.2500000 Va  O

1.3333333 -0.3333333 0.2500000 Va  O

1.3333333 0.6666666 0.2500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_72_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(24) * occ_func_2_0(10) *
              occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_2_0(29) * occ_func_2_0(13) *
              occ_func_2_0(27) +
          occ_func_2_0(0) * occ_func_2_0(26) * occ_func_2_0(12) *
              occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_2_0(25) * occ_func_2_0(11) *
              occ_func_2_0(29)) /
         4.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_72_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(24) * occ_func_2_0(10) *
              occ_func_2_0(28) +
          occ_func_2_0(22) * occ_func_2_0(0) * occ_func_2_0(8) *
              occ_func_2_0(26) +
          occ_func_2_0(4) * occ_func_2_0(6) * occ_func_2_0(0) *
              occ_func_2_0(12) +
          occ_func_2_0(18) * occ_func_2_0(20) * occ_func_2_0(2) *
              occ_func_2_0(0) +
          occ_func_2_0(0) * occ_func_2_0(26) * occ_func_2_0(12) *
              occ_func_2_0(28) +
          occ_func_2_0(2) * occ_func_2_0(8) * occ_func_2_0(0) *
              occ_func_2_0(10) +
          occ_func_2_0(20) * occ_func_2_0(0) * occ_func_2_0(6) *
              occ_func_2_0(24) +
          occ_func_2_0(18) * occ_func_2_0(22) * occ_func_2_0(4) *
              occ_func_2_0(0)) /
         4.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_72_0_at_1() const {
  return (occ_func_2_0(1) * occ_func_2_0(29) * occ_func_2_0(13) *
              occ_func_2_0(27) +
          occ_func_2_0(3) * occ_func_2_0(11) * occ_func_2_0(1) *
              occ_func_2_0(9) +
          occ_func_2_0(21) * occ_func_2_0(25) * occ_func_2_0(7) *
              occ_func_2_0(1) +
          occ_func_2_0(19) * occ_func_2_0(1) * occ_func_2_0(5) *
              occ_func_2_0(23) +
          occ_func_2_0(1) * occ_func_2_0(25) * occ_func_2_0(11) *
              occ_func_2_0(29) +
          occ_func_2_0(23) * occ_func_2_0(1) * occ_func_2_0(9) *
              occ_func_2_0(27) +
          occ_func_2_0(5) * occ_func_2_0(7) * occ_func_2_0(1) *
              occ_func_2_0(13) +
          occ_func_2_0(19) * occ_func_2_0(21) * occ_func_2_0(3) *
              occ_func_2_0(1)) /
         4.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_72_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(24) * occ_func_2_0(10) * occ_func_2_0(28) +
          occ_func_2_0(22) * occ_func_2_0(8) * occ_func_2_0(26) +
          occ_func_2_0(4) * occ_func_2_0(6) * occ_func_2_0(12) +
          occ_func_2_0(18) * occ_func_2_0(20) * occ_func_2_0(2) +
          occ_func_2_0(26) * occ_func_2_0(12) * occ_func_2_0(28) +
          occ_func_2_0(2) * occ_func_2_0(8) * occ_func_2_0(10) +
          occ_func_2_0(20) * occ_func_2_0(6) * occ_func_2_0(24) +
          occ_func_2_0(18) * occ_func_2_0(22) * occ_func_2_0(4)) /
         4.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_72_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(29) * occ_func_2_0(13) * occ_func_2_0(27) +
          occ_func_2_0(3) * occ_func_2_0(11) * occ_func_2_0(9) +
          occ_func_2_0(21) * occ_func_2_0(25) * occ_func_2_0(7) +
          occ_func_2_0(19) * occ_func_2_0(5) * occ_func_2_0(23) +
          occ_func_2_0(25) * occ_func_2_0(11) * occ_func_2_0(29) +
          occ_func_2_0(23) * occ_func_2_0(9) * occ_func_2_0(27) +
          occ_func_2_0(5) * occ_func_2_0(7) * occ_func_2_0(13) +
          occ_func_2_0(19) * occ_func_2_0(21) * occ_func_2_0(3)) /
         4.;
}

/**** Basis functions for orbit 73****
0.3333333 0.6666666 0.2500000 Va  O

1.3333333 -0.3333333 0.2500000 Va  O

1.3333333 0.6666666 -0.2500000 Va  O

2.3333333 1.6666667 0.2500000 Va  O

****/
template <typename Scalar>
Scalar ZrO_Clexulator::eval_bfunc_73_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(24) * occ_func_3_0(47) *
              occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_2_0(29) * occ_func_3_0(12) *
              occ_func_2_0(27) +
          occ_func_2_0(28) * occ_func_2_0(24) * occ_func_3_0(11) *
              occ_func_2_0(0) +
          occ_func_2_0(27) * occ_func_2_0(29) * occ_func_3_0(52) *
              occ_func_2_0(1) +
          occ_func_2_0(0) * occ_func_2_0(26) * occ_func_3_0(51) *
              occ_func_2_0(28) +
          occ_func_2_0(1) * occ_func_2_0(25) * occ_func_3_0(48) *
              occ_func_2_0(29) +
          occ_func_2_0(25) * occ_func_2_0(1) * occ_func_3_0(10) *
              occ_func_2_0(29) +
          occ_func_2_0(26) * occ_func_2_0(0) * occ_func_3_0(13) *
              occ_func_2_0(28)) /
         8.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_73_0_at_0() const {
  return (occ_func_2_0(0) * occ_func_2_0(24) * occ_func_3_0(47) *
              occ_func_2_0(28) +
          occ_func_2_0(22) * occ_func_2_0(0) * occ_func_3_0(43) *
              occ_func_2_0(26) +
          occ_func_2_0(18) * occ_func_2_0(20) * occ_func_3_0(31) *
              occ_func_2_0(0) +
          occ_func_2_0(3) * occ_func_2_0(11) * occ_func_3_0(0) *
              occ_func_2_0(9) +
          occ_func_2_0(28) * occ_func_2_0(24) * occ_func_3_0(11) *
              occ_func_2_0(0) +
          occ_func_2_0(26) * occ_func_2_0(0) * occ_func_3_0(9) *
              occ_func_2_0(22) +
          occ_func_2_0(0) * occ_func_2_0(20) * occ_func_3_0(3) *
              occ_func_2_0(18) +
          occ_func_2_0(43) * occ_func_2_0(47) * occ_func_3_0(0) *
              occ_func_2_0(31) +
          occ_func_2_0(0) * occ_func_2_0(26) * occ_func_3_0(51) *
              occ_func_2_0(28) +
          occ_func_2_0(20) * occ_func_2_0(0) * occ_func_3_0(39) *
              occ_func_2_0(24) +
          occ_func_2_0(18) * occ_func_2_0(22) * occ_func_3_0(35) *
              occ_func_2_0(0) +
          occ_func_2_0(35) * occ_func_2_0(39) * occ_func_3_0(0) *
              occ_func_2_0(51) +
          occ_func_2_0(7) * occ_func_2_0(5) * occ_func_3_0(0) *
              occ_func_2_0(13) +
          occ_func_2_0(26) * occ_func_2_0(0) * occ_func_3_0(13) *
              occ_func_2_0(28) +
          occ_func_2_0(0) * occ_func_2_0(20) * occ_func_3_0(7) *
              occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_2_0(18) * occ_func_3_0(5) *
              occ_func_2_0(0)) /
         8.;
}
template <typename Scalar>
Scalar ZrO_Clexulator::site_eval_bfunc_73_0_at_1() const {
  return (occ_func_2_0(36) * occ_func_2_0(40) * occ_func_3_0(1) *
              occ_func_2_0(52) +
          occ_func_2_0(1) * occ_func_2_0(29) * occ_func_3_0(12) *
              occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_2_0(25) * occ_func_3_0(6) *
              occ_func_2_0(1) +
          occ_func_2_0(19) * occ_func_2_0(1) * occ_func_3_0(4) *
              occ_func_2_0(23) +
          occ_func_2_0(12) * occ_func_2_0(6) * occ_func_3_0(1) *
              occ_func_2_0(4) +
          occ_func_2_0(27) * occ_func_2_0(29) * occ_func_3_0(52) *
              occ_func_2_0(1) +
          occ_func_2_0(1) * occ_func_2_0(25) * occ_func_3_0(40) *
              occ_func_2_0(21) +
          occ_func_2_0(23) * occ_func_2_0(1) * occ_func_3_0(36) *
              occ_func_2_0(19) +
          occ_func_2_0(32) * occ_func_2_0(44) * occ_func_3_0(1) *
              occ_func_2_0(48) +
          occ_func_2_0(1) * occ_func_2_0(25) * occ_func_3_0(48) *
              occ_func_2_0(29) +
          occ_func_2_0(23) * occ_func_2_0(1) * occ_func_3_0(44) *
              occ_func_2_0(27) +
          occ_func_2_0(19) * occ_func_2_0(21) * occ_func_3_0(32) *
              occ_func_2_0(1) +
          occ_func_2_0(25) * occ_func_2_0(1) * occ_func_3_0(10) *
              occ_func_2_0(29) +
          occ_func_2_0(1) * occ_func_2_0(23) * occ_func_3_0(8) *
              occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_2_0(19) * occ_func_3_0(2) *
              occ_func_2_0(1) +
          occ_func_2_0(8) * occ_func_2_0(2) * occ_func_3_0(1) *
              occ_func_2_0(10)) /
         8.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_73_0_at_0(int occ_i, int occ_f) const {
  return (m_occ_func_2_0[occ_f] - m_occ_func_2_0[occ_i]) *
         (occ_func_2_0(24) * occ_func_3_0(47) * occ_func_2_0(28) +
          occ_func_2_0(22) * occ_func_3_0(43) * occ_func_2_0(26) +
          occ_func_2_0(18) * occ_func_2_0(20) * occ_func_3_0(31) +
          occ_func_2_0(3) * occ_func_2_0(11) * occ_func_2_0(9) +
          occ_func_2_0(28) * occ_func_2_0(24) * occ_func_3_0(11) +
          occ_func_2_0(26) * occ_func_3_0(9) * occ_func_2_0(22) +
          occ_func_2_0(20) * occ_func_3_0(3) * occ_func_2_0(18) +
          occ_func_2_0(43) * occ_func_2_0(47) * occ_func_2_0(31) +
          occ_func_2_0(26) * occ_func_3_0(51) * occ_func_2_0(28) +
          occ_func_2_0(20) * occ_func_3_0(39) * occ_func_2_0(24) +
          occ_func_2_0(18) * occ_func_2_0(22) * occ_func_3_0(35) +
          occ_func_2_0(35) * occ_func_2_0(39) * occ_func_2_0(51) +
          occ_func_2_0(7) * occ_func_2_0(5) * occ_func_2_0(13) +
          occ_func_2_0(26) * occ_func_3_0(13) * occ_func_2_0(28) +
          occ_func_2_0(20) * occ_func_3_0(7) * occ_func_2_0(24) +
          occ_func_2_0(22) * occ_func_2_0(18) * occ_func_3_0(5)) /
         8.;
}

template <typename Scalar>
Scalar ZrO_Clexulator::site_deval_bfunc_73_0_at_1(int occ_i, int occ_f) const {
  return (m_occ_func_3_0[occ_f] - m_occ_func_3_0[occ_i]) *
         (occ_func_2_0(36) * occ_func_2_0(40) * occ_func_2_0(52) +
          occ_func_2_0(29) * occ_func_3_0(12) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_2_0(25) * occ_func_3_0(6) +
          occ_func_2_0(19) * occ_func_3_0(4) * occ_func_2_0(23) +
          occ_func_2_0(12) * occ_func_2_0(6) * occ_func_2_0(4) +
          occ_func_2_0(27) * occ_func_2_0(29) * occ_func_3_0(52) +
          occ_func_2_0(25) * occ_func_3_0(40) * occ_func_2_0(21) +
          occ_func_2_0(23) * occ_func_3_0(36) * occ_func_2_0(19) +
          occ_func_2_0(32) * occ_func_2_0(44) * occ_func_2_0(48) +
          occ_func_2_0(25) * occ_func_3_0(48) * occ_func_2_0(29) +
          occ_func_2_0(23) * occ_func_3_0(44) * occ_func_2_0(27) +
          occ_func_2_0(19) * occ_func_2_0(21) * occ_func_3_0(32) +
          occ_func_2_0(25) * occ_func_3_0(10) * occ_func_2_0(29) +
          occ_func_2_0(23) * occ_func_3_0(8) * occ_func_2_0(27) +
          occ_func_2_0(21) * occ_func_2_0(19) * occ_func_3_0(2) +
          occ_func_2_0(8) * occ_func_2_0(2) * occ_func_2_0(10)) /
         8.;
}

}  // namespace clexulator
}  // namespace CASM

extern "C" {
/// \brief Returns a clexulator::BaseClexulator* owning a ZrO_Clexulator
CASM::clexulator::BaseClexulator *make_ZrO_Clexulator() {
  return new CASM::clexulator::ZrO_Clexulator();
}
}
