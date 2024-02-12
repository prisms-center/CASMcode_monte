import numpy as np

import libcasm.monte.events as mcevents
import libcasm.xtal as xtal
import libcasm.xtal.prims as xtal_prims


def test_constructor_1():
    xtal_prim = xtal_prims.cubic(a=1.0, occ_dof=["A", "B"])
    T = np.array(
        [
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
        ]
    )
    convert = mcevents.Conversions(
        xtal_prim=xtal_prim, transformation_matrix_to_super=T
    )
    assert convert.l_size() == 27


def test_constructor_2():
    xtal_prim = xtal.Prim(
        lattice=xtal.Lattice(
            column_vector_matrix=np.array(
                [
                    [1.0, 0, 0],
                    [0, 1.0, 0],
                    [0, 0, 1.0],
                ]
            ).transpose()
        ),
        coordinate_frac=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
            ]
        ).transpose(),
        occ_dof=[
            ["A", "B"],
            ["B", "C"],
        ],
    )
    T = np.array(
        [
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
        ]
    )
    convert = mcevents.Conversions(
        xtal_prim=xtal_prim, transformation_matrix_to_super=T
    )
    assert convert.l_size() == 54

    f = convert.site_index_converter()

    def bijk(b, i, j, k):
        return xtal.IntegralSiteCoordinate.from_list([b, i, j, k])

    def trans(i, j, k):
        return np.array([i, j, k], dtype="int64")

    assert f.total_sites() == 54
    assert f.linear_site_index(bijk(1, 0, 0, 0)) == 27
    assert convert.bijk_to_l(bijk(1, 0, 0, 0)) == 27
    assert convert.l_to_bijk(27) == bijk(1, 0, 0, 0)

    assert convert.bijk_to_l(bijk(1, 0, 0, 0) + trans(1, 0, 0)) == convert.bijk_to_l(
        bijk(1, 1, 0, 0)
    )
