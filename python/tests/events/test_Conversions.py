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


def test_constructor_3():
    T = np.array(
        [
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
        ]
    )

    mol_x = xtal.Occupant(
        name="mol",
        atoms=[
            xtal.AtomComponent(name="B", coordinate=[-0.1, 0.0, 0.0], properties={}),
            xtal.AtomComponent(name="B", coordinate=[0.1, 0.0, 0.0], properties={}),
        ],
    )
    mol_y = xtal.Occupant(
        name="mol",
        atoms=[
            xtal.AtomComponent(name="B", coordinate=[0.0, -0.1, 0.0], properties={}),
            xtal.AtomComponent(name="B", coordinate=[0.0, 0.1, 0.0], properties={}),
        ],
    )
    mol_z = xtal.Occupant(
        name="mol",
        atoms=[
            xtal.AtomComponent(name="B", coordinate=[0.0, 0.0, -0.1], properties={}),
            xtal.AtomComponent(name="B", coordinate=[0.0, 0.0, 0.1], properties={}),
        ],
    )
    atom_A = xtal.Occupant(
        name="A",
        atoms=[
            xtal.AtomComponent(name="A", coordinate=[0.0, 0.0, 0.0], properties={}),
        ],
    )
    occupants = {"mol.x": mol_x, "mol.y": mol_y, "mol.z": mol_z, "A": atom_A}

    ## Initial prim ##
    occ_dof = [
        ["A"],
        ["A", "mol.x", "mol.y", "mol.z"],
        ["A", "mol.x", "mol.y", "mol.z"],
        ["A", "mol.x", "mol.y", "mol.z"],
    ]
    xtal_prim = xtal.Prim(
        lattice=xtal.Lattice(
            column_vector_matrix=np.eye(3),
        ),
        coordinate_frac=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        ).T,
        occ_dof=occ_dof,
        occupants=occupants,
    )
    convert = mcevents.Conversions(
        xtal_prim=xtal_prim, transformation_matrix_to_super=T
    )

    assert convert.l_size() == 27 * 4
    assert convert.asym_size() == 2

    ## Same symmetry, different order of occupants on one site ##
    occ_dof = [
        ["A"],
        ["A", "mol.x", "mol.y", "mol.z"],
        ["mol.x", "mol.y", "mol.z", "A"],
        ["A", "mol.x", "mol.y", "mol.z"],
    ]
    xtal_prim = xtal.Prim(
        lattice=xtal.Lattice(
            column_vector_matrix=np.eye(3),
        ),
        coordinate_frac=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        ).T,
        occ_dof=occ_dof,
        occupants=occupants,
    )
    convert = mcevents.Conversions(
        xtal_prim=xtal_prim, transformation_matrix_to_super=T
    )

    assert convert.l_size() == 27 * 4
    assert convert.asym_size() == 3
