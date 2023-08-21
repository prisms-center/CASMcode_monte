import math

import numpy as np

import libcasm.monte as mc


def test_BooleanValueMap_1():
    x = mc._monte.BooleanValueMap()
    assert len(x) == 0
    x["check"] = True
    assert len(x) == 1


def test_ScalarValueMap_1():
    x = mc._monte.ScalarValueMap()
    assert len(x) == 0
    x["check"] = 1.0
    assert len(x) == 1


def test_VectorValueMap_1():
    x = mc._monte.VectorValueMap()
    assert len(x) == 0
    x["check"] = np.array([1.0, 1.0])
    assert len(x) == 1


def test_MatrixValueMap_1():
    x = mc._monte.MatrixValueMap()
    assert len(x) == 0
    x["check"] = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert len(x) == 1


def test_ValueMap_1():
    values = mc.ValueMap()
    assert len(values.boolean_values) == 0

    # test boolean_values
    values.boolean_values["check"] = True
    assert len(values.boolean_values) == 1
    assert values.boolean_values["check"] is True

    # test scalar_values
    values.scalar_values["check"] = 1.0
    assert len(values.boolean_values) == 1
    assert math.isclose(values.scalar_values["check"], 1.0)

    # test vector_values
    v = np.array([1.0, 1.0])
    values.vector_values["check"] = v
    assert len(values.boolean_values) == 1
    assert np.allclose(values.vector_values["check"], v)

    values.vector_values["check"][0] = 2.0
    assert np.allclose(values.vector_values["check"], np.array([2.0, 1.0]))

    # test matrix_values
    M = np.array([[1.0, 1.0], [1.0, 1.0]])
    values.matrix_values["check"] = M
    assert len(values.boolean_values) == 1
    assert np.allclose(values.matrix_values["check"], M)

    values.matrix_values["check"][0, 1] = 2.0
    values.matrix_values["check"][1, 0] = 3.0
    values.matrix_values["check"][1, 1] = 4.0
    assert np.allclose(
        values.matrix_values["check"], np.array([[1.0, 2.0], [3.0, 4.0]])
    )


def test_ValueMap_tofrom_dict_1():
    s = 1.0
    v = np.array([1.0, 1.0])
    m = np.array([[1.0, 1.0], [1.0, 1.0]])

    values = mc.ValueMap()
    values.boolean_values["is_bool"] = True
    values.scalar_values["is_scalar"] = s
    values.vector_values["is_vector"] = v
    values.matrix_values["is_matrix"] = m

    # to_dict
    x = values.to_dict()
    assert len(x) == 4

    assert "is_bool" in x
    assert x["is_bool"] is True

    assert "is_scalar" in x
    assert math.isclose(x["is_scalar"], s)

    assert "is_vector" in x
    assert np.allclose(x["is_vector"], v)

    assert "is_matrix" in x
    assert np.allclose(x["is_matrix"], m)

    # from_dict
    values_2 = mc.ValueMap.from_dict(x)

    assert len(values_2.boolean_values) == 1
    assert "is_bool" in values_2.boolean_values
    assert values_2.boolean_values["is_bool"] is True

    assert len(values_2.scalar_values) == 1
    assert "is_scalar" in values_2.scalar_values
    assert math.isclose(values_2.scalar_values["is_scalar"], s)

    assert len(values_2.vector_values) == 1
    assert "is_vector" in values_2.vector_values
    assert np.allclose(values_2.vector_values["is_vector"], v)

    assert len(values_2.matrix_values) == 1
    assert "is_matrix" in values_2.matrix_values
    assert np.allclose(values_2.matrix_values["is_matrix"], m)
