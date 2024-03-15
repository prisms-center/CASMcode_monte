import numpy as np
import pytest

import libcasm.monte.sampling as sampling


def test_Sampler_scalar_1():
    sampler = sampling.Sampler(
        shape=[],
        component_names=["x"],
        capacity_increment=10000,
    )
    assert isinstance(sampler, sampling.Sampler)
    assert sampler.n_components() == 1
    assert sampler.n_samples() == 0
    assert sampler.component_names() == ["x"]

    # check list input
    for i in range(100):
        sampler.append([0.3])
    assert sampler.n_components() == 1
    assert sampler.n_samples() == 100
    sampler.clear()
    assert sampler.n_samples() == 0

    # check scalar_as_vector
    for i in range(100):
        sampler.append(sampling.scalar_as_vector(0.3))
    assert sampler.n_components() == 1
    assert sampler.n_samples() == 100
    sampler.clear()
    assert sampler.n_samples() == 0

    # check values
    for i in range(100):
        sampler.append([0.3])
    assert sampler.n_components() == 1
    assert sampler.n_samples() == 100
    values = sampler.values()
    assert isinstance(values, np.ndarray)
    assert values.shape == (100, 1)
    sampler.clear()
    assert sampler.n_samples() == 0

    n = 100000

    # add lots of values
    assert sampler.sample_capacity() == 10000
    for i in range(n):
        sampler.append([0.3])
    assert sampler.n_components() == 1
    assert sampler.n_samples() == n
    values = sampler.values()
    assert isinstance(values, np.ndarray)
    assert values.shape == (n, 1)
    assert sampler.sample_capacity() == n


def test_Sampler_vector_1():
    sampler = sampling.Sampler(
        shape=[2],
        component_names=["x1", "x2"],
        capacity_increment=10000,
    )
    assert isinstance(sampler, sampling.Sampler)
    assert sampler.n_components() == 2
    assert sampler.n_samples() == 0
    assert sampler.component_names() == ["x1", "x2"]

    # check list input
    for i in range(100):
        sampler.append([0.3, 0.5])
    assert sampler.n_components() == 2
    assert sampler.n_samples() == 100
    sampler.clear()
    assert sampler.n_samples() == 0

    # check matrix_as_vector (rename array_as_vector?)
    for i in range(100):
        sampler.append(sampling.matrix_as_vector([0.3, 0.5]))
    assert sampler.n_components() == 2
    assert sampler.n_samples() == 100
    sampler.clear()
    assert sampler.n_samples() == 0

    # check values
    for i in range(100):
        sampler.append([0.3, 0.5])
    assert sampler.n_components() == 2
    assert sampler.n_samples() == 100
    values = sampler.values()
    assert isinstance(values, np.ndarray)
    assert values.shape == (100, 2)
    sampler.clear()
    assert sampler.n_samples() == 0

    n = 100000

    # add lots of values
    assert sampler.sample_capacity() == 10000
    for i in range(n):
        sampler.append([0.3, 0.5])
    assert sampler.n_components() == 2
    assert sampler.n_samples() == n
    values = sampler.values()
    assert isinstance(values, np.ndarray)
    assert values.shape == (n, 2)
    assert sampler.sample_capacity() == n


def test_Sampler_matrix_1():
    sampler = sampling.Sampler(
        shape=[2, 2],
        component_names=["x1,y1", "x2,y1", "x1,y2", "x2,y2"],
        capacity_increment=10000,
    )

    x = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )
    assert isinstance(sampler, sampling.Sampler)
    assert sampler.n_components() == 4
    assert sampler.n_samples() == 0
    assert sampler.component_names() == ["x1,y1", "x2,y1", "x1,y2", "x2,y2"]

    # error:
    # # check list input
    # for i in range(100):
    #     sampler.append(x)
    # assert sampler.n_components() == 4
    # assert sampler.n_samples() == 100
    # sampler.clear()
    # assert sampler.n_samples() == 0

    # check matrix_as_vector (rename array_as_vector?)
    for i in range(100):
        sampler.append(sampling.matrix_as_vector(x))
    assert sampler.n_components() == 4
    assert sampler.n_samples() == 100
    sampler.clear()
    assert sampler.n_samples() == 0

    # check values
    for i in range(100):
        sampler.append(sampling.matrix_as_vector(x))
    assert sampler.n_components() == 4
    assert sampler.n_samples() == 100
    values = sampler.values()
    assert isinstance(values, np.ndarray)
    assert values.shape == (100, 4)
    sampler.clear()
    assert sampler.n_samples() == 0
    for i in range(100):
        assert np.isclose(values[i, :], np.array([0.1, 0.3, 0.2, 0.4])).all()

    n = 100000

    # add lots of values
    assert sampler.sample_capacity() == 10000
    for i in range(n):
        sampler.append(sampling.matrix_as_vector(x))
    assert sampler.n_components() == 4
    assert sampler.n_samples() == n
    values = sampler.values()
    assert isinstance(values, np.ndarray)
    assert values.shape == (n, 4)
    assert sampler.sample_capacity() == n


def test_Sampler_ndarray_1():
    component_names = [
        "x0,y0,z0",
        "x1,y0,z0",
        "x0,y1,z0",
        "x1,y1,z0",
        "x0,y0,z1",
        "x1,y0,z1",
        "x0,y1,z1",
        "x1,y1,z1",
    ]

    sampler = sampling.Sampler(
        shape=[2, 2, 2],
        component_names=component_names,
        capacity_increment=10000,
    )

    x = np.array(
        [
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            [
                [0.5, 0.6],
                [0.7, 0.8],
            ],
        ]
    )
    assert isinstance(sampler, sampling.Sampler)
    assert sampler.n_components() == 8
    assert sampler.n_samples() == 0
    assert sampler.component_names() == component_names

    # check np.ndarray.ravel, with order='F' (col-major)
    for i in range(100):
        sampler.append(x.ravel(order="F"))
    assert sampler.n_components() == 8
    assert sampler.n_samples() == 100
    sampler.clear()
    assert sampler.n_samples() == 0

    sample_expected = np.array(
        [
            x[0, 0, 0],
            x[1, 0, 0],
            x[0, 1, 0],
            x[1, 1, 0],
            x[0, 0, 1],
            x[1, 0, 1],
            x[0, 1, 1],
            x[1, 1, 1],
        ]
    )
    for i in range(100):
        assert np.isclose(sampler.sample(i), sample_expected).all()


def test_default_component_names():
    # scalar
    names = sampling.default_component_names([])
    assert names == ["0"]

    # vector
    names = sampling.default_component_names([3])
    assert names == ["0", "1", "2"]

    # matrix
    names = sampling.default_component_names([2, 3])
    assert names == ["0,0", "1,0", "0,1", "1,1", "0,2", "1,2"]

    # >2 dimensions not supported by this
    with pytest.raises(Exception):
        names = sampling.default_component_names([1, 2, 3])
