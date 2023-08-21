import libcasm.monte as mc


def test_constructor_1():
    rng = mc.RandomNumberGenerator()
    for i in range(int(1e5)):
        r = rng.random_int(9)
        assert r >= 0
        assert r <= 9


def test_constructor_2():
    e = mc.RandomNumberEngine()
    state = e.dump()

    rng = mc.RandomNumberGenerator(e)
    x = [rng.random_int(9) for i in range(10)]

    e.load(state)
    y = [rng.random_int(9) for i in range(10)]

    assert x == y


def test_constructor_3():
    e = mc.RandomNumberEngine()
    state = e.dump()

    rng = mc.RandomNumberGenerator(e)
    x = [rng.random_real(9) for i in range(10)]

    e.load(state)
    y = [rng.random_real(9) for i in range(10)]

    assert x == y
