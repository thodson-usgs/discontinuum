import pytest
import numpy as np

from loadest2.transform import Transform, ZTransform, LogZTransform, UnitTransform


@pytest.mark.parametrize('length', [2, 5, 10, 100, 1000, 10_000])
@pytest.mark.parametrize('range', [0.1, 1, 5, 10, 1e2, 1e4])
def test_transform(length, range):
    x = np.random.rand(length) * range
    t = Transform(x)
    assert np.allclose(t.transform(x), x)
    assert np.allclose(t.untransform(t.transform(x)), x)


@pytest.mark.parametrize('length', [2, 5, 10, 100, 1000, 10_000])
@pytest.mark.parametrize('range', [0.1, 1, 5, 10, 1e2, 1e4])
def test_ztransform(length, range):
    x = np.random.rand(length) * range
    zt = ZTransform(x)
    assert np.allclose(zt.transform(x), (x - np.mean(x)) / np.std(x))
    assert np.allclose(zt.untransform(zt.transform(x)), x)


@pytest.mark.parametrize('length', [2, 5, 10, 100, 1000, 10_000])
@pytest.mark.parametrize('range', [0.1, 1, 5, 10, 1e2, 1e4])
def test_logztransform(length, range):
    x = np.random.rand(length) * range

    if (x == 0).any():
        pytest.xfail("Drew an exact zero which is not implemented")

    lzt = LogZTransform(x)
    assert np.allclose(
        lzt.transform(x), (np.log(x) - np.mean(np.log(x))) / np.std(np.log(x))
    )
    assert np.allclose(lzt.untransform(lzt.transform(x)), x)


@pytest.mark.parametrize('length', [2, 5, 10, 100, 1000, 10_000])
@pytest.mark.parametrize('range', [0.1, 1, 5, 10, 1e2, 1e4])
def test_unittransform(length, range):
    x = np.random.rand(length) * range
    ut = UnitTransform(x)
    assert np.allclose(ut.transform(x), x / np.max(x))
    assert np.allclose(ut.untransform(ut.transform(x)), x)
