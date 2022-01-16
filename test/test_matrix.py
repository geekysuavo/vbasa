
import pytest

import vbasa
import numpy as np


@pytest.fixture(params=(2, 5, 10, 20))
def num_target(request) -> int:
    return request.param


@pytest.fixture(params=(2, 5, 10, 20))
def num_source(request) -> int:
    return request.param


@pytest.fixture(params=(0.1, 0.5, 1.0, 2.0, 10.0))
def scale(request) -> float:
    return request.param


def test_blosum62(num_target: int, num_source: int, scale: float):
    b62 = vbasa.matrices.blosum62(scale=scale)
    assert isinstance(b62, vbasa.SubstitutionMatrix)

    assert isinstance(b62.scale, float)
    assert b62.scale == scale

    assert isinstance(b62.vocab, tuple)
    assert all(isinstance(v, str) for v in b62.vocab)
    assert len(b62.vocab) == 20

    assert isinstance(b62.coeffs, np.ndarray)
    assert b62.coeffs.shape == (20, 20)

    a = np.random.choice(b62.vocab, size=(num_target,)).tolist()
    b = np.random.choice(b62.vocab, size=(num_source,)).tolist()
    prob = b62(a, b)

    assert isinstance(prob, np.ndarray)
    assert prob.shape == (num_target, num_source)
    assert np.all(np.isfinite(prob))
    assert np.all(prob > 0)
