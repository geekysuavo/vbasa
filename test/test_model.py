
import pytest

import vbasa
import numpy as np


@pytest.fixture(scope="module")
def b62() -> vbasa.SubstitutionMatrix:
    return vbasa.matrices.blosum62(scale=1)


@pytest.fixture(params=(2, 5, 10, 20))
def num_target(request) -> int:
    return request.param


@pytest.fixture(params=(2, 5, 10, 20))
def num_source(request) -> int:
    return request.param


@pytest.fixture(params=(2, 3))
def dims(request) -> int:
    return request.param


@pytest.fixture
def sizes(num_target: int, num_source: int, dims: int) -> tuple:
    return (num_target, num_source, dims)


@pytest.fixture
def points(sizes: tuple) -> tuple:
    (m, n, k) = sizes
    mn = max(m, n)

    dx_orient = np.random.normal(size=(mn, k))
    dx_orient /= np.linalg.norm(dx_orient, axis=-1, keepdims=True)
    dx_length = np.mean(np.random.normal(size=(mn, 1, 10)) ** 2, axis=-1)
    dx = dx_length * dx_orient

    Xa = np.random.normal(scale=10, size=(mn, k))
    Xb = Xa + dx

    Ya = Xa + np.random.normal(size=(mn, k))
    Yb = Xb + np.random.normal(size=(mn, k))

    i = np.random.permutation(mn)[:m]
    Xa = Xa[i, :]
    Xb = Xb[i, :]

    j = np.random.permutation(mn)[:n]
    Ya = Ya[j, :]
    Yb = Yb[j, :]

    return (Xa, Xb, Ya, Yb)


@pytest.fixture
def model(points: tuple, b62: vbasa.SubstitutionMatrix) -> vbasa.Model:
    m = points[0].shape[0]
    n = points[2].shape[0]

    sa = np.random.choice(b62.vocab, size=(m,)).tolist()
    sb = np.random.choice(b62.vocab, size=(n,)).tolist()

    return vbasa.model(*points, sa, sb, sub_matrix=b62)


def test_properties(model: vbasa.Model):
    # sizes
    assert model.sizes == (model.m, model.n, model.k)

    # point clouds
    assert isinstance(model.Xa, np.ndarray)
    assert isinstance(model.Xb, np.ndarray)

    assert isinstance(model.Ya, np.ndarray)
    assert isinstance(model.Yb, np.ndarray)

    assert model.Xa.shape == (model.m, model.k)
    assert model.Xb.shape == (model.m, model.k)

    assert model.Ya.shape == (model.n, model.k)
    assert model.Yb.shape == (model.n, model.k)

    # prior params
    assert isinstance(model.alpha0, float)
    assert isinstance(model.beta0, float)
    assert isinstance(model.gamma0, float)

    assert isinstance(model.omega0, np.ndarray)
    assert isinstance(model.A0, np.ndarray)

    assert model.alpha0 > 0
    assert model.beta0 > 0
    assert model.gamma0 > 0

    assert np.all(model.omega0 > 0)
    assert np.all(model.A0 >= 0)

    assert model.omega0.shape == (model.m, model.n)
    assert model.A0.shape == (model.k, model.k)

    # fit params
    assert isinstance(model.alpha, np.ndarray)
    assert isinstance(model.beta, np.ndarray)
    assert isinstance(model.eta, np.ndarray)
    assert isinstance(model.A, np.ndarray)
    assert isinstance(model.gamma, float)

    assert np.all(model.alpha == model.alpha0)
    assert np.all(model.beta == model.beta0)
    assert model.gamma == model.gamma0
    assert np.all(model.A == model.A0)
    assert np.all(model.eta == 0)

    # expectations
    assert isinstance(model.t, np.ndarray)
    assert isinstance(model.lmb, np.ndarray)

    assert isinstance(model.Z, np.ndarray)
    assert isinstance(model.R, np.ndarray)

    assert isinstance(model.bound, float)

    assert model.t.shape == (model.k,)
    assert model.lmb.shape == (model.n,)

    assert model.Z.shape == (model.m, model.n)
    assert model.R.shape == (model.k, model.k)

    assert np.all(model.Z > 0)


def test_step(model: vbasa.Model):
    elbo = model.step()

    assert isinstance(elbo, float)
    assert elbo == model.bound


def test_optimize(model: vbasa.Model):
    initial_elbo = model.bound
    elbo = model.optimize(iterations=10)

    assert isinstance(elbo, float)
    assert elbo >= initial_elbo
