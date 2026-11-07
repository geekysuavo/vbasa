import math

import pytest

import vbasa
import numpy as np


def assert_correct_model_properties(model: vbasa.Model):
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

    for x in (model.Xa, model.Xb, model.Ya, model.Yb):
        assert np.all(np.isfinite(x))

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

    assert np.all(np.isfinite(model.omega0))
    assert np.all(np.isfinite(model.A0))

    # fit params
    assert isinstance(model.alpha, float)
    assert isinstance(model.beta, float)
    assert isinstance(model.eta, np.ndarray)
    assert isinstance(model.A, np.ndarray)
    assert isinstance(model.gamma, float)

    assert math.isfinite(model.alpha)
    assert math.isfinite(model.beta)
    assert np.all(np.isfinite(model.eta))
    assert np.all(np.isfinite(model.A))

    assert model.alpha > 0
    assert model.beta > 0
    assert model.gamma > 0

    # expectations
    assert isinstance(model.t, np.ndarray)
    assert isinstance(model.lmb, float)

    assert isinstance(model.Z, np.ndarray)
    assert isinstance(model.R, np.ndarray)

    assert isinstance(model.bound, float)

    assert model.t.shape == (model.k,)
    assert model.lmb > 0

    assert model.Z.shape == (model.m, model.n + 1)
    assert model.R.shape == (model.k, model.k)

    for expect in (model.t, model.lmb, model.Z, model.R):
        assert np.all(np.isfinite(expect))

    assert np.all(model.Z >= 0)

    z_sums_err = np.abs(np.sum(model.Z, axis=1) - 1)
    assert np.all(z_sums_err < 1e-6)

    if model.argmax_rotation:
        R_dot_err = np.abs(model.R @ model.R.T - np.eye(3))
        assert np.all(R_dot_err < 1e-6)
        assert np.linalg.det(model.R) > 0
    else:
        sv = np.linalg.svd(model.R, compute_uv=False)
        assert np.all(sv <= 1 + 1.0e-9)


@pytest.fixture(scope="module")
def b62() -> vbasa.SubstitutionMatrix:
    return vbasa.matrices.blosum62(scale=1)


@pytest.fixture(params=(2, 5, 10, 20))
def num_target(request) -> int:
    return request.param


@pytest.fixture(params=(2, 5, 10, 20))
def num_source(request) -> int:
    return request.param


@pytest.fixture(params=(3,))
def dims(request) -> int:
    return request.param


@pytest.fixture(params=(0.0, 0.1))
def prior_unmatched(request) -> float:
    return request.param


@pytest.fixture(params=(False, True))
def argmax_rotation(request) -> bool:
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
def model(
    points: tuple,
    prior_unmatched: float,
    argmax_rotation: bool,
    b62: vbasa.SubstitutionMatrix,
) -> vbasa.Model:
    m = points[0].shape[0]
    n = points[2].shape[0]

    sa = np.random.choice(b62.vocab, size=(m,)).tolist()
    sb = np.random.choice(b62.vocab, size=(n,)).tolist()

    return vbasa.model(
        *points,
        sa,
        sb,
        prior_unmatched=prior_unmatched,
        argmax_rotation=argmax_rotation,
        sub_matrix=b62,
    )


def test_properties(model: vbasa.Model):
    # test general correctness.
    assert_correct_model_properties(model)

    # test initial conditions.
    assert model.alpha > model.alpha0
    assert model.beta > model.beta0
    assert model.gamma > model.gamma0
    assert np.all(np.isfinite(model.eta))


def test_step(model: vbasa.Model):
    elbo = model.step()

    assert isinstance(elbo, float)
    assert elbo == model.bound

    assert_correct_model_properties(model)


def test_optimize(model: vbasa.Model):
    initial_elbo = model.bound
    elbo = model.optimize(iterations=10)

    assert isinstance(elbo, float)
    assert elbo >= initial_elbo

    assert_correct_model_properties(model)


def test_monotonic(model: vbasa.Model):
    prev_elbo = model.bound
    for _ in range(50):
        elbo = model.step()
        assert elbo >= prev_elbo - 1.0e-9 * abs(prev_elbo)
        prev_elbo = elbo
