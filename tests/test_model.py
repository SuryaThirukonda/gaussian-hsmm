import numpy as np

from gaussian_hsmm import GaussianHSMM


def make_two_regime_data(seed=7):
    rng = np.random.default_rng(seed)
    blocks = [
        rng.normal([-2.0, -1.0], [0.35, 0.25], size=(35, 2)),
        rng.normal([2.0, 1.0], [0.30, 0.30], size=(45, 2)),
        rng.normal([-2.0, -1.0], [0.35, 0.25], size=(30, 2)),
        rng.normal([2.0, 1.0], [0.30, 0.30], size=(40, 2)),
    ]
    return np.vstack(blocks)


def test_diag_fit_score_decode_and_probabilities():
    X = make_two_regime_data()
    model = GaussianHSMM(
        n_components=2,
        covariance_type="diag",
        max_duration=70,
        n_iter=4,
        hmm_n_iter=100,
        random_state=42,
    ).fit(X)

    probabilities = model.predict_proba(X)
    states = model.predict(X)

    assert np.isfinite(model.score(X))
    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
    assert states.shape == (len(X),)
    assert np.allclose(np.diag(model.transmat_), 0.0)
    assert np.all(model.duration_means_ > 1.0)
    assert np.isfinite(model.aic(X))
    assert np.isfinite(model.bic(X))


def test_full_covariance_and_multiple_sequences():
    X = make_two_regime_data()
    lengths = [80, 70]
    model = GaussianHSMM(
        n_components=2,
        covariance_type="full",
        max_duration=60,
        n_iter=2,
        hmm_n_iter=80,
        random_state=3,
    ).fit(X, lengths=lengths)

    score, probabilities = model.score_samples(X, lengths=lengths)
    path_score, states = model.decode(X, lengths=lengths)

    assert np.isfinite(score)
    assert np.isfinite(path_score)
    assert probabilities.shape == (len(X), 2)
    assert states.shape == (len(X),)
    assert model.covars_.shape == (2, 2, 2)
    assert np.isclose(
        score,
        model.score(X[:80]) + model.score(X[80:]),
    )


def test_sklearn_parameter_round_trip():
    model = GaussianHSMM(n_components=4, max_duration=25, random_state=11)
    parameters = model.get_params()
    assert parameters["n_components"] == 4
    assert parameters["max_duration"] == 25
    assert parameters["random_state"] == 11


def test_fit_predict_and_information_criteria_parameter_penalty():
    X = make_two_regime_data(seed=19)
    model = GaussianHSMM(
        n_components=2,
        max_duration=60,
        n_iter=2,
        hmm_n_iter=60,
        random_state=5,
    )

    states = model.fit_predict(X)

    assert states.shape == (len(X),)
    assert model.monitor_.iter >= 1
    assert model.n_parameters_ > 0
    assert np.isclose(model.aic(X), 2 * model.n_parameters_ - 2 * model.score(X))
