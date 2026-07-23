import pickle

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from gaussian_hsmm import ConvergenceMonitor, GaussianHSMM


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("n_components", 1, "n_components"),
        ("covariance_type", "spherical", "covariance_type"),
        ("max_duration", 1, "max_duration"),
        ("n_iter", 0, "n_iter"),
        ("hmm_n_iter", 0, "n_iter"),
        ("tol", 0.0, "tol"),
        ("min_covar", 0.0, "min_covar"),
    ],
)
def test_invalid_hyperparameters_are_rejected(keyword, value, message):
    model = GaussianHSMM(**{keyword: value})
    with pytest.raises(ValueError, match=message):
        model._validate_hyperparameters()


@pytest.mark.parametrize("lengths", [[], [2, 0], [2, -1], [1, 1]])
def test_invalid_sequence_lengths_are_rejected(lengths):
    with pytest.raises(ValueError, match="lengths"):
        GaussianHSMM._validate_lengths(3, lengths)


def test_unfitted_inference_is_rejected():
    with pytest.raises(NotFittedError):
        GaussianHSMM().predict([[0.0], [1.0]])


def test_nonfinite_observations_are_rejected():
    with pytest.raises(ValueError):
        GaussianHSMM()._validate_X([[0.0], [np.nan]])


def test_duration_distribution_is_truncated_and_normalized():
    model = GaussianHSMM(n_components=2, max_duration=8)
    model.duration_means_ = np.array([2.0, 6.0])
    model._update_duration_probabilities()

    assert model.duration_probs_.shape == (2, 9)
    assert np.all(model.duration_probs_[:, 0] == 0.0)
    assert np.allclose(model.duration_probs_[:, 1:].sum(axis=1), 1.0)


def test_zero_off_diagonal_transition_row_gets_safe_fallback():
    model = GaussianHSMM(n_components=3)
    result = model._remove_self_transitions(np.eye(3))

    assert np.allclose(np.diag(result), 0.0)
    assert np.allclose(result.sum(axis=1), 1.0)
    assert np.allclose(result[result > 0], 0.5)


def test_convergence_monitor_and_pickle_round_trip():
    monitor = ConvergenceMonitor(tol=0.1, n_iter=10)
    monitor.report(-100.0)
    monitor.report(-99.95)

    restored = pickle.loads(pickle.dumps(monitor))
    assert restored.converged
    assert restored.iter == 2
    assert restored.history == [-100.0, -99.95]
