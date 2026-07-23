"""Gaussian hidden semi-Markov model with a scikit-learn-style API.

The :class:`GaussianHSMM` estimator uses ``hmmlearn.GaussianHMM`` only to find
an initial parameter basin.  All subsequent likelihood evaluation, posterior
inference, parameter updates, and decoding use an explicit-duration HSMM.

The implementation is intentionally pure Python/NumPy/SciPy so it is easy to
read, test, and package.  Numerical work is vectorized where practical.  Once
profiling identifies stable bottlenecks, the private dynamic-programming
methods can be moved to Cython/Rust/C++ without changing the public API.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from hmmlearn.hmm import GaussianHMM
from scipy.special import gammaln, logsumexp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


@dataclass
class ConvergenceMonitor:
    """Small convergence record modeled after ``hmmlearn``'s monitor object."""

    tol: float
    n_iter: int
    verbose: bool = False
    history: list[float] = field(default_factory=list)
    converged: bool = False

    @property
    def iter(self) -> int:
        """Number of completed explicit-duration EM iterations."""
        return len(self.history)

    def report(self, log_likelihood: float) -> None:
        self.history.append(float(log_likelihood))
        if self.verbose:
            change = np.nan if len(self.history) == 1 else self.history[-1] - self.history[-2]
            print(
                f"HSMM iteration {self.iter:>3}: "
                f"log likelihood={log_likelihood:>14.6f}, change={change:>12.6f}"
            )
        if len(self.history) >= 2:
            improvement = self.history[-1] - self.history[-2]
            self.converged = abs(improvement) < self.tol


class GaussianHSMM(BaseEstimator):
    """Explicit-duration hidden semi-Markov model with Gaussian emissions.

    Parameters
    ----------
    n_components : int, default=3
        Number of hidden states. At least two states are required because an
        HSMM transition describes which *different* state follows a completed
        segment.
    covariance_type : {"diag", "full"}, default="diag"
        Shape of each state's Gaussian covariance.
    max_duration : int, default=252
        Largest segment duration represented explicitly, in observations.
        With daily trading data, 252 is approximately one trading year.
    n_iter : int, default=20
        Maximum number of explicit-duration EM iterations after HMM
        initialization.
    tol : float, default=1e-3
        Absolute log-likelihood change used as the convergence threshold.
    min_covar : float, default=1e-4
        Diagonal regularization added to covariance estimates.
    random_state : int or None, default=None
        Random seed passed to the initializing Gaussian HMM.
    hmm_n_iter : int, default=300
        Maximum EM iterations used only by the initializing HMM.
    verbose : bool, default=False
        Print explicit-duration EM progress.

    Notes
    -----
    Durations follow a shifted Poisson distribution:

    ``D = 1 + Poisson(duration_mean - 1)``.

    The probability mass is normalized over ``1..max_duration``. Transitions
    have zero diagonal because staying in a state is represented by the
    duration distribution, not an HMM self-transition.

    Multiple sequences are supported through the ``lengths`` argument used by
    ``fit``, ``score``, ``decode``, ``predict``, and ``predict_proba``.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        max_duration: int = 252,
        n_iter: int = 20,
        tol: float = 1e-3,
        min_covar: float = 1e-4,
        random_state: int | None = None,
        hmm_n_iter: int = 300,
        verbose: bool = False,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_duration = max_duration
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        self.random_state = random_state
        self.hmm_n_iter = hmm_n_iter
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public estimator API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: npt.ArrayLike,
        lengths: Iterable[int] | None = None,
    ) -> GaussianHSMM:
        """Estimate an HSMM from one or more observation sequences.

        A Gaussian HMM is fitted first to initialize the parameter values. It
        is not the returned model. The following iterations use explicit
        duration forward-backward inference and HSMM parameter updates.
        """
        X_array = self._validate_X(X)
        sequence_lengths = self._validate_lengths(len(X_array), lengths)
        self._validate_hyperparameters()
        self.n_features_ = X_array.shape[1]

        initializer = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.hmm_n_iter,
            tol=min(self.tol, 1e-4),
            min_covar=self.min_covar,
            random_state=self.random_state,
        ).fit(X_array, lengths=sequence_lengths)
        self.hmm_initializer_ = initializer

        self.startprob_ = np.asarray(initializer.startprob_, dtype=float).copy()
        self.means_ = np.asarray(initializer.means_, dtype=float).copy()
        self.covars_ = self._extract_initializer_covariances(initializer)

        hmm_transition = np.asarray(initializer.transmat_, dtype=float)
        self.duration_means_ = np.clip(
            1.0 / np.maximum(1.0 - np.diag(hmm_transition), 1e-6),
            1.01,
            float(self.max_duration),
        )
        self.transmat_ = self._remove_self_transitions(hmm_transition)
        self._update_duration_probabilities()

        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
        best_snapshot = self._snapshot()
        best_log_likelihood = -np.inf

        for _ in range(self.n_iter):
            statistics = self._expectation_step(X_array, sequence_lengths)
            log_likelihood = float(statistics["log_likelihood"])
            self.monitor_.report(log_likelihood)

            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_snapshot = self._snapshot()

            if self.monitor_.converged:
                break

            self._maximization_step(X_array, statistics)

        # The truncated-duration mean update is a practical generalized M-step
        # and may occasionally lower likelihood slightly. Retain the strongest
        # evaluated iteration instead of returning a worse final update.
        self._restore(best_snapshot)
        self.log_likelihood_ = float(best_log_likelihood)
        self.n_parameters_ = self._n_parameters()
        return self

    def score(
        self,
        X: npt.ArrayLike,
        lengths: Iterable[int] | None = None,
    ) -> float:
        """Return total data log-likelihood under the fitted HSMM."""
        log_likelihood, _ = self.score_samples(X, lengths)
        return log_likelihood

    def score_samples(
        self,
        X: npt.ArrayLike,
        lengths: Iterable[int] | None = None,
    ) -> tuple[float, FloatArray]:
        """Return total log-likelihood and per-observation state posteriors."""
        self._check_fitted()
        X_array = self._validate_X(X)
        sequence_lengths = self._validate_lengths(len(X_array), lengths)
        posterior_blocks = []
        total_log_likelihood = 0.0

        for sequence in self._split_sequences(X_array, sequence_lengths):
            log_emissions = self._compute_log_emissions(sequence)
            alpha, log_likelihood, cumulative = self._forward(log_emissions)
            beta = self._backward(log_emissions, cumulative)
            gamma, _, _, _, _ = self._posterior_statistics(
                log_emissions, cumulative, alpha, beta, log_likelihood
            )
            posterior_blocks.append(gamma)
            total_log_likelihood += log_likelihood

        return float(total_log_likelihood), np.vstack(posterior_blocks)

    def decode(
        self,
        X: npt.ArrayLike,
        lengths: Iterable[int] | None = None,
    ) -> tuple[float, IntArray]:
        """Return the joint log probability and most likely HSMM state path."""
        self._check_fitted()
        X_array = self._validate_X(X)
        sequence_lengths = self._validate_lengths(len(X_array), lengths)
        paths = []
        total_path_log_probability = 0.0

        for sequence in self._split_sequences(X_array, sequence_lengths):
            path, path_log_probability = self._viterbi(self._compute_log_emissions(sequence))
            paths.append(path)
            total_path_log_probability += path_log_probability

        return float(total_path_log_probability), np.concatenate(paths)

    def predict(
        self,
        X: npt.ArrayLike,
        lengths: Iterable[int] | None = None,
    ) -> IntArray:
        """Return the most likely hidden-state sequence."""
        return self.decode(X, lengths)[1]

    def predict_proba(
        self,
        X: npt.ArrayLike,
        lengths: Iterable[int] | None = None,
    ) -> FloatArray:
        """Return smoothed posterior state probabilities for each observation."""
        return self.score_samples(X, lengths)[1]

    def fit_predict(
        self,
        X: npt.ArrayLike,
        lengths: Iterable[int] | None = None,
    ) -> IntArray:
        """Fit the model and return its most likely hidden-state sequence."""
        return self.fit(X, lengths).predict(X, lengths)

    def aic(self, X: npt.ArrayLike, lengths: Iterable[int] | None = None) -> float:
        """Akaike information criterion; lower values are preferred."""
        return 2 * self.n_parameters_ - 2 * self.score(X, lengths)

    def bic(self, X: npt.ArrayLike, lengths: Iterable[int] | None = None) -> float:
        """Bayesian information criterion; lower values are preferred."""
        X_array = self._validate_X(X)
        return float(np.log(len(X_array)) * self.n_parameters_ - 2 * self.score(X_array, lengths))

    # ------------------------------------------------------------------
    # Validation and representation helpers
    # ------------------------------------------------------------------
    def _validate_hyperparameters(self) -> None:
        if self.n_components < 2:
            raise ValueError("n_components must be at least 2 for an HSMM")
        if self.covariance_type not in {"diag", "full"}:
            raise ValueError("covariance_type must be 'diag' or 'full'")
        if self.max_duration < 2:
            raise ValueError("max_duration must be at least 2")
        if self.n_iter < 1 or self.hmm_n_iter < 1:
            raise ValueError("n_iter and hmm_n_iter must be positive")
        if self.tol <= 0 or self.min_covar <= 0:
            raise ValueError("tol and min_covar must be positive")

    def _validate_X(self, X: npt.ArrayLike) -> FloatArray:
        return np.asarray(
            check_array(X, ensure_2d=True, dtype=np.float64, ensure_all_finite=True),
            dtype=np.float64,
        )

    @staticmethod
    def _validate_lengths(
        n_observations: int,
        lengths: Iterable[int] | None,
    ) -> IntArray:
        if lengths is None:
            return np.asarray([n_observations], dtype=int)
        output = np.asarray(list(lengths), dtype=int)
        if output.ndim != 1 or len(output) == 0 or np.any(output <= 0):
            raise ValueError("lengths must contain positive sequence lengths")
        if int(output.sum()) != n_observations:
            raise ValueError("lengths must sum to the number of observations")
        return output

    @staticmethod
    def _split_sequences(
        X: FloatArray,
        lengths: IntArray,
    ) -> Iterator[FloatArray]:
        boundary = 0
        for length in lengths:
            yield X[boundary : boundary + length]
            boundary += int(length)

    def _check_fitted(self) -> None:
        check_is_fitted(
            self,
            ["startprob_", "transmat_", "means_", "covars_", "duration_probs_"],
        )

    def _extract_initializer_covariances(self, initializer: GaussianHMM) -> FloatArray:
        covariances = np.asarray(initializer.covars_, dtype=float)
        if self.covariance_type == "diag" and covariances.ndim == 3:
            covariances = np.diagonal(covariances, axis1=1, axis2=2)
        return covariances.copy()

    def _remove_self_transitions(self, transition_matrix: FloatArray) -> FloatArray:
        output = np.asarray(transition_matrix, dtype=float).copy()
        np.fill_diagonal(output, 0.0)
        row_sums = output.sum(axis=1, keepdims=True)
        uniform_off_diagonal = (np.ones_like(output) - np.eye(self.n_components)) / (
            self.n_components - 1
        )
        # NumPy's return annotation differs across supported NumPy releases.
        # Converting explicitly keeps our public return type stable and avoids
        # an `Any` leaking from older NumPy stubs during Python 3.10/3.11 CI.
        return np.asarray(
            np.divide(
                output,
                row_sums,
                out=uniform_off_diagonal,
                where=row_sums > 0,
            ),
            dtype=np.float64,
        )

    def _snapshot(self) -> dict[str, FloatArray]:
        return {
            "startprob": self.startprob_.copy(),
            "transmat": self.transmat_.copy(),
            "means": self.means_.copy(),
            "covars": self.covars_.copy(),
            "duration_means": self.duration_means_.copy(),
            "duration_probs": self.duration_probs_.copy(),
        }

    def _restore(self, snapshot: dict[str, FloatArray]) -> None:
        self.startprob_ = snapshot["startprob"]
        self.transmat_ = snapshot["transmat"]
        self.means_ = snapshot["means"]
        self.covars_ = snapshot["covars"]
        self.duration_means_ = snapshot["duration_means"]
        self.duration_probs_ = snapshot["duration_probs"]
        with np.errstate(divide="ignore"):
            self._log_duration_probs_ = np.log(self.duration_probs_)

    # ------------------------------------------------------------------
    # Emission and duration distributions
    # ------------------------------------------------------------------
    def _compute_log_emissions(self, X: FloatArray) -> FloatArray:
        n_observations, n_features = X.shape
        output = np.empty((n_observations, self.n_components))

        for state in range(self.n_components):
            difference = X - self.means_[state]
            if self.covariance_type == "diag":
                variance = np.maximum(self.covars_[state], self.min_covar)
                log_determinant = np.log(variance).sum()
                quadratic = ((difference**2) / variance).sum(axis=1)
            else:
                covariance = self.covars_[state] + np.eye(n_features) * self.min_covar
                sign, log_determinant = np.linalg.slogdet(covariance)
                if sign <= 0:
                    raise np.linalg.LinAlgError("Covariance is not positive definite")
                precision = np.linalg.inv(covariance)
                quadratic = np.einsum("ti,ij,tj->t", difference, precision, difference)
            output[:, state] = -0.5 * (n_features * np.log(2 * np.pi) + log_determinant + quadratic)
        return output

    def _update_duration_probabilities(self) -> None:
        durations = np.arange(1, self.max_duration + 1, dtype=float)
        log_probabilities = np.empty((self.n_components, self.max_duration + 1))
        log_probabilities[:, 0] = -np.inf

        for state, mean_duration in enumerate(self.duration_means_):
            rate = max(float(mean_duration) - 1.0, 1e-6)
            counts = durations - 1.0
            log_pmf = counts * np.log(rate) - rate - gammaln(counts + 1.0)
            log_probabilities[state, 1:] = log_pmf - logsumexp(log_pmf)
        self.duration_probs_ = np.exp(log_probabilities)
        self._log_duration_probs_ = log_probabilities

    @staticmethod
    def _cumulative_emissions(log_emissions: FloatArray) -> FloatArray:
        return np.vstack([np.zeros((1, log_emissions.shape[1])), np.cumsum(log_emissions, axis=0)])

    # ------------------------------------------------------------------
    # Explicit-duration dynamic programming
    # ------------------------------------------------------------------
    def _forward(
        self,
        log_emissions: FloatArray,
    ) -> tuple[FloatArray, float, FloatArray]:
        """Segment-end forward recursion in log space."""
        n_observations = len(log_emissions)
        cumulative = self._cumulative_emissions(log_emissions)
        alpha = np.full((n_observations, self.n_components), -np.inf)
        log_start = np.log(np.maximum(self.startprob_, np.finfo(float).tiny))
        with np.errstate(divide="ignore"):
            log_transition = np.log(self.transmat_)

        for end in range(n_observations):
            durations = np.arange(1, min(self.max_duration, end + 1) + 1)
            starts = end - durations + 1
            for state in range(self.n_components):
                segment = (
                    self._log_duration_probs_[state, durations]
                    + cumulative[end + 1, state]
                    - cumulative[starts, state]
                )
                at_start = starts == 0
                values = segment.copy()
                values[at_start] += log_start[state]
                if np.any(~at_start):
                    incoming = alpha[starts[~at_start] - 1] + log_transition[:, state]
                    values[~at_start] += logsumexp(incoming, axis=1)
                alpha[end, state] = logsumexp(values)

        return alpha, float(logsumexp(alpha[-1])), cumulative

    def _backward(
        self,
        log_emissions: FloatArray,
        cumulative: FloatArray,
    ) -> FloatArray:
        """Suffix likelihood conditional on a segment ending at each date/state."""
        n_observations = len(log_emissions)
        beta = np.full((n_observations, self.n_components), -np.inf)
        beta[-1] = 0.0
        with np.errstate(divide="ignore"):
            log_transition = np.log(self.transmat_)

        for current_end in range(n_observations - 2, -1, -1):
            next_start = current_end + 1
            durations = np.arange(
                1,
                min(self.max_duration, n_observations - next_start) + 1,
            )
            next_ends = next_start + durations - 1
            for current_state in range(self.n_components):
                next_state_values = np.full(self.n_components, -np.inf)
                for next_state in range(self.n_components):
                    if next_state == current_state:
                        continue
                    segment = (
                        self._log_duration_probs_[next_state, durations]
                        + cumulative[next_ends + 1, next_state]
                        - cumulative[next_start, next_state]
                    )
                    continuation = np.where(
                        next_ends == n_observations - 1,
                        0.0,
                        beta[next_ends, next_state],
                    )
                    next_state_values[next_state] = log_transition[
                        current_state, next_state
                    ] + logsumexp(segment + continuation)
                beta[current_end, current_state] = logsumexp(next_state_values)
        return beta

    def _posterior_statistics(
        self,
        log_emissions: FloatArray,
        cumulative: FloatArray,
        alpha: FloatArray,
        beta: FloatArray,
        log_likelihood: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """Compute occupancy, initial, transition, and duration sufficient stats."""
        n_observations = len(log_emissions)
        gamma_difference = np.zeros((n_observations + 1, self.n_components))
        initial_counts = np.zeros(self.n_components)
        transition_counts = np.zeros((self.n_components, self.n_components))
        duration_sums = np.zeros(self.n_components)
        segment_counts = np.zeros(self.n_components)
        log_start = np.log(np.maximum(self.startprob_, np.finfo(float).tiny))
        with np.errstate(divide="ignore"):
            log_transition = np.log(self.transmat_)

        for end in range(n_observations):
            durations = np.arange(1, min(self.max_duration, end + 1) + 1)
            starts = end - durations + 1
            at_start = starts == 0

            for state in range(self.n_components):
                segment_remainder = (
                    self._log_duration_probs_[state, durations]
                    + cumulative[end + 1, state]
                    - cumulative[starts, state]
                    + beta[end, state]
                )
                prefix = np.empty(len(durations))
                prefix[at_start] = log_start[state]

                if np.any(~at_start):
                    incoming = alpha[starts[~at_start] - 1] + log_transition[:, state]
                    prefix[~at_start] = logsumexp(incoming, axis=1)

                    transition_log_weights = (
                        incoming + segment_remainder[~at_start, None] - log_likelihood
                    )
                    transition_counts[:, state] += np.exp(transition_log_weights).sum(axis=0)

                segment_probabilities = np.exp(prefix + segment_remainder - log_likelihood)
                np.add.at(gamma_difference[:, state], starts, segment_probabilities)
                gamma_difference[end + 1, state] -= segment_probabilities.sum()

                initial_counts[state] += segment_probabilities[at_start].sum()
                duration_sums[state] += np.dot(segment_probabilities, durations)
                segment_counts[state] += segment_probabilities.sum()

        gamma = np.cumsum(gamma_difference[:-1], axis=0)
        gamma = np.maximum(gamma, 0.0)
        row_sums = gamma.sum(axis=1, keepdims=True)
        gamma = np.divide(
            gamma,
            row_sums,
            out=np.full_like(gamma, 1.0 / self.n_components),
            where=row_sums > 0,
        )
        return (
            gamma,
            initial_counts,
            transition_counts,
            duration_sums,
            segment_counts,
        )

    def _viterbi(self, log_emissions: FloatArray) -> tuple[IntArray, float]:
        """Explicit-duration Viterbi decoder."""
        n_observations = len(log_emissions)
        cumulative = self._cumulative_emissions(log_emissions)
        scores = np.full((n_observations, self.n_components), -np.inf)
        predecessor = np.full((n_observations, self.n_components), -1, dtype=int)
        chosen_duration = np.ones((n_observations, self.n_components), dtype=int)
        log_start = np.log(np.maximum(self.startprob_, np.finfo(float).tiny))
        with np.errstate(divide="ignore"):
            log_transition = np.log(self.transmat_)

        for end in range(n_observations):
            durations = np.arange(1, min(self.max_duration, end + 1) + 1)
            starts = end - durations + 1
            at_start = starts == 0
            for state in range(self.n_components):
                values = (
                    self._log_duration_probs_[state, durations]
                    + cumulative[end + 1, state]
                    - cumulative[starts, state]
                )
                previous_states = np.full(len(durations), -1, dtype=int)
                values[at_start] += log_start[state]
                if np.any(~at_start):
                    incoming = scores[starts[~at_start] - 1] + log_transition[:, state]
                    best_previous = np.argmax(incoming, axis=1)
                    previous_states[~at_start] = best_previous
                    values[~at_start] += incoming[np.arange(len(best_previous)), best_previous]
                winner = int(np.argmax(values))
                scores[end, state] = values[winner]
                predecessor[end, state] = previous_states[winner]
                chosen_duration[end, state] = durations[winner]

        path = np.empty(n_observations, dtype=int)
        end = n_observations - 1
        state = int(np.argmax(scores[-1]))
        path_log_probability = float(scores[-1, state])
        while end >= 0:
            duration = int(chosen_duration[end, state])
            start = end - duration + 1
            path[start : end + 1] = state
            state = int(predecessor[end, state])
            end = start - 1
        return path, path_log_probability

    # ------------------------------------------------------------------
    # EM sufficient statistics and parameter updates
    # ------------------------------------------------------------------
    def _expectation_step(
        self,
        X: FloatArray,
        lengths: IntArray,
    ) -> dict[str, FloatArray | float]:
        gamma_blocks = []
        initial_counts = np.zeros(self.n_components)
        transition_counts = np.zeros((self.n_components, self.n_components))
        duration_sums = np.zeros(self.n_components)
        segment_counts = np.zeros(self.n_components)
        total_log_likelihood = 0.0

        for sequence in self._split_sequences(X, lengths):
            log_emissions = self._compute_log_emissions(sequence)
            alpha, log_likelihood, cumulative = self._forward(log_emissions)
            beta = self._backward(log_emissions, cumulative)
            (
                gamma,
                sequence_initial,
                sequence_transitions,
                sequence_duration_sums,
                sequence_segment_counts,
            ) = self._posterior_statistics(log_emissions, cumulative, alpha, beta, log_likelihood)
            gamma_blocks.append(gamma)
            initial_counts += sequence_initial
            transition_counts += sequence_transitions

            duration_sums += sequence_duration_sums
            segment_counts += sequence_segment_counts
            total_log_likelihood += log_likelihood

        duration_means = np.divide(
            duration_sums,
            segment_counts,
            out=self.duration_means_.copy(),
            where=segment_counts > 0,
        )
        return {
            "gamma": np.vstack(gamma_blocks),
            "initial_counts": initial_counts,
            "transition_counts": transition_counts,
            "duration_means": duration_means,
            "log_likelihood": float(total_log_likelihood),
        }

    def _maximization_step(
        self,
        X: FloatArray,
        statistics: dict[str, FloatArray | float],
    ) -> None:
        gamma = np.asarray(statistics["gamma"])
        state_weights = gamma.sum(axis=0)

        initial = np.asarray(statistics["initial_counts"]) + 1e-8
        self.startprob_ = initial / initial.sum()

        transitions = (
            np.asarray(statistics["transition_counts"])
            + (np.ones((self.n_components, self.n_components)) - np.eye(self.n_components)) * 1e-8
        )
        np.fill_diagonal(transitions, 0.0)
        self.transmat_ = transitions / transitions.sum(axis=1, keepdims=True)

        for state in range(self.n_components):
            if state_weights[state] <= self.n_features_:
                continue
            weights = gamma[:, state]
            self.means_[state] = np.average(X, axis=0, weights=weights)
            difference = X - self.means_[state]
            if self.covariance_type == "diag":
                self.covars_[state] = (
                    np.average(difference**2, axis=0, weights=weights) + self.min_covar
                )
            else:
                self.covars_[state] = (
                    difference * weights[:, None]
                ).T @ difference / state_weights[state] + np.eye(self.n_features_) * self.min_covar

        self.duration_means_ = np.clip(
            np.asarray(statistics["duration_means"]),
            1.01,
            float(self.max_duration),
        )
        self._update_duration_probabilities()

    def _n_parameters(self) -> int:
        covariance_parameters = (
            self.n_components * self.n_features_
            if self.covariance_type == "diag"
            else self.n_components * self.n_features_ * (self.n_features_ + 1) // 2
        )
        return int(
            (self.n_components - 1)
            + self.n_components * max(self.n_components - 2, 0)
            + self.n_components * self.n_features_
            + covariance_parameters
            + self.n_components
        )


__all__ = ["ConvergenceMonitor", "GaussianHSMM"]
