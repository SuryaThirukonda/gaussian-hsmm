"""Fit and inspect a two-state Gaussian HSMM on simulated data."""

from __future__ import annotations

import numpy as np

from gaussian_hsmm import GaussianHSMM


def main() -> None:
    rng = np.random.default_rng(42)
    observations = np.vstack(
        [
            rng.normal([-2.0, 0.0], [0.35, 0.25], size=(45, 2)),
            rng.normal([2.0, 1.5], [0.40, 0.30], size=(60, 2)),
            rng.normal([-2.0, 0.0], [0.35, 0.25], size=(40, 2)),
        ]
    )

    model = GaussianHSMM(
        n_components=2,
        covariance_type="diag",
        max_duration=80,
        n_iter=5,
        random_state=42,
        verbose=True,
    ).fit(observations)

    states = model.predict(observations)
    probabilities = model.predict_proba(observations)

    print("means:\n", model.means_)
    print("segment transitions:\n", model.transmat_)
    print("mean durations:", model.duration_means_)
    print("log likelihood:", model.score(observations))
    print("first 10 states:", states[:10])
    print("first 3 posterior rows:\n", probabilities[:3])


if __name__ == "__main__":
    main()
