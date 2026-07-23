"""Demonstrate independent-sequence boundaries through ``lengths``."""

from __future__ import annotations

import numpy as np

from gaussian_hsmm import GaussianHSMM


def make_sequence(rng: np.random.Generator, first_length: int, second_length: int) -> np.ndarray:
    return np.vstack(
        [
            rng.normal([-1.5, 0.0], [0.3, 0.2], size=(first_length, 2)),
            rng.normal([1.5, 1.0], [0.3, 0.2], size=(second_length, 2)),
        ]
    )


def main() -> None:
    rng = np.random.default_rng(7)
    first = make_sequence(rng, 30, 40)
    second = make_sequence(rng, 25, 35)
    observations = np.vstack([first, second])
    lengths = [len(first), len(second)]

    model = GaussianHSMM(
        n_components=2,
        max_duration=55,
        n_iter=4,
        random_state=7,
    ).fit(observations, lengths=lengths)

    score, probabilities = model.score_samples(observations, lengths=lengths)
    states = model.predict(observations, lengths=lengths)

    print("total log likelihood:", score)
    print("posterior shape:", probabilities.shape)
    print("decoded shape:", states.shape)


if __name__ == "__main__":
    main()
