# gaussian-hsmm

[![CI](https://github.com/SuryaThirukonda/gaussian-hsmm/actions/workflows/ci.yml/badge.svg)](https://github.com/SuryaThirukonda/gaussian-hsmm/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Typed](https://img.shields.io/badge/typing-py.typed-blue.svg)](src/gaussian_hsmm/py.typed)

`gaussian-hsmm` is a scikit-learn-style implementation of a multivariate Gaussian hidden semi-Markov model (HSMM) with explicit state-duration distributions.

It is designed for regime segmentation problems where an ordinary hidden Markov model's memoryless duration assumption is too restrictive. Examples include financial regimes, machine operating modes, behavioral stages, and biological sequences.

> **Development status:** Alpha research software. The public API may change before version 1.0. Validate results against simulations and an independent implementation before using the model for high-stakes or publication-critical conclusions.

## Why an HSMM?

An HMM decides at every observation whether to remain in its current state. If its self-transition probability is (p_{jj}), the implied duration is geometric:

\[
P(D=d)=(p_{jj})^{d-1}(1-p_{jj}).
\]

That distribution is memoryless: after 100 days in a regime, the probability of leaving tomorrow is exactly the same as it was after one day.

An HSMM instead models the duration of a complete state segment directly. This package currently uses a shifted Poisson distribution:

\[
D=1+\operatorname{Poisson}(\lambda_j),
\qquad E[D]=1+\lambda_j.
\]

The distribution is truncated and normalized over `1..max_duration`. Once a segment ends, the transition matrix determines which *different* state begins next. Consequently, the fitted segment-transition matrix has a zero diagonal.

## What the package does

- Multivariate Gaussian emissions.
- Diagonal and full covariance structures.
- Shifted-Poisson explicit state durations.
- Log-space explicit-duration forward-backward inference.
- Smoothed posterior state probabilities.
- Explicit-duration Viterbi decoding.
- EM-style parameter updates.
- HMM-based parameter initialization through `hmmlearn`.
- Multiple independent sequences through a `lengths` argument.
- AIC and BIC model-selection helpers.
- scikit-learn-compatible `get_params` and `set_params` behavior.
- Convergence history through `model.monitor_`.

## What the package does not currently do

- It does not support missing values.
- It does not yet offer duration families other than shifted Poisson.
- It does not provide online/filter-only state probabilities; `predict_proba` returns smoothed probabilities using the complete supplied sequence.
- It does not currently implement sampling from a fitted model.
- It does not guarantee that state labels match across separate fits. Hidden-state numbers are arbitrary.
- It is not yet backed by a compiled Cython/C++ kernel, so large `max_duration`, state grids, and long sequences can be computationally expensive.

These limitations are intentional and documented rather than hidden behind approximations.

## Installation

### Local development

Clone the repository, create an isolated environment, and install the package in editable mode:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### PyPI

The following command is reserved for after the project has actually been published:

```bash
python -m pip install gaussian-hsmm
```

No package has been uploaded as part of this repository setup.

## Quick start

```python
import numpy as np
from gaussian_hsmm import GaussianHSMM

rng = np.random.default_rng(42)
X_train = np.vstack([
    rng.normal([-2.0, 0.0], [0.4, 0.3], size=(80, 2)),
    rng.normal([2.0, 1.5], [0.5, 0.4], size=(100, 2)),
    rng.normal([-2.0, 0.0], [0.4, 0.3], size=(70, 2)),
])

model = GaussianHSMM(
    n_components=2,
    covariance_type="diag",
    max_duration=120,
    n_iter=10,
    random_state=42,
    verbose=True,
)

model.fit(X_train)

states = model.predict(X_train)
probabilities = model.predict_proba(X_train)
log_likelihood = model.score(X_train)

print("state means:\n", model.means_)
print("segment transitions:\n", model.transmat_)
print("mean durations:", model.duration_means_)
print("log likelihood:", log_likelihood)
```

`probabilities[t, j]` is the smoothed probability that observation `t` belongs to state `j`, conditional on the entire sequence passed to `predict_proba`.

## Multiple independent sequences

Concatenate the observations and pass their separate lengths. No transition is inferred across a sequence boundary.

```python
X = np.vstack([sequence_a, sequence_b, sequence_c])
lengths = [len(sequence_a), len(sequence_b), len(sequence_c)]

model.fit(X, lengths=lengths)
states = model.predict(X, lengths=lengths)
probabilities = model.predict_proba(X, lengths=lengths)
```

## Choosing model complexity

Compare several state counts and covariance structures chronologically:

```python
rows = []

for covariance_type in ("diag", "full"):
    for n_states in range(2, 6):
        candidate = GaussianHSMM(
            n_components=n_states,
            covariance_type=covariance_type,
            max_duration=120,
            n_iter=8,
            random_state=42,
        ).fit(X_train)

        rows.append({
            "n_states": n_states,
            "covariance_type": covariance_type,
            "train_log_likelihood": candidate.score(X_train),
            "test_log_likelihood_per_observation": (
                candidate.score(X_test) / len(X_test)
            ),
            "aic": candidate.aic(X_train),
            "bic": candidate.bic(X_train),
        })
```

- Larger log-likelihood means the observations are more plausible under the model.
- Lower AIC/BIC is preferred after penalizing extra parameters.
- Held-out likelihood is essential because a more flexible model can fit training data while generalizing poorly.
- Repeat multiple random seeds. Compare economic/statistical profiles rather than numeric state labels.

## Parameters

### `GaussianHSMM`

| Parameter | Default | Meaning |
|---|---:|---|
| `n_components` | `3` | Number of hidden states. Must be at least 2. |
| `covariance_type` | `"diag"` | `"diag"` or `"full"` Gaussian covariance. |
| `max_duration` | `252` | Largest explicitly represented segment duration. |
| `n_iter` | `20` | Maximum duration-aware EM iterations after initialization. |
| `tol` | `1e-3` | Absolute likelihood-change convergence threshold. |
| `min_covar` | `1e-4` | Covariance regularization floor. |
| `random_state` | `None` | Seed passed to the HMM initializer. |
| `hmm_n_iter` | `300` | Maximum iterations for HMM initialization. |
| `verbose` | `False` | Print HSMM likelihood progress. |

## Learned attributes

| Attribute | Shape/type | Meaning |
|---|---|---|
| `startprob_` | `(K,)` | Initial state probabilities. |
| `transmat_` | `(K, K)` | Between-segment transition matrix; diagonal is zero. |
| `means_` | `(K, F)` | Gaussian state means. |
| `covars_` | `(K, F)` or `(K, F, F)` | Diagonal or full state covariances. |
| `duration_means_` | `(K,)` | Shifted-Poisson mean duration for each state. |
| `duration_probs_` | `(K, max_duration+1)` | Truncated duration probabilities; index 0 is impossible. |
| `log_likelihood_` | `float` | Best evaluated training likelihood retained by `fit`. |
| `n_parameters_` | `int` | Parameter count used by AIC/BIC. |
| `monitor_` | `ConvergenceMonitor` | Iterations, likelihood history, and convergence flag. |
| `hmm_initializer_` | `GaussianHMM` | Initializer retained for inspection; not used for HSMM predictions. |

Here `K` is the number of states and `F` is the number of observed features.

## Methods

| Method | Result |
|---|---|
| `fit(X, lengths=None)` | Estimate model parameters. |
| `score(X, lengths=None)` | Total sequence log-likelihood. |
| `score_samples(X, lengths=None)` | `(log_likelihood, posterior_probabilities)`. |
| `decode(X, lengths=None)` | `(MAP_path_log_probability, state_path)`. |
| `predict(X, lengths=None)` | Most likely explicit-duration state path. |
| `predict_proba(X, lengths=None)` | Smoothed state probabilities. |
| `fit_predict(X, lengths=None)` | Fit and decode the supplied sequence. |
| `aic(X, lengths=None)` | Akaike information criterion. |
| `bic(X, lengths=None)` | Bayesian information criterion. |

## HMM initialization versus HSMM fitting

`fit` deliberately separates two stages:

1. `hmmlearn.GaussianHMM` finds an initial basin for means, covariances, initial probabilities, and transitions.
2. The estimator converts HMM persistence into initial duration means, removes self-transitions, and runs explicit-duration inference and updates.

All calls to `score`, `predict`, `predict_proba`, and `decode` use the final HSMM parameters and HSMM dynamic programs. `hmm_initializer_` is exposed only for diagnostics.

## Computational complexity

Let:

- `T` be sequence length;
- `K` be number of states;
- `D` be `max_duration`;
- `F` be feature count.

The explicit-duration recursions scale approximately with `T`, `K`, and `D`, with additional state-transition work. This is substantially more expensive than an ordinary HMM. Start with a defensible duration cap and a small state grid. Profile before increasing both.

## Reproducibility and interpretation

- Standardize features using training data only.
- Preserve temporal ordering; do not shuffle regime sequences.
- Use chronological validation.
- Fit multiple seeds because initialization can find local optima.
- Treat state numbers as arbitrary identifiers.
- Plot probabilities as well as hard states.
- Check whether fitted duration means approach `max_duration`.
- Avoid assigning causal or business-cycle names without external validation.

## Examples and design documentation

- [`examples/basic_usage.py`](examples/basic_usage.py)
- [`examples/multiple_sequences.py`](examples/multiple_sequences.py)
- [`docs/algorithm.md`](docs/algorithm.md)
- [`docs/api.md`](docs/api.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`RELEASING.md`](RELEASING.md)

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src/gaussian_hsmm
pytest --cov=gaussian_hsmm
python -m build
python -m twine check dist/*
```

The GitHub Actions workflow runs tests across supported Python versions and validates the wheel and source distribution. It does **not** publish anything.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and [SECURITY.md](SECURITY.md) for responsible vulnerability reporting. All participation is covered by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

If this package contributes to academic work, cite the software metadata in [`CITATION.cff`](CITATION.cff) and archive a specific release so your analysis can identify an immutable version.

## License

MIT. See [LICENSE](LICENSE).
