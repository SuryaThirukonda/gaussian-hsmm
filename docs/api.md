# API reference

## `GaussianHSMM`

```python
GaussianHSMM(
    n_components=3,
    covariance_type="diag",
    max_duration=252,
    n_iter=20,
    tol=1e-3,
    min_covar=1e-4,
    random_state=None,
    hmm_n_iter=300,
    verbose=False,
)
```

The estimator follows scikit-learn naming conventions: constructor arguments are configuration, learned attributes end in `_`, and `fit` returns `self`.

### Input format

`X` must be a finite numeric two-dimensional array with shape `(n_observations, n_features)`.

For multiple sequences, concatenate them and pass positive `lengths` whose sum equals `len(X)`.

### `fit(X, lengths=None)`

Fit HMM initialization followed by explicit-duration HSMM parameter updates.

### `score(X, lengths=None)`

Return total log-likelihood after summing across sequences.

### `score_samples(X, lengths=None)`

Return `(log_likelihood, probabilities)`, where probabilities have shape `(n_observations, n_components)`.

### `decode(X, lengths=None)`

Return `(path_log_probability, states)` for the most likely explicit-duration path.

### `predict(X, lengths=None)`

Return the decoded state path.

### `predict_proba(X, lengths=None)`

Return smoothed posterior probabilities. These use all observations within each supplied sequence.

### `fit_predict(X, lengths=None)`

Fit and decode in one call.

### `aic(X, lengths=None)` and `bic(X, lengths=None)`

Return information criteria based on the package's documented free-parameter count. Lower values are preferred when comparing models fit to the same observations.

## `ConvergenceMonitor`

`model.monitor_` exposes:

- `history`: evaluated HSMM log-likelihoods;
- `iter`: completed duration-aware iterations;
- `converged`: whether the last absolute likelihood change was below `tol`.

The initializing HMM has its own monitor at `model.hmm_initializer_.monitor_`.
