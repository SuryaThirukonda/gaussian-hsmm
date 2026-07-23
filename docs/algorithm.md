# Algorithm and statistical design

## Model

For observations \(x_1,\ldots,x_T\), an HSMM represents the sequence as contiguous segments. Each segment has:

- a hidden state \(z_s\);
- a duration \(d_s\);
- observations emitted by the state-specific Gaussian distribution.

The segment-level joint probability contains an initial-state probability, between-segment transitions, explicit duration probabilities, and Gaussian emission densities.

## Gaussian emissions

For state \(j\):

\[
x_t \mid z_t=j \sim \mathcal{N}(\mu_j,\Sigma_j).
\]

`covariance_type="diag"` estimates only diagonal variances. `"full"` estimates all within-state covariances and therefore uses substantially more parameters.

## Durations

The current duration family is shifted Poisson:

\[
D_j=1+Y_j,\qquad Y_j\sim\operatorname{Poisson}(\lambda_j).
\]

Probabilities are calculated for durations 1 through `max_duration` and normalized over that support. A fitted mean near the duration cap indicates truncation may be affecting inference.

## HMM initialization

The estimator first fits `hmmlearn.GaussianHMM` to initialize Gaussian and transition parameters. HMM self-transition probabilities initialize duration means through \(1/(1-p_{jj})\). Self-transitions are then removed and the remaining transition probabilities are normalized across other states.

The HMM does not perform final inference.

## Forward recursion

The forward variable stores the log probability that all observations through time \(t\) have been generated and a state-\(j\) segment ends at \(t\). It sums over possible segment durations and preceding states. Calculations use `logsumexp` to prevent probability underflow.

## Backward recursion and posteriors

The backward variable represents the likelihood of observations after a segment ending at a given time/state. Combining forward, backward, transition, duration, and segment-emission terms yields posterior probabilities for candidate segments. Segment probabilities are accumulated into per-observation state probabilities.

## Parameter updates

- Gaussian means/covariances use posterior observation weights.
- Initial probabilities use posterior first-segment counts.
- Transitions use expected between-segment transition counts.
- Duration means use expected duration totals divided by expected segment counts.

Because duration probabilities are truncated, duration-mean updates are a practical generalized M-step. `fit` retains the evaluated parameter snapshot with the greatest likelihood.

## Viterbi decoding

The explicit-duration Viterbi recursion maximizes over predecessor states and entire candidate durations. Backtracking assigns states a segment at a time.

## Complexity

The dynamic programs depend on sequence length, state count, and maximum duration. Large duration caps and extensive tuning grids can be expensive. The public API is separated from private recursions so performance-critical code can later move to a compiled extension without breaking callers.

## Identifiability

State labels are exchangeable. Two statistically equivalent fits can permute state numbers. Evaluation should align states by parameters or decoded overlap rather than requiring identical labels.
