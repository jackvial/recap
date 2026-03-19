# Value Network

Given an observation at time $t$, the goal is to predict the cumulative reward-to-go, i.e. the value of being in the current state at the current time.

## Training Objective

The value network is trained as a classification problem over binned returns:

```math
\min_{\phi} \; \mathbb{E}_{\tau \in \mathcal{D}}
\left[
\sum_{o_t \in \tau}
H\!\left(R_t^B(\tau),\, p_{\phi}(V \mid o_t, \ell)\right)
\right].
```

Where:

- $o_t$ is the observation at time $t$
- $\ell$ is the language task instruction
- $R_t^B(\tau)$ is the binned return-to-go target (represented as a one-hot vector)
- $p_{\phi}(V \mid o_t, \ell)$ policy $p$ where the output is the predicted binned return-to-go. Not be confused with the stochastic dynamics from preliminaries which is also notated as $p$.
- H is cross-entropy loss
- $\mathcal{D}$ is the dataset of trajectories

## Inputs and Outputs

- Inputs: $o_t, \ell$
- Outputs: logits over return bins
- Targets: bin id/class ids

## Standard RL Objective

The standard reinforcement learning objective is to maximize expected return:

```math
\begin{aligned}
J(\pi)
&= \mathbb{E}_{\tau \sim p_{\pi}}[R(\tau)] \\
&= \mathbb{E}_{\tau \sim p_{\pi}} \left[\sum_{t=0}^{T} r_t\right].
\end{aligned}
```

## Value Function

The value function is the expected return starting from the current state and then following policy $\pi$:

```math
V^{\pi}(o_t)
=
\mathbb{E}_{t+1:T}
\left[
\sum_{t'=t}^{T} r_{t'}
\right].
```

This can be interpreted as the expected return-to-go from state $o_t$ under policy $\pi$.


## Reward Definition

The reward sequence is defined as:

```math
r_t =
\begin{cases}
0, & \text{if } t = T \text{ and success} \\
-C_{\mathrm{fail}}, & \text{if } t = T \text{ and failure} \\
-1, & \text{otherwise.}
\end{cases}
```

This means:

- successful episodes receive terminal reward $0$
- failed episodes receive terminal reward $-C_{\mathrm{fail}}$
- every non-terminal step incurs a reward of $-1$

## Return and Binned Return

Trajectory return is:

```math
R(\tau) = \sum_{t=0}^{T} r_t.
```

This is then quantized/binned.

```math
R_t^B(\tau).
```

```python
# Binned return's example
import numpy as np

NUM_BINS = 5

def value_targets_from_success(success: bool, episode_len: int, max_episode_len: int, c_fail: int):
    assert NUM_BINS <= episode_len
    assert max_episode_len >= episode_len
    if success:
        # t = 0..episode_len-1, target is negative remaining steps normalized to [-1, 0]
        returns = -np.arange(episode_len - 1, -1, -1, dtype=np.float32) / max_episode_len
    else:
        # failed episodes get a large negative value, then clamp into training range if desired
        returns = np.full(episode_len, -c_fail / max_episode_len, dtype=np.float32)

    # paper says values are predicted in (-1, 0)
    returns = np.clip(returns, -1.0, 0.0)

    bin_edges = np.linspace(-1.0, 0.0, NUM_BINS + 1)

    # Categorical label that will used as the target for cross entropy loss
    # we are learning to predict bin ids not returns. the predicted bin
    # will be used to recover an approximated return-to-go at inference
    target_bin_ids = np.clip(np.digitize(returns, bin_edges) - 1, 0, NUM_BINS - 1)

    return target_bin_ids, returns
```

Cross-entropy is applied per timestep. Assume belong represents a batch of timesteps.

```python
import torch
import torch.nn.functional as F

target_bin_ids, returns = value_targets_from_success(
    success=True,
    episode_len=10,
    max_episode_len=11,
    c_fail=20,
)

# Shape: (10, NUM_BINS)
# One NUM_BINS-bin prediction for each timestep.
logits = torch.randn(len(target_bin_ids), NUM_BINS)

# Shape: (10,)
# One target class index for each timestep.
targets = torch.tensor(target_bin_ids, dtype=torch.long)

loss = F.cross_entropy(logits, targets)
```

At inference time, the model predicts a categorical distribution over bins, and we convert that back into a scalar value estimate by taking the expectation over bin values:

```math
\hat V_{\phi}(o_t, \ell) = \sum_{b=0}^{B-1} p_\phi(V=b \mid o_t, \ell)\, v(b),
```

where $v(b)$ denotes the representative scalar value for bin $b$. In `value_network.py`, this is approximated with evenly spaced values in $[-1, 0]$.

```python
predicted_probs = F.softmax(logits, dim=-1)

# One option is to use representative values derived from the bins.
vb_maybe = returns.reshape(NUM_BINS, -1)[:, -1]

# A simple approximation is to use evenly spaced values in [-1, 0].
vb = torch.linspace(-1, 0, NUM_BINS)

v_ref = (predicted_probs * vb).sum(dim=-1)
```
## Worked Example

Take the same settings as `value_network.py`:

- `success = True`
- `episode_len = 10`
- `max_episode_len = 11`
- `NUM_BINS = 5`

By Eq. (5), a successful episode of length 10 has reward sequence:

$$
[-1,\,-1,\,-1,\,-1,\,-1,\,-1,\,-1,\,-1,\,-1,\,0].
$$

The unnormalized return-to-go from each timestep is therefore:

$$
[-9,\,-8,\,-7,\,-6,\,-5,\,-4,\,-3,\,-2,\,-1,\,0].
$$

What `value_network.py` actually trains on is the normalized return:

$$
\tilde R_t
=
\mathrm{clip}\!\left(\frac{R_t}{\texttt{max\_episode\_len}}, -1, 0\right)
=
\left[
\frac{-9}{11},\,
\frac{-8}{11},\,
\frac{-7}{11},\,
\frac{-6}{11},\,
\frac{-5}{11},\,
\frac{-4}{11},\,
\frac{-3}{11},\,
\frac{-2}{11},\,
\frac{-1}{11},\,
0
\right]
$$

which numerically is:

$$
[-0.818,\,-0.727,\,-0.636,\,-0.545,\,-0.455,\,-0.364,\,-0.273,\,-0.182,\,-0.091,\,0].
$$

These values are then discretized into bins. With `NUM_BINS = 5`, the code uses:

$$
\texttt{bin\_edges}=\mathrm{linspace}(-1, 0, 6)=[-1,\,-0.8,\,-0.6,\,-0.4,\,-0.2,\,0].
$$

Applying `np.digitize(...)-1` gives the target class for each timestep:

$$
[0,\,1,\,1,\,2,\,2,\,3,\,3,\,4,\,4,\,4].
$$

These class ids are the targets in Eq. (1). For timestep $t$, if the network outputs logits $z_t \in \mathbb{R}^5$, then the per-timestep loss is:

$$
H\!\left(R_t^B(\tau),\, p_\phi(V \mid o_t, \ell)\right)
=
\mathrm{CE}\!\left(\text{onehot}(b_t),\, \mathrm{softmax}(z_t)\right),
$$

and in code `F.cross_entropy(logits, targets)` averages this over all timesteps in the batch.

For example, suppose one timestep has predicted probabilities

$$
[0.045,\;0.447,\;0.068,\;0.366,\;0.074].
$$

Using the representative bin values from the script,

$$
v(b)=\mathrm{linspace}(-1,0,5)=[-1,\,-0.75,\,-0.5,\,-0.25,\,0],
$$

the scalar value estimate is:

$$
\hat V_{\phi}(o_t,\ell)
=
\sum_{b=0}^{4} p_\phi(V=b \mid o_t,\ell)\, v(b)
$$

$$
=
(0.045)(-1) + (0.447)(-0.75) + (0.068)(-0.5) + (0.366)(-0.25) + (0.074)(0)
\approx -0.50.
$$

So the full pipeline in the note now matches the code:

$$
\text{rewards}
\rightarrow
\text{return-to-go}
\rightarrow
\text{normalize to } [-1,0]
\rightarrow
\text{bin id target}
\rightarrow
\text{cross-entropy training}
\rightarrow
\text{expected scalar value at inference}.
$$