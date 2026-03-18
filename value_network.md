# Value Network

Given an observation at time $t$, the goal is to predict the cumulative reward-to-go, i.e. the value of being in the current state at the current time.

## Training Objective

The value network is trained as a classification problem over binned returns:

$$
\min_{\phi} \; \mathbb{E}_{\tau \in \mathcal{D}}
\left[
\sum_{o_t \in \tau}
H\!\left(R_t^B(\tau),\, p_{\phi}(V \mid o_t, \ell)\right)
\right].
\tag{1}
$$

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
- Targets: one-hot vectors of binned returns-to-go

## Standard RL Objective

The standard reinforcement learning objective is to maximize expected return:

$$
J(\pi)
=
\mathbb{E}_{\tau \sim p_{\pi}}[R(\tau)]
=
\mathbb{E}_{\tau \sim p_{\pi}}
\left[\sum_{t=0}^{T} r_t\right].
$$

## Value Function

The value function is the expected return starting from the current state and then following policy $\pi$:

$$
V^{\pi}(o_t)
=
\mathbb{E}_{t+1:T}
\left[
\sum_{t'=t}^{T} r_{t'}
\right].
$$

This can be interpreted as the expected return-to-go from state $o_t$ under policy $\pi$.


## Reward Definition

The reward sequence is defined as:

$$
r_t =
\begin{cases}
0, & \text{if } t = T \text{ and success} \\
-C_{\mathrm{fail}}, & \text{if } t = T \text{ and failure} \\
-1, & \text{otherwise.}
\end{cases}
\tag{5}
$$

This means:

- successful episodes receive terminal reward $0$
- failed episodes receive terminal reward $-C_{\mathrm{fail}}$
- every non-terminal step incurs a reward of $-1$

## Return and Binned Return

Trajectory return is:

$$
R(\tau) = \sum_{t=0}^{T} r_t.
$$

This is then quantized/binned.

$$
R_t^B(\tau).
$$

```python
# Binned return's example
import numpy as np

NUM_BINS = 201

def value_targets_from_success(success: bool, T: int, max_episode_len: int, c_fail: int):
    if success:
        # t = 0..T, target is negative remaining steps normalized to [-1, 0]
        returns = -np.arange(T, -1, -1, dtype=np.float32) / max_episode_len
    else:
        # failed episodes get a large negative value, then clamp into training range if desired
        returns = np.full(T + 1, -c_fail / max_episode_len, dtype=np.float32)

    # paper says values are predicted in (-1, 0)
    returns = np.clip(returns, -1.0, 0.0)

    bin_edges = np.linspace(-1.0, 0.0, NUM_BINS + 1)
    
    # tell us which bin each step belongs to
    bin_ids = np.clip(np.digitize(returns, bin_edges) - 1, 0, NUM_BINS - 1)

    return returns, bin_ids
```

Cross-entropy is applied per timestep, not once per whole trajectory.

If a trajectory has `T + 1` observations, then:

- `returns.shape == (T + 1,)`
- `bin_ids.shape == (T + 1,)`
- `logits.shape == (T + 1, NUM_BINS)`

Each row `logits[t]` is the model's predicted distribution over the 201 bins for observation `o_t`, and `bin_ids[t]` is the index of the correct bin for that same timestep.

```python
import torch
import torch.nn.functional as F

# Example successful trajectory with 5 observations: t = 0..4
returns, bin_ids = value_targets_from_success(
    success=True,
    T=4,
    max_episode_len=10,
    c_fail=20,
)

# Shape: (5, 201)
# One 201-bin prediction for each timestep.
logits = torch.randn(len(bin_ids), NUM_BINS)

# Shape: (5,)
# One target class index for each timestep.
targets = torch.tensor(bin_ids, dtype=torch.long)

loss = F.cross_entropy(logits, targets)
```

This works because PyTorch interprets the inputs as:

- `logits[t, :]`: scores for all bins at timestep `t`
- `targets[t]`: the correct bin index at timestep `t`

So the loss is effectively:

$$
\frac{1}{T+1}\sum_{t=0}^{T} -\log p_{\phi}(V = \text{bin\_ids}[t] \mid o_t, \ell).
$$


## Intuition with a Reward Sequence

For a successful episode, a reward sequence might look like:

$$
[-1,\,-1,\,-1,\,-1,\,-1,\,-1,\,-1,\,-1,\,-1,\,0].
$$

The corresponding cumulative returns-to-go would be:

$$
[-9,\,-8,\,-7,\,-6,\,-5,\,-4,\,-3,\,-2,\,-1,\,0].
$$

Those scalar returns are then discretized into bins (for example `bin 0`, `bin 1`, `bin 2`, `bin 3`), and the network predicts logits over those bins.
