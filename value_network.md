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


$$
R_t^B(\tau).
$$

This is a one-hot representation of the return bucket assigned to timestep $t$.

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
