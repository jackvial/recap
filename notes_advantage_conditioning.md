# RECAP Advantage Conditioning

## Overview
Start with the standard KL regularized RL objective ([SAC]((https://arxiv.org/pdf/1801.01290)), [TRPO](https://arxiv.org/pdf/1502.05477))

```math
\mathcal{J}(\pi, \pi^{ref}) = \mathbb{E}_{\tau \sim \rho_{\pi_\theta}}[\sum_{t=0}^T \gamma^t r_t] - \beta \mathbb{E}_{\mathbf{o} \sim \rho_{\pi_\theta}} [D(\pi(\cdot \mid \mathbf{o}) \| \pi_{\text{ref}}(\cdot \mid \mathbf{o}))]
```

> Choose a policy that maximizes expected return minus a penalty for being different from the reference policy

 with the goal of getting to an NLL (token prediction) or MSE (Flow matching) supervised learning objective conditioned on our advantage derived from our value network.
 
 ```math
 \begin{equation}
\begin{aligned}
\min_\theta \: &\mathbb{E}_{\mathcal{D}_{\pi_{\mathrm{ref}}}} \Big[ -\log \pi_\theta(\mathbf{a}_t \mid \mathbf{o}_t, \ell) - \alpha \log \pi_\theta(\mathbf{a}_t \mid I_t, \mathbf{o}_t, \ell)\Big], \\
& \text{where } I_t = \mathbb{1}\big(A^{\pi_{\mathrm{ref}}}(\mathbf{o}_t, \mathbf{a}_t, \ell) > \epsilon_\ell \big).
\end{aligned}
\end{equation}
```
## Steps

KL regularized RL objective is based on max entropy RL. Max entropy builds on the Boltzmann-Gibbs distribution being the maximizer of max entropy. (Softmax is an example of the Boltzmann-Gibbs distribution). So the following term is the maximimzer of the KL reguluated objective. This is the Boltzmann Gibbs form with the normalizer omitted, hence porportional.

```math
\hat{\pi}(\mathbf{a} \mid \mathbf{o}) \propto \pi_{\mathrm{ref}}(\mathbf{a} \mid \mathbf{o}) \exp(A^{\pi_{\mathrm{ref}}}(\mathbf{o}, \mathbf{a}) / \beta)
```

Replace the exponential weight term with a policy improvement term and indicator $I$. This policy improvement term appears to be related to [CFGRL](https://arxiv.org/pdf/2505.23458)
```math
\hat{\pi}(\mathbf{a} \mid \mathbf{o}) \propto \pi_{\mathrm{ref}}(\mathbf{a} \mid \mathbf{o}) p(I \mid A^{\pi_{\mathrm{ref}}}(\mathbf{o}, \mathbf{a}))^\beta
```
where 
```math
p(I \mid A^{\pi_{\mathrm{ref}}}(\mathbf{o}, \mathbf{a})) = \frac{g(A^{\pi_{\mathrm{ref}}}(\mathbf{o}, \mathbf{a}))}{\int g(A^{\pi_{\mathrm{ref}}}(\mathbf{o}, \mathbf{a}')) \, \mathrm{d}\mathbf{a}'}
```
is the probability of any action a improving over $\pi_{ref}$.

Using Bayes rule, rewrite the policy improvement term so that actions are conditioned on the observation and indicator rather than indicator being conditioned on the observation and action. This is the conditional dependency we want for our supervised learning objective.
```math
p(I \mid A^{\pi_{\mathrm{ref}}}(\mathbf{o}, \mathbf{a})) = \frac{\pi_{\mathrm{ref}}(\mathbf{a} \mid I, \mathbf{o})}{\pi_{\mathrm{ref}}(\mathbf{a} \mid \mathbf{o})}
```

Plugging this into our language conditioned policy setting
```math
\begin{equation}
\hat{\pi}(\mathbf{a} \mid \mathbf{o}, \ell) \propto \pi_{\mathrm{ref}}(\mathbf{a} \mid \mathbf{o}, \ell) \left( \frac{\pi_{\mathrm{ref}}(\mathbf{a} \mid I, \mathbf{o}, \ell)}{\pi_{\mathrm{ref}}(\mathbf{a} \mid \mathbf{o}, \ell)} \right)^\beta
\end{equation}
```

Then set $\beta=1$ to achieve a few things:
- Removes the weighted term and moves the policy indictor into the policy term.
- Allows the same paramterized policy to be used for conditioned and unconditioned policy. They reference [CFGRL]([CFGRL](https://arxiv.org/pdf/2505.23458)) and say "This principle is similar to the approach in classifier-free guidance, where a diffusion model is trained
to model the data both with and without a conditioning
variable"

Then apply threshold to get a delta ($\delta$) distribution (Dirac delta is the continous case, Kronecker is the discrete case). That is to say it's a distribution that has a signle value of 1 and is 0 everywhere else.

```math
p(I \mid A^{\pi_{\mathrm{ref}}}(o, a, \ell)) = \delta\left(A^{\pi_{\mathrm{ref}}}(o, a, \ell) > \epsilon_\ell\right)
```

So at this point we can derive an NLL (equivalently cross entropy) objective suitable for supervised learning that is conditioned on the binary indictor derivied from the advantage we get from the trained value network and justified as policy improvement via the steps outlined above. Note that this only shows $\pi_{\theta}$ this is a consequence of the [CFGRL]([CFGRL](https://arxiv.org/pdf/2505.23458)) related move of using the same policy for the unconditioned and conditioned policy.

 ```math
 \begin{equation}
\begin{aligned}
\min_\theta \: &\mathbb{E}_{\mathcal{D}_{\pi_{\mathrm{ref}}}} \Big[ -\log \pi_\theta(\mathbf{a}_t \mid \mathbf{o}_t, \ell) - \alpha \log \pi_\theta(\mathbf{a}_t \mid I_t, \mathbf{o}_t, \ell)\Big], \\
& \text{where } I_t = \mathbb{1}\big(A^{\pi_{\mathrm{ref}}}(\mathbf{o}_t, \mathbf{a}_t, \ell) > \epsilon_\ell \big).
\end{aligned}
\end{equation}
```

For Flow-matching VLA we want an MSE objective over velocity. Details on the loss MSE are described in appendix C resulting in the squared error term in equation 9.

The $\pi0.5$ and $\pi^*0.6$ describe a NLL/CE loss for token prediction and the MSE loss for the Flow-matching head. Current implemetations of $\pi0.5$ in OpenPI and LeRobot appear to only support training with an MSE loss. The FAST versions of the models support the cross entropy loss training on tokens.

## References
- [π*0.6 : a VLA That Learns From Experience](https://arxiv.org/pdf/2511.14759)

- [Soft Actor-Critic:
Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290)
- [Diffusion Guidance Is a Controllable
Policy Improvement Operator (CFGRL)](https://arxiv.org/pdf/2505.23458)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477)
- [π0.5: a Vision-Language-Action Model with
Open-World Generalization](https://arxiv.org/pdf/2504.16054)