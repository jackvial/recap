# Advantage Conditioning

## KL Regularized RL
- We want something like [PPO](https://arxiv.org/abs/1707.06347)/[TRPO](https://arxiv.org/abs/1502.05477)/[MAXIMUM A POSTERIORI POLICY OPTIMISATION](https://arxiv.org/pdf/1806.06920) regularization so we don't take too large of a step away from our reference policy but we want to end up with something we can use in a supervised learning objective (the advantage indicator).
- Maximum Entropy Inverse Reinforcement Learning https://www.ri.cmu.edu/pub_files/2008/7/AAAI2008-bziebart.pdf?utm_source=chatgpt.com
- RL regularization notes https://rl-vs.github.io/rlvs2021/class-material/regularized_mdp/Regularization_RL_RLVS.pdf


KL regularized RL objective. Choose a policy that maximizes expected return minus a penalty for being different from the reference policy. The divergence D is usually the KL divergence.

```math
\mathcal{J}(\pi, \pi^{ref}) = \mathbb{E}_{\tau \sim \rho_{\pi_\theta}}[\sum_{t=0}^T \gamma^t r_t] - \beta \mathbb{E}_{\mathbf{o} \sim \rho_{\pi_\theta}} [D(\pi(\cdot \mid \mathbf{o}) \| \pi_{\text{ref}}(\cdot \mid \mathbf{o}))]
```
We do not want to optimize the above directly, instead we use  the following Boltzmann/softmax closed-form optimizer. 

```math
\hat{\pi}(a \mid o)=
\frac{\pi_{\text{ref}}(a \mid o)\exp\left(A^{\pi_{\text{ref}}}(o, a) / \beta\right)}
{\sum_{a'} \pi_{\text{ref}}(a' \mid o)\exp\left(A^{\pi_{\text{ref}}}(o, a') / \beta\right)}
```

This is often written with the denominator omitted 
```math
\hat{\pi}(\mathbf{a} \mid \mathbf{o}) \propto \pi_{\text{ref}}(\mathbf{a} \mid \mathbf{o}) \exp\left(A^{\pi_{\text{ref}}}(\mathbf{o}, \mathbf{a}) / \beta\right)
```

For continuous action spaces, replace the sum in the denominator with the corresponding integral / partition function.

This is related to the fact that softmax is the solution to the KL regularized argmax. Some useful references:
- [On the Properties of the Softmax Function with Application in
Game Theory and Reinforcement Learning](https://arxiv.org/pdf/1704.00805)
- [Soft Actor-Critic:
Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290)

For some intuition, consider that softmax is the "soft" version of the argmax. Argmax says take the arg with highest value with absolute certainty, softmax says prefer higher scores but still leave some uncertainty. As temperature goes to zero, softmax concentrates on the argmax set; when there is a unique maximizer it converges to the corresponding one-hot distribution.
