# Advantage Conditioning

## KL Regularized RL
- We want something like [PPO](https://arxiv.org/abs/1707.06347)/[TRPO](https://arxiv.org/abs/1502.05477)/[MAXIMUM A POSTERIORI POLICY OPTIMISATION](https://arxiv.org/pdf/1806.06920) regularization so we don't take too large of a step away from our reference policy but we want to end up with something we can use in a supervised learning objective (the advantage indicator).
- The KL regularization is informated by information geometry and natural policy gradients
- Maximum Entropy Inverse Reinforcement Learning https://www.ri.cmu.edu/pub_files/2008/7/AAAI2008-bziebart.pdf?utm_source=chatgpt.com
- RL regularization notes https://rl-vs.github.io/rlvs2021/class-material/regularized_mdp/Regularization_RL_RLVS.pdf


KL regularized RL objective

```math
\mathcal{J}(\pi, \pi^{ref}) = \mathbb{E}_{\tau \sim \rho_{\pi_\theta}}[\sum_{t=0}^T \gamma^t r_t] - \beta \mathbb{E}_{\mathbf{o} \sim \rho_{\pi_\theta}} [D(\pi(\cdot \mid \mathbf{o}) \| \pi_{\text{ref}}(\cdot \mid \mathbf{o}))]
```
choose a policy that maximizes expected return minus a penalty for being different from the reference policy