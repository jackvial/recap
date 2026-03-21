# RECAP / π∗0.6

## Resources
- [π∗0.6 paper](https://arxiv.org/abs/2511.14759)
- [LeRobot RL refactoring call for contributions](https://github.com/huggingface/lerobot/issues/3076#issuecomment-4035545530)
- [DRTC](https://jackvial.com/posts/distributed-real-time-chunking.html)

## Roadmap

### Milestone 1
- [x] Notes on the main value function equations from the paper
    - [value network notes](./notes_value_network.md)
    - [value network code example](./value_network.py)
- [ ] Notes on base model architecture needed for the value function/network
- [x] Notes on RECAP advantage conditioning
    - [advantage conditioning notes](./notes_advantage_conditioning.md)
- [ ] Collect a small dataset of a pick and place task on an SO-101. e.g. 20 positive examples, 20 negative examples.
- [ ] Label the episodes with binary reward labels. Figure out where to keep the reward labels. 
- [ ] Implement the value network
- [ ] Train the value network
- [ ] Implement the main model with advantage conditioning
- [ ] Train the main model
- [ ] Evaluate the trained policy. We should expect the policy to show some better generalization and corrective behavior compared to the same model trained without advantage conditioning.
- [ ] Documentation. We should aim to get this much merged and provide getting started documentation. Should include our datasets, trained models, and experimental results as this will be useful reference material.
### Milestone 2
- [ ] Add recording during rollouts. Current plan is to build this on [DRTC](https://jackvial.com/posts/distributed-real-time-chunking.html) since record_episode does not currently support RTC afaik.
- [ ] Reward labeling for rollouts. This should be able to use the same tooling as milestone 1.
- [ ] Dataset mixing. Support mixing reward labeled and non-labeled data.
- [ ] Collect data during rollouts. Reward label the episodes. Mix the data. Train the value network. Train the advantage conditioned model. Much like milestone one, we should expect to see some improvements over the baseline model.
- [ ] Documentation. We should aim to get this much merged and provide getting started documentation. Should include our datasets, trained models, and experimental results as this will be useful reference material.
### Milestone 3 
- [ ] Add intervention during rollouts. Intervention will be done via game control. Interventions will nudge the robot by delta from its current position.
- [ ] Intervention labeling during recording. If intervention is engaged add an intervention true label for that step.
- [ ] Update the advantage conditioned model to handle per intervention labels and force indicator true when intervention is true regardless of the computed advantage.
- [ ] Collect episodes with intervention. Mix data, train value network, train advantage conditioned network. Evaluate, expect to see improvement over baseline.
- [ ] Documentation. We should aim to get this much merged and provide getting started documentation. Should include our datasets, trained models, and experimental results as this will be useful reference material.

## Notes
- [value network notes](./notes_value_network.md)
- [value network code example](./value_network.py)