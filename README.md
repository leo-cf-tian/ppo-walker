# PPO Walker Agent In Unity

In this project, I am training an agent in Unity to be able to walk around and navigate to locations based on raycast "visual" inputs.

The idea is to develop a simple walker, and try to extend it to some more complicated movements.

Here are the main resources referenced in this project:

**Unity Walker**: https://github.com/Unity-Technologies/ml-agents/tree/develop/Project/Assets/ML-Agents/Examples/Walker

## Demonstration

**Final Agent Locating and Walking Towards Target**
![Walking Test](/demo/Walking.gif)

Raycast visualised using lines, red indicating it "sees" the target.

**Separate Agent Standing (Test Run)**
![Walking Test](/demo/Standing.gif)

## Implentations

I am experimenting with the following algorithms / methods to complete this project.

- [x] PPO
- [ ] SAC
- [ ] Imitation Learning

## Action and Observation Space

I have written down some more notable things, some are left blank as they are repetive. This is more of a mental note to myself.

### Actions

| Action         | Range                       | Notes                                                            |
| -------------- | --------------------------- | ---------------------------------------------------------------- |
| Current angles | Normalized to $[0, 1]$      |                                                                  |
| Strength       |                             |                                                                  |
| Direction      | Normalized to $[-0.5, 0.5]$ | Represents how much the agent should adjust its current rotation |

### Observations

| Observation                        | Range                      | Notes                                             |
| ---------------------------------- | -------------------------- | ------------------------------------------------- |
| Current angles                     | Unity Vector3 / Quaternion | Should be relative to some general direction      |
| Current angular velocity           |                            |                                                   |
| Current position                   |                            | With reference to hips but also general direction |
| Current velocity                   |                            |                                                   |
| Target direction                   |                            | Unity's idea of a box is pretty nice              |
| Torso direction                    |                            | Relative to said box                              |
| Current torque                     | $(-\infty,\infty)$         |                                                   |
| Head direction                     |                            |                                                   |
| Average velocity of all body parts |                            |                                                   |
| Target speed                       | $[0, 10]$                  |                                                   |
| Body part touching the ground      | 0 or 1                     | Or just end the episode when it happens lol       |
| Feet raycast                       | Unity raycast              |                                                   |
| Head raycast                       |                            |                                                   |
| Current target                     | Normalized to $[0, 1]$     |                                                   |
| Current strength                   |                            |                                                   |

## Rewards & Curriculum

- [x] Teach agent to walk
	- Reward for time alive
	- Large punishment for falling down
	- Reward for forward velocity
	- Reward for lifting one foot off ground
- [x] Teach agent to reach target
	- Reward for decreasing distance from target
	- Reward for reaching target
	- Reward for facing the correct direction
- [x] Re-learn walking posture (was good before target training, but lost knowledge)
	- Reward for alternating feet
	- Reward for walking straight
- [ ] Terrain training
- [ ] Obstacle training


## Reinforcement Learning Implementations in Python

In this project I have also implemented **Deep Q Networks (DQN)** and **Proximal Policy Optimization (PPO-Clip)** from scratch using PyTorch, on the cartpole, pendulum, and bipedal walker gym environments. 

This does not really impact the Unity aspects of the projects, but I did it so that I could form a more foundational understanding of these algorithms before applying them to more complicated problems.

PPO implementation heavily references [this medium article](https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8).