# A reinforcement learning adventure
This repository contains a compilation of reinforcement learning algorithms and a test suite. 

## List of algorithms
- Action-value method (to solve the k-armed Bandit problem)
- Q-Learning (using a linear model and a deep neural network)
- Monte Carlo Control
- Actor-Critic with eligibility traces 
- DDPG 
- PPO (WIP)

## Related works
- OpenAI baseline (https://github.com/openai/baselines/tree/master/baselines )
- rllab (https://github.com/rll/rllab)
- Implementation of selected reinforcement learning algorithms in Tensorflow. A3C, DDPG, REINFORCE, DQN, etc. (https://github.com/stormmax/reinforcement_learning)
- Implementation of Reinforcement Learning Algorithms. Python, OpenAI Gym, Tensorflow. (https://github.com/dennybritz/reinforcement-learning)
- A set of Deep Reinforcement Learning Agents implemented in Tensorflow. (https://github.com/awjuliani/DeepRL-Agents)
- TensorFlow implementation of Deep Reinforcement Learning papers (https://github.com/carpedm20/deep-rl-tensorflow)
- Pytorch Implementation of DQN / DDQN / Prioritized replay/ noisy networks/ distributional values/ Rainbow/ hierarchical RL (https://github.com/higgsfield/RL-Adventure)
- Highly modularized implementation of popular deep RL algorithms in PyTorch (https://github.com/ShangtongZhang/DeepRL)


## Hyperparameters tunning 
This repository also includes an abstraction (HyperFitter) that allows a Grid or Random search 
on the hyperparameter space of the agent. It is inspired by HyperOpt and sklearn GridSearchCV. 


## Dependencies 
- PyTorch (for neural networks)
- TensorFlow 
- OpenAI Gym (for the environments)
- pytest (for testing)
- sklearn (for feature mapping)
- Matplotlib (for plotting)
- NumPy 

### Testing 
Testing is an interesting issue in RL. 
It is not clear what assertions should be made about an agent. 
We know that it should learn, but it is hard to predict what the learning curve should look like. 
See this great talk by Dr. Pineau : https://www.youtube.com/watch?v=-0G98MYUtjI

This repo mainly use integration testing (testing the general behavior instead of every method of every class).
To do so, tests check if the return of the agent (the metric we try to maximize) 
is greater then the return of an agent that takes random decisions.
 
```
PYTHONPATH=. pytest 
```
You can add the flag ``--pdb`` to debug 
