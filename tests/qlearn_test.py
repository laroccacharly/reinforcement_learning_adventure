import gym
from src.agent import QlearnAgent

step_size = 0.1
env = gym.make('FrozenLake-v0')
agent = QlearnAgent(env=env, nb_episodes=100, step_size=step_size)


def test_learn():
    state = 0
    action = 0
    reward = 1
    next_state = 1
    next_action = 1
    agent.learn(state, action, reward, next_state, next_action, done=False)
    assert agent.model(state, action) == step_size * reward


def test_score():
    agent.reset()
    assert agent.model(state=0, action=0) == 0
    score = agent.score()
    assert agent.model(state=0, action=0) != 0
    assert isinstance(score, float)
