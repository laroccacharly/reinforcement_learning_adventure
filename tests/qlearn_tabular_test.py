import gym
from src.agent import QlearnAgent

learning_rate = 0.1
env = gym.make('FrozenLake-v0')
agent = QlearnAgent(env=env, nb_episodes=1000, learning_rate=learning_rate)


def test_learn():
    state = 0
    action = 0
    reward = 1
    next_state = 1
    next_action = 1
    agent.learn(state, action, reward, next_state, next_action, done=False)
    assert agent.model(state, action) == learning_rate * reward


def test_score():
    agent.reset()
    assert agent.model(state=0, action=0) == 0
    score = agent.score()
    assert agent.model(state=0, action=0) != 0
    assert score > 0
