import gym
from src.agent import QlearnAgent


def test_on_blackjack():
    env = gym.make('FrozenLake-v0')
    agent = QlearnAgent(env=env, nb_episodes=100)
    score = agent.score()

    assert agent.values != {}
    assert isinstance(score, float)
