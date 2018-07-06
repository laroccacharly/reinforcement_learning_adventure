import gym
from src.agent import MonteAgent


def test_on_blackjack():
    env = gym.make('Blackjack-v0')
    agent = MonteAgent(env=env, nb_episodes=100)
    score = agent.score()

    assert agent.returns_dict != {}
    assert isinstance(score, float)


