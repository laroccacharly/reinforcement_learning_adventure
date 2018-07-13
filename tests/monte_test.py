import gym
from src.agent import MonteAgent

env = gym.make('Blackjack-v0')
agent = MonteAgent(env=env, nb_episodes=100)

def test_score():
    score = agent.score()
    assert agent.returns_dict != {}
    assert isinstance(score, float)


def test_learn():
    agent.reset()
    agent.learn(state=0, action=0, reward=0, next_state=0, next_action=0, done=False)
    assert len(agent.transitions) == 1
    agent.learn(state=0, action=1, reward=0, next_state=0, next_action=0, done=False)
    assert len(agent.transitions) == 2
    agent.learn(state=0, action=1, reward=0, next_state=0, next_action=0, done=False)
    assert len(agent.transitions) == 2
