import mock
import pytest
from src.env import BanditEnv
from src.env.bandit import Bandit
from src.agent import BanditAgent
BANDIT_RETURN_VALUE = 1


def new_play(args):
    return BANDIT_RETURN_VALUE


def test_bandit_env():
    with mock.patch.object(Bandit, 'play', new=new_play): # patch the method play on the object Bandit
        env = BanditEnv(averages=[1, 2, 3])
        reward = env.step(action=0)
        assert reward == BANDIT_RETURN_VALUE


def make_agent(averages, nb_iter, epsilon=0.1):
    env = BanditEnv(averages=averages)
    agent = BanditAgent(env=env, nb_iter=nb_iter, epsilon=epsilon)
    return agent


def test_bandit_agent_simple():
    averages = [3]
    nb_iter = 1000
    agent = make_agent(averages, nb_iter)
    score = agent.score()

    assert pytest.approx(averages[0] * nb_iter, 0.1) == score
    for index, average in enumerate(averages):
        assert pytest.approx(average,  0.1) == agent.action_values[index]

    assert agent.action_selection_count == [nb_iter]


def test_bandit_agent_multiple():
    averages = [3, 100]
    nb_iter = 2000
    agent = make_agent(averages, nb_iter)
    score = agent.score()

    assert isinstance(score, float)
    for index, average in enumerate(averages):
        assert pytest.approx(average,  0.1) == agent.action_values[index]

    assert agent.action_selection_count[0] < agent.action_selection_count[1]  # Because 3 < 100



