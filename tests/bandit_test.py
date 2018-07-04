import mock

from src.env import BanditEnv
from src.env.bandit import Bandit

BANDIT_RETURN_VALUE = 1


def new_play(args):
    return BANDIT_RETURN_VALUE


def test_bandit_env():
    with mock.patch.object(Bandit, 'play', new=new_play): # patch the method play on the object Bandit
        env = BanditEnv(averages=[1, 2, 3])
        reward = env.step(action=0)
        assert reward == BANDIT_RETURN_VALUE

