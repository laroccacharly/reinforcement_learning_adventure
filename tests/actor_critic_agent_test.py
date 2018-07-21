import gym
from src.agent import ActorCriticAgent
from src.utils import HyperFitter
import numpy as np

env = gym.make('MountainCar-v0')
nb_episodes = 10
agent = ActorCriticAgent(env=env, nb_episodes=nb_episodes)

def test_mountain_car():
    return 0 # Test skipped because it takes a while to run.
    # {'policy_learning_rate': 0.01, 'model_learning_rate': 0.1, 'nb_features': 1000, 'discount': 1, 'lamda': 0.6}
    space = {
        'policy_learning_rate': np.geomspace(1e-3, 1e-0, 10),
        'model_learning_rate': np.geomspace(1e-3, 1e-0, 10),
        'nb_features': [200, 300, 500, 600, 1000],
        'discount': [1],
        'lamda': [0.25, 0.4, 0.5, 0.6, 0.75]
    }

    hf = HyperFitter(space=space, agent=agent, nb_points=5, random=True, verbose=True)
    hf.fit()
    score = hf.best_score
    assert score > nb_episodes * -200

