from src.utils import Featurizer
import gym

env = gym.make("MountainCar-v0")
nb_features = 400


def test_basic():
    featurizer = Featurizer(env=env, nb_features=nb_features)
    transformed = featurizer.transform(env.observation_space.sample())
    assert len(transformed) == nb_features
    assert max(transformed) < 1


def test_with_action():
    featurizer = Featurizer(env=env, nb_features=nb_features, with_action=True)
    transformed = featurizer.transform(env.observation_space.sample(), action=env.action_space.sample())
    assert len(transformed) == nb_features
    assert max(transformed) < 1