from src.utils import Featurizer
import gym


def test_basic():
    env = gym.make("MountainCar-v0")
    nb_features = 400
    featurizer = Featurizer(env=env, nb_features=nb_features)
    transformed = featurizer.transform(env.observation_space.sample())
    assert len(transformed) == nb_features
    assert max(transformed) < 1
