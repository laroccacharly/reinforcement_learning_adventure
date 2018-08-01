import numpy as np

from .model_base import ModelBase
from src.utils import Featurizer


class LinearModel(ModelBase):
    def __init__(self, env, learning_rate):
        self.env = env
        self.learning_rate = learning_rate
        self.featurizer = Featurizer(env=env, nb_features=400)
        self.models = [SubModel(self.featurizer, learning_rate=self.learning_rate) for _ in range(self.env.action_space.n)]


# This needs a better name.
class SubModel:
    def __init__(self, featurizer, learning_rate):
        self.weights = np.zeros(featurizer.nb_features)
        self.learning_rate = learning_rate
        self.featurizer = featurizer

    def __call__(self, state):
        x = self.featurizer.transform(state)
        return x.dot(self.weights)  # We model the Q function as linear combination of an extended state vector.

    def update(self, state, target):
        x = self.featurizer.transform(state)
        current_value = x.dot(self.weights)
        delta = target - current_value
        # The update rule.
        self.weights = self.weights + self.learning_rate * delta * x
        return abs(delta)
