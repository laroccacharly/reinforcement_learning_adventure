import numpy as np


class EligibilityModel:
    """
    Linear model with eligibility traces
    """
    def __init__(self, featurizer, learning_rate, lamda, discount=1):
        self.learning_rate = learning_rate
        self.discount = discount
        self.featurizer = featurizer
        self.nb_features = featurizer.nb_features
        self.weights = np.zeros(self.nb_features)
        self.trace = np.zeros(self.nb_features)
        self.lamda = lamda
        self.I = 1

    def __call__(self, s):
        x = self.featurizer.transform(s)
        return x.dot(self.weights)

    def update(self, state, delta):
        x = self.featurizer.transform(state)
        self.trace = self.discount * self.lamda * self.trace + self.I * x
        self.weights = self.weights + self.learning_rate * delta * self.trace
        self.I = self.I * self.discount
        return delta

