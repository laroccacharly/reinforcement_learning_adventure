from math import exp
from src.utils import Featurizer
import numpy as np


class Policy:
    """
    Policy with eligibility traces. Action probabilities are modeled by an exponential softmax distribution.
    """
    def __init__(self, env, learning_rate, discount, lamda, nb_features):
        self.action_space = [i for i in range(env.action_space.n)]
        self.nb_features = nb_features
        self.featurizer = Featurizer(nb_features=self.nb_features, env=env, with_action=True)
        self.learning_rate = learning_rate
        self.weights = np.zeros(self.nb_features)
        self.lamda = lamda
        self.discount = discount
        self.trace = np.zeros(self.nb_features)
        self.I = 1

    def choose_action(self, state):
        action_probs = []
        for action in self.action_space:
            action_probs.append(self.eval(state, action))

        action = np.random.choice(self.action_space, p=action_probs)
        return action

    def eval(self, state, action):
        denominator = 0
        for a in self.action_space:
            result = exp(self.param_fct(state, a))
            denominator += result
            if a == action:
                numerator = result

        return numerator / denominator

    def param_fct(self, state, action):
        x = self.featurizer.transform(state, action)
        return x.dot(self.weights)

    def update(self, state, action, target):
        self.trace = self.discount * self.lamda * self.trace + \
                     self.I * self.dln(state, action)
        self.I = self.I * self.discount
        self.weights = self.weights + self.learning_rate * target * self.trace

    # ∇θ ln π(a|s, θ)
    def dln(self, state, action):
        sum = 0
        for a in self.action_space:
            x = self.featurizer.transform(state, a)
            sum += self.eval(state, a) * x
            if a == action:
                xa = x

        return xa - sum
