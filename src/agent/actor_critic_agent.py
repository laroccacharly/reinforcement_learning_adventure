from src.agent.agent_base import AgentBase
import numpy as np
from math import exp
from src.utils import Featurizer


class ActorCriticAgent(AgentBase):
    """
    Policy gradient agent,  One-step Actor–Critic
    """
    def __init__(self, env, nb_episodes):
        AgentBase.__init__(self, env, nb_episodes)
        self.policy_learning_rate = 0.01
        self.model_learning_rate = 0.01
        self.nb_features = 400
        self.discount = 1
        self.I = 1
        self.lamda = 0.5
        self.reset()

    def reset(self):
        self.policy = Policy(env=self.env, learning_rate=self.policy_learning_rate, nb_features=self.nb_features,
                             lamda=self.lamda, discount=self.discount)
        self.model = EligibilityModel(featurizer=Featurizer(nb_features=self.nb_features, env=self.env),
                                      learning_rate=self.model_learning_rate, lamda=self.lamda, discount=self.discount)

    def choose_action(self, state):
        return self.policy.choose_action(state)

    def learn(self,  state, action, reward, next_state, next_action, done):
        delta = reward + self.discount * self.model(next_state) - self.model(state)
        self.model.update(state, delta)
        self.policy.update(state, action, delta)


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


class Policy(object):
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
