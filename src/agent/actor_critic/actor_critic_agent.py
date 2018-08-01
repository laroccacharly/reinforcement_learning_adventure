from src.agent.agent_base import AgentBase
from src.utils import Featurizer
from .policy import Policy
from .model import EligibilityModel


class ActorCriticAgent(AgentBase):
    """
    Policy gradient agent,  One-step Actor–Critic
    """
    def __init__(self, env, nb_episodes,
                 policy_learning_rate,
                 model_learning_rate,
                 nb_features,
                 discount,
                 lamda):
        self.policy_learning_rate = policy_learning_rate
        self.model_learning_rate = model_learning_rate
        self.nb_features = nb_features
        self.discount = discount
        self.lamda = lamda
        AgentBase.__init__(self, env, nb_episodes)
        self.reset()

    def reset(self):
        super(ActorCriticAgent, self).reset()
        self.policy = Policy(env=self.env, learning_rate=self.policy_learning_rate, nb_features=self.nb_features,
                             lamda=self.lamda, discount=self.discount)
        self.model = EligibilityModel(featurizer=Featurizer(nb_features=self.nb_features, env=self.env),
                                      learning_rate=self.model_learning_rate, lamda=self.lamda, discount=self.discount)
        self.I = 1

    def choose_action(self, state):
        return self.policy.choose_action(state)

    def learn(self,  state, action, reward, next_state, next_action, done):
        delta = reward + self.discount * self.model(next_state) - self.model(state)
        self.model.update(state, delta)
        self.policy.update(state, action, delta)

