from abc import ABCMeta, abstractmethod
from .tabular_model import TabularModel
from src.utils import EGreedyPolicy


class AgentBase:
    """
    The base of every agent. This is the interface with the env.
    Every child implements a different way of choosing an action (policy) and learning from data.
    """
    __metaclass__ = ABCMeta

    def __init__(self, env, nb_episodes=1):
        self.env = env
        self.nb_episodes = nb_episodes

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def learn(self,  state, action, reward, next_state, next_action, done):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def set_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)
            if self.model is not None: # Kinda hacky. Allows me to broadcast attributes down to the model.
                setattr(self.model, k, v)
        self.reset()

    def play_one_episode(self, render=False):
        state = self.env.reset()
        action = self.choose_action(state)

        total_reward = 0
        done = False
        while not done:
            if render:
                self.env.render()
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            if done:
                self.learn(state, action, reward, next_state, None, done)
                break

            next_action = self.choose_action(next_state)

            self.learn(state, action, reward, next_state, next_action, done)

            action = next_action
            state = next_state

        return total_reward

    def score(self):
        total_reward = 0
        for _ in range(self.nb_episodes):
            total_reward += self.play_one_episode()

        return total_reward


class GreedyAgentBase(AgentBase):
    """
    Because most agents are greedy.
    It implements the choose_action method using the EGreedyPolicy.
    His children would have different ways of learning the action_values.
    """
    def __init__(self, env, nb_episodes, epsilon=0.1):
        AgentBase.__init__(self, env, nb_episodes)
        self.epsilon = epsilon
        self.action_space_dim = env.action_space.n
        self.action_space = [i for i in range(self.action_space_dim)]
        self.model = TabularModel()

    def reset(self):
        self.policy = EGreedyPolicy(epsilon=self.epsilon)
        self.model.reset()

    def choose_action(self, state):
        action_values = self.action_values(state)
        action = self.policy(action_values)
        return action

    def action_values(self, state, action=None):
        if action is not None:
            return self.model(state, action)
        else:
            values = [self.action_values(state, a) for a in self.action_space]
            return dict(zip(self.action_space, values))

    def learn(self,  state, action, reward, next_state, next_action, done):
        raise NotImplementedError

