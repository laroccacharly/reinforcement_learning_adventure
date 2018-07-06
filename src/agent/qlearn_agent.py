from src.utils import EGreedyPolicy
from .agent_base import AgentBase


class QlearnAgent(AgentBase):
    """
       Q-Learning
    """
    def __init__(self, env, nb_episodes, epsilon=0.1, step_size=0.1, discount=1,  verbose=False):
        AgentBase.__init__(self, env, nb_episodes)
        self.epsilon = epsilon
        self.step_size = step_size
        self.discount = discount
        self.verbose = verbose
        self.action_space = [i for i in range(env.action_space.n)]
        self.transitions = []
        self.default_action_value = 0
        self.reset()

    def reset(self):
        self.values = {}
        self.policy = EGreedyPolicy(epsilon=self.epsilon)

    def action_values(self, state, action=None):
        if action is not None:
            return self.values.setdefault(state, {}).setdefault(action, self.default_action_value)
        else:
            values = [self.action_values(state, a) for a in self.action_space]
            return dict(zip(self.action_space, values))

    def choose_action(self, state):
        action_values = self.action_values(state)
        action = self.policy(action_values)
        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        if self.verbose:
            print(state, action, reward, next_state, next_action, done)
        target = reward + self.discount * max(self.action_values(next_state).values())
        current_value = self.action_values(state, action)
        new_value = current_value + self.step_size * (target - current_value)
        self.update_action_value(state, action, new_value)

    def update_action_value(self, state, action, value):
        self.values[state][action] = value

