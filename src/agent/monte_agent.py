from .agent_base import AgentBase
from collections import namedtuple
import numpy as np
from statistics import mean
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'next_action', 'done'))


class MonteAgent(AgentBase):
    """
    MONTE CARLO CONTROL WITHOUT EXPLORING STARTS
    """
    def __init__(self, env, nb_episodes=1, epsilon=0.1, verbose=False):
        AgentBase.__init__(self, env, nb_episodes)
        self.epsilon = epsilon
        self.verbose = verbose
        self.action_space = env.action_space
        self.transitions = []
        self.reset()

    def reset(self):
        self.returns_dict = {}

    def choose_action(self, state):
        if not self.is_observed_state(state) or self.is_exploring():
            return self.random_action()

        return self.best_action(state)

    def best_action(self, state):
        actions_dict = self.returns_dict[state]
        expected_returns = []
        actions = []
        for action, returns in actions_dict.items():
            actions.append(action)
            expected_returns.append(mean(returns))

        best_action = actions[np.argmax(expected_returns)]
        return best_action

    def learn(self, state, action, reward, next_state, next_action, done):
        self.transitions.append(Transition(state, action, reward, next_state, next_action, done))
        if done:  # MonteCarlo algo updates when the episode is done
            _return = 0
            for t in reversed(self.transitions):
                # TODO: Check if transition t is first, if not skip.
                _return = _return + t.reward
                self.append_return(t.state, t.action, _return)
            self.transitions = []

    def is_exploring(self):
        return np.random.uniform(0, 1) < self.epsilon

    def is_observed_state(self, state):
        if self.returns_dict.get(state) is None:
            return False
        else:
            return True

    def random_action(self):
        return self.action_space.sample()

    def append_return(self, state, action, _return):
        returns = self.returns_dict.setdefault(state, {}).setdefault(action, [])
        returns.append(_return)
        self.returns_dict[state][action] = returns
        if self.verbose:
            print(self.returns_dict)



