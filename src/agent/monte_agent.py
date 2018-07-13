from src.utils import EGreedyPolicy
from .agent_base import AgentBase
from .model import Model
from collections import namedtuple
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
        self.action_space = [i for i in range(env.action_space.n)]
        self.reset()

    def reset(self):
        self.transitions = []
        self.returns_dict = {}
        self.model = Model()
        self.policy = EGreedyPolicy(epsilon=self.epsilon)

    def action_values(self, state, action=None):
        if action is not None:
            return self.model(state, action)
        else:
            values = [self.action_values(state, a) for a in self.action_space]
            return dict(zip(self.action_space, values))

    def choose_action(self, state):
        action_values = self.action_values(state)
        action = self.policy(action_values)
        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        transition_present = False
        for t in self.transitions:
            if t.state == state and t.action == action:
                transition_present = True

        if not transition_present:
            self.transitions.append(Transition(state, action, reward, next_state, next_action, done))

        if done:
            _return = 0
            for t in reversed(self.transitions):
                _return = _return + t.reward
                self.append_return(t.state, t.action, _return)
            self.transitions = []


    def append_return(self, state, action, _return):
        returns = self.returns_dict.setdefault(state, {}).setdefault(action, [])
        returns.append(_return)
        self.returns_dict[state][action] = returns
        self.update_action_value(state, action, mean(returns))
        if self.verbose:
            print(self.returns_dict)

    def update_action_value(self, state, action, value):
        return self.model.update(state, action, value)



