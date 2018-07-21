from src.agent.agent_base import GreedyAgentBase
from statistics import mean
from src.utils.misc import Transition


class MonteAgent(GreedyAgentBase):
    """
    Famous Monte Carlo agent.
    """
    def __init__(self, env, nb_episodes=1, epsilon=0.1, verbose=False):
        GreedyAgentBase.__init__(self, env, nb_episodes, epsilon)
        self.verbose = verbose
        self.reset()

    def reset(self):
        super(MonteAgent, self).reset()
        self.transitions = []
        self.returns_dict = {}

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



