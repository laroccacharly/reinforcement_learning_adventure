
from src.agent.agent_base import GreedyAgentBase

class QlearnAgent(GreedyAgentBase):
    """
       Q-Learning. Discrete state and action space.
    """
    def __init__(self, env, nb_episodes=1, epsilon=0.1, step_size=0.1, discount=1,  verbose=False):
        GreedyAgentBase.__init__(self, env, nb_episodes, epsilon)
        self.step_size = step_size
        self.discount = discount
        self.verbose = verbose
        self.reset()

    def learn(self, state, action, reward, next_state, next_action, done):
        if self.verbose:
            print(state, action, reward, next_state, next_action, done)
        target = reward + self.discount * max(self.action_values(next_state).values())
        current_value = self.action_values(state, action)
        new_value = current_value + self.step_size * (target - current_value)
        self.model.update(state, action, new_value)





