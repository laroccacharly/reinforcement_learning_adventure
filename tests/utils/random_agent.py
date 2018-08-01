from src.agent.agent_base import AgentBase


class RandomAgent(AgentBase):
    """
    Simple agent that acts randomly
    """
    def __init__(self, env, nb_episodes):
        AgentBase.__init__(self, env, nb_episodes)

    def choose_action(self, state):
        return self.env.action_space.sample()

    def learn(self,  state, action, reward, next_state, next_action, done):
        return 0
