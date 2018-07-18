from src.agent.agent_base import GreedyAgentBase
from src.utils.memory import ReplayMemory
import numpy as np

class QlearnAgent(GreedyAgentBase):
    """
       Q-Learning.
    """
    def __init__(self, env, nb_episodes=1, epsilon=0.1, learning_rate=0.1, discount=1,  verbose=False):
        GreedyAgentBase.__init__(self, env, nb_episodes, epsilon)
        self.learning_rate = learning_rate
        self.discount = discount
        self.verbose = verbose
        self.memory = ReplayMemory(capacity=1e6)
        self.reset()

    def reset(self):
        self.model.learning_rate = self.learning_rate # Send the learning rate to the model.
        super(QlearnAgent, self).reset()

    def stats(self):
        """
            A method to help debug. It gets the logs from the training session in memory.
        """
        batch = self.memory.sample_zipped(len(self.memory))
        print(np.bincount(np.array(batch.action)))
        return batch

    def learn(self, state, action, reward, next_state, next_action, done):
        # We save the transition in memory (database)
        self.memory.push(state, action, reward, next_state, next_action, done)
        # Q-Learning algo down here. Kinda simple isn't it?
        target = reward + self.discount * max(self.action_values(next_state).values())
        # The update returns the loss. By how much we updated the function, that should converge to 0.
        loss = self.model.update(state, action, target)
        if self.verbose:
            print('action values : ', self.action_values(state))
            print('loss : ', loss)





