from src.agent.agent_base import GreedyAgentBase
from .linear_model import LinearModel
from .deep_model import DeepModel

from src.agent.tabular_model import TabularModel
from src.utils.memory import ReplayMemory

class QlearnTabularAgent(GreedyAgentBase):
    """
        Q-Learning.
        observation space : discrete
        action space : discrete
    """
    def __init__(self, env, nb_episodes=1, epsilon=0.1, learning_rate=0.1, discount=1,  verbose=False):
        self.learning_rate = learning_rate
        self.discount = discount
        self.verbose = verbose
        GreedyAgentBase.__init__(self, env, nb_episodes, epsilon)
        self.reset()

    def reset(self):
        super(QlearnTabularAgent, self).reset()
        self.memory = ReplayMemory(capacity=1e6)
        self.model = TabularModel(learning_rate=self.learning_rate)

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


class QlearnLinearAgent(QlearnTabularAgent):
    """
        Q-Learning with a linear model
        observation space : continuous
        action space : discrete
    """
    def __init__(self, env, nb_episodes=1, epsilon=0.1, learning_rate=0.1, discount=1,  verbose=False):
        QlearnTabularAgent.__init__(self, env, nb_episodes, epsilon, learning_rate, discount, verbose)
        self.reset()

    def reset(self):
        super(QlearnLinearAgent, self).reset()
        self.model = LinearModel(self.env, self.learning_rate)


class QlearnDeepAgent(QlearnTabularAgent):
    """
        Q-Learning with a neural network model
        observation space : continuous
        action space : discrete
    """

    def __init__(self, env, nb_episodes=1, epsilon=0.1, learning_rate=0.1, discount=1, verbose=False,
                 nb_hidden_1=100, nb_hidden_2=100, activation='relu', optim='SGD'):
        self.nb_hidden_1 = nb_hidden_1
        self.nb_hidden_2 = nb_hidden_2
        self.activation = activation
        self.optim = optim
        QlearnTabularAgent.__init__(self, env, nb_episodes, epsilon, learning_rate, discount, verbose)
        self.reset()

    def reset(self):
        super(QlearnDeepAgent, self).reset()
        self.model = DeepModel(self.env,
                               self.learning_rate,
                               self.nb_hidden_1,
                               self.nb_hidden_2,
                               self.activation,
                               self.optim)


