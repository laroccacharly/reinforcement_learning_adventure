import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from src.agent.model_base import ModelBase
from src.utils import Featurizer


def to_var(x):
    """
        Because PyTorch works with Variables
    """
    x = np.array(x)
    x = torch.Tensor(x)
    x = Variable(x)
    return x


class DeepModel(ModelBase):
    """
    A layer on top of the neural networks. A NN for each action.
    """
    def __init__(self, env, learning_rate):
        self.env = env
        self.learning_rate = learning_rate
        self.featurizer = Featurizer(env=env, nb_features=400)
        self.nb_hidden_1 = 100
        self.nb_hidden_2 = 100
        self.activation = 'relu'
        self.optim = 'SGD'
        self.reset()

    def reset(self):
        self.models = [self.make_model() for _ in range(self.env.action_space.n)]

    def make_model(self):
        model = Net(self.featurizer, learning_rate=self.learning_rate,
                    nb_hidden_1=self.nb_hidden_1, nb_hidden_2=self.nb_hidden_2,
                    activation=self.activation, optim=self.optim)

        return model


class Net(nn.Module):
    """
        A deep neural network using pytorch.
        2 linear layers, configurable activation layers and optim.
    """
    def __init__(self, featurizer, nb_hidden_1, nb_hidden_2, learning_rate, activation='relu', optim='SGD'):
        super(Net, self).__init__()

        self.featurizer = featurizer
        self.nb_features = int(self.featurizer.nb_features)
        self.nb_hidden_1 = int(nb_hidden_1)
        self.nb_hidden_2 = int(nb_hidden_2)
        self.learning_rate = learning_rate

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.loss = nn.MSELoss()

        self.linear_1 = nn.Linear(self.nb_features, self.nb_hidden_1)
        self.linear_2 = nn.Linear(self.nb_hidden_1, self.nb_hidden_2)
        self.linear_3 = nn.Linear(self.nb_hidden_2, 1)
        self.epsilon = 3e-5
        # Init the last layer to get an output close to zero.
        self.linear_3.weight.data.uniform_(-self.epsilon, self.epsilon)
        self.linear_3.bias.data.uniform_(-self.epsilon, self.epsilon)

        if activation == 'relu':
            self.act = self.relu
        else:
            self.act = self.sig

        self.layers = [
            self.linear_1,
            self.act,
            self.linear_2,
            self.act,
            self.linear_3,
        ]

        if optim == 'SGD':
            self.opt = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            self.opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward_to_var(self, state):
        x = self.featurizer.transform(state)
        x = to_var(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, state):
        """
            This returns a float instead of a Variable (the default output of pytorch).
        """
        x = self.forward_to_var(state)
        return x.data[0]

    def update(self, state, target):
        prediction = self.forward_to_var(state)
        target = to_var([target])
        loss = self.loss(prediction, target)
        self.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.data[0]


