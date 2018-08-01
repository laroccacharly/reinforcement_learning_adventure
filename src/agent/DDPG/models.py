import torch
import torch.nn as nn
from math import sqrt
INIT_PARAM_MIN = -3e-3
INIT_PARAM_MAX = 3e-3

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(INIT_PARAM_MIN, INIT_PARAM_MAX)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        if torch.cuda.is_available():
            print('Gpu active')
            self.cuda()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_to_numpy(self, *args):
        return self.forward(*args).data.cpu().numpy()

    def init_params_from_model(self, model):
        self.load_state_dict(model.state_dict())

    def update_params_from_model(self, model, tau):
        for param1, param2 in zip(self.parameters(), model.parameters()):
            param1.data = tau * param2.data + (1 - tau) * param1.data

class CriticNet(Net):
    def __init__(self, nb_features, nb_actions, nb_hidden_1, nb_hidden_2, learning_rate, weight_decay=1e-2, batch_norm=False):
        super(CriticNet, self).__init__()

        self.relu = nn.ReLU()
        self.soft = nn.Softplus()
        self.tanh = nn.Tanh()

        # self.batch_norm_1 = nn.BatchNorm1d(nb_features)
        self.linear_1 = nn.Linear(nb_features, nb_hidden_1)
        self.linear_1.weight.data.uniform_(-1/sqrt(nb_features), 1/sqrt(nb_features))
        self.linear_1.bias.data.uniform_(-1/sqrt(nb_features), 1/sqrt(nb_features))

        self.batch_norm_2 = nn.BatchNorm1d(nb_hidden_1)
        self.linear_2 = nn.Linear(nb_hidden_1, nb_hidden_2)
        self.linear_2.weight.data.uniform_(-1/sqrt(nb_hidden_1), 1/sqrt(nb_hidden_1))
        self.linear_2.bias.data.uniform_(-1/sqrt(nb_hidden_1), 1/sqrt(nb_hidden_1))

        self.batch_norm_3 = nn.BatchNorm1d(nb_hidden_2)
        self.linear_3 = nn.Linear(nb_hidden_2, 1)
        self.linear_3.weight.data.uniform_(INIT_PARAM_MIN, INIT_PARAM_MAX)
        self.linear_3.bias.data.uniform_(INIT_PARAM_MIN, INIT_PARAM_MAX)

        self.linear_action = nn.Linear(nb_actions, nb_hidden_2, bias=False)


        if batch_norm:
            self.layers_1= [
                self.linear_1,
                self.relu,
                self.batch_norm_2,
                self.linear_2,
            ]
            self.layers_2 = [
                self.relu,
                self.batch_norm_3,
                self.linear_3
            ]
        else:
            self.layers_1 = [
                self.linear_1,
                self.soft,
                self.linear_2
            ]
            self.layers_2 = [
                self.tanh,
                self.linear_3
            ]

        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def forward(self, states, actions):
        x = states
        for layer in self.layers_1:
            x = layer(x)

        x = x + self.linear_action(actions)
        for layer in self.layers_2:
            x = layer(x)

        return x



class ActorNet(Net):
    def __init__(self, nb_features, nb_actions, nb_hidden_1, nb_hidden_2, learning_rate, batch_norm=False):
        super(ActorNet, self).__init__()

        self.relu = nn.ReLU()
        #self.batch_norm_1 = nn.BatchNorm1d(nb_features)
        self.linear_1 = nn.Linear(nb_features, nb_hidden_1)
        self.linear_1.weight.data.uniform_(-1/sqrt(nb_features), 1/sqrt(nb_features))
        self.linear_1.bias.data.uniform_(-1/sqrt(nb_features), 1/sqrt(nb_features))

        self.batch_norm_2 = nn.BatchNorm1d(nb_hidden_1)
        self.linear_2 = nn.Linear(nb_hidden_1, nb_hidden_2)
        self.linear_2.weight.data.uniform_(-1/sqrt(nb_hidden_1), 1/sqrt(nb_hidden_1))
        self.linear_2.bias.data.uniform_(-1/sqrt(nb_hidden_1), 1/sqrt(nb_hidden_1))

        self.batch_norm_3 = nn.BatchNorm1d(nb_hidden_2)
        self.linear_3 = nn.Linear(nb_hidden_2, nb_actions)
        self.linear_3.weight.data.uniform_(INIT_PARAM_MIN, INIT_PARAM_MAX)
        self.linear_3.bias.data.uniform_(INIT_PARAM_MIN, INIT_PARAM_MAX)

        self.soft = nn.Softplus()
        self.tanh = nn.Tanh()

        if batch_norm:
            self.layers = [
                self.linear_1,
                self.relu,
                self.batch_norm_2,
                self.linear_2,
                self.relu,
                self.batch_norm_3,
                self.linear_3,
            ]
        else:
            self.layers = [
                self.linear_1,
                self.soft,
                self.linear_2,
                self.tanh,
                self.linear_3,
            ]

        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        #self.apply(weight_init)


