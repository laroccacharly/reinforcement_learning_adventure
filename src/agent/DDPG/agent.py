from src.utils.misc import to_var
from .models import CriticNet, ActorNet
from .tf_grad_processer import grad_inverter
from ..agent_base import AgentBase

from src.utils.memory import ReplayMemory
from src.utils.noise import OUNoise
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import copy


class DDPGAgent(AgentBase):
    def __init__(self, env, nb_episodes):
        AgentBase.__init__(self, env, nb_episodes)

        # Parameters
        self.batch_size = 64
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.critic_weight_decay = 1e-2
        self.discount = 0.99
        self.tau = 0.001
        self.hidden_size_1 = 400
        self.hidden_size_2 = 300
        self.buffer_size = 1e6
        self.monitoring = False
        self.batch_norm = False
        self.with_del_q = True

        # Env
        self.state_space_dim = env.observation_space.shape[0]
        self.action_space_dim = env.action_space.shape[0]

        # Setting up utilities
        self.set_utils()

        # Setting neural networks
        self.set_neural_networks()

        self.reset()

    def set_utils(self):
        self.grad_inverter = self.make_grad_inv()
        self.memory = ReplayMemory(self.buffer_size)
        self.noise = OUNoise(self.action_space_dim)
        if self.monitoring:
          self.writer = SummaryWriter()

    def make_grad_inv(self):
        action_max = np.array(self.env.action_space.high).tolist()
        action_min = np.array(self.env.action_space.low).tolist()
        action_bounds = [action_max, action_min]
        return grad_inverter(action_bounds)

    def set_neural_networks(self):
        self.critic = CriticNet(nb_features=self.state_space_dim, nb_actions=self.action_space_dim, nb_hidden_1=self.hidden_size_1,
                           nb_hidden_2=self.hidden_size_2,
                           learning_rate=self.critic_lr, weight_decay=self.critic_weight_decay, batch_norm=self.batch_norm)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor = ActorNet(nb_features=self.state_space_dim, nb_actions=self.action_space_dim, nb_hidden_1=self.hidden_size_1,
                         nb_hidden_2=self.hidden_size_2,
                         learning_rate=self.actor_lr, batch_norm=self.batch_norm)
        self.target_actor = copy.deepcopy(self.actor)

    def choose_action(self, state):
        state = to_var([state])
        self.actor.eval()
        action = self.actor.forward_to_numpy(state)
        action = action[0]
        noisy_action = action + self.noise.sample()
        if self.monitoring:
            self.writer.add_scalar('action/with_noise', noisy_action, self.global_step)
            self.writer.add_scalar('action/without_noise', action, self.global_step)
        return noisy_action

    def learn(self, state, action, reward, next_state, next_action, done):
        self.memory.push(state, action, reward, next_state, next_action, done)
        self.optimize()
        if done:
            self.noise.reset()

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return

        # Extract and process raw data.
        batch = self.memory.sample_zipped(self.batch_size)
        states, rewards, next_states, actions = to_var(batch.state), to_var(batch.reward), \
                                                to_var(batch.next_state), to_var(batch.action)
        dones = np.array(batch.done)
        rewards = rewards.view(self.batch_size, 1)

        # Training the critic
        self.critic.train()
        self.target_actor.eval()
        self.target_critic.eval()

        # Build target action values to train the critic
        target_action_values = Variable(torch.Tensor(self.batch_size, 1))
        for i in range(self.batch_size):
            if dones[i]:
                target_action_values[i] = rewards[i]
            else:
                ns = next_states[i].view(1, len(next_states[i]))
                target_action_values[i] = rewards[i] + self.discount * self.target_critic(ns, self.target_actor(ns))

        action_values = self.critic(states, actions)
        mse_loss_calculator = nn.MSELoss(size_average=False)  # Sum the loss vector instead of averaging.
        loss_critic = mse_loss_calculator(action_values,
                                          target_action_values.detach())  # Detach because no grad on target
        loss_critic.backward()
        self.critic.opt.step()
        self.critic.zero_grad()

        # Training the actor
        self.actor.train()
        self.critic.eval()

        if self.with_del_q:
            # Compute del_q_a
            actions_for_del_q_a = self.actor.forward_to_numpy(states)
            actions_for_del_q_a = to_var(actions_for_del_q_a)
            actions_for_del_q_a.requires_grad = True

            q_a = self.critic(states, actions_for_del_q_a)
            loss = q_a.sum()
            loss.backward()

            # Invert del_q_a. Make sure that we don't go beyond the action space.
            del_q_a = actions_for_del_q_a.grad
            del_q_a = actions_for_del_q_a.grad.data.cpu().numpy()
            del_q_a = self.grad_inverter.invert(del_q_a, actions_for_del_q_a.data.cpu().numpy())
            del_q_a = to_var(del_q_a)

            # Backprop
            actor_out = self.actor(states)
            actor_out.backward(-del_q_a)
            self.actor.opt.step()
            self.actor.opt.zero_grad()
        else:
            loss = -self.critic(states, self.actor(states))
            loss = loss.sum()
            loss.backward()
            self.actor.opt.step()
            self.actor.opt.zero_grad()

        # Update the target networks
        self.target_critic.update_params_from_model(self.critic, self.tau)
        self.target_actor.update_params_from_model(self.actor, self.tau)

        # Summary writing
        if self.monitoring:
            self.writer.add_scalar('loss/critic', loss_critic.data[0], self.global_step)
            t = self.memory.memory[-1]
            for index, s in enumerate(t.state):
                self.writer.add_scalar('state/' + str(index), t.state[index], self.global_step)

            self.writer.add_scalar('reward/current', t.reward, self.global_step)
            self.writer.add_scalar('reward/total', self.total_reward, self.global_step)
            performance = self.critic(to_var([t.state]), self.actor(to_var([t.state]))).data[0][0]
            self.writer.add_scalar('performance', performance, self.global_step)





