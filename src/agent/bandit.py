import numpy as np


class BanditAgent:
    def __init__(self, env, nb_iter, epsilon=0.1, verbose=False):
        self.env = env
        self.action_space = env.action_space
        self.nb_bandit = len(self.action_space)
        self.nb_iter = nb_iter
        self.epsilon = epsilon
        self.action_value_initial = 0
        self.verbose = verbose
        self.__reset()

    def score(self):
        total_reward = 0
        for _ in range(self.nb_iter):
            total_reward += self.__play_one()
        return total_reward

    def __play_one(self):
        action = self.__choose_action()
        reward = self.env.step(action)
        self.__learn(action, reward)
        return reward

    def __choose_action(self):
        if self.__is_exploring():
            return self.__random_action()
        else:
            return self.__best_action()

    def __is_exploring(self):
        return np.random.uniform(0, 1) < self.epsilon

    def __random_action(self):
        return np.random.choice(self.action_space)

    def __best_action(self):
        return np.argmax(self.action_values)

    def __learn(self, action, reward):
        self.action_selection_count[action] += 1
        count = self.action_selection_count[action]
        self.action_values[action] = (self.action_values[action] * (count - 1) + reward) / count

        if self.verbose:
            print(self.action_selection_count)
            print(self.action_values)

    def __reset(self):
        self.action_values = [self.action_value_initial for _ in range(self.nb_bandit)]
        self.action_selection_count = [0 for _ in range(self.nb_bandit)]

    def set_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)
        self.__reset()


