import numpy as np

class Bandit(object):
    def __init__(self, average=0, std=1):
        self.average = average
        self.std = std

    def play(self):
        return np.random.normal(self.average, self.std)


class BanditEnv(object):
    """Implementation of the multi armed bandit env
    See : https://en.wikipedia.org/wiki/Multi-armed_bandit
    """
    def __init__(self, averages):
        self.bandits = [Bandit(average=average) for average in averages]

    def step(self, action):
        bandit = self.bandits[action]
        reward = bandit.play()

        return reward

