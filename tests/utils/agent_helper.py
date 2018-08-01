import numpy as np


class AgentHelper(object):
    """
        This is just a small class to help testing
    """
    def __init__(self, with_noise=False, noise_std=1):
        self.foo = 1
        self.bar = 1
        self.with_noise = with_noise
        self.noise_std = noise_std

    def set_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def score(self):
        summation = self.bar + self.foo
        if self.with_noise:
            noise = np.random.normal(1, self.noise_std)
            return summation + noise
        else:
            return summation

