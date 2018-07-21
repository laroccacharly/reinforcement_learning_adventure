import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
import numpy as np

class Featurizer(object):
    """
        Wrapper around sklearn scaler and RBFSamplers.
        It samples the observation space and action space of the env.
        After, it scales the data by removing the mean and scaling to unit variance.
        Finally, a feature map is made using RBFSampler.
    """
    def __init__(self, nb_features, env, with_action=False):
        self.with_action = with_action
        self.env = env
        self.nb_samples = 10000
        samples = self.make_samples()
        remaining = nb_features % 4
        n_components = int(nb_features / 4)
        featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components)),
            ("rbf5", RBFSampler(gamma=0.3, n_components=remaining))
        ])
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(samples)
        featurizer.fit(scaler.transform(samples))

        self.nb_features = nb_features
        self.scaler = scaler
        self.featurizer = featurizer

    def make_samples(self):
        return np.array([self.make_sample() for _ in range(self.nb_samples)])

    def make_sample(self):
        state = self.env.observation_space.sample().tolist()
        action = self.env.action_space.sample()
        if self.with_action:
            return state + [action]
        else:
            return state

    def transform(self, observation, action=None):
        if action is not None:
            observation = observation.tolist() + [action]
        scaled = self.scaler.transform([observation])
        return self.featurizer.transform(scaled)[0]