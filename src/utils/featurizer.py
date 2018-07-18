import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
import numpy as np

class Featurizer(object):
    """
        Wrapper around sklearn scaler and RBFSamplers.
    """
    def __init__(self, nb_features, env):
        observation_examples = np.array([env.observation_space.sample().tolist() for _ in range(10000)])
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
        scaler.fit(observation_examples)
        featurizer.fit(scaler.transform(observation_examples))

        self.nb_features = nb_features
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observation):
        scaled = self.scaler.transform([observation])
        return self.featurizer.transform(scaled)[0]