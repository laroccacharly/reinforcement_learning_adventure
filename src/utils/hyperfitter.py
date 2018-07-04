import numpy as np
from sklearn.model_selection import ParameterGrid

class HyperFitter(object):
    """ Finds the best hyperparameters for an agent

    Attributes:
        results : A list with the results
    """
    def __init__(self, space, agent, nb_iter=None, random=False):
        """
        :param space: A dict with parameters to explore. Ex: {a: [1,2], b: [2,3]}
        :param agent: the object to find the best parameters for. The agent must implement two methods : score and set_params
        :param random: Use random search instead of grid search
        :param nb_iter: number of parameters to check the score on. Useful only when random is True.
        """
        self.space = space
        self.agent = agent
        self.nb_iter = nb_iter
        self.best_params = None
        self.best_score = None
        self.results = []
        self.random = random
        self.scores = []
        if random:
            assert isinstance(nb_iter, int)

    def fit(self):
        """
        Finds the score for different parameters and stores the results
        :return: None
        """
        if self.random:
            for _ in range(self.nb_iter):
                params = self.__sample_params()
                self.__process_params(params)
        else:
            for params in list(ParameterGrid(self.space)):
                self.__process_params(params)

        best_result = self.results[np.argmax(self.scores)]
        self.best_score = best_result['score']
        self.best_params = best_result['params']

    def __score(self, params):
        self.agent.set_params(params)
        return self.agent.score()

    def __sample_params(self):
        sample_param = {}
        for key, value in self.space.items():
            sample_param[key] = np.random.choice(value)

        return sample_param

    def __process_params(self, params):
        score = self.__score(params)
        self.scores.append(score)
        self.results.append({'score': score, 'params': params})






