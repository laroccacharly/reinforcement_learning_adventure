from sklearn.model_selection import ParameterGrid
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
# This functions finds the mean and the confidence interval of a set of data.
def mean_confidence_interval(data, confidence):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


class HyperFitter(object):
    """
    Finds the best hyperparameters for an agent

    """
    def __init__(self, space, agent, nb_iter=1, confidence=0.9, nb_points=None, random=False, verbose=False):
        """
        :param space: A dict with parameters to explore. Ex: {a: [1,2], b: [2,3]}
        :param agent: the object to find the best parameters for. The agent must implement two methods : score and set_params
        :param random: Use random search instead of grid search
        :param nb_points: number of parameters to check the score on. Useful only when random is True.
               By default, all possible combination of parameters are tested.
        :param nb_iter: number of times the score is computed for a specific point. Useful if there is variability on the score.
        """
        self.space = space
        self.agent = agent
        self.nb_iter = nb_iter
        self.nb_points = nb_points
        self.confidence = confidence
        self.best_params = None
        self.best_score = None
        self.results = []
        self.random = random
        self.scores = []
        self.errors = []
        self.verbose = verbose
        if random:
            assert isinstance(nb_points, int)

    def fit(self):
        """
        Finds the score for different parameters and stores the results
        """
        if self.random:
            for _ in range(self.nb_points):
                params = self.sample_params()
                self.__process_params(params)
        else:
            for params in list(ParameterGrid(self.space)):
                self.__process_params(params)

        best_result = self.results[np.argmax(self.scores)]
        self.best_score = best_result['score']
        self.best_params = best_result['params']

    def plot(self, x_axis_name, compare_on):
        """
        Plots the scores for specified parameters.

        :param x_axis_name: the name of the parameter you want to plot the score for. Usually it is epsilon or the learning rate
        :param compare_on: the name of the parameter to compare the score on.
        """

        compare_on_values = self.space[compare_on]
        x_values = self.space[x_axis_name]

        for value in compare_on_values:
            scores = []
            errors = []
            for result in self.results:
                if result['params'][compare_on] == value:
                    scores.append(result['score'])
                    errors.append(result['error'])
            plt.errorbar(x_values, scores, errors, label=compare_on + ' is ' + str(value))
        plt.legend()
        plt.xlabel(x_axis_name)
        plt.ylabel('score')
        return plt

    def score(self, params):
        self.agent.set_params(params)
        if self.nb_iter == 1:
            return self.agent.score(), 0
        else:
            scores = [self.agent.score() for _ in range(self.nb_iter)]
            return mean_confidence_interval(scores, confidence=self.confidence)

    def sample_params(self):
        sample_param = {}
        for key, value in self.space.items():
            sample_param[key] = np.random.choice(value)

        return sample_param

    def __process_params(self, params):
        score, error = self.score(params)
        self.scores.append(score)
        self.errors.append(error)
        self.results.append({'score': score, 'params': params, 'error': error})
        if self.verbose:
            print({'score': score, 'params': params, 'error': error})


