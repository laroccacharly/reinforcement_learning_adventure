from src.utils import HyperFitter


class Agent(object):
    """This is just a small class to help testing

    """
    def __init__(self):
        self.foo = 1
        self.bar = 1

    def set_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)

    def score(self):
        return self.bar + self.foo


agent = Agent()
space = {
    'bar': [0, 1],
    'foo': [0, 1]
}


def test_random():
    nb_iter = 500
    hf = HyperFitter(agent=agent, space=space, nb_iter=nb_iter, random=True)
    hf.fit()
    assert len(hf.results) == nb_iter
    assert hf.best_score == 2
    assert hf.best_params == {'bar': 1, 'foo': 1}


def test_grid_search():
    hf = HyperFitter(agent=agent, space=space, random=False)
    hf.fit()
    assert len(hf.results) == 4
    assert hf.best_score == 2
    assert hf.best_params == {'bar': 1, 'foo': 1}

