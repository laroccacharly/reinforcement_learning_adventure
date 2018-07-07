from src.utils import HyperFitter
from src.utils import AgentHelper


agent = AgentHelper()

space = {
    'bar': [0, 1],
    'foo': [0, 1]
}


def test_random_search():
    nb_points = 500
    hf = HyperFitter(agent=agent, space=space, nb_points=nb_points, random=True)
    hf.fit()
    assert len(hf.results) == nb_points
    assert hf.best_score == 2
    assert hf.best_params == {'bar': 1, 'foo': 1}


def test_grid_search():
    hf = HyperFitter(agent=agent, space=space, random=False)
    hf.fit()
    assert len(hf.results) == 4
    assert hf.best_score == 2
    assert hf.best_params == {'bar': 1, 'foo': 1}


def test_with_noise():
    hf = HyperFitter(agent=AgentHelper(with_noise=True), space=space, nb_iter=20)
    hf.fit()
    assert hf.best_params == {'bar': 1, 'foo': 1}


def test_plot():
    space = {
        'with_noise': [True, False],
        'noise_std': [0.1, 0.5, 0.8, 1],
        'bar': [1],
        'foo': [1]
    }
    hf = HyperFitter(agent=agent, space=space, nb_iter=20)
    hf.fit()
    assert hf.plot(x_axis_name='noise_std', compare_on='with_noise')

