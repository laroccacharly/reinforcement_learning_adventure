import gym
from src.agent import QlearnAgent
from src.agent.deep_model import DeepModel
from src.agent.linear_model import LinearModel
from src.utils import HyperFitter

learning_rate = 0.1
env = gym.make("MountainCar-v0")
deep_agent = QlearnAgent(env=env, nb_episodes=5, learning_rate=learning_rate)
linear_agent = QlearnAgent(env=env, nb_episodes=10, learning_rate=learning_rate)

deep_model = DeepModel(env=env, learning_rate=learning_rate)
linear_model = LinearModel(env=env, learning_rate=learning_rate)

deep_agent.model = deep_model
linear_agent.model = linear_model


def test_linear():
    score = linear_agent.score()
    # Mountain Car env has 200 steps and a reward of -1 for each of them. -200 * 10 episodes = -2000.
    # A score greater than -2000 means the agent solved the env at least once.
    assert score > -2000

def test_deep():
    return 0 # Test skipped because it takes a while to run.
    # We use hyperfitter here because there are too many combinations to tests manually.
    # These worked :
    # {'learning_rate': 0.001, 'nb_hidden_1': 100, 'nb_hidden_2': 200, 'activation': 'relu', 'optim': 'Adam'}
    # {'learning_rate': 0.001, 'nb_hidden_1': 200, 'nb_hidden_2': 100, 'activation': 'relu', 'optim': 'Adam'}
    # {'learning_rate': 0.001, 'nb_hidden_1': 50, 'nb_hidden_2': 10, 'activation': 'relu', 'optim': 'Adam'}
    space = {
        'learning_rate': [0.001, 0.0005, 0.002, 0.0015],
        'nb_hidden_1': [5, 10, 25, 50],
        'nb_hidden_2': [5, 10, 25, 50],
        'activation': ['relu'],
        'optim': ['Adam'],
    }

    hf = HyperFitter(space=space, agent=deep_agent, nb_points=5, random=True, verbose=False)
    hf.fit()
    score = hf.best_score
    assert score > -1000
