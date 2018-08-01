import gym

from src.agent import MonteTabularAgent, QlearnTabularAgent, QlearnLinearAgent, QlearnDeepAgent, ActorCriticAgent, \
    DDPGAgent
from src.agent.dql_agent import DQLAgent
from tests.utils import RandomAgent

blackjack_env = gym.make('Blackjack-v0')
frozen_env = gym.make('FrozenLake-v0')
mountain_car_env = gym.make("MountainCar-v0")
mountain_car_continuous_env = gym.make('MountainCarContinuous-v0')


def make_agents():
    agents = [MonteTabularAgent(blackjack_env, nb_episodes=1000),
              QlearnTabularAgent(frozen_env, nb_episodes=1000),
              QlearnLinearAgent(mountain_car_env, nb_episodes=10, learning_rate=0.1),
              QlearnDeepAgent(mountain_car_env, nb_episodes=10, learning_rate=0.001, nb_hidden_1=50, nb_hidden_2=10,
                              activation='relu', optim='Adam'),
              ActorCriticAgent(mountain_car_env, nb_episodes=10, lamda=0.6, discount=1, nb_features=400,
                               policy_learning_rate=0.01, model_learning_rate=0.1),
              DDPGAgent(mountain_car_continuous_env, nb_episodes=2)]
    return agents


def test_agents():
    agents = make_agents()

    for agent in agents:
        random_agent = RandomAgent(env=agent.env, nb_episodes=agent.nb_episodes)
        score = agent.score()
        baseline = random_agent.score()
        assert score > baseline
