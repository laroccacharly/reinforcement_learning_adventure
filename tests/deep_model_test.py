from src.agent.qlearning import DeepModel
import pytest
import gym

env = gym.make("MountainCar-v0")
learning_rate = 0.01
model = DeepModel(env=env, learning_rate=learning_rate)
state = env.observation_space.sample()
action = env.action_space.sample()


def test_forward():
    value = model(state, action)
    assert pytest.approx(0, abs=0.1) == value


def test_update():
    target = 2
    prev_value = model(state, action)
    model.update(state, action, target=target)
    value = model(state, action)
    assert prev_value < value
    assert value < target
