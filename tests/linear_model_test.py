from src.agent.qlearning import LinearModel
import gym

env = gym.make("MountainCar-v0")
learning_rate = 0.01
model = LinearModel(env=env, learning_rate=learning_rate)
state = env.observation_space.sample()
action = env.action_space.sample()


def test_forward():
    value = model(state, action)
    assert value == 0


def test_update():
    target = 1
    model.update(state, action, target=target)
    value = model(state, action)
    assert value > 0
    assert value < target

