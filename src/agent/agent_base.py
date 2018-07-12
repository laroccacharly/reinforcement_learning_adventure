from abc import ABCMeta, abstractmethod


class AgentBase:
    __metaclass__ = ABCMeta

    def __init__(self, env, nb_episodes=1):
        self.env = env
        self.nb_episodes = nb_episodes

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def learn(self,  state, action, reward, next_state, next_action, done):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def set_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)
        self.reset()

    def play_one_episode(self, render=False):
        state = self.env.reset()
        action = self.choose_action(state)

        total_reward = 0
        done = False
        while not done:
            if render:
                self.env.render()
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            if done:
                self.learn(state, action, reward, next_state, None, done)
                break

            next_action = self.choose_action(next_state)

            self.learn(state, action, reward, next_state, next_action, done)

            action = next_action
            state = next_state

        return total_reward

    def score(self):
        total_reward = 0
        for _ in range(self.nb_episodes):
            total_reward += self.play_one_episode()

        return total_reward
