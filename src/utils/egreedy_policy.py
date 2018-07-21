import numpy as np


class EGreedyPolicy(object):
    """
        Epsilon greedy policy.
        You call the object with a dict of action values ex : {action_1: value_1, action_2: value_2, action_3: value_3}
        It returns the chosen action.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, action_values):
        actions = []
        values = []
        indexes = []
        for index, (action, value) in enumerate(action_values.items()):
            actions.append(action)
            values.append(value)
            indexes.append(index)

        nb_actions = len(actions)
        values = np.array(values)
        best_action_index = np.random.choice(np.flatnonzero(values == max(values)))
        action_probs = np.ones(nb_actions, dtype=float) * self.epsilon / float(nb_actions)
        action_probs[best_action_index] += (1.0 - self.epsilon)
        chosen_action_index = np.random.choice(indexes, p=action_probs)
        chosen_action = actions[chosen_action_index]
        return chosen_action
