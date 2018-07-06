from src.utils import EGreedyPolicy
import numpy as np
import pytest


def test_egreedy_policy():
    action_values = {0: 44, 1: 42, 3: 100}
    nb_actions = 3
    epsilon = 0.1
    nb_iter = 10000
    policy = EGreedyPolicy(epsilon=epsilon)
    chosen_action = []
    for _ in range(nb_iter):
        action = policy(action_values)
        chosen_action.append(action)

    unique, counts = np.unique(chosen_action, return_counts=True)
    unique_count_dict = dict(zip(unique, counts))

    assert pytest.approx(1-epsilon + epsilon/nb_actions, 0.2) == unique_count_dict[3]/float(nb_iter)
    assert pytest.approx(epsilon/nb_actions, 0.2) == unique_count_dict[0]/float(nb_iter)
    assert pytest.approx(epsilon/nb_actions, 0.2) == unique_count_dict[1]/float(nb_iter)






