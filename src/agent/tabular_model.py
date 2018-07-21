class TabularModel(object):
    """
    Basic model for discrete state and action space.
    Implements the update rule.
    """
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate
        self.reset()

    def __call__(self, state, action):
        return self.values.setdefault(state, {}).setdefault(action, 0)

    def update(self, state, action, target):
        current_value = self.__call__(state, action)
        new_value = current_value + self.learning_rate * (target - current_value)
        self.values[state][action] = new_value
        return abs(current_value - new_value)

    def reset(self):
        self.values = {}






