
class TabularModel(object):
    def __init__(self):
        self.values = {}

    def __call__(self, state, action):
        return self.values.setdefault(state, {}).setdefault(action, 0)

    def update(self, state, action, value):
        prev_value = self.__call__(state, action)
        self.values[state][action] = value
        return abs(prev_value - value)

    def reset(self):
        self.values = {}






