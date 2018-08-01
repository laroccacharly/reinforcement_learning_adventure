class ModelBase:
    """
       I see this class like a mother. She takes care of SubModels.
       She calls them when we ask her to and she updates them to be better.
    """

    def __call__(self, state, action):
        model = self.models[action]
        return model(state)

    def update(self, state, action, target):
        model = self.models[action]
        return model.update(state, target)