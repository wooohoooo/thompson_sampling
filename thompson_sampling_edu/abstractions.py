# AUTOGENERATED! DO NOT EDIT! File to edit: 00_abstractions.ipynb (unless otherwise specified).

__all__ = ['AbstractSolver', 'AbstractContextualSolver']

# Cell
class AbstractSolver(object):
    def choose_arm(self):
        """choose an arm to play according to internal policy"""
        raise NotImplementedError

    def update(self, arm, reward):
        """ update internal policy to reflect changed knowledge"""
        raise NotImplementedError


# Cell
class AbstractContextualSolver(object):
    def __init__(self, model, num_arms):
        self.model = model
        self.num_arms = num_arms


    def choose_arm(self,context):
        """choose an arm to play according to internal policy"""
        raise NotImplementedError

    def update(self, arm, context, reward):
        """ update internal policy to reflect changed knowledge"""
        raise NotImplementedError