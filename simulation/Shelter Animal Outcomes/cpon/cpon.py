from ._controller_ import *


class PbClassKernel:
    pass


class PbClass:
    def __init__(self, target):
        self._target = target
        pass

    def target(self):
        return self._name
    pass


class CPON:
    def __init__(self, name):
        self._name = name
        self._pbclasses = {}
        pass

    @property
    def name(self):
        return self._name

    def fit(self, data, target):
        pass

    def predict(self, data):
        result = []
        # for sample in data:
        #     pvalues = {k: k, v.predict(sample) for k, v in self._pbclasses.items()}
    pass