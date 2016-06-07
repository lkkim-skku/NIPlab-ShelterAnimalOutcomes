from ._controller_ import *
from ._model_ import *
import kaggleio


class PbClassKernel:
    def __init__(self, target, idx):
        self._target = target

    @property
    def target(self):
        return self._target
    pass


class PbClass:
    def __init__(self, target):
        self._target = target
        pass

    @property
    def target(self):
        return self._target

    def fit(self, data, header):
        for key in zip(*data):
            yield
        pass

    def predict(self, data):
        data_pe = self.__pe__(data)
        # probability =
        return probability
    pass


class BayesianNeuralNetwork(ProbabilityEstimationNeuralNetwork):
    def __init__(self, name):
        super().__init__()
        self._name = name
        self._pbclasses = list()
        self._targets = set()
        pass

    @property
    def name(self):
        return self._name

    def __suite__(self, dataset):
        dataset_pe = probability_estimate(dataset)
        # makes The feature 'week' the average of continuous 5 items
        cls_week_ave = average({dataset_pe[x]['week'] for x in self._targets}, average=5)
        for cls in self._targets:
            dataset_pe[cls]['week'] = cls_week_ave[cls]
        return dataset_pe

    def fit(self, data, target):
        self._targets = set(target)
        dataset = {x: [] for x in self._targets}
        for d, t in zip(data, target):
            dataset[t].append(d)

        dataset = self.__suite__(dataset)
        for cls in self._targets:
            pbc = PbClass(cls)
            pbc.fit(dataset[cls])
            self._pbclasses.append(pbc)

    @property
    def classset(self):
        return self._targets if len(self._targets) > 0 else KeyError

    def pbclass(self, target):
        for pbclass in self.pbclasses:
            if target == pbclass.targetfeature:
                return pbclass
            else:
                return KeyError

    def predict(self, data):
        _pred = []
        for sample in data:
            pvalues = {pbc.targetfeature: pbc.predict(sample) for pbc in self._pbclasses}
            _pred.append(max(pvalues, key=lambda x: pvalues.get(x)))  # TODO 이거 맞는지 확인할 것
        return _pred

    def predict_table(self, data):
        """
        predict의 결과를 table of class probability로 출력합니다.
        :param data:
        :return:
        """
        result = []
        for sample in data:
            pvalues = {pbc.targetfeature: pbc.predict(sample) for pbc in self._pbclasses}
            result.append(pvalues)

        return result
    pass
