from ._controller_ import *
from ._model_ import *
from sklearn.cluster import KMeans
import numpy as np
import math
from simulation import statistics


class PbKernel:
    def __init__(self, target, idx):
        self._target = target

    def fit(self, data):
        zipdata = list(zip(*data))
        self.mean = [np.mean(x) for x in zipdata]
        self.var = [np.var(x, ddof=1) for x in zipdata]
        kdata = [math.exp(-.5 * sum(1 if v == 0 else ((s - m) ** 2) / v for s, m, v in zip(sample, self.mean, self.var))) for sample in data]
        self.kmin, self.kmax = min(kdata), max(kdata)
        if self.kmin == self.kmax:
            fdata = [.99 for _ in kdata]
        else:
            fdata = [.99 * (k - self.kmin) / (self.kmax - self.kmin) if self.kmin < k < self.kmax else 0 if k <= self.kmin else .99 for k in kdata]
        fdata.sort()
        cdf = statistics.histo_cudif(fdata, bins=100)
        self.cdf = cdf

    def predict(self, sample):
        ksample = math.exp(-.5 * sum(1 if v == 0 else ((s - m) ** 2) / v for s, m, v in zip(sample, self.mean, self.var)))
        if self.kmin == self.kmax:
            fsample = .99 if ksample == self.kmin else 0
        else:
            fsample = .99 * (ksample - self.kmin) / (self.kmax - self.kmin) if self.kmin < ksample < self.kmax else 0 if ksample <= self.kmin else .99
        p = self.cdf[int(fsample * 100)]

        return p


class PbClass(ProbEstimator):
    def __init__(self, target):
        super().__init__(target)
        self._kernel = []
        self._clustersize = 1

    def fit(self, data):
        # clustering
        kms = KMeans(n_clusters=self._clustersize)
        cluster = {x: [] for x in range(self._clustersize)}
        for c, s in zip(kms.fit_predict(data, self._target), data):
            cluster[c].append(s)
        for index in range(self._clustersize):
            pbk = PbKernel(self._target, index)
            pbk.fit(cluster[index])
            self._kernel.append(pbk)
        # feature scaling
        # beta model estimation
        pass

    def predict(self, data):
        superresult = super().predict(data)
        p = 0
        for kernel in self._kernel:
            p += kernel.predict(superresult)
        return p

    @staticmethod
    def factory(probestimator: ProbEstimator):
        this = PbClass(probestimator.target)
        this.header = probestimator.header
        return this
    pass


class BayesianNeuralNetwork(ProbabilityEstimationNeuralNetwork):
    def __init__(self, name):
        super().__init__(name)
        self._pbclass = list()
        pass

    def fit(self, data, target):
        super().fit(data, target)

        datadict = {x: [] for x in set(target)}
        for d, t in zip(data, target):
            datadict[t].append(d)

        for node in self._node.values():
            pbc = PbClass(node.target)
            pbc.header = node.header
            pbc(node())
            pbc.fit([node.predict(sample) for sample in datadict[node.target]])
            # pbc.fit2([node.predict(sample) for sample in data])
            self._pbclass.append(pbc)

    @property
    def classset(self):
        return self._targets if len(self._targets) > 0 else KeyError

    def pbclass(self, target):
        for pbclass in self.pbclasses:
            if target == pbclass.target:
                return pbclass
            else:
                return KeyError

    def predict(self, data):
        _pred = []
        for sample in data:
            pvalues = {pbc.target: pbc.predict(sample) for pbc in self._pbclass}
            _pred.append(max(pvalues, key=lambda x: pvalues.get(x)))  # TODO 이거 맞는지 확인할 것..나중에.
        return _pred

    def predict_table(self, data):
        """
        predict의 결과를 table of class probability로 출력합니다.
        :param data:
        :return:
        """
        result = []
        for sample in data:
            pvalues = {pbc.target: pbc.predict(sample) for pbc in self._pbclass}
            sumpvalue = sum(pvalues.values())
            if sumpvalue != 0:
                for key in pvalues:
                    pvalues[key] /= sumpvalue
            result.append(pvalues)

        return result
