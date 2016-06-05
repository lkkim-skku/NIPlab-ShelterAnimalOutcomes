class FullSearchClustering:
    def __init__(self, header):
        pass

    def fit(self, data):
        pass

    def fit_predict(self, data):
        pass

    def predict(self, data):
        pass

    pass


class BaseDataset:
    def __init__(self, dataset: dict or Dataset):
        self._dataset = dataset

    @property
    def header(self):
        return [x for x in self._dataset.keys()]

    def fit(self, arr: list):
        header = arr.pop(0)

        self._dataset = {h: [] for h in header}

        for row in arr:
            for h, col in zip(header, row):
                self._dataset[h].append(col)

        return self

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._dataset[item]
        elif isinstance(item, int):
            return [self._dataset[x][item] for x in self.header]

    def __sizeof__(self):
        return len(self._dataset)

    def __call__(self):
        """
        array로 바꾸기
        :return:
        """
        return self._dataset


class ProbEstimator:
    """
    모든 class가 가지고 있는 하나의 feature에
    """
    def __init__(self, target):
        self._data = []
        self._target = target

    @property
    def target(self):
        return self._target

    def fit(self, data: list or tuple):
        """
        각 Random Variable의 갯수를 구해서 리턴합니다.
        :param data:
        :return:
        """
        self._ranvarset = list(set(data))
        self._ranvarset.sort()
        self._data = data
        self._estimate_rv = {rv: data.count(rv) for rv in self._ranvarset}
        self._min, self._max = min(self._ranvarset), max(self._ranvarset)
        return self._estimate_rv

    def fit_predict(self, criteria: list or tuple):
        estidata = KeyError
        if isinstance(criteria, list) or isinstance(criteria, tuple):
            for rv in self._estimate_rv:
                if rv not in criteria:
                    return estidata

            for rv in self._estimate_rv:
                self._estimate_rv[rv] /= criteria[rv]

            estidata = [self._estimate_rv[rv] for i, rv in enumerate(self._data)]

        return estidata

    def predict(self, ranvar: int or float):
        """
        ProbEstimatorNetwork에서 계산한 criteria로 Random Variable을
        Random Variable의 probability로 바꿉니다.
        :param ranvar:
        만약 int나 float라면 rv라는 것이니 모두 동일하게,
        list나 tuple이라면 새롭게 작성하는 것이니 각 Randon Variable마다 다르게
        :return:
        """
        estidata = 0
        if isinstance(ranvar, int) or isinstance(ranvar, float):
            if ranvar in self._ranvarset:
                estidata = self._estimate_rv[ranvar]
            elif self._min < ranvar < self._max:
                # 만약 rv의 범위 내에는 존재하나 rv가 나타난 적이 없다면
                # 앞뒤의 rv간의 평균으로 계산합니다.
                i = 0
                for i, rv in enumerate(self._ranvarset):
                    if rv > ranvar:
                        break
                x0, x2 = self._ranvarset[i - 1], self._ranvarset[i]
                d = x2 - x0
                w0, w2 = (ranvar - x0) / d, (x2 - ranvar) / d
                estidata = self._estimate_rv[x0] * w0 + self._estimate_rv[x2] * w2
        return estidata
    pass


class ProbabilityEstimationNeuralNetwork:
    def __init__(self, name):
        self._name = name
        pass

    @property
    def name(self):
        return self._name

    def fit(self, data, target):
        """
        data를 읽어서 각 target별로 data를 분할
        :param data:
        :param target:
        :return:
        """
        self._targets = set(target)
        datadict = {x: [] for x in self._targets}
        for d, t in zip(data, target):
            datadict[t].append(d)
        self._nodes = [ProbEstimator(k) for k in datadict]
        rv_amount = {node.target: node.fit(datadict[node.target]) for node in self._nodes}
        rvcdict = {set(x) for x in rv_amount}
        """
        rvcdict로 하나의 feature의 모든 random variable을 알아낸 후
        각 random variable의 합을 구한 후 각 class의 fit_predict를 실행
        """
        for node in self._nodes:
            node.fit_predict(rvcdict)
        pass

    @property
    def targets(self):
        return self._targets

    def predict(self, ranvar):
        """
        각 class별로 해당 ranvar의 probability를 계산해서 리턴
        노드 순서대로 계산한 후 노드 순서대로 리턴
        :param ranvar:
        :return:
        """
        return (x.predict(ranvar) for x in self._nodes)
        pass
    pass
