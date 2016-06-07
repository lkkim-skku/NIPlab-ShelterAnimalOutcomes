from Tools.scripts.make_ctype import values


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


class Dataset:
    """
    하나의 class에 대한 dataset을 구성합니다.
    """
    def __init__(self, dataset: dict or Dataset):
        pass

    @property
    def header(self):
        return [x for x in self._dataset]

    def fit(self, data, header):
        self._dataset = {x: [] for x in header}
        for key, values in zip(header, zip(*data)):
            self._dataset[key] = values
            setattr(self, key, self._dataset[key])

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
        self._data = {}
        self._target = target

    @property
    def target(self):
        return self._target

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, header):
        self._header = header

    def fit(self, data: list or tuple):
        """
        각 Random Variable의 갯수를 구해서 리턴합니다.
        :param data:
        :return:
        """
        for key, *values in zip(self._header, *data):
            # values = trdata[i]
            valueset = {k: 0 for k in set(values)}
            for v in values:
                valueset[v] += 1
            self._data[key] = valueset

    def __getitem__(self, item):
        return self._data[item]

    def fit_predict(self, criteria: dict):
        for key in self._header:
            cit = criteria[key]
            feature = self._data[key]
            for rv in feature:
                feature[rv] /= cit[rv]
            self._data[key] = feature

    def predict(self, sample):
        """
        ProbEstimatorNetwork에서 계산한 criteria로 Random Variable을
        Random Variable의 probability로 바꿉니다.
        :param sample:
        :return:
        """
        sample_p = []
        for s, h in zip(sample, self._header):
            feature = self._data[h]
            sortedfeature = [x for x in feature]
            sortedfeature.sort()
            if s in feature:
                sample_p.append(feature[s])
            elif sortedfeature[0] < s < sortedfeature[-1]:
                i = 0
                for i, rv in enumerate(sortedfeature):
                    if rv < s:
                        break

                sample_p.append(feature[sortedfeature[i]])
            else:
                sample_p.append(0)

        if len(sample_p) < 7:
            a  =1
        return tuple(sample_p)
        # estidata = 0
        # if isinstance(ranvar, int) or isinstance(ranvar, float):
        #     if ranvar in self._ranvarset:
        #         estidata = self._estimate_rv[ranvar]
        #     elif self._min < ranvar < self._max:
        #         # 만약 rv의 범위 내에는 존재하나 rv가 나타난 적이 없다면
        #         # 앞뒤의 rv간의 평균으로 계산합니다.
        #         i = 0
        #         for i, rv in enumerate(self._ranvarset):
        #             if rv > ranvar:
        #                 break
        #         x0, x2 = self._ranvarset[i - 1], self._ranvarset[i]
        #         d = x2 - x0
        #         w0, w2 = (ranvar - x0) / d, (x2 - ranvar) / d
        #         estidata = self._estimate_rv[x0] * w0 + self._estimate_rv[x2] * w2
        # return estidata


class ProbabilityEstimationNeuralNetwork:
    def __init__(self, name):
        self._name = name
        self._node = {}

    @property
    def name(self):
        return self._name

    def append(self, probestimator: ProbEstimator):
        self._node[probestimator.name] = probestimator
        setattr(self, probestimator.name, self._node[probestimator.name])

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, header):
        self._header = header

    def fit(self, data, target):
        """
        data를 읽어서 각 target별로 data를 분할
        :param data:
        :param target:
        :return:
        """
        datadict = {x: [] for x in set(target)}
        for d, t in zip(data, target):
            datadict[t].append(d)

        for key in datadict:
            pe = ProbEstimator(key)
            pe.header = self._header
            pe.fit(datadict[key])
            self._node[key] = pe
            setattr(self, key, pe)
        """
       rvcdict로 하나의 feature의 모든 random variable을 알아낸 후
       각 random variable의 합을 구한 후 각 class의 fit_predict를 실행
       """
        rvcdict = {x: {} for x in self._header}
        for pe in self._node.values():
            for key in self._header:
                rvc = rvcdict[key]
                for rv in pe[key]:
                    if rv not in rvc:
                        rvc[rv] = 0
                    rvc[rv] += pe[key][rv]

        for pe in self._node.values():
            pe.fit_predict(rvcdict)

    def predict(self, data):
        """
        각 class별로 해당 ranvar의 probability를 계산해서 리턴
        노드 순서대로 계산한 후 노드 순서대로 리턴
        :param data:
        :return:
        """
        predata = []
        for sample in data:
            pred = {k: x.predict(sample) for k, x in self._node.items()}
            predata.append(pred)
        return tuple(predata)
