import os
import csv
import pandas
import pickle
from datetime import datetime
import numpy


def load(projectname, filename, file_extension):
    """
    kaggle data를 읽어옵니다
    :param projectname:
    :param filename: filename과 확장명까지 입력해야 합니다.
    :param file_extension: pandas의 reader 중 어떤 reader를 사용할 지 결정합니다.
    :return:
    """
    path = os.path.join((os.path.split(os.path.dirname(__file__))[0]), 'dataset', projectname)
    filepath = os.path.join(path, filename)
    if file_extension == 'csv':
        pandaran = pandas.read_csv(filepath)
    else:
        raise KeyError

    return pandaran


class DataSet:
    """
    Scikit-Learn에 적합하지만 Utility Function이 추가된 dataset model입니다.
    """
    def __init__(self, dataset=None):
        self._dataset = dataset if dataset else dict()
        pass

    @property
    def header(self):
        return [x for x in self._dataset.keys()]

    def fit(self, arr: list):
        header = arr.pop(0)

        self._dataset = {h: [] for h in header}

        for row in arr:
            for h, col in zip(header, row):
                self._dataset[h].append(tuple(col))

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
        arr = (self._dataset[h] for h in self.header)
        return arr


class BaseFeatureParser:
    """
    Feature를 parse하는 객체입니다.
    :class:BaseDatasetParser에 callback으로 넣어주면 됩니다.
    """
    def __init__(self, header):
        self.header = header
        self.index = -1

    def fit(self, setname):
        if self.index == -1:
            raise AttributeError
        return self.index

    def predict(self, value):
        raise NotImplementedError

    def __call__(self, value):
        return self.predict(value)


class AnimalParser(BaseFeatureParser):
    def predict(self, value):
        return 0 if value == 'Cat' else 1


class NomineParser(BaseFeatureParser):
    def predict(self, value):
        return 0 if pandas.isnull(value) else 1


class BreedMixParser(BaseFeatureParser):
    def predict(self, value):
        return 1 if 'Mix' in value or '/' in value else 0


class AgeParser(BaseFeatureParser):
    def predict(self, value):
        if pandas.isnull(value):
            y = -1
        else:
            y = int(value[:2])
            if 'day' in value:
                y *= 1 / 365
            elif 'week' in value:
                y *= 1 / 52
            elif 'month' in value:
                y *= 1 / 12
        return y


class SexParser(BaseFeatureParser):
    def predict(self, value):
        return .5 if pandas.isnull(value) else .5 if 'Unknown' in x else 0 if 'Female' in value else 1


class NeuterParser(BaseFeatureParser):
    def predict(self, value):
        return 0 if pandas.isnull(value) else 0 if 'Intact' in value or 'Unknown' in value else 1


class WeekNumParser(BaseFeatureParser):
    def predict(self, value):
        x = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return int(x.strftime('%W')) / 53


class KaggleDatasetParser:
    """
    dataset을 parse합니다.
    """

    def __init__(self, obj=None):
        if isinstance(obj, KaggleDatasetParser):
            self._targetfeature = obj.targetfeature
            self._parsers = obj.parsers
            self._header_train = obj.header_train
            self._header_parsed = obj.header_parsed
            self._header_test = obj.header_test
        else:
            self._targetfeature = str()
            self._parsers = dict()
            self._header_train = list()
            self._header_test = list()
        pass

    @property
    def targetfeature(self):
        return self._targetfeature

    @property
    def parsers(self):
        return self.parsers

    @property
    def header_train(self):
        return self._header_train

    @property
    def header_parsed(self):
        return list(self._parsers.keys())

    @property
    def header_test(self):
        return self._header_test

    def find_index_train(self, header):
        """
        header의 이름을 보고 index를 찾아줍니다.
        :param header:
        :return:
        """
        return self._header_train.index(header)

    def find_index_parsed(self, header):
        """
        header의 이름을 보고 index를 찾아줍니다.
        :param header:
        :return:
        """
        return self.header_parsed.index(header)

    def register(self, **parsers):
        """
        parser들을 등록합니다.
        :param parsers:
        :return:
        """
        for key, parser in parsers.items():
            if isinstance(parser, BaseFeatureParser):
                self._parsers[key] = parser
            else:
                raise ValueError

    def find_index_test(self, header):
        """
        header의 이름을 보고 index를 찾아줍니다.
        :param header:
        :return:
        """
        return self._header_test.index(header)

    def fit(self, pandas_train: pandas, targetfeature):
        """
        어떻게 parsing할 지 결정합니다.
        :param pandas_train:
        :param targetfeature:
        :return:
        """
        self._targetfeature = targetfeature
        self._header_train = list(pandas_train)

        for key, parser in self._parsers.items():
            parser.index = self.find_index_train(parser.header)

        for key, parser in self._parsers.items():
            value = parser.predict(pandasample[parser.index])

        return dataset.target, dataset.data

    def predict(self, pandas_test):
        """
        test sample을 parsing합니다.
        :param pandas_test:
        :return:
        """
        self._header_test = list(pandas_test)
        for key, parser in self._parsers.items():
            index = self.find_index_test(parser.header)

        return [self._closure[key](pandas_test) for key in self._closure]

if __name__ == '__main__':
    pandaran = load('Shelter Animal Outcomes', 'train.csv', 'csv')

    parser = KaggleDatasetParser()
    parser.register(animal=AnimalParser('AnimalType'), nomine=NomineParser('Name'), breedmix=BreedMixParser('Breed'),
                    realage=AgeParser('AgeuponOutcome'), sex=SexParser('SexuponOutcome'),
                    neuter=NeuterParser('SexuponOutcome'), weeknum=WeekNumParser('DateTime'))
    parser.fit(pandaran, 'OutcomeType')

    pickle.dump(parser)
    pp = pickle.load('parser.pickle')

    result = []
    for sample in testdataset:
        result.append(parser.predict(sample))
