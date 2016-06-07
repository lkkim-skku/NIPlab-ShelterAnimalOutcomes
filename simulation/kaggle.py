import os
import csv
import pandas
import pickle
from datetime import datetime
import numpy


def path_project(projectname):
    return os.path.join((os.path.split(os.path.dirname(__file__))[0]), 'dataset', projectname)


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
        pandaran = pandas.read_csv(filepath, header=0)
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

    def predict(self, value):
        raise NotImplementedError

    def __call__(self, value):
        return self.predict(value)


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

    def fit(self, pandas_train: pandas, header_target):
        """
        어떻게 parsing할 지 결정합니다.
        :param pandas_train:
        :param header_target:
        :return:
        """
        self._targetfeature = header_target
        self._header_train = list(pandas_train)

        for key, parser in self._parsers.items():
            parser.index = self.find_index_train(parser.header)

        target = tuple(pandas_train[header_target])
        data = [tuple(parser.predict(value[parser.index]) for parser in self._parsers.values()) for value in pandas_train.values]

        return tuple(data), target

    def predict(self, pandas_test):
        """
        test sample을 parsing합니다.
        :param pandas_test:
        :return:
        """
        self._header_test = list(pandas_test)
        for key, parser in self._parsers.items():
            parser.index = self.find_index_test(parser.header)

        data = [tuple(parser.predict(value[parser.index]) for parser in self._parsers.values()) for value in pandas_test.values]

        return tuple(data)


def pickle_parser_dump(projectname, picklename):
    pass


def pickle_parser_load(projectname, picklename):
    pass


# if __name__ == '__main__':
    # projname = 'Shelter Animal Outcomes'
    # pandaran = load(projname, 'train.csv', 'csv')
    #
    # parser = KaggleDatasetParser()
    # parser.register(animal=AnimalParser('AnimalType'), nomine=NomineParser('Name'), breedmix=BreedMixParser('Breed'),
    #                 realage=AgeParser('AgeuponOutcome'), sex=SexParser('SexuponOutcome'),
    #                 neuter=NeuterParser('SexuponOutcome'), weeknum=WeekNumParser('DateTime'))
    # target, data = parser.fit(pandaran, 'OutcomeType')
    #
    # with open(os.path.join(path_project(projname), '{}.kpp'.format(projname)), 'wb') as file:
    #     pickle.dump(parser, file)
    #
    # with open(os.path.join(path_project(projname), '{}.kpp'.format(projname)), 'rb') as file:
    #     pp = pickle.load(file)

