import os
import csv
import pandas
import pickle
from datetime import datetime
import numpy

def load_train(name):
    path = os.path.join((os.path.split(os.path.dirname(__file__))[0]), 'dataset', name)

    with open(os.path.join(path, 'train.csv'), 'r') as file:
        csver = csv.reader(file, delimiter=',', quotechar='|')
        arr = [x for x in csver]

    return arr


def load_test(name):
    path = os.path.join((os.path.split(os.path.dirname(__file__))[0]), 'dataset', name)

    with open(os.path.join(path, 'test.csv'), 'r') as file:
        csver = csv.reader(file, delimiter=',', quotechar='|')
        arr = [x for x in csver]

    return arr


def load(name):
    """
    kaggle data를 읽어옵니다
    :param name:
    :return:
    """
    return load_train(name), load_test(name)


class DataSet:
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




def week(header_index):
    index = header_index

    def wrapper(sample):
        nonlocal index
        x = datetime.strptime(sample[index], "%Y-%m-%d %H:%M:%S")
        y = int(x.strftime('%W')) / 53
        return y

    return wrapper


def week_array(timeinfo):
    """
    발견된 시점을 주단위로 변환합니다.
    :param timeinfo:
    :return: int, range: 1 to 53
    """
    timeinfo = timeinfo['DateTime']
    dtlist = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in timeinfo]
    weeks = [int(x.strftime('%W')) for x in dtlist]
    return weeks


def sex(header_index):
    index = header_index

    def wrapper(sample):
        nonlocal index
        x = sample[index]
        y = .5 if pandas.isnull(x) else .5 if 'Unknown' in x else 0 if 'Female' in x else 1
        return y

    return wrapper


def sex_array(sexinfo):
    sexinfo = sexinfo['SexuponOutcome']
    # sexes = [0 if 'Female' in x else 1 if 'Male' in x else .5 for x in sexinfo]
    sexes = [0 if pandas.isnull(x) else 0 if 'Unknown' in x else 0 if 'Female' in x else 1 for x in sexinfo]
    return sexes


def neuter(header_index):
    index = header_index

    def wrapper(sample):
        nonlocal index
        x = sample[index]
        y = 0 if pandas.isnull(x) else 0 if 'Intact' in x or 'Unknown' in x else 1
        return y

    return wrapper


def neuter_array(sexinfo):
    sexinfo = sexinfo['SexuponOutcome']
    neuters = [0 if pandas.isnull(x) else 0 if 'Intact' in x or 'Unknown' in x else 1 for x in sexinfo]
    return neuters


def age(header_index):
    index = header_index

    def wrapper(sample):
        nonlocal index
        x = sample[index]

        if pandas.isnull(x):
            y = -1
        else:
            y = int(x[:2])
            if 'day' in x:
                y *= 1 / 365
            elif 'week' in x:
                y *= 1 / 52
            elif 'month' in x:
                y *= 1 / 12
        return y

    return wrapper


def age_array(ageinfo):
    """
    나이를 숫자로 변환합니다. 1살을 기준으로 1로 변환합니다.
    :param ageinfo:
    :return:
    """
    ageinfo = ageinfo['AgeuponOutcome']
    ages = []
    for x in ageinfo:
        if pandas.isnull(x):
            ages.append(-1)
        else:
            _age = int(x[:2])
            if 'day' in x:
                _age *= 1 / 365
            elif 'week' in x:
                _age *= 1 / 52
            elif 'month' in x:
                _age *= 1 / 12
            ages.append(_age)
    _age = [x for x in ages if x > 0]
    agemean = numpy.mean(_age)
    for i in range(len(ages)):
        if ages[i] == -1:
            ages[i] = agemean
    return ages


def mix(header_index):
    index = header_index

    def wrapper(sample):
        nonlocal index
        x = sample[index]
        y = 1 if 'Mix' in x or '/' in x else 0
        return y

    return wrapper


def breed_array(breedinfo):
    """
    종의 정보를 표시합니다.
    :param breedinfo:
    :return: 0 if single breed else 1
    """
    breeds = [1 if 'Mix' in x or '/' in x else 0 for x in breedinfo]
    return breeds


def name(header):
    name.header = header

    # name.index = 1
    # index = header_index

    def wrapper(sample):
        # nonlocal name.index
        x = sample[wrapper.index]
        y = 0 if pandas.isnull(x) else 1
        return y

    return wrapper


def name2(header):
    name.header = header

    # name.index = 1
    # index = header_index

    def wrapper(sample):
        x = sample[name.header]
        y = 0 if pandas.isnull(x) else 1
        return y

    return wrapper


def name_array(nameinfo):
    """
    이름의 존재 여부를 검사합니다.
    :param nameinfo:
    :return: 0 if it doesn't have name else 1
    """
    namelens = [0 if len(x) == 0 else 1 for x in nameinfo]
    return namelens


def featurescaling(values: list or tuple):
    """

    :param values:
    :return:
    """
    vx, vn = max(values), min(values)
    vdiff = vx - vn
    fscaled = [(x - vn) / vdiff for x in values]
    return fscaled


class FeatureParser:
    def header_train(self, *args):
        self._header_train = args

    @property
    def header_train(self):
        return self._header_train

    def header_test(self, *args):
        self._header_test = args

    @property
    def header_test(self):
        return self._header_test

    pass


class BaseDatasetParser:
    """
    dataset을 parse합니다.
    """

    def __init__(self, *header):
        self._header = header
        pass

    @property
    def header(self):
        return self._header

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = value

    def fit(self, pandasobj: list, closure: dict):
        """
        어떻게 parsing할 지 결정합니다.
        :param pandasobj:
        :param closure:
        :return:
        """
        self._closure = closure
        if not hasattr(pandasobj, self._target):
            raise KeyError
        target = getattr(pandasobj, self._target)
        data = [tuple(x for x in closure)]

        a = closure['name'].header
        for sample in pandasobj.values:
            data.append(tuple(closure[key](sample) for key in closure))

        return tuple(target), tuple(data)

        # targetidx = pandasobj[self._target_header]
        # data = []
        # for key in closures:
        #     data.append(closures[key](pandasobj))
        # pass

    def predict(self, sample):
        """
        test sample을 parsing합니다.
        :param sample:
        :return:
        """
        return [self._closure[key](sample) for key in self._closure]

if __name__ == '__main__':
    dataset = pandas.read_csv('train.csv')

    parser = BaseDatasetParser('AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'AnimalType',
                               'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color')

    parser.target = 'OutcomeType'
    name2.index = 1
    parser.fit(dataset, {
        # 'week': week(parser.header.index('DateTime')),
        # 'sex': sex(parser.header.index('SexuponOutcome')),
        # 'neuter': neuter(parser.header.index('SexuponOutcome')),
        # 'age': age(parser.header.index('AgeuponOutcome')),
        # 'mix': mix(parser.header.index('Breed')),
        'name': name2
    })
    for sampleset in dataset:
        parser.predict(sampleset)

