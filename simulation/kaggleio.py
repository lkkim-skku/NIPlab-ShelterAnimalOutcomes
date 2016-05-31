import os
import csv


def load_learn(name):
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
    return load_learn(name), load_test(name)


class DataSet:
    def __init__(self):
        self._set = dict()

    @property
    def header(self):
        return [x for x in self._set.keys()]

    def fit(self, arr: list):
        header = arr.pop(0)

        self._set = {h: [] for h in header}

        for row in arr:
            for h, col in zip(header, row):
                self._set[h].append(col)

        return self

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._set[item]
        elif isinstance(item, int):
            return [self._set[x][item] for x in self.header]

    def __call__(self):
        """
        array로 바꾸기
        :return:
        """
        arr = [self._set[h] for h in self.header]
        return zip(*arr)
