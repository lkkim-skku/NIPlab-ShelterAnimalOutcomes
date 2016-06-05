"""
feature를 숫자로 변환합니다.
"""
from datetime import datetime
import numpy


def parse_week(timeinfo):
    """
    발견된 시점을 주단위로 변환하고 변환합니다. 이 때 각 주차는 0에서 1사이값으로 바뀝니다.
    예를들어 첫번쨰주는 0.01886792452830, 24번째 주는 0.4528301886792453, 마지막주는 1입니다.
    :param timeinfo:
    :return: int, range: 1 to 53
    """
    dtlist = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in timeinfo]
    return [int(x.strftime('%W')) / 53 for x in dtlist]


def parse_sex(sexinfo):
    """
    성별을 숫자로 변환합니다.
    :param sexinfo:
    :return:
    0 if female else 1
    """
    return [0 if 'Female' in x else 1 if 'Male' in x else .5 for x in sexinfo]


def parse_neuter(sexinfo):
    """
    중성화여부를 숫자로 변환합니다.
    :param sexinfo:
    :return:
    0 if intact or unknwon else 1
    """
    return [0 if 'Intact' in x or 'Unknown' in x else 1 for x in sexinfo]


def parse_age(ageinfo):
    """
    나이를 숫자로 변환합니다. 1살을 기준으로 1로 변환합니다.
    :param ageinfo:
    :return:
    """
    ages = []
    for x in ageinfo:
        if len(x) < 1:
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


def parse_breed(breedinfo):
    """
    종의 정보를 표시합니다.
    :param breedinfo:
    :return: 0 if single breed else 1
    """
    return [1 if 'Mix' in x or '/' in x else 0 for x in breedinfo]


def parse_name(nameinfo):
    """
    이름의 존재 여부를 검사합니다.
    :param nameinfo:
    :return: 0 if it doesn't have name else 1
    """
    return [0 if len(x) == 0 else 1 for x in nameinfo]


class Parser:
    header_test = ("ID", "Name", "DateTime", "AnimalType", "SexuponOutcome", "AgeuponOutcome", "Breed", "Color")
    header_train = ("AnimalID", "Name", "DateTime", "OutcomeSubtype", "AnimalType",
                    "SexuponOutcome", "AgeuponOutcome", "Breed", "Color")

    def __init__(self):
        self._header_parsed = tuple()
        self._parsers = tuple()
        pass

    def header_parsed(self):
        return self._header_parsed

    def fit_target(self, database):
        """
        주어진 database는 header를 포함하며, class attribute가 data 안에 포함되어 있습니다.
        이 함수는 class attribute를 첫번째
        :param database:
        :return:
        """
        header_train = ("AnimalID", "Name", "DateTime", "OutcomeType", "OutcomeSubtype", "AnimalType",
                        "SexuponOutcome", "AgeuponOutcome", "Breed", "Color")
        header = database[0]

        for key in header:
            if key not in header_train:
                return KeyError

        target, data = [], []
        for sample_raw in database:
            target.append(sample_raw.pop(header_train.index("OutcomeType")))
            data.append(sample_raw)

        data.pop(0), target.pop(0)  # 첫번째 index에 header가 있으므로 다 털어냄

        return data, target

    def fit(self, database, callbacks: list or tuple):
        """

        :param database: header까지 포함된 전체 데이터
        :param callbacks: parse할 함수와 그 이름
        :return:
        """
        data, target = self.fit_target(database)
        parsedata = []
        for a, b in zip(self.header_train, zip(*data)):
            c = a
            d = b
        for callback in callbacks:
            parsedata.append(callback(data))
        pass

    def predict(self, sample: list or tuple):
        return (p(x) for x, p in zip(sample, self._parsers))
