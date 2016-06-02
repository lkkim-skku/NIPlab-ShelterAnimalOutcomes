from simulation import kaggleio
from simulation import statistics
from datetime import datetime
import numpy


def week(timeinfo):
    dtlist = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in timeinfo]
    weeks = [int(x.strftime('%W')) for x in dtlist]
    return weeks


def sexneuter(sexinfo):
    sexes = [0 if 'Female' in x else 1 if 'Male' in x else .5 for x in sexinfo]
    neuters = [0 if 'Intact' in x or 'Unknown' in x else 1 for x in sexinfo]
    return sexes, neuters


def age(ageinfo):
    """
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


def breed(breedinfo):
    breeds = [1 if 'Mix' in x or '/' in x else 0 for x in breedinfo]
    return breeds


def namelen(nameinfo):
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


class ShelterAnimal(kaggleio.DataSet):
    """
    input: AnimalID,Name,DateTime,OutcomeType,OutcomeSubtype,AnimalType,SexuponOutcome,AgeuponOutcome,Breed,Color
    """
    def predict(self, arr):
        """
        feature: animal, month, week, sex, neuter, age, breed, color
        :param arr:
        :return:
        """
        # super().fit(arr)

        plain_features = {'outcome': self._set['OutcomeType']}
        plain_features['animal'] = [1 if x == 'Dog' else 0 for x in self._set['AnimalType']]
        plain_features['week'] = week(self._set['DateTime'])
        plain_features['sex'], plain_features['neuter'] = sexneuter(self._set['SexuponOutcome'])
        plain_ages = age(self._set['AgeuponOutcome'])
        plain_features['age'] = featurescaling(plain_ages)
        plain_features['mix'] = breed(self._set['Breed'])
        plain_features['namelen'] = namelen(self._set['Name'])

        features = kaggleio.DataSet(plain_features)

        # build class data
        classes = set(features['outcome'])
        cdheader = [x for x in features.header if 'outcome' not in x]
        classdatasets = {x: {y: [] for y in cdheader} for x in classes}

        for sample in features:
            for name in classes:
                if name in sample:
                    for key, value in zip(features.header, sample):
                        if key in cdheader:
                            classdatasets[name][key].append(value)
        for key in classdatasets:
            classdatasets[key] = kaggleio.DataSet(classdatasets[key])

        # The feature 'Week' average
        weekrange = range(0, 53)
        tempdict = {x: {y: 0 for y in weekrange} for x in classdatasets}
        for idxwk in weekrange:
            amount = 0

            for key in classdatasets:
                size = classdatasets[key]['week'].count(idxwk)
                amount += size
                tempdict[key][idxwk] = size

            for key in tempdict:
                tempdict[key][idxwk] /= amount

        for idxwk in weekrange:
            for key in classdatasets:
                average5 = sum(tempdict[key][52 - x - 2 if idxwk + x - 2 < 0 else idxwk + x - 2 - 52 if idxwk + x - 2 > 52 else idxwk + x - 2] for x in range(5)) / 5

            # for key in tempdict:
                for i, x in enumerate(classdatasets[key]['week']):
                    if x == idxwk:
                        classdatasets[key]['week'][i] = average5

        return classdatasets
