from simulation import kaggleio
from datetime import datetime


def monthweek(timeinfo):
    dtlist = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in timeinfo]
    months = [x.month for x in dtlist]
    weeks = [x.isocalendar()[1] for x in dtlist]
    return months, weeks


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
    return ages


def breed(breedinfo):
    breeds = [1 if 'Mix' in x or '/' in x else 0 for x in breedinfo]
    return breeds


def namelen(nameinfo):
    namelens = [0 if len(x) == 0 else 1 for x in nameinfo]
    return namelens


class ShelterAnimal(kaggleio.DataSet):
    """
    input: AnimalID,Name,DateTime,OutcomeType,OutcomeSubtype,AnimalType,SexuponOutcome,AgeuponOutcome,Breed,Color
    """
    def suite(self):
        """
        feature: animal, month, week, sex, neuter, age, breed, color
        :return:
        """
        features = {'outcome': self._set['OutcomeType']}
        features['animal'] = [1 if x == 'Dog' else 0 for x in self._set['AnimalType']]
        features['month'], features['week'] = monthweek(self._set['DateTime'])
        features['sex'], features['neuter'] = sexneuter(self._set['SexuponOutcome'])
        features['age'] = age(self._set['AgeuponOutcome'])
        features['mix'] = breed(self._set['Breed'])
        features['namelen'] = namelen(self._set['Name'])
        self._features = kaggleio.DataSet(features)

    def predict(self):
        targets = set(self._features['outcome'])
        outcomeindex = list(x for x in self._features.header).index('outcome')
        classes = {t: [data for data in self._features if t in data] for t in targets}
        for target in classes:
            for data in classes[target]:
                data.pop(outcomeindex)
        pass

    @property
    def features(self):
        return self._features

    pass
