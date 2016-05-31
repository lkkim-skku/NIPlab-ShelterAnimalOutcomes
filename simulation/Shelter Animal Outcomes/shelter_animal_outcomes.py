from simulation import kaggleio
from datetime import datetime


def monthweek(strptimes):
    dtlist = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in strptimes]
    weeks = [x.isocalendar()[1] for x in dtlist]
    months = [x.month for x in dtlist]
    return weeks, months


class SAO160530(kaggleio.DataSet):
    """
    input: AnimalID,Name,DateTime,OutcomeType,OutcomeSubtype,AnimalType,SexuponOutcome,AgeuponOutcome,Breed,Color
    """
    def __init__(self):
        super().__init__()
        self._set = dict()
        self.callbacks = dict()

    def suite(self):
        """
        feature: animal, month, week, sex, neuter, age, breed, color
        :return:
        """
        features['month'], features['week'] = monthweek(self._set['DateTime'])

    pass
