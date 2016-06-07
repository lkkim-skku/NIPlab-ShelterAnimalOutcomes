import pandas
from datetime import datetime
from kaggle import BaseFeatureParser


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
        return .5 if pandas.isnull(value) else .5 if 'Unknown' in value else 0 if 'Female' in value else 1


class NeuterParser(BaseFeatureParser):
    def predict(self, value):
        return 0 if pandas.isnull(value) else 0 if 'Intact' in value or 'Unknown' in value else 1


class WeekNumParser(BaseFeatureParser):
    def predict(self, value):
        x = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return int(x.strftime('%W')) / 53
