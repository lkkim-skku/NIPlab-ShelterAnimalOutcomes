"""
input: AnimalID,Name,DateTime,OutcomeType,OutcomeSubtype,AnimalType,SexuponOutcome,AgeuponOutcome,Breed,Color
output: ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer
"""
import os
import sys
sys.path.append(os.path.abspath('../'))
import kaggle
# from shelter_animal_outcomes import ShelterAnimal
from shelteranimal import *
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import cross_validation as cv
from bnn import ProbabilityEstimationNeuralNetwork as PENN

if __name__ == '__main__':
    projname = os.path.split(sys.path[0])[-1]

    pandas_train_raw = kaggle.load(projname, 'train.csv', 'csv')

    parser = kaggle.KaggleDatasetParser()
    parser.register(animal=AnimalParser('AnimalType'), namexist=NomineParser('Name'), breedmix=BreedMixParser('Breed'),
                    realage=AgeParser('AgeuponOutcome'), sex=SexParser('SexuponOutcome'),
                    neuter=NeuterParser('SexuponOutcome'), weeknum=WeekNumParser('DateTime'))

    data, target = parser.fit(pandas_train_raw, 'OutcomeType')
    svc = SVC()
    svc.fit(data, target)

    pandas_test_raw = kaggle.load(projname, 'test.csv', 'csv')
    data_test = parser.predict(pandas_test_raw)
    for key, sample in enumerate(data_test):
        print(key + 1, svc.predict([sample]))

    penn = PENN(projname)
    penn.resister(*parser.header_parsed)
    data, target = penn.fit(data, target)

    svc = SVC()
    svc.fit(data, target)
    result = svc.predict(data)
    acc = metrics.accuracy_score(result, target)
    print('accuracy: ', acc)
    crossvalidation = cv.cross_val_score(svc, data, target, cv=10)

