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
    # parser.register(animal=AnimalParser('AnimalType'), namexist=NomineParser('Name'), breedmix=BreedMixParser('Breed'),
    #                 realage=AgeParser('AgeuponOutcome'), sex=SexParser('SexuponOutcome'),
    #                 neuter=NeuterParser('SexuponOutcome'), weeknum=WeekNumParser('DateTime'))
    #
    # data, target = parser.fit(pandas_train_raw, 'OutcomeType')

    parser.header_train = 'AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color'
    parser.header_test = 'AnimalID', 'Name', 'DateTime', 'AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color'

    parser.fit('OutcomeType', animal=AnimalParser('AnimalType'), namexist=NomineParser('Name'),
               breedmix=BreedMixParser('Breed'), realage=AgeParser('AgeuponOutcome'),
               sex=SexParser('SexuponOutcome'), neuter=NeuterParser('SexuponOutcome'),
               weeknum=WeekNumParser('DateTime'))
    target = pandas_train_raw['OutcomeType']
    parsed_train = parser.predict(pandas_train_raw)

    penn = PENN(projname)
    penn.header = parser.header_parsed
    penn.fit(parsed_train, target)
    penn_train = penn.predict(parsed_train)
    penn_train = [x[k] for x, k in zip(penn_train, target)]

    svc = SVC()
    svc.fit(penn_train, target)

    pandas_test_raw = kaggle.load(projname, 'test.csv', 'csv')
    parsed_test = parser.predict(pandas_test_raw)
    penn_test = penn.predict(parsed_test)
    for key, sample in enumerate(penn_test):
        print(key + 1, svc.predict([sample['Adoption']]), svc.predict([sample['Died']]), svc.predict([sample['Euthanasia']]), svc.predict([sample['Return_to_owner']]), svc.predict([sample['Transfer']]))

    # result = svc.predict(data_penn)
    # result = svc.predict(data)
    acc = metrics.accuracy_score(result, target)
    print('accuracy: ', acc)
    crossvalidation = cv.cross_val_score(svc, data, target, cv=10)

