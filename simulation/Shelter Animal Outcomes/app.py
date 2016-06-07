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

if __name__ == '__main__':
    projname = os.path.split(sys.path[0])[-1]

    pandas_train_raw = kaggle.load(projname, 'train.csv', 'csv')

    parser = kaggle.KaggleDatasetParser()
    parser.register(animal=AnimalParser('AnimalType'), nomine=NomineParser('Name'), breedmix=BreedMixParser('Breed'),
                    realage=AgeParser('AgeuponOutcome'), sex=SexParser('SexuponOutcome'),
                    neuter=NeuterParser('SexuponOutcome'), weeknum=WeekNumParser('DateTime'))

    data, target = parser.fit(pandas_train_raw, 'OutcomeType')

    svc = SVC()
    # svc.fit(data, target)
    # result = svc.predict(data)
    # acc = metrics.accuracy_score(result, target)
    # print('accuracy: ', acc)
    crossvalidation = cv.cross_val_score(svc, data, target, cv=10)

    learn_raw = kaggleio.load_train(projname)
    trainset = ShelterAnimal()
    trainset.fit(learn_raw)
    trainclassset = trainset.predict(1)
    X, y = [], []

    for name in trainclassset:
        X.extend(trainclassset[name])
        y.extend([name for x in trainclassset[name]])

    test_raw = kaggleio.load_test(projname)
    testset = ShelterAnimal()
    testset.fit(test_raw)

    svc = SVC()
    svc.fit(X, y)
    result = svc.predict(X)
    acc = metrics.accuracy_score(result, y)
    crossvalidation = cv.cross_val_score(svc, X, y)
    print('Accuracy:', acc)