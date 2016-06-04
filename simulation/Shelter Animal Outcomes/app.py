"""
input: AnimalID,Name,DateTime,OutcomeType,OutcomeSubtype,AnimalType,SexuponOutcome,AgeuponOutcome,Breed,Color
output: ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer
"""
import os
import sys
sys.path.append(os.path.abspath('../'))
import kaggleio
from shelter_animal_outcomes import ShelterAnimal
from sklearn import metrics
from sklearn.svm import SVC

if __name__ == '__main__':
    print(sys.path)
    project_name = os.path.split(sys.path[0])[-1]

    learn_raw = kaggleio.load_learn(project_name)
    trainset = ShelterAnimal()
    trainset.fit(learn_raw)
    trainclassset = trainset.predict(1)
    X, y = [], []

    for name in trainclassset:
        X.extend(trainclassset[name])
        y.extend([name for x in trainclassset[name]])

    test_raw = kaggleio.load_test(project_name)
    testset = ShelterAnimal()
    testset.fit(test_raw)

    svc = SVC()
    svc.fit(X, y)
    result = svc.predict(X)
    acc = metrics.accuracy_score(result, y)
    print('Accuracy:', acc)