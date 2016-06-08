"""
input: AnimalID,Name,DateTime,OutcomeType,OutcomeSubtype,AnimalType,SexuponOutcome,AgeuponOutcome,Breed,Color
output: ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer
"""
import os
import sys
sys.path.append(os.path.abspath('../'))
import kaggle
from shelteranimal import *
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import cross_validation as cv
from bnn import BayesianNeuralNetwork as BNN

if __name__ == '__main__':
    projname = os.path.split(sys.path[0])[-1]

    pandas_train_raw = kaggle.load(projname, 'train.csv', 'csv')

    parser = kaggle.KaggleDatasetParser()
    parser.header_train = 'AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color'
    parser.header_test = 'AnimalID', 'Name', 'DateTime', 'AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color'

    parser.fit('OutcomeType', animal=AnimalParser('AnimalType'), namexist=NomineParser('Name'),
               breedmix=BreedMixParser('Breed'), realage=AgeParser('AgeuponOutcome'),
               sex=SexParser('SexuponOutcome'), neuter=NeuterParser('SexuponOutcome'),
               weeknum=WeekNumParser('DateTime'))
    target = pandas_train_raw['OutcomeType']
    parsed_train = parser.predict(pandas_train_raw)

    bnn = BNN(projname)
    bnn.header = parser.header_parsed
    bnn.fit(parsed_train, target)
    # bnn_train = bnn.predict_table(parsed_train)
    # bnn_train = [x[k] for x, k in zip(bnn_train, target)]

    pandas_test_raw = kaggle.load(projname, 'test.csv', 'csv')
    parsed_test = parser.predict(pandas_test_raw)
    bnn_test = bnn.predict_table(parsed_test)

    with open('subminssion.csv', 'w') as file:
        file.write('ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer\n')
        # for key, sample in enumerate(penn_train):
            # file.write(str(key + 1) + ',' + repr(sum(sample) / 7) + '\n')
        for key, sample in enumerate(bnn_test):
            file.write(','.join([str(key + 1), repr(sample['Adoption']), repr(sample['Died']), repr(sample['Euthanasia']), repr(sample['Return_to_owner']), repr(sample['Transfer'])])+'\n')

    acc = metrics.accuracy_score(result, target)
    print('accuracy: ', acc)
    crossvalidation = cv.cross_val_score(svc, data, target, cv=10)

