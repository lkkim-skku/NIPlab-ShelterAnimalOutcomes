"""
input: AnimalID,Name,DateTime,OutcomeType,OutcomeSubtype,AnimalType,SexuponOutcome,AgeuponOutcome,Breed,Color
output: ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer
"""
import os
import sys

sys.path.append(os.path.abspath('../'))
import kaggleio
import control
import shelteranimal
import bnn
from bnn import BayesianNeuralNetwork as BNN


if __name__ == '__main__':
    project_name = os.path.split(sys.path[0])[-1]

    # train_raw = kaggleio.load_train(project_name)
    a = kaggleio.load_train("Facebook V-Predicting Check Ins")
    saparser = shelteranimal.Parser()
    d, t = saparser.fit(train_raw, [])
    learner = BNN(project_name)
    # train_target, train_data = control.targetdata(train_raw, 'OutcomeType')
    learner.fit(train_data, train_target)
    test_raw = kaggleio.load_test(project_name)
    test_data = learner.normalize(test_raw)
    learner.predict(test_data)
