"""
input: AnimalID,Name,DateTime,OutcomeType,OutcomeSubtype,AnimalType,SexuponOutcome,AgeuponOutcome,Breed,Color
output: ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer
"""
import os
import sys

sys.path.append(os.path.abspath('../'))
import kaggleio
import control
from cpon import CPON

if __name__ == '__main__':
    project_name = os.path.split(sys.path[0])[-1]

    train_raw = kaggleio.load_learn(project_name)
    cpm = CPON(project_name)
    train_target, train_data = control.targetdata(train_raw, 'OutcomeType')
    cpm.fit(train_data, train_target)
