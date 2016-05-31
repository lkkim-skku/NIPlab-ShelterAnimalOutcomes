"""
input: AnimalID,Name,DateTime,OutcomeType,OutcomeSubtype,AnimalType,SexuponOutcome,AgeuponOutcome,Breed,Color
output: ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer
"""
import sys
import os
from simulation import kaggleio
from shelter_animal_outcomes import SAO160530


if __name__ == '__main__':
    project_name = os.path.split(sys.path[0])[-1]

    learn_raw = kaggleio.load_learn(project_name)
    a = SAO160530().fit(learn_raw)
    feature_dict = a.suite()
    pass
