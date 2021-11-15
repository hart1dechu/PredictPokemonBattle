import csv
import math
from functools import reduce
import operator
from numpy import sqrt, true_divide
import random
import numpy as np
from sklearn.model_selection import KFold

typeTable = {
    "Steel" : {
        "x2" : ["Fairy","Ice","Rock"],
        "x0.5" : ["Steel","Water","Electric","Fire"],
        "x0" : []
    },
    "Fighting": {
        "x2" : ["Steel","Ice","Normal","Rock","Dark"],
        "x0.5" : ["Fairy","Bug","Poison","Psychic","Flying"],
        "x0" : ["Ghost"] 
    },
    "Dragon": {
        "x2" : ["Dragon"],
        "x0.5" : ["Steel"],
        "x0" : ["Fairy"]
    },
    "Water": {
        "x2" : ["Fire","Rock","Ground"],
        "x0.5" : ["Dragon","Water","Grass"],
        "x0" : []
    },
    "Electric": {
        "x2" : ["Water","Flying"],
        "x0.5" : ["Dragon","Electric","Grass"],
        "x0" : ["Ground"]
    },
    "Fairy": {
        "x2" : ["Fighting","Dragon","Dark"],
        "x0.5" : ["Steel","Fire","Poison"],
        "x0" : []
    },
    "Fire": {
        "x2" : ["Steel","Ice","Bug","Grass"],
        "x0.5" : ["Dragon","Water","Fire","Rock"],
        "x0" : []
    },
    "Ice": {
        "x2" : ["Dragon","Grass","Ground","Flying"],
        "x0.5" : ["Steel","Water","Fire","Ice"],
        "x0" : []
    },
    "Bug": {
        "x2" : ["Grass","Psychic","Dark"],
        "x0.5" : ["Steel","Fighting","Fairy","Fire","Poison","Ghost","Flying"],
        "x0" : []
    },
    "Normal": {
        "x2" : [],
        "x0.5" : ["Steel","Rock"],
        "x0" : ["Ghost"]
    },
    "Grass": {
        "x2" : ["Water","Rock","Ground"],
        "x0.5" : ["Steel","Dragon","Fire","Grass","Poison","Flying"],
        "x0" : []
    },
    "Poison": {
        "x2" : ["Fairy","Grass"],
        "x0.5" : ["Poison","Rock","Ground","Ghost"],
        "x0" : ["Steel"]
    },
    "Psychic": {
        "x2" : ["Fire","Rock","Ground"],
        "x0.5" : ["Dragon","Water","Grass"],
        "x0" : []
    },
    "Rock": {
        "x2" : ["Fire","Ice","Bug","Flying"],
        "x0.5" : ["Steel","Fighting","Ground"],
        "x0" : []
    },
    "Ground": {
        "x2" : ["Steel","Electric","Fire","Rock","Poison"],
        "x0.5" : ["Bug","Grass"],
        "x0" : ["Flying"]
    },
    "Ghost": {
        "x2" : ["Psychic","Ghost"],
        "x0.5" : ["Dark"],
        "x0" : ["Normal"]
    },
    "Dark": {
        "x2" : ["Psychic","Ghost"],
        "x0.5" : ["Fighting","Fairy","Dark"],
        "x0" : []
    },
    "Flying": {
        "x2" : ["Fighting","Bug","Grass"],
        "x0.5" : ["Steel","Electric","Rock"],
        "x0" : []
    },
}

def split_lines(input, seed, output1, output2):
  random.seed(seed)
  output1 = open(output1,'a')
  output1.truncate(0)
  output2 = open(output2,'a')
  output2.truncate(0)
  for line in open(input, 'r').readlines():
      if (random.random() < 0.5):
          write = output1;
      else:
          write = output2;
      write.write(line);

def read_data(filename):
  X = []
  Y = []
  with open(filename) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      next(csv_reader,None)
      for line in csv_reader:
          X.append(line[:1])
          Y.append(line[2] == line[0])

  return (X,Y);

#Extraire les données du fichier pokemon.csv sans la colonne 'Development stage'
def read_data_pokemon(filename):
    X = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader,None)

        for line in csv_reader:
            pokemon = line[1:]
            del pokemon[-2]
            X.append(pokemon)
            
    return X
#Regarde si le type1 est efficace sur le type2
def istypeEffective (type1,type2):
    type = typeTable[type1]
    
    if type2 in type["x2"]:
        return 2.0
    elif type2 in type["x0.5"]:
        return 0.5
    elif type2 in type["x0"]:
        return 0.0
    else:
        return 1.0

#Regarde si le type à un avantage par rapport au type adverse
#typeAttack est un type, tandis que typeDefense est un tableau de type
def isTypeAdvantage(typeAttack,typeDefense):
    damageMultiplier = 1.0 * istypeEffective(typeAttack,typeDefense[0])
    return damageMultiplier * istypeEffective(typeAttack,typeDefense[1]) > 1.0 if len(typeDefense) == 2 else damageMultiplier > 1