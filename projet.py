import csv
import math
from functools import reduce
import operator
from numpy import sqrt, true_divide
import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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
  avoidHeader = 0
  for line in open(input, 'r').readlines():
      if avoidHeader != 0:
        if (random.random() < 0.5):
            write = output1;
        else:
            write = output2;
        write.write(line);
      else:
          avoidHeader+=1

def read_data(filename):
  X = []
  Y = []
  with open(filename) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for line in csv_reader:
          X.append(line[:2])
          Y.append(line[2] == line[0])

  return (X,Y);

#Extraire les données du fichier pokemon.csv sans la colonne 'Development stage'
def read_data_pokemon(filename):
    X = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for line in csv_reader:
            pokemon = line[1:]
            del pokemon[-2]
            X.append(pokemon)

    return X

pokemon = read_data_pokemon('pokemon.csv')
split_lines('combats.csv',0,'train','test')
train_raw_x,train_raw_y = read_data('train')
test_raw_x,test_raw_y = read_data('test')
#Regarde si le type1 est efficace sur le type2
#return theMultiplier
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
#return le damageMultiplier
def isTypeAdvantage(typeAttack,typeDefense):
    damageMultiplier = 1.0 * istypeEffective(typeAttack,typeDefense[0])
    damageMultiplier = damageMultiplier * (istypeEffective(typeAttack,typeDefense[1]) if (len(typeDefense) == 2) & (typeDefense[1] != '') else 1)
    return damageMultiplier

#Permet de savoir si le type d'un pokemon est avantagé ou immunisé face au type d'una utre pokemon
#type1 et type2 sont des tableaux de type de pokemon
#return isAdvantaged, isImmune
def doubleTypeAdvantage(type1,type2):
    advantage1= isTypeAdvantage(type1[0],type2)
    advantage2= isTypeAdvantage(type1[1],type2) if (len(type1) > 1) & (type1[1] != '') else 0
    result = max(advantage1,advantage2)
    return result > 1, result == 0
#Prend deux pokémon, et renvoie le typeAdvantage du pkm1 face au pkm2
def typeBattle(pkm1,pkm2):
    typePkm1 = pokemon[pkm1][1:3]
    typePkm2 = pokemon[pkm2][1:3]
    return doubleTypeAdvantage(typePkm1,typePkm2)
#fait la somme des elements d'une table
def sumInTable(tab):
    count = 0
    for elt in tab:
        count += elt
    return count
#Fonction principale pour le training d'arbre de decision
#Elle convertit les données des combats, en une table décrivant les situations entre les deux pokémons
#Exemple: Le type était-il plus avantageux ? Ses stats étaient-ils meilleures ?
def tableDecision(train_x,train_y):
    table_x = []
    table_y = []
    for i in range(len(train_x)):
        table_x_elt = []
        advantage,immune = typeBattle(int(train_x[i][0]),int(train_x[i][1]))
        table_x_elt.append(advantage)
        table_x_elt.append(immune)
        #Récupération des pokémons vi la table pokémon
        pokemon1 = pokemon[int(train_x[i][0])]
        pokemon2 = pokemon[int(train_x[i][1])]
        #tableaux des stats des pokémons
        statsp1 = list(map(lambda x:int(x), pokemon1[3:9]))
        statsp2 = list(map(lambda x:int(x),pokemon2[3:9]))
        for j in range (len(statsp1)):
            table_x_elt.append(statsp1[j] > statsp2[j])
        table_x_elt.append(sumInTable(statsp1) > sumInTable(statsp2))
        
        table_y.append(train_y[i])
        table_x.append(table_x_elt)
    return table_x,table_y

train_x,train_y = tableDecision(train_raw_x,train_raw_y)
test_x,test_y = tableDecision(test_raw_x,test_raw_y)

#Fonction pour l'arbre de décision
def randomForestClassifier(train_x,train_y,X):
    clf = DecisionTreeClassifier(max_depth = 6,random_state=0)
    clf.fit(train_x,train_y)
    return clf.predict(np.reshape(X,[1,-1]))

#Evaluation sur le fichier test
def eval_pokemon_battle_prediction(test_x,test_y,classifier):
    count = 0;
    for i in range(len(test_x)):
      if (classifier(test_x[i]) != test_y[i]):
          count+=1;

    return count/len(test_x)

def test_eval_pokemon_battle():
    return eval_pokemon_battle_prediction(test_x,test_y,lambda x : randomForestClassifier(train_x,train_y,x))

"""
dict["has advantage ?"] = advantage
        dict["is immune ?"] = immune
        dict["gagné ?"] = train_y[i]
        """


#test_eval_pokemon_battle Decision ==> 0.4592106316547914
#test_eval_pokemon_battle Random ==>
# Decision : 1 ==> 5 min, k = 2 ==>0.46354624670237426 , k = 5 ==>0.46354624670237426, k = 10 ==> 0.46354624670237426
# Random : 1 ==> ? min, k = 2 ==>
def test_cross_validation_pokemon_battle():
    erreur = 0
    total = 0

    #Utilisation du KFold pour avoir 5 permutations du fichier train_x et train_y
    X = np.array(train_x)
    Y = np.array(train_y)
    kf = KFold(n_splits=2)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        #Parcourir X_test et comparer le resultat obtenu avec untrained_classifier avec la réponse puis calcule le nombre d'eerreur
        for i in range(len(X_test)) :
            total +=1
            if randomForestClassifier(train_x,train_y,X_test[i]) != Y_test[i] :
                erreur += 1
        print(erreur)
    print(erreur)
    print(total)
    #Retourne le pourcentage d'erreur
    return erreur/total
