import csv
import math
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

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

#Diviser un fichier en deux fichier : fichier_x et fichier_y
def split_lines(input, seed, output1, output2):
  random.seed(seed)
  output1 = open(output1,'a')
  output1.truncate(0)
  output2 = open(output2,'a')
  output2.truncate(0)
  avoidHeader = 0

  for line in open(input, 'r').readlines():
      if avoidHeader != 0:
        if (random.random() < 0.8):
            write = output1;
        else:
            write = output2;
        write.write(line);
      else:
          avoidHeader+=1


#Extraire les données d'un fichier (combats.csv) sous forme (X,Y)
#X est un tableau contenant les deux pokemons
#Y est un tableau de boolean, contenant True si le 1er pokemon gagne, False sinon
def read_data(filename):
  X = []
  Y = []

  with open(filename) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for line in csv_reader:
          X.append(list(map(int,line[:2])))
          Y.append(line[2] == line[0])

  return (X,Y);

#Diviser le fichier combats.csv en fichier train et test
split_lines('combats.csv',0,'train','test')

#train_raw_x est le tableau des pokemons en combats pour l'apprentissage
#train_raw_y est un tableau de boolean qui indique si le 1er pokémon est le gagnant pour l'apprentissage
train_raw_x,train_raw_y = read_data('train')

#test_raw_x est le tableau des pokemons en combats pour le test
#test_raw_y est un tableau de boolean qui indique si le 1er pokémon est le gagnant pour le test
test_raw_x,test_raw_y = read_data('test')

def winrate(pkm,allBattle,allBattleVictory):
    pkm = int(pkm) if type(pkm) == str else pkm
    count = 0
    win = 0
    for i in range(len(allBattle)):
        if pkm == allBattle[i][0]:
            if allBattleVictory[i]:
                win+=1
            count+=1
        elif pkm == allBattle[i][1]:
            if not allBattleVictory[i]:
                win+=1
            count+=1
    if count == 0 :return 0.0
    return win/count

#Extraire les données d'un fichier (pokemon.csv)
#X est un tableau contenant tous les colonnes du fichier sauf la colonne 'Development stage'
def read_data_pokemon(filename):
    X = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader,None)
        X.append([])
        for line in csv_reader:
            pokemon = line[1:]
            del pokemon[-2]
            pokemon.append(winrate(line[0],train_raw_x,train_raw_y))
            X.append(pokemon)

    return X

#Tableau pokemon contenant les caractéristiques des pokémons
pokemon = read_data_pokemon('pokemon.csv')
def getPokemonName(pkm):
    return pokemon[pkm][0]

#Retourne le coefficient d'efficacité du type1 sur le type2 d'après le typeTable
# 2.0, si le type1 est super efficace sur le type2
# 0.5, si le type1 est pas efficace sur le type2
# 0.0, si le type2 est immunisé contre le type1
# 1.0, sinon
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


# Permet de savoir si le type à un avantage par rapport aux types adverse
#typeAttack est un type, tandis que typeDefense est un tableau de type
#Retourne le damageMultiplier
def isTypeAdvantage(typeAttack,typeDefense):
    damageMultiplier = 1.0 * istypeEffective(typeAttack,typeDefense[0])
    damageMultiplier = damageMultiplier * (istypeEffective(typeAttack,typeDefense[1]) if (len(typeDefense) == 2) & (typeDefense[1] != '') else 1)
    return damageMultiplier


#Permet de savoir si le type d'un pokemon est avantagé ou immunisé face au type d'un autre pokemon
#type1 et type2 sont des tableaux de type de pokemon
#return isAdvantaged, isImmune
def doubleTypeAdvantage(type1,type2):
    advantage1= isTypeAdvantage(type1[0],type2)
    advantage2= isTypeAdvantage(type1[1],type2) if (len(type1) > 1) & (type1[1] != '') else 0
    result = max(advantage1,advantage2)
    return result


#Renvoie des boolean (isAdvantaged, isImmune) permet de savoir si le pkm1 est un TypeAdvantage face au pkm2,
def typeBattle(pkm1,pkm2):
    typePkm1 = pokemon[pkm1][1:3]   #tableau des types du pokemon1
    typePkm2 = pokemon[pkm2][1:3]   #tableau des types du pokemon2
    return doubleTypeAdvantage(typePkm1,typePkm2)


#Faire la somme des elements d'une table
def sumInTable(tab):
    count = 0
    for elt in tab:
        count += elt
    return count

#Renvoie les stats de base du pokémon
def getBaseStats(pkm):
    return list(map(lambda x:int(x), pkm[3:9]))


#Formule pour avoir la stat d'HP selon le level 50, et IV 31 EV 0
def HP(hp):
    return math.floor(0.01 * (2.0 * hp + 31) * 50) + 50 + 10


#Formule pour avoir les autres stats selon le level 50, IV 31 EV 0 Nature Neutre
def otherStat(stat):
    return math.floor((((2.0*stat + 31) * 50)/ 100) +5)


#Renvoie les stats d'un pokémon au niv 50, IV 50 EV 0 Nature Neutre
def getStats(baseStat):
    stats = []
    stats.append(HP(baseStat[0]))
    for i in baseStat[1:7]:
        stats.append(otherStat(i))
    return stats

#Fonction principale pour le training d'arbre de decision
#Elle convertit les données des combats, en une table décrivant les situations entre les deux pokémons
#Exemple: Le type était-il plus avantageux ? Ses stats étaient-ils meilleures ?
def tableDecision(train_x,train_y):
    table_x = []    #Tableaux de table_x_elt
    table_y = []    #Tableaux des resultats (train_y ??=

    for i in range(len(train_x)):
        # tableau = [isAdvantaged,isImmune,HP1 > HP2,Attack1 > Attack2,Defense1 > Defense2,Sp.Atk1 > Sp.Stk2, Sp.Df1 > Sp.Df2,Speed1 > Speed2, SommeStat1 > SommeStat2]
        table_x_elt = []

        damageMultiplier = typeBattle(train_x[i][0],train_x[i][1])
        table_x_elt.append(damageMultiplier)

        #Récupération des données des pokémons via la table pokémon
        pokemon1 = pokemon[train_x[i][0]]
        pokemon2 = pokemon[train_x[i][1]]

        #Tableaux des stats des pokémons
        statsP1 = getBaseStats(pokemon1)
        statsP2 = getBaseStats(pokemon2)

        #On compare les statistiques et la somme des statistiques des pokemon entre eux
        #True si les stats du pkm1 sont supérieur au pkm2
        for j in range (len(statsP1)):
            table_x_elt.append(statsP1[j])
            table_x_elt.append(statsP2[j])
            table_x_elt.append(statsP1[j] > statsP2[j])

        sumStatsP1 = sumInTable(statsP1)
        sumStatsP2 = sumInTable(statsP2)   
        table_x_elt.append(sumStatsP1)
        table_x_elt.append(sumStatsP2)
        table_x_elt.append(sumStatsP1 > sumStatsP2)

        winRatepkm1 = pokemon1[-1]
        winRatepkm2 = pokemon2[-1]
        table_x_elt.append(winRatepkm1)
        table_x_elt.append(winRatepkm2)
        table_y.append(train_y[i])
        table_x.append(table_x_elt)

    return table_x,train_y


#train_x est un tableau de tableau de boolean contenant les caractéristiques du combats de l'apprentissage
#train_y est un tableau de boolean contenant les resultats de l'apprentissage
train_x,train_y = tableDecision(train_raw_x,train_raw_y)

#train_x est un tableau de tableau de booleancontenant les caractéristiques du combats du test
#train_y est un tableau de boolean contenant les resultats du test
test_x,test_y = tableDecision(test_raw_x,test_raw_y)

#Random Forest:


#Fonction pour l'arbre de décision
def eval_Random_Forest(train_x,train_y,X,y,k):
    clf = RandomForestClassifier(n_estimators=250)
    clf.fit(train_x,train_y)
    return clf.score(X,y)


#Evaluation sur le fichier test
def eval_pokemon_battle_prediction(test_x,test_y,classifier):
    count = 0;
    for i in range(len(test_x)):
      if (classifier(test_x[i]) != test_y[i]):
          count+=1;

    return count/len(test_x)


##Retourne le pourcentage d'erreur avec la méthode KFold
def test_cross_validation_pokemon_battle(k):
    total = 0   #nombre total d'apprentissage
    meanSum = 0
    X = np.array(train_x)
    Y = np.array(train_y)
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        meanSum += round(eval_Random_Forest(X_train,Y_train,X_test,Y_test,k) * 100,2)
        total+=1
    return meanSum/total

def test_find_best_k():
    k = [100,150,200,250,300,350,400] #On pioche 10 valeurs entre 1 et len(train_x)
    print(k)

    min_list = []   #liste de tous les valeurs de cross_validation
    min_indice = 0  #indice de la plus petit valeur

    #Calculer le pourcentage d'erreur en fonction de k
    for val in k:
        val = test_cross_validation_pokemon_battle(val)
        print(val)
        min_list.append(val)

    min_list = np.array(min_list)   #Transformer la liste en array

    #Retourne le meileur K ,celui qui renvoie la plus petit erreur
    return k[np.argmax(min_list)]
