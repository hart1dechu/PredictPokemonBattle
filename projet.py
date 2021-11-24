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
          X.append(line[:2])
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
    pkm = str(pkm) if type(pkm) == int else pkm
    count = 0
    win = 0
    for i in range(len(allBattle)):
        if pkm in allBattle[i][0]:
            if allBattleVictory[i]:
                win+=1
            count+=1
        elif pkm in allBattle[i][1]:
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
    return pokemon[int(pkm)][0]
def isGhost(pkm):
    if 'Ghost' in (pokemon[int(pkm)][1:3]):
        return True
    return False
def isFighting(pkm):
    if 'Fighting' in (pokemon[int(pkm)][1:3]):
        return True
    return False
def battleAnalyzing():
    allBatlle = []
    for i in range(len(train_raw_x)):
        x = []
        pkm1 = train_raw_x[i][0]
        pkm2 = train_raw_x[i][1]
        if (isGhost(pkm1) and isFighting(pkm2)) or (isGhost(pkm2) and isFighting(pkm1)):
            x.append(getPokemonName(pkm1))
            x.append(getPokemonName(pkm2))
            x.append(train_raw_y[i])
        if (len(x) != 0):
            allBatlle.append(x)
    return allBatlle

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
    """result = doubleTypeAdvantage(typePkm1,typePkm2)
    return result > 1, result == 0"""
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


def damageCalculator(atkStat,defStat):
    return ((((2*50)/5) * 40 * (atkStat/defStat)) /50) +2


def supposedDamageDealt(pkm1,pkm2):
    atkPkm1 = damageCalculator(pkm1[1],pkm2[2])
    atkSpePkm1 = damageCalculator(pkm1[3],pkm2[4])
    atkPkm2 = damageCalculator(pkm2[1],pkm1[2])
    atkSpePkm2 = damageCalculator(pkm2[3],pkm1[4])
    return max(atkPkm1,atkSpePkm1),max(atkPkm2,atkSpePkm2)


#Fait une simulation naive d'un combat entre deux pokémon, avec une attaque qui fait 40 de degats
#Renvoie le gagnant
def battleSimulation(pkm1,pkm2):
    statsPokemon1 = getStats(getBaseStats(pkm1))
    statsPokemon2 = getStats(getBaseStats(pkm2))
    hp1 = int(statsPokemon1[0])
    hp2 = int(statsPokemon2[0])
    damagePkm1,damagePkm2 = supposedDamageDealt(statsPokemon1,statsPokemon2)
    damagePkm1 *= doubleTypeAdvantage(pkm1[1:3],pkm2[1:3])
    damagePkm2 *= doubleTypeAdvantage(pkm2[1:3],pkm1[1:3])
    if damagePkm2 == 0:
        return True
    if damagePkm1 == 0:
        return False
    result1 = math.ceil(hp2/damagePkm1)
    result2 = math.ceil(hp1/damagePkm2)
    return result1 <= result2 if statsPokemon1[5] > statsPokemon2[5] else not(result2 <= result1)


#Fonction principale pour le training d'arbre de decision
#Elle convertit les données des combats, en une table décrivant les situations entre les deux pokémons
#Exemple: Le type était-il plus avantageux ? Ses stats étaient-ils meilleures ?
def tableDecision(train_x,train_y):
    table_x = []    #Tableaux de table_x_elt
    table_y = []    #Tableaux des resultats (train_y ??=

    for i in range(len(train_x)):
        # tableau = [isAdvantaged,isImmune,HP1 > HP2,Attack1 > Attack2,Defense1 > Defense2,Sp.Atk1 > Sp.Stk2, Sp.Df1 > Sp.Df2,Speed1 > Speed2, SommeStat1 > SommeStat2]
        table_x_elt = []

        #Ajouter les boolean (isAdvantaged, isImmune)
        """advantage,immune = typeBattle(int(train_x[i][0]),int(train_x[i][1]))
        table_x_elt.append(advantage)
        table_x_elt.append(immune)"""
        damageMultiplier = typeBattle(int(train_x[i][0]),int(train_x[i][1]))
        table_x_elt.append(damageMultiplier)

        #Récupération des données des pokémons via la table pokémon
        pokemon1 = pokemon[int(train_x[i][0])]
        pokemon2 = pokemon[int(train_x[i][1])]

        #Tableaux des stats des pokémons
        statsp1 = getBaseStats(pokemon1)
        statsp2 = getBaseStats(pokemon2)

        #On compare les statistiques et la somme des statistiques des pokemon entre eux
        #True si les stats du pkm1 sont supérieur au pkm2
        for j in range (len(statsp1)):
            table_x_elt.append(statsp1[j] > statsp2[j])
        table_x_elt.append(sumInTable(statsp1))
        table_x_elt.append(sumInTable(statsp2))

        """# attaque1 > defense2
        table_x_elt.append(statsp1[1] > statsp2[2])
        #attaque2 < defense1
        table_x_elt.append(statsp2[1] < statsp1[2])
        #attaqueSp1 > defenseSpe2
        table_x_elt.append(statsp1[3] > statsp2[4])
        #attaqueSpe2 < defenseSpe1
        table_x_elt.append(statsp2[3] < statsp1[4])"""
        naiveBattleSimulator = battleSimulation(pokemon1,pokemon2)
        table_x_elt.append(naiveBattleSimulator)
        winRatepkm1 = pokemon1[-1]
        winRatepkm2 = pokemon2[-1]
        table_x_elt.append(winRatepkm1)
        table_x_elt.append(winRatepkm2)
        table_y.append(train_y[i])
        table_x.append(table_x_elt)

    return table_x,table_y


#train_x est un tableau de tableau de boolean contenant les caractéristiques du combats de l'apprentissage
#train_y est un tableau de boolean contenant les resultats de l'apprentissage
train_x,train_y = tableDecision(train_raw_x,train_raw_y)

#train_x est un tableau de tableau de booleancontenant les caractéristiques du combats du test
#train_y est un tableau de boolean contenant les resultats du test
test_x,test_y = tableDecision(test_raw_x,test_raw_y)


#Fonction pour l'arbre de décision
def eval_DecisionTreeClassifier(train_x,train_y,X,y,k):
    clf = RandomForestClassifier(max_depth=k)
    clf.fit(train_x,train_y)
    return clf.score(X,y)


#Evaluation sur le fichier test
def eval_pokemon_battle_prediction(test_x,test_y,classifier):
    count = 0;
    for i in range(len(test_x)):
      if (classifier(test_x[i]) != test_y[i]):
          count+=1;

    return count/len(test_x)


#Fonction de test simple
#test_eval_pokemon_battle Decision ==> 0.4592106316547914
def test_eval_pokemon_battle():
    return eval_pokemon_battle_prediction(test_x,test_y,lambda x : eval_DecisionTreeClassifier(train_x,train_y,x))


##Retourne le pourcentage d'erreur avec la méthode KFold
def test_cross_validation_pokemon_battle(k):
    erreur = 0  #nombre d'erreur lors de l'apprentissage
    total = 0   #nombre total d'apprentissage
    meanSum = 0
    X = np.array(train_x)
    Y = np.array(train_y)
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        #Parcourir X_test et comparer le resultat obtenu avec la bonne réponse puis calcule le nombre d'eerreur
        """for i in range(len(X_test)) :
            total +=1
            if eval_DecisionTreeClassifier(train_x,train_y,X_test[i],k) != Y_test[i] :
                erreur += 1"""
        meanSum += round(eval_DecisionTreeClassifier(X_train,Y_train,X_test,Y_test,k) * 100,2)
        total+=1

    print(k)
    return meanSum/total


#Retourne un tableau d'entier entre min et max de longueur num
def sampled_range(mini, maxi, num):
  if not num:
    return []
  lmini = math.log(mini)
  lmaxi = math.log(maxi)
  ldelta = (lmaxi - lmini) / (num - 1)
  out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
  out.sort()
  return out


def test_find_best_k():
    k = sampled_range(37,50,20) #On pioche 10 valeurs entre 1 et len(train_x)
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
