# **Journal du projet**

[Le projet se trouve sur git](git@github.com:hart1dechu/PredictPokemonBattle.git)

## Semaine 1
#### Travail :
- Creation du fichier train et test avec la méthode split().
- Remplissage manuelle d'une case vide du fichier pokemon.csv.
-Modélisation des données du fichier combats.csv sous forme de (X,Y), où :
  - X est un tableau contenant des tableaux des pokémons qui combattent.
  - Y est un tableau de booléan, vaut True si le vainqueur est le 1 pokémon , False sinon.
-Modélisation des donnés du fichier pokemon.csv sous forme de tableau X, où :
  - X contient des tableaux correspondant aux caractéristiques des pokémons sauf la colonne 'Development stage'

#### A voir:
- Le fonctionnement des arbres de décisions de sklearn
- Réfléchir au question possible:
  - Si le pokémon est un pokémon légendaire
  - S'il a un avantage en fonction de ses types
  - S'il a de meilleur statistique


## Semaine 2
####Travail :
- La fonction principale pour le training d'arbre de decision  tableDecision(train_x,train_y), permet de convertir les données des combats, en table décrivant les situations entre les deux pokémons :
  - table_x est tableau de tableau de boolean contenant les information pour l'arbre de décision : [isAdvantaged, isImmune, HP1>HP2, Attack1>Attack2, Defense1>Defense2, Sp.Atk1>Sp.Stk2, Sp.Df1>Sp.Df2, Speed1>Speed2, SommeStat1>SommeStat2]
  - table_y est le tableau des résultats
- Des tests simple avec la méthode test_eval_pokemon_battle() utilisant DecisionTreeClassifier
  - Les premiers test :  
      - max_depth = 2 , 0.4592106316547914 en environ 15 minutes (environ 11 minutes)
- Des tests avec la méthode est_cross_validation_pokemon_battle() utilisant KFold et DecisionTreeClassifier
  - Les premiers test avec max_depth=2
    - k = 5, 0.46354624670237426
    - k = 10,  0.46354624670237426
  - Les derniers test avec max_depth=6 et ajout des statistiques des pokemons:
    - k = 6, 0.05... (environ 15 minutes)

#### A voir:
  - Fonction pour trouver le best k pour max_depth
  - Implémenter quelques questions en plus pour l'arbre de décision



>Mémo : pas oublier de commenter le code
