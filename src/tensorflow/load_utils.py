import numpy as np
from sklearn import datasets
from sklearn import cross_validation as cv

# Chemin relatif jusqu'aux dataset
PREFIX = "../../"

# Liste des jeux de donnees
dataset_list = {'polynome', 'boston', 'airfoil', 'onlinepop', 'compactiv', 'spacega'}

# GEN
# La fonction generatrice sert a generer les indices
# permettant d'effectuer une 5-fold cross-validation
# On effectue 4 fois ce partitionnement de maniere pseudo-aleatoire 
# Pour cela on utilise la librairie cross_validation de sklearn
# On retourne ensuite ce k-fold
# NB : retourne en fait un generateur

def gen_index(n_elem):
    kf_array = []

    for random_state in [42, 1994, 69, 314]:
        kf_array.append(cv.KFold(n_elem, n_folds=5, shuffle=True, random_state=random_state))
        
    return kf_array


def load_polynome():
    # Creation des donnees artificielles representant un polynome
    dataX = []
    dataY = []

    for i in range(-10,10):
        x = i/10.0
        y = x**4 + x**3 + x**2 + x
        dataX.append([x])
        dataY.append([y])

    # Convertion en numpy array
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # Generation des indices
    kf_array = gen_index(len(dataX))

    # Temporairement : on ne run que sur un seul 5-fold pour le moment
    return dataX, dataY, kf_array


def load_boston():
    # Recuperation des donnees
    boston = datasets.load_boston()
    dataX = boston.data
    dataY = boston.target

    # On transpose la matrice Y
    dataY = np.reshape(dataY, (len(dataY), -1))
    
    # Generation des indices
    kf_array = gen_index(len(dataX))
    
    # Temporairement : on ne run que sur un seul 5-fold pour le moment
    return dataX, dataY, kf_array


def load_airfoil():
    # Recuperation des donnees du fichier csv
    path = PREFIX + "datasets/airfoil/airfoil.dat"
    airfoil = np.genfromtxt(path, delimiter='\t', skip_header=1)
    dataX = airfoil[:, :5]
    dataY = airfoil[:, -1:]

    # Generation des indices
    kf_array = gen_index(len(dataX))
    
    return dataX, dataY, kf_array


def load_onlinepop():
    # Recuperation des donnees du fichier csv
    path = PREFIX + "datasets/OnlineNewsPopularity/OnlineNewsPopularity.csv"
    onlinepop = np.genfromtxt(path, delimiter=',', skip_header=1)
    dataX = onlinepop[:, 2:60]
    dataY = onlinepop[:, -1:]

    # Generation des indices
    kf_array = gen_index(len(dataX))
    
    return dataX, dataY, kf_array


def load_compactiv():
    # Recuperation des donnees du fichier csv
    path = PREFIX + "datasets/compactiv/compactiv.data"
    compactiv = np.genfromtxt(path, delimiter=' ')
    dataX = compactiv[:, :21]
    dataY = compactiv[:, -1:]

    # Generation des indices
    kf_array = gen_index(len(dataX))

    return dataX, dataY, kf_array


def load_spacega():
    # Recuperation des donnees du fichier csv
    path = PREFIX + "datasets/spacega/spacega.csv"
    spacega = np.genfromtxt(path, delimiter=',')
    dataX = spacega[:, 1:]
    dataY = spacega[:, [0]]

    # Generation des indices
    kf_array = gen_index(len(dataX))

    return dataX, dataY, kf_array
