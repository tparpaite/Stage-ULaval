import numpy as np
import random as rd

from sklearn import datasets
from sklearn import cross_validation as cv

# Chemin relatif jusqu'aux dataset
PREFIX = "../../"

# Dictionnaire faisant le lien entre l'argument de main et la fonction de load
dict_load = {'polynome':'loadPolynome()', 'boston':'loadBoston()', 'airfoil':'loadAirfoil()', 'onlinepop':'loadOnlinepop()', 'compactiv':'loadCompactiv()', 'spacega':'loadSpacega()'}

# GEN
# La fonction generatrice sert a generer les indices
# permettant d'effectuer une 5-fold cross-validation
# On effectue 4 fois ce partitionnement de maniere pseudo-aleatoire 
# Pour cela on utilise la librairie cross_validation de sklearn
# On retourne ensuite ce k-fold
# NB : retourne en fait un generateur


def genIndex(n_elem):
    kf_array = []

    for random_state in [42, 1994, 69, 314]:
        kf_array.append(cv.KFold(n_elem, n_folds=5, shuffle=True, random_state=random_state))
        
    return kf_array


def loadPolynome():
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
    kf_array = genIndex(len(dataX))

    # Temporairement : on ne run que sur un seul 5-fold pour le moment
    return dataX, dataY, kf_array[0]
    

def loadBoston():
    # Recuperation des donnees
    boston = datasets.load_boston()
    dataX = boston.data
    dataY = boston.target

    # On transpose la matrice Y
    dataY = np.reshape(dataY, (len(dataY), -1))
    
    # Generation des indices
    kf_array = genIndex(len(dataX))
    
    # Temporairement : on ne run que sur un seul 5-fold pour le moment
    return dataX, dataY, kf_array[0]


def loadAirfoil():
    # Recuperation des donnees du fichier csv
    path = PREFIX + "res/airfoil/airfoil.dat"
    airfoil = np.genfromtxt(path, delimiter='\t', skip_header=1)
    dataX = airfoil[:, :5]
    dataY = airfoil[:, -1:]

    # Generation des indices
    kf_array = genIndex(len(dataX))
    
    return dataX, dataY, kf_array[0]


def loadOnlinepop():
    # Recuperation des donnees du fichier csv
    path = PREFIX + "res/OnlineNewsPopularity/OnlineNewsPopularity.csv"
    onlinepop = np.genfromtxt(path, delimiter=',', skip_header=1)
    dataX = onlinepop[:, 2:60]
    dataY = onlinepop[:, -1:]

    # Generation des indices
    kf_array = genIndex(len(dataX))
    
    return dataX, dataY, kf_array[0]


def loadCompactiv():
    # Recuperation des donnees du fichier csv
    path = PREFIX + "res/compactiv/compactiv.data"
    compactiv = np.genfromtxt(path, delimiter=' ')
    dataX = compactiv[:, :21]
    dataY = compactiv[:, -1:]

    # Generation des indices
    kf_array = genIndex(len(dataX))

    return dataX, dataY, kf_array[0]


def loadSpacega():
    # Recuperation des donnees du fichier csv
    path = PREFIX + "res/spacega/spacega.csv"
    spacega = np.genfromtxt(path, delimiter=',')
    dataX = spacega[:, 1:]
    dataY = spacega[:, [0]]

    # Generation des indices
    kf_array = genIndex(len(dataX))

    return dataX, dataY, kf_array[0]
