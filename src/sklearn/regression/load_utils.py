import numpy as np
import random as rd

from sklearn import datasets
from sklearn import cross_validation as cv

# Chemin relatif jusqu'aux dataset
PREFIX = "../../../"

# Dictionnaire faisant le lien entre l'argument de main et la fonction de load
dict_load = {'polynome':'loadPolynome()', 'boston':'loadBoston()', 'airfoil':'loadAirfoil()', 'onlinepop':'loadOnlinepop()', 'compactiv':'loadCompactiv()', 'spacega':'loadSpacega()'}


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

    # On transpose la matrice Y
    dataY = np.reshape(dataY, -1)

    return dataX, dataY
    

def loadBoston():
    # Recuperation des donnees
    boston = datasets.load_boston()
    dataX = boston.data
    dataY = boston.target

    return dataX, dataY


def loadAirfoil():
    # Recuperation des donnees du fichier csv
    path = PREFIX + "res/airfoil/airfoil.dat"
    airfoil = np.genfromtxt(path, delimiter='\t', skip_header=1)
    dataX = airfoil[:, :5]
    dataY = airfoil[:, -1:]

    # On transpose la matrice Y
    dataY = np.reshape(dataY, -1)
    
    return dataX, dataY


def loadOnlinepop():
    # Recuperation des donnees du fichier csv
    path = PREFIX + "res/OnlineNewsPopularity/OnlineNewsPopularity.csv"
    onlinepop = np.genfromtxt(path, delimiter=',', skip_header=1)
    dataX = onlinepop[:, 2:60]
    dataY = onlinepop[:, -1:]

    # On transpose la matrice Y
    dataY = np.reshape(dataY, -1)
    
    return dataX, dataY


def loadCompactiv():
    # Recuperation des donnees du fichier csv
    path = PREFIX + "res/compactiv/compactiv.data"
    compactiv = np.genfromtxt(path, delimiter=' ')
    dataX = compactiv[:, :21]
    dataY = compactiv[:, -1:]
    
    # On transpose la matrice Y
    dataY = np.reshape(dataY, -1)

    return dataX, dataY


def loadSpacega():
    # Recuperation des donnees du fichier csv
    path = PREFIX + "/res/spacega/spacega.csv"
    spacega = np.genfromtxt(path, delimiter=',')
    dataX = spacega[:, 1:]
    dataY = spacega[:, [0]]

    # On transpose la matrice Y
    dataY = np.reshape(dataY, -1)
    
    return dataX, dataY
