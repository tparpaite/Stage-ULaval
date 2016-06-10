import numpy as np
from sklearn import datasets
from sklearn import cross_validation

def loadBoston():
    boston = datasets.load_boston()
    trX, teX, trY, teY = cross_validation.train_test_split(boston.data, boston.target, 
                                                           test_size=0.33, random_state=42)
    trY, teY = trY.reshape(len(trY), -1), teY.reshape(len(teY), -1)
    return trX, trY, teX, teY


def loadAirfoil():
    # Recuperation des donnees du fichier csv
    path = "../../../res/airfoil/airfoil.dat"
    airfoil = np.genfromtxt(path, delimiter='\t', skip_header=1)
    airfoil_X = airfoil[:, :5]
    airfoil_y = airfoil[:, -1:]

    trX, teX, trY, teY = cross_validation.train_test_split(airfoil_X, airfoil_y, 
                                                           test_size=0.33, random_state=42)
    trY, teY = trY.reshape(len(trY), -1), teY.reshape(len(teY), -1)
    return trX, trY, teX, teY


def loadOnlinepop():
    # Recuperation des donnees du fichier csv
    path = "../../../res/OnlineNewsPopularity/OnlineNewsPopularity.csv"
    onlinepop = np.genfromtxt(path, delimiter=',', skip_header=1)
    onlinepop_X = onlinepop[:, 2:60]
    onlinepop_y = onlinepop[:, -1:]

    trX, teX, trY, teY = cross_validation.train_test_split(onlinepop_X, onlinepop_y, 
                                                           test_size=0.33, random_state=42)
    trY, teY = trY.reshape(len(trY), -1), teY.reshape(len(teY), -1)
    return trX, trY, teX, teY
