#######################################################################
# load_utils.py                                                       #
# Ce module permet de charger des donnees existantes                  #
# Il permet egalement de creer un generateur d'indices pour realiser  #
# une 5-fold x4 cross-validation                                      #
#######################################################################

import os
import numpy as np
import random as rd

from sklearn import cross_validation as cv

# Chemin relatif du repertoire datasets (lors de l'execution)
PREFIX = os.path.dirname(__file__) + '/'

# Liste des jeux de donnees
dataset_list = { 'polynome', 'keijzer_6', 'nguyen_7', 'pagie_1', 'vladislavleva_4',
                 'boston', 'bioavailability', 'airfoil', 'onlinepop', 'compactiv', 'spacega' }


# gen_index
# La fonction generatrice sert a generer les indices
# permettant d'effectuer une 5-fold cross-validation
# On effectue 4 fois ce partitionnement de maniere pseudo-aleatoire 
# Pour cela on utilise la librairie cross_validation de sklearn
# On retourne ensuite ce tableau de k-fold
# NB : retourne en fait un tableau de generateurs

def gen_index(n_elem):
    kf_array = []

    for random_state in [42, 1994, 69, 314]:
        kf_array.append(cv.KFold(n_elem, n_folds=5, shuffle=True, random_state=random_state))
        
    return kf_array


#################################
# Jeux de donnees synthetiques  #
#################################

def load_polynome():
    # Recuperation des donnees
    path = PREFIX + "polynome.csv"
    dataset = np.genfromtxt(path, delimiter=',')
    dataX = dataset[:, [0]]
    dataY = dataset[:, [1]]

    # Generation des indices
    kf_array = gen_index(len(dataX))

    # Temporairement : on ne run que sur un seul 5-fold pour le moment
    return dataX, dataY, kf_array[0]


def load_keijzer_6():
    # Recuperation des donnees
    path = PREFIX + "keijzer_6.csv"
    dataset = np.genfromtxt(path, delimiter=',')
    dataX = dataset[:, [0]]
    dataY = dataset[:, [1]]

    # Generation des indices
    kf_array = gen_index(len(dataX))

    return dataX, dataY, kf_array[0]


def load_nguyen_7():
    # Recuperation des donnees
    path = PREFIX + "nguyen_7.csv"
    dataset = np.genfromtxt(path, delimiter=',')
    dataX = dataset[:, [0]]
    dataY = dataset[:, [1]]
    
    # Generation des indices
    kf_array = gen_index(len(dataX))

    return dataX, dataY, kf_array[0]


def load_pagie_1():
    # Recuperation des donnees
    path = PREFIX + "pagie_1.csv"
    dataset = np.genfromtxt(path, delimiter=',')
    dataX = dataset[:, :2]
    dataY = dataset[:, -1:]
    
    # Generation des indices
    kf_array = gen_index(len(dataX))

    return dataX, dataY, kf_array[0]


def load_vladislavleva_4():
    # Recuperation des donnees
    path = PREFIX + "vladislavleva_4.csv"
    dataset = np.genfromtxt(path, delimiter=',')
    dataX = dataset[:, :5]
    dataY = dataset[:, -1:]

    # Generation des indices
    kf_array = gen_index(len(dataX))

    return dataX, dataY, kf_array[0]
    

##############################
# Jeux de donnees reels      #
##############################

def load_boston():
    # Recuperation des donnees
    path = PREFIX + "boston.csv"
    dataset = np.genfromtxt(path, delimiter=',')
    dataX = dataset[:, :13]
    dataY = dataset[:, -1:]

    # Generation des indices
    kf_array = gen_index(len(dataX))
    
    return dataX, dataY, kf_array[0]


def load_bioavailability():
    # Recuperation des donnees
    path = PREFIX + "bioavailability.csv"
    dataset = np.genfromtxt(path, delimiter='\t')
    dataX = dataset[:, :241]
    dataY = dataset[:, -1:]

    # Generation des indices
    kf_array = gen_index(len(dataX))
    
    return dataX, dataY, kf_array[0]


def load_airfoil():
    # Recuperation des donnees
    path = PREFIX + "airfoil.data"
    airfoil = np.genfromtxt(path, delimiter='\t', skip_header=1)
    dataX = airfoil[:, :5]
    dataY = airfoil[:, -1:]
    
    # Generation des indices
    kf_array = gen_index(len(dataX))
    
    return dataX, dataY, kf_array[0]


def load_onlinepop():
    # Recuperation des donnees
    path = PREFIX + "onlinepop.csv"
    onlinepop = np.genfromtxt(path, delimiter=',', skip_header=1)
    dataX = onlinepop[:, 2:60]
    dataY = onlinepop[:, -1:]

    # Generation des indices
    kf_array = gen_index(len(dataX))
    
    return dataX, dataY, kf_array[0]


def load_compactiv():
    # Recuperation des donnees
    path = PREFIX + "compactiv.data"
    compactiv = np.genfromtxt(path, delimiter=' ')
    dataX = compactiv[:, :21]
    dataY = compactiv[:, -1:]

    # Generation des indices
    kf_array = gen_index(len(dataX))

    return dataX, dataY, kf_array[0]


def load_spacega():
    # Recuperation des donnees
    path = PREFIX + "spacega.csv"
    spacega = np.genfromtxt(path, delimiter=',')
    dataX = spacega[:, 1:]
    dataY = spacega[:, [0]]

    # Generation des indices
    kf_array = gen_index(len(dataX))

    return dataX, dataY, kf_array[0]
