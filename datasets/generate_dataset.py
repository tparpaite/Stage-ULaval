##########################################################################
# generate_dataset.py                                                    #
# Ce module permet de generer des jeux de donnees de maniere synthetique #
##########################################################################

import sys
import numpy as np
import random as rd

from sklearn import datasets

# Chemin relatif jusqu'aux dataset
PREFIX = "../../datasets/"

# Liste des jeux de donnees qu'on peut generer
dataset_list = { 'polynome', 'keijzer_6', 'nguyen_7', 'pagie_1', 'vladislavleva_4', 'boston' }


def generate_polynome():
    dataset = []

    for i in range(-10, 10):
        x = i / 10.0
        y = x**4 + x**3 + x**2 + x
        dataset.append([x, y])

    dataset = np.array(dataset)
    
    # Sauvegarde du dataset en dur au format csv
    path = PREFIX + "polynome.csv"
    np.savetxt(path, dataset, delimiter=',', fmt='%10.5f')
    print "polynome.csv successfully generated"


def generate_keijzer_6():
    dataset = []

    sum = 0

    for i in range(1, 200):
        x = i
        sum += 1.0 / x
        dataset.append([x, sum])

    dataset = np.array(dataset)
        
    # Sauvegarde du dataset en dur au format csv
    path = PREFIX + "keijzer_6.csv"
    np.savetxt(path, dataset, delimiter=',', fmt='%10.5f')
    print "keijzer_6.csv successfully generated"


def generate_nguyen_7():
    dataset = []

    # Generation des inputs
    dataX = np.random.uniform(0, 2, size=1000)

    # Calcul des outputs
    for x in dataX:
        y = np.log(x + 1) + np.log(x ** 2 + 1)
        dataset.append([x, y])

    dataset = np.array(dataset)
        
    # Sauvegarde du dataset en dur au format csv
    path = PREFIX + "nguyen_7.csv"
    np.savetxt(path, dataset, delimiter=',', fmt='%10.5f')
    print "nguyen_7.csv successfully generated"


def generate_pagie_1():
    dataset = []

    # Generation des inputs
    dataX = np.random.uniform(-5, 5, size=1000)
    dataY = np.random.uniform(-5, 5, size=1000)

    # Calcul des outputs
    for x, y in zip(dataX, dataY):
        z = 1 / (1 + x ** (-4)) + 1 / (1 + y ** (-4))
        dataset.append([x, y, z])

    dataset = np.array(dataset)
        
    # Sauvegarde du dataset en dur au format csv
    path = PREFIX + "pagie_1.csv"
    np.savetxt(path, dataset, delimiter=',', fmt='%10.5f')
    print "pagie_1.csv successfully generated"


def generate_vladislavleva_4():
    dataset = []

    # Generation des inputs
    dataX = [None] * 5
    for i in range(5):
        dataX[i] = np.random.uniform(-0.25, 6.05, 1000)

    # On transpose la matrice : 1024 x 5
    dataX = np.array(dataX)
    dataX = dataX.T
    
    # Calcul des outputs
    for x_array in dataX:
        sum = 0
        line = []

        for x in x_array:
            sum += (x - 3) ** 2
            line.append(x)

        y = 10 / (5 + sum)
        line.append(y)     
        dataset.append(line)

    dataset = np.array(dataset)
        
    # Sauvegarde du dataset en dur au format csv
    path = PREFIX + "vladislavleva_4.csv"
    np.savetxt(path, dataset, delimiter=',', fmt='%10.5f')
    print "vladislavleva_4.csv successfully generated"


# generate_boston
# Permet de generer le dataset boston au format csv a partir du format initial

def generate_boston():
    dataset = []

    # Recuperation des donnees
    boston = datasets.load_boston()
    dataX = boston.data
    dataY = boston.target

    for x_array, y in zip(dataX, dataY):
        line = []

        for x in x_array:
            line.append(x)

        line.append(y)
        dataset.append(line)

    dataset = np.array(dataset)

    # Sauvegarde du dataset en dur au format csv
    path = PREFIX + "boston.csv"
    np.savetxt(path, dataset, delimiter=',', fmt='%10.5f')
    print "boston.csv successfully generated"


###############
# Main call   #
###############

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in dataset_list): 
        err_msg = "Usage : python generate_dataset.py data_name\n"
        err_msg += "Jeux de donnees generables : "
        err_msg += str([dataset for dataset in dataset_list]) + "\n"
        sys.stderr.write(err_msg)
        sys.exit(1)


def main():
    usage(sys.argv)
    dataset = sys.argv[1]
    run = "generate_" + dataset + "()"
    eval(run)


if __name__ == "__main__":
    main()

