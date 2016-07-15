##########################################################################
# generate_dataset.py                                                    #
# Ce module permet de generer des jeux de donnees de maniere synthetique #
##########################################################################


import sys
import numpy as np
import random as rd

# Chemin relatif jusqu'aux dataset
PREFIX = "../../datasets/"

# Liste des jeux de donnees qu'on peut generer
dataset_list = { 'polynome', 'keijzer-6', 'nguyen-7', 'pagie-1', 'vladislavleva-4' }


def generate_polynome():
    polynome = []

    for i in range(-10, 10):
        x = i / 10.0
        y = x**4 + x**3 + x**2 + x
        polynome.append([x, y])

    polynome = np.array(polynome)

    path = PREFIX + "polynome.csv"
    np.savetxt(path, polynome, delimiter=',', fmt='%10.5f')
    print "polynome.csv successfully generated"


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

