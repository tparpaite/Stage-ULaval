import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics


########################################################################
# Deap / Tensorflow run                                                #
########################################################################

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in load.dict_load): 
        err_msg = "Usage : python symbreg_sklearn_linear.py data_name\n"
        err_msg += "Jeux de donnees disponibles : "
        err_msg += str([key for key in load.dict_load.keys()]) + "\n"
        sys.stderr.write(err_msg)
        sys.exit(1)


def main():
    # Chargement des donnees
    usage(sys.argv)
    arg = sys.argv[1]
    run = "load." + load.dict_load[arg]
    dataX, dataY, kfold = eval(run)

    # Aprentissage automatique
    begin = time.time()
    mse, logbook = run(dataX, dataY, kfold)
    runtime = "(done in {:.2f} seconds)".format(time.time() - begin)

    # On sauvegarde le mse en dur
    log_mse = arg + " | MSE : " + str(mse) + " " + runtime + "\n"
    file = open("logbook_mse_sklearnlinear.txt", "a")
    file.write(log_mse)


if __name__ == "__main__":
    main()
