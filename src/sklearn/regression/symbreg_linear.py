import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import load_utils as load

from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics


#########################################
# Boucle principale regression lineaire #
#########################################

def linear_run(dataX, dataY, kf_array):
    # Creation de l'objet permettant d'effectuer la regression lineaire
    regr = linear_model.LinearRegression()

    # On boucle sur le 5-fold x4 (cross validation)
    mse_sum = 0
    for kfold in kf_array:
        for tr_index, te_index in kfold:
            trX, teX = dataX[tr_index], dataX[te_index]
            trY, teY = dataY[tr_index], dataY[te_index]
            
            # On entraine notre svm
            regr.fit(trX, trY)

            # Evaluation du mse sur l'ensemble test
            predicted = regr.predict(teX)
            mse = metrics.mean_squared_error(teY, predicted)
            mse_sum += mse

    # On retourne le mse moyen
    return mse_sum / 20
    

###############
# Main call   #
###############

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in load.dataset_list): 
        err_msg = "Usage : python symbreg_linear.py data_name\n"
        err_msg += "Jeux de donnees disponibles : "
        err_msg += str([dataset for dataset in load.dataset_list]) + "\n"
        sys.stderr.write(err_msg)
        sys.exit(1)


def main():
    # Chargement des donnees
    usage(sys.argv)
    dataset = sys.argv[1]
    run = "load.load_" + dataset + "()"
    dataX, dataY, kf_array = eval(run)

    # Execution
    begin = time.time()
    mse = linear_run(dataX, dataY, kf_array)
    runtime = "{:.2f} seconds".format(time.time() - begin)

    # On sauvegarde le mse en dur
    log_mse = dataset + " | MSE : " + str(mse) + " | " + runtime + "\n"
    file = open("./logbook/logbook_mse_linear.txt", "a")
    file.write(log_mse)


if __name__ == "__main__":
    main()
