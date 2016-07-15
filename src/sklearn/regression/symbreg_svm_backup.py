import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import load_utils as load

from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from sklearn.grid_search import GridSearchCV


###########################
# Boucle principale SVM   #
###########################

def svm_run(dataX, dataY):
    # Create support vector machine (regression) object
    svr = svm.SVR(kernel='rbf')
    params = {"C":[0.1, 1, 10], "gamma":[0.1, 0.01, 0.001]}
    svr_opti = GridSearchCV(svr, params)

    # Using 5-fold-cross validation
    predicted = cross_validation.cross_val_predict(svr_opti, dataX, dataY, cv=5)
    mse = metrics.mean_squared_error(dataY, predicted)

    return mse
    

###############
# Main call   #
###############

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in load.dict_load): 
        err_msg = "Usage : python symbreg_svm.py data_name\n"
        err_msg += "Jeux de donnees disponibles : "
        err_msg += str([key for key in load.dict_load.keys()]) + "\n"
        sys.stderr.write(err_msg)
        sys.exit(1)


def main():
    # Chargement des donnees
    usage(sys.argv)
    arg = sys.argv[1]
    run = "load." + load.dict_load[arg]
    dataX, dataY = eval(run)

    # Execution
    begin = time.time()
    mse = svm_run(dataX, dataY)
    runtime = "{:.2f} seconds".format(time.time() - begin)

    # On sauvegarde le mse en dur
    log_mse = arg + " | MSE : " + str(mse) + " | " + runtime + "\n"
    file = open("logbook_mse_svm.txt", "a")
    file.write(log_mse)


if __name__ == "__main__":
    main()
