import time
import sys
import os
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import load_utils as load

from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from sklearn import grid_search


######################################
# Optimisation des hyperparametres   #
######################################

def svm_hyperparameters(dataset, dataX, dataY, kf_array):
    filepath = "./hyperparameters/hypers_svm_" + dataset + ".pickle"

    # On regarde si on n'a pas deja calcule les hyperparametres optimaux
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            best_params = pickle.load(f)
        return best_params
        
    ######################################################
    # Debut de la recherche des hyperparametres optimaux #
    ######################################################
        
    best_params = { "C":None, "gamma":None, "mse": sys.float_info.max }

    # On recupere le premier pli
    kfold = list(kf_array[0])
    tr_index, te_index = kfold[0]
    trX, teX = dataX[tr_index], dataX[te_index]
    trY, teY = dataY[tr_index], dataY[te_index]
    

    # Echantillonnage
    # On echantillone de maniere uniforme dans l'espace logarithmique
    # Cela correspond a determiner les indices i pour les utiliser dans l'expression 10^i

    # [-10, 5]
    C_log_sample = np.random.uniform(-10, 6, size=100)

    # [-10, 2]
    G_log_sample = np.random.uniform(-10, 3, size=100)

    # On lance 100 fois l'entrainement du SVM sur le pli train/test avec les echantillons
    for C_log, G_log in zip(C_log_sample, G_log_sample):
        # On sort de l'echelle logarithmique
        C = 10 ** (C_log)
        gamma = 10 ** (G_log)

        # Creation du svr et entrainement
        svr = svm.SVR(kernel='rbf', C=C, gamma=gamma)
        svr.fit(trX, trY)
        
        # Evaluation du mse sur l'ensemble test
        predicted = svr.predict(teX)
        mse = metrics.mean_squared_error(teY, predicted)

        # Sauvegarde des hyperparametres s'ils sont meilleurs
        if mse < best_params["mse"]:
            best_params["C"] = C
            best_params["gamma"] = gamma
            best_params["mse"] = mse

    # On sauvegarde les hyperparametres en dur avec pickle
    with open(filepath, 'wb') as f:
        pickle.dump(best_params, f)

    return best_params


###########################
# Boucle principale SVM   #
###########################

def svm_run(hyperparameters, dataX, dataY, kf_array):
    # On recupere les infos sur les hyperparametres optimaux
    C = hyperparameters["C"]
    gamma = hyperparameters["gamma"]

    # Creation de l'objet SVR
    svr = svm.SVR(kernel='rbf', C=C, gamma=gamma)

    # On boucle sur le 5-fold x4 (cross validation)
    mse_sum = 0
    for kfold in kf_array:
        for tr_index, te_index in kfold:
            trX, teX = dataX[tr_index], dataX[te_index]
            trY, teY = dataY[tr_index], dataY[te_index]
            
            # On entraine notre svm
            svr.fit(trX, trY)

            # Evaluation du mse sur l'ensemble test
            predicted = svr.predict(teX)
            mse = metrics.mean_squared_error(teY, predicted)
            mse_sum += mse

    # On retourne le mse moyen
    return mse_sum / 20
    

###############
# Main call   #
###############

def usage(argv):
    if len(sys.argv) != 2 or not(sys.argv[1] in load.dataset_list): 
        err_msg = "Usage : python symbreg_svm.py data_name\n"
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

    # Recherche des hyperparametres optimaux (ou chargement si deja calcule)
    hyperparameters = svm_hyperparameters(dataset, dataX, dataY, kf_array)

    # Execution
    begin = time.time()
    mse = svm_run(hyperparameters, dataX, dataY, kf_array)
    runtime = "{:.2f} seconds".format(time.time() - begin)

    # On sauvegarde le mse en dur
    log_mse = dataset + " | MSE : " + str(mse) + " | " + runtime + "\n"
    file = open("./logbook/logbook_mse_svm.txt", "a")
    file.write(log_mse)


if __name__ == "__main__":
    main()
