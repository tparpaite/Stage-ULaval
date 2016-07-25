import time
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../../')
from datasets import load_utils as load
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics


# Chemin relatif du repertoire logbook
LOGBOOK_PATH = "../../../stats/logbook/"


#########################################
# Boucle principale regression lineaire #
#########################################

def linear_run(dataX, dataY, kf_array):
    # Creation de l'objet permettant d'effectuer la regression lineaire
    regr = linear_model.LinearRegression()

    # On boucle sur le 5-fold x4 (cross validation)
    stats_dic = { 'mse_train_array': [], 'mse_test_array': [] }

    for kfold in kf_array:
        for tr_index, te_index in kfold:
            trX, teX = dataX[tr_index], dataX[te_index]
            trY, teY = dataY[tr_index], dataY[te_index]
            
            # On entraine notre svm
            regr.fit(trX, trY)
        
            # Evaluation du mse
            predicted_train = regr.predict(trX)
            mse_train = metrics.mean_squared_error(trY, predicted_train)
            predicted_test = regr.predict(teX)
            mse_test = metrics.mean_squared_error(teY, predicted_test)

            # On recupere les informations dans le dictionnaire de stats
            stats_dic['mse_train_array'].append(mse_train)
            stats_dic['mse_test_array'].append(mse_test)

    # On retourne le dictionnaire contenant les informations sur les stats
    return stats_dic
    

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
    stats_dic = linear_run(dataX, dataY, kf_array)
    runtime = "{:.2f} seconds".format(time.time() - begin)

    # Sauvegarde du dictionnaire contenant les stats
    logbook_filename = LOGBOOK_PATH + "logbook_stats/logbook_stats_linear_" + dataset + ".pickle"
    pickle.dump(stats_dic, open(logbook_filename, 'w'))

    # Sauvegarde du mse
    mse_train_mean = np.mean(stats_dic['mse_train_array'])
    mse_test_mean = np.mean(stats_dic['mse_test_array'])
    log_mse = dataset + " | MSE (train) : " + str(mse_train_mean) + " | MSE (test) : " + str(mse_test_mean) 
    log_mse += " | " + runtime + "\n"
    logbook_filename = LOGBOOK_PATH + "logbook_mse/logbook_mse_linear.txt"
    fd = open(logbook_filename, 'a')
    fd.write(log_mse)


if __name__ == "__main__":
    main()
