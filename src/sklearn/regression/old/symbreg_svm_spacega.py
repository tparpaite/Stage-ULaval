import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics

# Recuperation des donnees du fichier csv
path = "../../../res/spacega/spacega.csv"
spacega = np.genfromtxt(path, delimiter=',')
spacega_X = spacega[:, 1:]
spacega_y = spacega[:, [0]]

# Create linear regression object
svr = svm.SVR(kernel='linear', C=1)

# Using 5-fold-cross validation
predicted = cross_validation.cross_val_predict(svr, spacega_X, spacega_y, cv=5)
mse = metrics.mean_squared_error(spacega_y, predicted)

# Fitness
print("MSE : %0.2f " % mse)

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(spacega_y, predicted)
ax.plot([spacega_y.min(), spacega_y.max()], [spacega_y.min(), spacega_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
