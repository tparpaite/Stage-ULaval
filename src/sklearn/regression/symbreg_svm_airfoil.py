import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics

# Recuperation des donnees du fichier csv
airfoil = np.genfromtxt("../../res/airfoil/airfoil.dat", delimiter='\t', skip_header=1)
airfoil_X = airfoil[:, :5]
airfoil_y = airfoil[:, -1:]

# Create linear regression object
svr = svm.SVR(kernel='linear', C=1)

# Using 5-fold-cross validation
predicted = cross_validation.cross_val_predict(svr, airfoil_X, airfoil_y.ravel(), cv=5)
mse = metrics.mean_squared_error(airfoil_y, predicted)

# Fitness
print("MSE : %0.2f " % mse)

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(airfoil_y, predicted)
ax.plot([airfoil_y.min(), airfoil_y.max()], [airfoil_y.min(), airfoil_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()