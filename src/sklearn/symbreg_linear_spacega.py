import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

# Recuperation des donnees du fichier csv
spacega = np.genfromtxt("../../res/spacega/spacega.csv", delimiter=',')
spacega_X = spacega[:, 1:]
spacega_y = spacega[:, [0]]

# Create linear regression object
regr = linear_model.LinearRegression()

# Using 5-fold-cross validation
predicted = cross_validation.cross_val_predict(regr, spacega_X, spacega_y, cv=5)
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
