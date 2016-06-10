import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

# Recuperation des donnees du fichier csv
path = "../../../res/abalone/abalone.data"
abalone = np.genfromtxt(path, delimiter=',')
abalone_X = abalone[:, 1:8]
abalone_y = abalone[:, -1:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Using 5-fold-cross validation
predicted = cross_validation.cross_val_predict(regr, abalone_X, abalone_y, cv=5)
mse = metrics.mean_squared_error(abalone_y, predicted)

# Fitness
print("MSE : %0.2f " % mse)

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(abalone_y, predicted)
ax.plot([abalone_y.min(), abalone_y.max()], [abalone_y.min(), abalone_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
