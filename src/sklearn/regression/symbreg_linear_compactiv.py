import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

# Recuperation des donnees du fichier csv
path = "../../../res/compactiv/compactiv.data"
compactiv = np.genfromtxt(path, delimiter=' ')
compactiv_X = compactiv[:, :21]
compactiv_y = compactiv[:, [21]]

# Create linear regression object
regr = linear_model.LinearRegression()

# Using 5-fold-cross validation
predicted = cross_validation.cross_val_predict(regr, compactiv_X, compactiv_y, cv=5)
mse = metrics.mean_squared_error(compactiv_y, predicted)

# Fitness
print("MSE : %0.2f " % mse)

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(compactiv_y, predicted)
ax.plot([compactiv_y.min(), compactiv_y.max()], [compactiv_y.min(), compactiv_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
