import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

# Recuperation des donnees du fichier csv
path = "../../../res/wine_quality/winequality-red.csv"
wine = np.genfromtxt(path, delimiter=';', skip_header=1)
wine_X = wine[:, 0:11]
wine_y = wine[:, -1:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Using 5-fold-cross validation
predicted = cross_validation.cross_val_predict(regr, wine_X, wine_y, cv=5)
mse = metrics.mean_squared_error(wine_y, predicted)

# Fitness
print("MSE : %0.2f " % mse)

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(wine_y, predicted)
ax.plot([wine_y.min(), wine_y.max()], [wine_y.min(), wine_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
