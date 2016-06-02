import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

# Recuperation des donnees du fichier csv
onlinepop = np.genfromtxt("../../res/OnlineNewsPopularity/OnlineNewsPopularity.csv", delimiter=',', skip_header=1)
onlinepop_X = onlinepop[:, 2:60]
onlinepop_y = onlinepop[:, -1:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Using 5-fold-cross validation
predicted = cross_validation.cross_val_predict(regr, onlinepop_X, onlinepop_y, cv=5)
mse = metrics.mean_squared_error(onlinepop_y, predicted)

# Fitness
print("MSE : %0.2f " % mse)

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(onlinepop_y, predicted)
ax.plot([onlinepop_y.min(), onlinepop_y.max()], [onlinepop_y.min(), onlinepop_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
