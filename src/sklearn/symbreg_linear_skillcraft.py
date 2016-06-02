import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics

# Recuperation des donnees du fichier csv
skillcraft = np.genfromtxt("../../res/skillcraft/skillcraft.csv", delimiter=',', skip_header=1)
skillcraft_X = skillcraft[:, 2:]
skillcraft_y = skillcraft[:, [1]]

# Create linear regression object
regr = linear_model.LinearRegression()

# Using 5-fold-cross validation
predicted = cross_validation.cross_val_predict(regr, skillcraft_X, skillcraft_y, cv=5)
mse = metrics.mean_squared_error(skillcraft_y, predicted)

# Fitness
print("MSE : %0.2f " % mse)

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(skillcraft_y, predicted)
ax.plot([skillcraft_y.min(), skillcraft_y.max()], [skillcraft_y.min(), skillcraft_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
