import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics

# Load the boston dataset
boston = datasets.load_boston()
boston_X = boston.data
boston_y = boston.target

# Create linear regression object
regr = linear_model.LinearRegression()

# Using 5-fold-cross validation
predicted = cross_validation.cross_val_predict(regr, boston_X, boston_y, cv=5)
mse = metrics.mean_squared_error(boston_y, predicted)

# Fitness
print("MSE : %0.2f " % mse)

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(boston_y, predicted)
ax.plot([boston_y.min(), boston_y.max()], [boston_y.min(), boston_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

