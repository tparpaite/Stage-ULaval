import matplotlib.pyplot as plt
import numpy as np

from sklearn import grid_search
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
# svr = svm.SVR(kernel='rbf', gamma=0.1)
# svr = svm.SVR(kernel='poly', C=1, degree=2)
svr = svm.SVR(kernel='linear', C=1)

# Using 5-fold-cross validation
predicted = cross_validation.cross_val_predict(svr, boston_X, boston_y, cv=5, n_jobs=-1)
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
