import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics

# Creation des donnees artificielles representant un polynome
polynome_X = []
polynome_y = []

for i in range(-10,10):
    x = i/10.0
    y = x**4 + x**3 + x**2 + x
    polynome_X.append([x])
    polynome_y.append([y])

# Convertion en numpy array
polynome_X = np.array(polynome_X)
polynome_y = np.array(polynome_y)

# Create linear regression object
regr = linear_model.LinearRegression()

# Using 5-fold-cross validation
predicted = cross_validation.cross_val_predict(regr, polynome_X, polynome_y, cv=5)
mse = metrics.mean_squared_error(polynome_y, predicted)

# Fitness
print("MSE : %0.2f " % mse)

# Plot outputs
fig, ax = plt.subplots()
ax.scatter(polynome_y, predicted)
ax.plot([polynome_y.min(), polynome_y.max()], [polynome_y.min(), polynome_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
