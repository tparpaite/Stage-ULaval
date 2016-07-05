import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm


# Load the boston dataset
boston = datasets.load_boston()
nb_instances = len(boston.data)

# Create linear regression object
regr = linear_model.LinearRegression()

# Split the data into training/testing sets
boston_X_train, boston_X_test, boston_y_train, boston_y_test = cross_validation.train_test_split(boston.data, boston.target, test_size=0.33, random_state=random.randint(0, 1000))

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(boston_X_train, boston_y_train)

# The coefficients
print("Coefficients : ")
print(regr.coef_)

# The mean square error
print("MSE : %.2f"
      % np.mean((boston_y_test - regr.predict(boston_X_test)) ** 2))

# Explained variance score : 1 is perfect prediction
print('Variance score, 1 is perfect prediction : %.2f' % regr.score(boston_X_test, boston_y_test))

# The coefficients
# print('Coefficients: \n', regr.coef_)

# Plot outputs
# fig, ax = plt.subplots()
# ax.scatter(y, predicted)
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()
