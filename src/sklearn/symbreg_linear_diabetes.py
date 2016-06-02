import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm


# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
# Classical way
# regr.fit(diabetes.data, diabetes.target)

# Using 5-fold-cross validation
scores = cross_validation.cross_val_score(regr, diabetes.data, diabetes.target, cv=5)

# The coefficients
# print('Coefficients: \n', regr.coef_)

# Accuracy
print("Accuracy : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Plot outputs
# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
#          linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
