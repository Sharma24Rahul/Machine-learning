#simple Linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('E:\machine-learning\P14-Machine-Learning-AZ-Template-Folder\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 4 - Simple Linear Regression\Simple_Linear_Regression\Salary_Data.csv')

X = dataset.iloc[:, :-1].values # age

y = dataset.iloc[:, 1] # salary

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

#fitting the simple regression on training set it is machine
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict the test result it is learning
y_pred = regressor.predict(X_test)

#visualization the training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience [Training Set]')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

#visualizing the test set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience [Test Set]')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()














