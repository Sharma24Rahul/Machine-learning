import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("E:\\machine-learning\\P14-Machine-Learning-AZ-Template-Folder\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\Multiple_Linear_Regression\\50_Startups.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap

X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#multiple regression model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#optimize model

import statsmodels.formula.api as st

import statsmodels.regression.linear_model as sm


X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]

#regressor for new multiple regression

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()
#Backward selection

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()













