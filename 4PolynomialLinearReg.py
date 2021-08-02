import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

data= pd.read_csv('Position_Salaries.csv')

x= data.iloc[:, 1:-1].values
y=data.iloc[:, -1].values
print(x)

# ct= ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
# x=np.array(ct.fit_transform(x))
# print(x)

#TRAINING THE LINEAR REGRESSION ON THE WHOLE DATASET
regressor=LinearRegression()
regressor.fit(x, y)

#TRAINING THE POLYNOMIAL REGRESSION ON THE WHOLE DATASET
polyregressor= PolynomialFeatures(degree=4)
x_poly= polyregressor.fit_transform(x)
regressor2= LinearRegression()
regressor2.fit(x_poly, y)

#VISUALISING LINEAR REGRESSION RESULT
plt.scatter(x, y, color='red')
plt.plot(x,regressor.predict(x), color='blue')
plt.title("True or Bluf (Linear Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
print(plt.show())

#VISUALISING POLYNOMIAL LINEAR REGRESSION RESULT
plt.scatter(x, y, color='red')
plt.plot(x,regressor2.predict(x_poly), color='blue')
plt.title("True or Bluf (Polynomial Regression)")
plt.xlabel('Position Level')
plt.ylabel('Salary')
print(plt.show())

#PREDICTING NEW VALUES FROM LINEAR REGRESSION
print(regressor.predict([[6.5]]))

#PREDICTING NEW VALUES FROM POLYNOMIAL LINEAR REGRESSION
print(regressor2.predict(polyregressor.fit_transform([[6.5]])))