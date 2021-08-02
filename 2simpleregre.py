import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset= pd.read_csv("Salary_Data.csv")
x= dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=1/3,random_state=0)

# Model train
lr= LinearRegression()
lr.fit(x_train, y_train)

#predicting test dataset
y_predict= lr.predict(x_test)

#visualising training dataset
plt.scatter(x_train,y_train, color= 'red')
plt.plot(x_train, lr.predict(x_train), color='blue')
plt.title('Years of Experience vs Salary (Training Dataset)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
print(plt.show())

#visualising testing dataset
plt.scatter(x_test,y_test, color= 'orange')
plt.plot(x_train, lr.predict(x_train), color='green')
plt.title('Years of Experience vs Salary (Testing Dataset)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
print(plt.show())