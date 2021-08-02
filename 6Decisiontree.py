import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

data=pd.read_csv('Position_Salaries.csv')
x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values
print(x)
regressor= DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

print(regressor.predict([[6.5]]))

x_grid= np.arange(min(x), max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y, color='red')
plt.plot(x_grid,regressor.predict(x_grid), color='blue')
plt.title('True vs Predicted (Decision Tree Regressor')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()