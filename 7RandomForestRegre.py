import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data=pd.read_csv('Position_Salaries.csv')
x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values
print(x)
regressor= RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x,y)

print(regressor.predict([[6.5]]))

x_grid= np.arange(min(x), max(x),0.1)
# print(x_grid)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y, color='red')
plt.plot(x_grid,regressor.predict(x_grid), color='blue')
plt.title('True vs Predicted (Random forest Tree Regressor')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# print(x_grid)

#EVALUATING THE MODEL PERFORMANCE
print(r2_score(y,regressor.predict(x))) #r2_score(y_test, Y_predict)