import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
data= pd.read_csv('Position_Salaries.csv')

x= data.iloc[:, 1:-1].values
y=data.iloc[:, -1].values
print(x)
print(y)
y= y.reshape(len(y), 1)
print(y)

fs_x= StandardScaler()
x= fs_x.fit_transform(x)
fs_y= StandardScaler()
y= fs_y.fit_transform(y)

print(x)
print(y)

regressor= SVR(kernel= 'rbf')
regressor.fit(x,y)

#INVERSE SCALING TO PREDICT NEW VALUE
print(fs_y.inverse_transform(regressor.predict(fs_x.transform([[6.5]]))))

plt.scatter(fs_x.inverse_transform(x),fs_y.inverse_transform(y), color='red')
plt.plot(fs_x.inverse_transform(x), fs_y.inverse_transform(regressor.predict(x)))
plt.title('True and Predict (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
