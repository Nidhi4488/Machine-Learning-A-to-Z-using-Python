import numpy as np
import pandas as pd
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

data= pd.read_csv('50_Startups.csv')
x= data.iloc[:, :-1].values
y= data.iloc[:, -1].values

ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])] , remainder='passthrough')
x= np.array(ct.fit_transform(x))
print(x)

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2,random_state=0)

regressor= LinearRegression()
regressor.fit(x_train, y_train)

y_predict= regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_predict.reshape(len(y_predict),1),y_test.reshape(len(y_test),1)),1))