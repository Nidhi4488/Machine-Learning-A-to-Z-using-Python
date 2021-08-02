import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#importing the dataset
data= pd.read_csv("Data.csv")
x= data.iloc[ : , :-1].values
y= data.iloc[ : , -1].values
print(x)
print(y)
print(data.head())
print(data.tail())
#Taking care of missing value
impute= SimpleImputer( missing_values= np.nan, strategy='mean')
impute.fit(x[:, 1:3])
x[:, 1:3]= impute.transform(x[:, 1:3])
print(x)
#Encoding Categorical data
#One Hot Encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)
#Label encoding
le= LabelEncoder()
y=le.fit_transform(y)
print(y)

#Splitting training and testing dataset
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#Feature scaling
sc= StandardScaler()
x_train[:,3:]=sc.fit_transform(x_train[:,3:])
x_test[:,3:]=sc.transform(x_test[:,3:])
print(x_train)
print(x_test)