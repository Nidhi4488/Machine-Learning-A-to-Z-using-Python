import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

data= pd.read_csv('Social_Network_Ads.csv')

x= data.iloc[:, :-1].values
y= data.iloc[:, -1].values

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.25, random_state=0)

sc= StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

print(x_train)
print(x_test)

classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
classifier.fit(x_train, y_train)

print(classifier.predict(sc.transform([[30,16000]])))

y_pred= classifier.predict(x_test)
print(np.concatenate((y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)),1))

print(confusion_matrix(y_test, y_pred))
print("The accuracy of the model is:")
print(accuracy_score(y_test,y_pred))