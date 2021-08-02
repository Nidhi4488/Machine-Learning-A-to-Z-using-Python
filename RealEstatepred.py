import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

data= pd.read_csv("realestatedata.csv")
# x=data.iloc[:,:-1].values
# y=data.iloc[:,-1].values
print(data.head())
print(data.describe())
print(data.info())
data.hist(bins=5, figsize=(30,20), color='yellow', rwidth= 0.95)
plt.show()

train_set,test_set= train_test_split(data, test_size=0.2, random_state=0)
print(len(train_set))
print(len(test_set))

split= StratifiedShuffleSplit()


# print(len(y_train))
# print(len(y_test))
