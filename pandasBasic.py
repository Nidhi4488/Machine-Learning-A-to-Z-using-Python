import pandas as pd
import numpy as np
#Creating Series #np.nan for null values
s=pd.Series([1,2,3,np.nan,5,6,7])
print(s)

#Creating dataframe for date
d= pd.date_range('20200929', periods=20)
print(d)

#dataframe with random numbers
df= pd.DataFrame(np.random.randn(20,4),index=d, columns=['A','B','C','D'])
print(df)

#creating dataframe using dictionary with various objects
df1=pd.DataFrame({'A':[1,2,3,4],
                  'B': pd.Timestamp('20200930'),
                  'C':pd.Series(2,index= list(range(4)),dtype=float),
                  'D':np.array([4]*4, dtype=int),
                  'E':'Employee'})
print(df1)

#Creating my own customised dataframe
inventory=pd.DataFrame({'Product':['Maize','Rice','Pulse','Mustard Oil'],
                        'Product Code':['0023','456e3','242f','ghs123'],
                        'Expiry Date':pd.date_range('20210923', periods=4),
                        'Manufacture Date':pd.date_range('20200912', periods=4),
                        'Quantity':np.array([1]*4,dtype=int),
                        'Cost':['230','434','124','341']})
print(inventory)
print(inventory.dtypes)

#view data
print (inventory.head())
print (inventory.tail())
print (inventory.index)
print (inventory.columns)
print (inventory.to_numpy())
print (inventory.describe())
#sorting the dataframe by index
sort=inventory.sort_index(axis=0, ascending=True)
sort1=inventory.sort_index(axis=0, ascending=False)
print(sort)
print(sort1)

#sorting the dataframe by values
sort3=inventory.sort_values(by='Cost',ascending=False)
print(sort3)

#getting value of particular column
print(inventory['Product'])
print(inventory['Expiry Date'])
print(inventory['Manufacture Date'])

#slicing
print(inventory[2:4])

#select data using labels/ pass values by labels
a=df.loc[d[0]]
print (a)
b=inventory.loc[0]
print (b)

#selecting data on a multi axis by label
print(inventory.loc[:,['Product','Expiry Date']])
print(inventory.loc[0:2,['Product','Expiry Date']])
print(inventory.at[0,'Cost'])
print(inventory.iloc[0:4,3])
print(inventory.shape[0]) #for number of rows
print(inventory.shape[1]) #for number of columns

#boolean indexing
print(inventory['Cost']> '200')