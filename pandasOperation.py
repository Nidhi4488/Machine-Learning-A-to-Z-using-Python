import pandas as pd
import numpy as np

d= pd.date_range('20200929', periods=10)
df= pd.DataFrame(np.random.randn(10,4),index=d, columns=['A','B','C','D'])
print(df)

print(df.mean())
print(df.mean(1))

#creating a series
s= pd.Series([1,2,3,4,np.nan,6,7,8,9,10], index=d).shift(2)
print(s)

print(df.sub(s,axis='index'))

#applying different functions to our data
print(df.apply(np.sum))
print(df.apply(lambda x: x.max()- x.min()))

#value counts for histogramming
print(s.value_counts())

#String method
s1=pd.Series(['my','name',np.nan,'is','nidhi'])
print(s1.str.upper()) #lower
s2=pd.Series(['MY','FATHERS','NAME' ,'IS',np.nan,'DHRUBPRASADGUPTA'])
print(s2.str.lower())

