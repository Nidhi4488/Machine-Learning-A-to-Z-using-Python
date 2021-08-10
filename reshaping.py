import pandas as pd
import numpy as np

#creating tuples
my_tuple=list(zip(*[[1,2,3,4,5,17,18,19],[11,12,13,6,7,8,9,10]]))
#multiindexing
index=pd.MultiIndex.from_tuples(my_tuple, names=['First','Second'])
df=pd.DataFrame(np.random.randn(8,2), index=index, columns=['A','B'])
print(df)
df2= df[:4]
print(df2)

#stacking
a=df2.stack()
print(a)
print(df.stack())
print(a.unstack())

#pivot table
df3=pd.DataFrame({'A':['a','b','c','d']*3,
                  'B':['A','B','C']*4,
                  'C':['P','P','P','Q','Q','Q']*2,
                  'D':np.random.randn(12),
                  'E':np.random.randn(12)})
print(df3)
print(pd.pivot_table(df3, values='D',index=['A','B'],columns=['C']))
print(pd.pivot_table(df3, values=['D','E',],index=['A','B'],columns=['C']))