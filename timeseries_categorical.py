import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
date= pd.date_range('20201118', periods=10, freq='W')  #freq= H,W,T,S,M,N
ts= pd.DataFrame(np.random.randn(len(date)), date)
print(ts)
print(ts.to_period())

df= pd.DataFrame(np.random.randn(100), index= pd.date_range('20201012',periods=100))
df= df.cumsum()
df.plot()
plt.show()

