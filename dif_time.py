import pandas as pd
import numpy  as np
from   utils.diffrential_timeseries import _get_first_difference,_get_log_returns
from   fracdiff.sklearn import Fracdiff


df = pd.read_csv('data.csv')
print(df)

#log diffrential 
'''a = _get_log_returns(df)
print(a)'''

#fractali diffrential
'''b = _get_first_difference(df)
print(b)'''

#N order diffrential
fracdiff = Fracdiff(0.5)
dif      = fracdiff.fit_transform(df)
rt       = pd.DataFrame(dif)
print(rt)

