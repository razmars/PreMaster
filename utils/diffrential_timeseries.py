import pandas as pd
import numpy  as np 
from  fracdiff.sklearn import Fracdiff


def _get_first_difference(data_df):
    d = data_df.pct_change().fillna(0)
    return d.iloc[1:]

def _get_log_returns(data_df):
    d = np.log(data_df / data_df.shift(1)).fillna(0)
    return d.iloc[1:]

def _get_dif_order(df,order=1):
    fracdiff = Fracdiff(order)
    return fracdiff.fit_transform(df)

