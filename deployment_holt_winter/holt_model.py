#model training:

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pickle import dump,load

df=pd.read_csv("Gold_data.csv")

df.drop_duplicates(inplace=True)



from statsmodels.tsa.holtwinters import ExponentialSmoothing 
holt = ExponentialSmoothing(df["price"],seasonal="mul",trend="add",seasonal_periods=86).fit()


# save the model to disk
dump(holt, open('holt_model_deploy.sav', 'wb'))















