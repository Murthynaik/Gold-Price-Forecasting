#model training:

import pystan
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pickle import dump

df=pd.read_csv("Gold_data.csv")

df.drop_duplicates(inplace=True)


# converting date to timestamp:
df["date"]=pd.to_datetime(df["date"])



# fbprophet needs date column name as ds and other column i.e price as y:
df.columns=['ds','y']


#!pip install prophet
#!pip install pystan~=2.14
#!pip install fbprophet


from fbprophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

prophet=Prophet()
model=prophet.fit(df)



# save the model to disk
dump(model, open('fbp_model_deploy.sav', 'wb'))















