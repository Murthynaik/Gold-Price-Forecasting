# using streamlit
import numpy as np
import pickle
from pickle import load
import pandas as pd
import streamlit as st 


# load the model from disk:
loaded_model=load(open('holt_model_deploy.sav', 'rb'))


#number_of_days
def forecast_price(number_of_days):
    
    prediction=loaded_model.forecast(number_of_days).round(2)
    #print(prediction)
    #return list(prediction)
    return prediction


def main():
    st.title("Gold Price Forecasting")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Gold Price Predictor ML App </h2>
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    number_of_days = st.text_input("NUMBER_OF_DAYS")
    
    days = 0
    if number_of_days!="":
        days=int(number_of_days)
    result=""
    if st.button("Predict"):
        result=forecast_price(days)
        
        dates=pd.date_range(start='22/12/2021', freq='D', periods=days).date  # taking only date from datetimeindex
        
        result=pd.DataFrame({"Date":dates,"price":result})
        st.subheader("the predicted price for next {} days are :".format(days))
    st.write(result)
   

if __name__=='__main__':
    main()  
    
    
    
    
    
    
    
    