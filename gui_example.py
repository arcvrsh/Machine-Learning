import streamlit as st
import pickle as pkl
import numpy as np


def load_model(path):
    with open(path,'rb') as f:
       return pkl.load(f)

st.title('House Price Predictor')
with st.spinner('Please Wait'):
    feature_scaler = load_model('models/rfr_feature_scaler_house_price.pkl')
    model = load_model('models/rfr_house_price_model.pkl')
    st.success('AI models are loaded')

st.header('Enter the house details')
beds = st.number_input(label='No of Bed',value=1)
baths = st.number_input(label='No of Bathrooms/toilets',value=1)
area = st.number_input(label='Area in Sq.ft',value=1000)
btn = st.button('Get Price (beta)') 

if btn:
    try:
        x = np.array([[beds,baths,area]])
        x = feature_scaler.transform(x)
        result = model.predict(x)[0]
        st.success(result)
        st.write('Price predicted')
    except Exception as e:
        st.error(e)
