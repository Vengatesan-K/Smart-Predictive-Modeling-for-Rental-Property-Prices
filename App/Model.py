import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
import joblib as joblib

st.set_page_config(page_title="House Rent Prediction", page_icon="🏠", layout="wide", initial_sidebar_state="auto")
st.markdown(f'<h1 style="text-align: center;"> House Rent Prediction </h1>', unsafe_allow_html=True)
loaded_model = joblib.load('trained_house_data.pkl')

col1, col2 = st.columns(2)
with col1:
    type = st.selectbox("Type", ['BHK2', 'BHK3', 'BHK1','RK1','BHK4','BHK4PLUS','bhk2','bhk3','1BHK1'])
    type_dict = {'BHK2': 0, 'BHK3': 1, 'BHK1': 2,'RK1':3,'BHK4':4,'BHK4PLUS':5,'bhk2':6,'bhk3':7,'1BHK1':8}
    
    lift_no = st.number_input('Lift Count', 0.0, 10.0, 1.0)
    parking = st.selectbox('Parking', ["Two Wheeler", "Four Wheeler", "Both","None"])
    parking_dict = {'Two Wheeler': 0, 'Four Wheeler': 1, 'Both': 2,'None':3}
    bathroom = st.number_input('Bathroom', 0.0, 10.0, 1.0)
    cupboard = st.number_input('Cupboard', 0.0, 10.0, 1.0)
    
with col2:
    building_type = st.selectbox('Building Type', ["IF", "IH", "AP","GC"])
    building_type_dict = {'IF': 0, 'IH': 1, 'AP': 2,'GC':3}
    balcony = st.number_input('Balcony', 0.0, 10.0, 1.0)
    property_size = st.number_input('Property Size', 0.0, 5000.0, 1.0)
    lift_available = st.selectbox('Lift Available', ["True", "False"])
    lift_dict = {'True': 0, 'False': 1}
    PB = st.selectbox('PB', ["True", "False"])
    PB_dict = {'True': 0, 'False': 1}
    
st.write('')
st.write('')
col1,col2 = st.columns(2)
with col2:
    submit = st.button(label='Submit')
st.write('')

if submit :
    try :
        user_data = np.array([[type_dict[type],lift_dict[lift_available],parking_dict[parking],bathroom,cupboard,building_type_dict[building_type],balcony,property_size,lift_no,PB_dict[PB]]])
        output = loaded_model.predict(user_data)
        
        st.info(f"🏠The predicted House Rent is :🔖{output[0]}")
    except:
        st.write('Something went wrong')
        
