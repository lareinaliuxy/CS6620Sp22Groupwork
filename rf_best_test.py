"""
To run this app, in your terminal:
> streamlit run streamlit_demo.py
Source: https://is.gd/SobJvL
"""

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from PIL import Image

clf = joblib.load('./joblib/rf-best-prediction-model.joblib')

# Create title and sidebar
st.title("Disease Prediction App")
st.sidebar.title("Features")

# Intializing parameter values
BMI_mean = 28.325399
BMI_std = 6.356100
PhysicalHealth_mean = 3.37171
PhysicalHealth_std = 7.95085
MentalHealth_mean = 3.898366
MentalHealth_std = 7.955235
SleepTime_mean = 7.097075
SleepTime_std = 1.436007

correlation = Image.open('./joblib/correlation.png')

parameter_list = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'Smoking_No', 'Smoking_Yes', 'AlcoholDrinking_No', 'AlcoholDrinking_Yes', 'Stroke_No', 'Stroke_Yes', 'DiffWalking_No', 'DiffWalking_Yes', 'Sex_Female', 'Sex_Male', 'AgeCategory_18-24', 'AgeCategory_25-29', 'AgeCategory_30-34', 'AgeCategory_35-39', 'AgeCategory_40-44', 'AgeCategory_45-49', 'AgeCategory_50-54', 'AgeCategory_55-59', 'AgeCategory_60-64', 'AgeCategory_65-69', 'AgeCategory_70-74', 'AgeCategory_75-79',
                  'AgeCategory_80 or older', 'Race_American Indian/Alaskan Native', 'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other', 'Race_White', 'Diabetic_No', 'Diabetic_No, borderline diabetes', 'Diabetic_Yes', 'Diabetic_Yes (during pregnancy)', 'PhysicalActivity_No', 'PhysicalActivity_Yes', 'GenHealth_Excellent', 'GenHealth_Fair', 'GenHealth_Good', 'GenHealth_Poor', 'GenHealth_Very good', 'Asthma_No', 'Asthma_Yes', 'KidneyDisease_No', 'KidneyDisease_Yes', 'SkinCancer_No', 'SkinCancer_Yes']
parameter_type = ['float64', 'float64', 'float64', 'float64', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8',
                  'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8', 'uint8']
parameter_input_values = []
parameter_default_values = [0] * 50
source = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex',
          'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer']


def load_data():
    data = pd.read_csv("heart_2020_cleaned.csv")
    return data


df = load_data()
if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Heart Disease Data Set (Classification)")
    st.write(df)
    st.markdown(
        "The dataset contains 18 variables (9 booleans, 5 strings and 4 decimals)")
    st.markdown("In machine learning projects, HeartDisease can be used as the explonatory variable, but note that the classes are heavily unbalanced.")

if st.sidebar.checkbox("Show correlation chart", False):
    st.subheader("Heart Disease Correlation")
    st.image(correlation)

option1 = st.sidebar.number_input("BMI", 0.01, 100.0, step=0.01)
option2 = st.sidebar.number_input("PhysicalHealth", 0, 30, step=1)
option3 = st.sidebar.number_input("MentalHealth", 0, 30, step=1)
option4 = st.sidebar.number_input("SleepTime", 0, 30, step=1)

option5 = st.sidebar.selectbox(
    'Smoking',
    ('Yes', 'No'))


option6 = st.sidebar.selectbox(
    'AlcoholDrinking',
    ('Yes', 'No'))

option7 = st.sidebar.selectbox(
    'Stroke',
    ('No', 'Yes'))

option8 = st.sidebar.selectbox(
    'DiffWalking',
    ('No', 'Yes'))

option9 = st.sidebar.selectbox(
    'Sex',
    ('Female', 'Male'))


option10 = st.sidebar.selectbox(
    'AgeCategory',
    ('18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'))

option11 = st.sidebar.selectbox(
    'Race',
    ('American Indian/Alaskan Native', 'Asian', 'Black', 'Hispanic', 'Other', 'White'))

option12 = st.sidebar.selectbox(
    'Diabetic',
    ('No', 'No, borderline diabetes', 'Yes', 'Yes (during pregnancy)'))

option13 = st.sidebar.selectbox(
    'PhysicalActivity',
    ('No', 'Yes'))

option14 = st.sidebar.selectbox(
    'GenHealth',
    ('Excellent', 'Fair', 'Good', 'Poor', 'Very good'))

option15 = st.sidebar.selectbox(
    'Asthma',
    ('No', 'Yes'))

option16 = st.sidebar.selectbox(
    'KidneyDisease',
    ('No', 'Yes'))
option17 = st.sidebar.selectbox(
    'SkinCancer',
    ('No', 'Yes'))

input_variables = pd.DataFrame(
    [parameter_default_values], columns=parameter_list, dtype=float)
input_variables['BMI'] = (option1 - BMI_mean) / BMI_std
input_variables['PhysicalHealth'] = (
    option2 - PhysicalHealth_mean) / PhysicalHealth_std
input_variables['MentalHealth'] = (
    option3 - MentalHealth_mean) / MentalHealth_std
input_variables['SleepTime'] = (option4 - SleepTime_mean) / SleepTime_std

# y = (x – mean) / standard_deviation

if option5 == 'Yes':
    input_variables['Smoking_Yes'] = 1
elif option5 == 'No':
    input_variables['Smoking_No'] = 1

if option6 == 'Yes':
    input_variables['AlcoholDrinking_Yes'] = 1
elif option6 == 'No':
    input_variables['AlcoholDrinking_No'] = 1

if option7 == 'Yes':
    input_variables['Stroke_Yes'] = 1
elif option7 == 'No':
    input_variables['Stroke_No'] = 1

if option8 == 'Yes':
    input_variables['DiffWalking_Yes'] = 1
elif option8 == 'No':
    input_variables['DiffWalking_No'] = 1

if option9 == 'Female':
    input_variables['Sex_Female'] = 1
elif option9 == 'Male':
    input_variables['Sex_Male'] = 1

if option10 == '18-24':
    input_variables['AgeCategory_18-24'] = 1
elif option10 == '25-29':
    input_variables['AgeCategory_25-29'] = 1
elif option10 == '30-34':
    input_variables['AgeCategory_30-34'] = 1
elif option10 == '35-39':
    input_variables['AgeCategory_35-39'] = 1
elif option10 == '40-44':
    input_variables['AgeCategory_40-44'] = 1
elif option10 == '45-49':
    input_variables['AgeCategory_45-49'] = 1
elif option10 == '50-54':
    input_variables['AgeCategory_50-54'] = 1
elif option10 == '55-59':
    input_variables['AgeCategory_55-59'] = 1
elif option10 == '60-64':
    input_variables['AgeCategory_60-64'] = 1
elif option10 == '65-69':
    input_variables['AgeCategory_65-69'] = 1
elif option10 == '70-74':
    input_variables['AgeCategory_70-74'] = 1
elif option10 == '75-79':
    input_variables['AgeCategory_75-79'] = 1
elif option10 == '80 or older':
    input_variables['AgeCategory_80 or older'] = 1

if option11 == 'American Indian/Alaskan Native':
    input_variables['Race_American Indian/Alaskan Native'] = 1
elif option11 == 'Asian':
    input_variables['Race_Asian'] = 1
elif option11 == 'Black':
    input_variables['Race_Black'] = 1
elif option11 == 'Hispanic':
    input_variables['Race_Hispanic'] = 1
elif option11 == 'Other':
    input_variables['Race_Other'] = 1
elif option11 == 'White':
    input_variables['Race_White'] = 1

if option12 == 'No':
    input_variables['Diabetic_No'] = 1
elif option12 == 'No, borderline diabetes':
    input_variables['Diabetic_No, borderline diabetes'] = 1
elif option12 == 'Yes':
    input_variables['Diabetic_Yes'] = 1
elif option12 == 'Yes (during pregnancy)':
    input_variables['Diabetic_Yes (during pregnancy)'] = 1

if option13 == 'Yes':
    input_variables['PhysicalActivity_Yes'] = 1
elif option13 == 'No':
    input_variables['PhysicalActivity_No'] = 1

elif option14 == 'Excellent':
    input_variables['GenHealth_Excellent'] = 1
elif option14 == 'Fair':
    input_variables['GenHealth_Fair'] = 1
elif option14 == 'Good':
    input_variables['GenHealth_Good'] = 1
elif option14 == 'Poor':
    input_variables['GenHealth_Poor'] = 1
elif option14 == 'Very good':
    input_variables['GenHealth_Very good'] = 1

if option15 == 'Yes':
    input_variables['Asthma_Yes'] = 1
elif option15 == 'No':
    input_variables['Asthma_No'] = 1

if option16 == 'Yes':
    input_variables['KidneyDisease_Yes'] = 1
elif option16 == 'No':
    input_variables['KidneyDisease_No'] = 1

if option17 == 'Yes':
    input_variables['SkinCancer_Yes'] = 1
elif option17 == 'No':
    input_variables['SkinCancer_No'] = 1

print(input_variables['BMI'])
print(input_variables['PhysicalHealth'])


# Button that triggers the actual prediction
if st.button("Click Here to Classify"):
    prediction = clf.predict(input_variables)
    st.write('Heart Disease Prediction $', prediction[0] > 0.5)
