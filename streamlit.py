import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("trained_model.pk1")

st.write("""
Are you depressed?
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 22)  # Adjust based on range of your age data
    academic_pressure = st.sidebar.slider('Academic Pressure', 0.0, 10.0, 5.0)
    cgpa = st.sidebar.slider('CGPA', 0.0, 10.0, 3.0)
    study_satisfaction = st.sidebar.slider('Study Satisfaction', 0.0, 10.0, 5.0)
    work_study_hours = st.sidebar.slider('Work/Study Hours', 0.0, 24.0, 8.0)
    financial_stress = st.sidebar.slider('Financial Stress', 0.0, 10.0, 5.0)
    dietary_habits_healthy = st.sidebar.selectbox('Dietary Habits - Healthy', [0, 1], index=1)
    dietary_habits_unhealthy = st.sidebar.selectbox('Dietary Habits - Unhealthy', [0, 1], index=0)
    suicidal_thoughts_no = st.sidebar.selectbox('Have you ever had suicidal thoughts? - No', [0, 1], index=1)
    suicidal_thoughts_yes = st.sidebar.selectbox('Have you ever had suicidal thoughts? - Yes', [0, 1], index=0)
    
    # Create the data dictionary for display purposes
    data = {
        'Age': age,
        'Academic Pressure': academic_pressure,
        'CGPA': cgpa,
        'Study Satisfaction': study_satisfaction,
        'Work/Study Hours': work_study_hours,
        'Financial Stress': financial_stress,
        'Dietary Habits_Healthy': dietary_habits_healthy,
        'Dietary Habits_Unhealthy': dietary_habits_unhealthy,
        'Have you ever had suicidal thoughts ?_No': suicidal_thoughts_no,
        'Have you ever had suicidal thoughts ?_Yes': suicidal_thoughts_yes
    }

    # Create input_features as a flat list of all the input values
    input_features = [
        age, 
        academic_pressure, 
        cgpa, 
        study_satisfaction, 
        work_study_hours, 
        financial_stress, 
        dietary_habits_healthy, 
        dietary_habits_unhealthy, 
        suicidal_thoughts_no, 
        suicidal_thoughts_yes
    ]
    
    features = pd.DataFrame(data, index=[0])
    
    return data, input_features

# Get user input data and features
data, input_features = user_input_features()

st.subheader('User Input Parameters')
st.write(pd.DataFrame([data]))

if st.button("Predict Depression Likelihood"):
    prediction = model.predict([input_features])[0]
    st.subheader('Prediction')
    st.success(f"Predicted Depression Likelihood: {prediction:.2f}")
