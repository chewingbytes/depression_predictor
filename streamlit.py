import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

st.write("""
# Mental Health Prediction App
This app predicts the mental health status based on user input!
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
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

# Assuming you have a trained RandomForest model with the right columns
# (You would train your model on the data similar to the input features here)
# For example purposes, we'll use random data for this part:

GBR = GradientBoostingRegressor(
    n_estimators=3000, 
    min_samples_leaf=5, 
    max_features=0.3, 
    max_depth=4, 
    loss='huber', 
    learning_rate=0.01
)

# Mock training with dummy data (replace this with actual model training)
X_dummy = pd.DataFrame({
    'Age': [20, 22, 24, 26],
    'Academic Pressure': [5.0, 4.0, 6.0, 7.0],
    'CGPA': [3.5, 3.8, 3.0, 3.9],
    'Study Satisfaction': [7.0, 6.0, 8.0, 5.0],
    'Work/Study Hours': [10, 12, 8, 14],
    'Financial Stress': [6.0, 5.0, 7.0, 8.0],
    'Dietary Habits_Healthy': [1, 0, 1, 1],
    'Dietary Habits_Unhealthy': [0, 1, 0, 0],
    'Have you ever had suicidal thoughts ?_No': [1, 1, 0, 1],
    'Have you ever had suicidal thoughts ?_Yes': [0, 0, 1, 0]
})

Y_dummy = [0, 1, 0, 1]  # Dummy target for training (adjust based on actual target)

GBR.fit(X_dummy, Y_dummy)

# Make prediction based on the user input
prediction = GBR.predict(df)
prediction_proba = GBR.predict_proba(df)

st.subheader('Prediction')
st.write(f"Predicted Class: {prediction[0]}")  # Output the predicted class

st.subheader('Prediction Probability')
st.write(f"Prediction Probability: {prediction_proba[0]}")  # Display probabilities for each class
